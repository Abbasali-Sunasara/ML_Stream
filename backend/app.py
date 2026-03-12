from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib 

# --- Plotly Charting Library ---
import plotly.express as px

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, mean_absolute_error

# Algorithms
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------------------------------------------------
# --- 1. UPLOAD ROUTE (Complete with Data Health Logic) ---
# ------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            rows = df.shape[0]
            
            missing_values = df.isnull().sum().to_dict()
            preview = df.head(5).where(pd.notnull(df), None).to_dict(orient='records')
            
            missing_counts = df.isnull().sum()
            max_missing = int(missing_counts.max())
            missing_percentage = (max_missing / rows) * 100 if rows > 0 else 0
            
            if missing_percentage == 0 or missing_percentage < 5:
                recommendation = "drop"
            else:
                worst_column = missing_counts.idxmax()
                col_type = df[worst_column].dtype
                recommendation = "mode" if col_type == 'object' else ("median" if abs(df[worst_column].skew()) > 1 else "mean")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': file.filename,
                'columns': df.shape[1],
                'rows': rows,
                'column_names': df.columns.tolist(),
                'dtypes': {k: str(v) for k, v in df.dtypes.items()},
                'missing_values': missing_values, 
                'preview': preview,
                'missing_recommendation': recommendation 
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
# --- 2. PLOTLY CHARTING ENGINE ---
# ------------------------------------------------------------------
@app.route('/plot', methods=['POST'])
def generate_plot():
    try:
        data = request.json
        chart_type, x_col, y_col = data.get('type'), data.get('x'), data.get('y')
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'train.csv'))
        fig = None

        if chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, template="plotly_dark", color_discrete_sequence=['#22d3ee'])
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, template="plotly_dark", color_discrete_sequence=['#8b5cf6'])
        elif chart_type == 'box':
            fig = px.box(df, y=x_col, template="plotly_dark", color_discrete_sequence=['#6366f1'])
        elif chart_type == 'bar':
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, 'count']
            fig = px.bar(counts, x=x_col, y='count', template="plotly_dark", color_discrete_sequence=['#8b5cf6'])
        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr().fillna(0).round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto", template="plotly_dark", color_continuous_scale='Viridis')

        if fig:
            return app.response_class(fig.to_json(), mimetype='application/json')
        return jsonify({'error': 'Invalid chart type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
# --- 3. THE MODEL ARENA ---
# ------------------------------------------------------------------
@app.route('/train', methods=['POST'])
def train_model():
    try:
        config = request.json
        filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')
        df = pd.read_csv(filepath)
        target = config.get('target')

        strategy = config.get('preprocessing', {}).get('missing_value_strategy', 'drop')
        if strategy == 'drop':
            df = df.dropna()
        else:
            num_cols = df.select_dtypes(include=['number']).columns
            df[num_cols] = SimpleImputer(strategy='mean' if strategy=='mean' else 'median').fit_transform(df[num_cols])

        X = df.drop(columns=[target]).select_dtypes(include=['number'])
        y = df[target]

        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 15
        if not is_regression and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaling = config.get('preprocessing', {}).get('scaling', 'none')
        if scaling != 'none':
            scaler = StandardScaler() if scaling == 'standard' else MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        algorithm = config.get('algorithm', 'random_forest')
        
        if is_regression:
            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100),
                'xgboost': HistGradientBoostingRegressor(),
                'knn': KNeighborsRegressor(n_neighbors=5),
                'svr': SVR()
            }
        else:
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100),
                'xgboost': HistGradientBoostingClassifier(),
                'knn': KNeighborsClassifier(n_neighbors=5),
                'svm': SVC(probability=True)
            }

        model = models.get(algorithm, models['random_forest'])
        model.fit(X_train, y_train)
        
        joblib.dump(model, os.path.join(UPLOAD_FOLDER, 'trained_model.pkl'))
        
        predictions = model.predict(X_test)
        metrics = {'algorithm_used': algorithm}
        
        try:
            if hasattr(model, 'feature_importances_'):
                imp = [{"feature": col, "importance": float(v)} for col, v in zip(X.columns, model.feature_importances_)]
                metrics['feature_importance'] = sorted(imp, key=lambda x: x['importance'], reverse=True)
            elif hasattr(model, 'coef_'):
                coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                imp = [{"feature": col, "importance": float(abs(v))} for col, v in zip(X.columns, coefs)]
                metrics['feature_importance'] = sorted(imp, key=lambda x: x['importance'], reverse=True)
        except:
            metrics['feature_importance'] = None
        
        if not is_regression:
            metrics['accuracy'] = float(accuracy_score(y_test, predictions))
            metrics['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()
            metrics['classes'] = [str(c) for c in (le.classes_ if 'le' in locals() else np.unique(y))]
        else:
            metrics['r2'] = float(r2_score(y_test, predictions))
            metrics['rmse'] = float(mean_squared_error(y_test, predictions) ** 0.5)
            metrics['mae'] = float(mean_absolute_error(y_test, predictions))
            
        return jsonify({'status': 'success', 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
# --- 5. UTILITY ROUTES ---
# ------------------------------------------------------------------
@app.route('/download', methods=['GET'])
def download_model():
    path = os.path.join(UPLOAD_FOLDER, 'trained_model.pkl')
    return send_file(path, as_attachment=True) if os.path.exists(path) else (jsonify({'error': 'No model'}), 404)

if __name__ == '__main__':
    app.run(debug=True, port=5000)