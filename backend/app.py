from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import json 
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

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


# ------------------------------------------------------------------
# --- 6. AI ANALYSIS ROUTE ---
# ------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_results():
    try:
        data = request.get_json(silent=True) or {}
        metrics = data.get('metrics') or {}
        context = data.get('context') or {}  # e.g., filename, target variable

        if not isinstance(metrics, dict) or not metrics:
            return jsonify({'error': 'Invalid payload: metrics object is required'}), 400

        if not isinstance(context, dict):
            return jsonify({'error': 'Invalid payload: context must be an object'}), 400

        filename = context.get('filename', 'Unknown Dataset')
        target = context.get('target', 'Unknown Target')
        algorithm_used = metrics.get('algorithm_used', 'unknown')
        
        prompt = f"""
        You are a Senior Data Scientist. Analyze these ML model results:
        Dataset: {filename}
        Target Variable: {target}
        Algorithm: {algorithm_used}
        Metrics: {json.dumps(metrics)}
        
        Provide a 3-sentence expert summary:
        1. Is the model's performance (Accuracy/R2) good for this dataset?
        2. What do the top 2 features tell us about the data?
        3. One specific recommendation to improve this specific model.
        Keep it professional and technical.
        """

        if client is None:
            feature_imp = metrics.get('feature_importance') or []
            top_two = feature_imp[:2]
            top_text = ', '.join([f"{f.get('feature', 'unknown')} ({f.get('importance', 0):.3f})" for f in top_two]) if top_two else 'feature importance is unavailable for this model.'
            if metrics.get('accuracy') is not None:
                score_text = f"The model achieved accuracy {float(metrics['accuracy']):.3f}, and this should be compared with class distribution and a baseline classifier."
            elif metrics.get('r2') is not None:
                score_text = f"The model achieved R2 {float(metrics['r2']):.3f} with RMSE={float(metrics.get('rmse', 0)):.3f} and MAE={float(metrics.get('mae', 0)):.3f}; validate against domain error tolerance."
            else:
                score_text = "Model metrics are incomplete; validate with cross-validation and baseline comparison."

            return jsonify({
                'analysis': (
                    f"For dataset '{filename}' targeting '{target}' using '{algorithm_used}', {score_text} "
                    f"Top feature signals: {top_text}. "
                    "Recommendation: tune hyperparameters with cross-validation and compare against a simple baseline before deployment."
                ),
                'warning': 'GEMINI_API_KEY not configured. Returned local fallback analysis.'
            }), 200
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",  # Using the latest 2026 model
                contents=prompt
            )
        except Exception as api_error:
            error_text = str(api_error)
            if 'RESOURCE_EXHAUSTED' in error_text or '429' in error_text:
                feature_imp = metrics.get('feature_importance') or []
                top_two = feature_imp[:2]
                top_text = ', '.join([f"{f.get('feature', 'unknown')} ({f.get('importance', 0):.3f})" for f in top_two]) if top_two else 'feature importance is unavailable for this model.'

                score_text = ""
                if metrics.get('accuracy') is not None:
                    score_text = f"The model achieved accuracy {float(metrics['accuracy']):.3f}, which should be interpreted against class balance and baseline performance."
                elif metrics.get('r2') is not None:
                    score_text = f"The model achieved R2 {float(metrics['r2']):.3f}, and error metrics RMSE={float(metrics.get('rmse', 0)):.3f}, MAE={float(metrics.get('mae', 0)):.3f} should be evaluated against domain tolerance."
                else:
                    score_text = "The available metrics suggest reviewing validation strategy and a baseline model before deployment."

                fallback_analysis = (
                    f"For dataset '{filename}' targeting '{target}' with algorithm '{algorithm_used}', {score_text} "
                    f"Top feature signals: {top_text}. "
                    f"Recommendation: run cross-validation with hyperparameter tuning and compare against a simple baseline to confirm generalization."
                )

                return jsonify({
                    'analysis': fallback_analysis,
                    'warning': 'Gemini quota exceeded (429 RESOURCE_EXHAUSTED). Returned local fallback analysis.'
                }), 200
            raise

        analysis_text = getattr(response, 'text', None)
        if not analysis_text and getattr(response, 'candidates', None):
            try:
                analysis_text = response.candidates[0].content.parts[0].text
            except Exception:
                analysis_text = None

        if not analysis_text:
            return jsonify({'error': 'No analysis text returned from Gemini API'}), 502

        return jsonify({'analysis': analysis_text})
    except Exception as e:
        app.logger.exception('Error in /analyze route')
        return jsonify({'error': f'Analyze failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)