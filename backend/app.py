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
from imblearn.over_sampling import SMOTE

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
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold

# Algorithms
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# --- FRONTEND SERVING ---
import os.path as osp
frontend_build_path = osp.join(osp.dirname(__file__), '../frontend/dist')

app = Flask(__name__, static_folder=frontend_build_path, static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def parse_params(algo, sent_params):
    if not sent_params:
        return {}
    parsed = {}
    for key, value in sent_params.items():
        try:
            # Handle Integers (most sliders)
            if key in ['n_estimators', 'max_depth', 'n_neighbors', 'max_iter', 'min_samples_split']:
                parsed[key] = int(value)
            # Handle Floats
            elif key in ['learning_rate', 'C', 'gamma', 'var_smoothing', 'epsilon']:
                parsed[key] = float(value)
            # Handle Booleans from dropdowns
            elif value in ['True', 'False']:
                parsed[key] = (value == 'True')
            # Handle Strings (Kernels/Weights)
            else:
                parsed[key] = value
        except:
            continue
    return parsed


def build_model(algorithm, is_regression, hp_parsed):
    if is_regression:
        if algorithm == 'linear_regression':
            return LinearRegression(**{k: v for k, v in hp_parsed.items() if k in ['fit_intercept', 'copy_X', 'n_jobs']})
        if algorithm == 'random_forest':
            return RandomForestRegressor(**{k: v for k, v in hp_parsed.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']})
        if algorithm == 'decision_tree':
            return DecisionTreeRegressor(**{k: v for k, v in hp_parsed.items() if k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'random_state']})
        if algorithm == 'knn':
            return KNeighborsRegressor(**{k: v for k, v in hp_parsed.items() if k in ['n_neighbors', 'weights', 'algorithm', 'p']})
        if algorithm == 'adaboost':
            return AdaBoostRegressor(**{k: v for k, v in hp_parsed.items() if k in ['n_estimators', 'learning_rate', 'loss', 'random_state']})
        if algorithm == 'xgboost':
            return HistGradientBoostingRegressor(**{k: v for k, v in hp_parsed.items() if k in ['learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf', 'l2_regularization', 'random_state']})
        if algorithm == 'svr':
            return SVR(**{k: v for k, v in hp_parsed.items() if k in ['C', 'kernel', 'gamma', 'epsilon', 'degree', 'coef0']})
        return RandomForestRegressor(**{k: v for k, v in hp_parsed.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']})

    if algorithm == 'logistic_regression':
        filtered = {k: v for k, v in hp_parsed.items() if k in ['C', 'penalty', 'solver', 'fit_intercept', 'max_iter', 'class_weight', 'random_state']}
        filtered['max_iter'] = filtered.get('max_iter', 1000)
        return LogisticRegression(**filtered)
    if algorithm == 'random_forest':
        return RandomForestClassifier(**{k: v for k, v in hp_parsed.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']})
    if algorithm == 'decision_tree':
        return DecisionTreeClassifier(**{k: v for k, v in hp_parsed.items() if k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'random_state']})
    if algorithm == 'knn':
        return KNeighborsClassifier(**{k: v for k, v in hp_parsed.items() if k in ['n_neighbors', 'weights', 'algorithm', 'p']})
    if algorithm == 'adaboost':
        return AdaBoostClassifier(**{k: v for k, v in hp_parsed.items() if k in ['n_estimators', 'learning_rate', 'algorithm', 'random_state']})
    if algorithm == 'xgboost':
        return HistGradientBoostingClassifier(**{k: v for k, v in hp_parsed.items() if k in ['learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf', 'l2_regularization', 'random_state']})
    if algorithm == 'svm':
        filtered = {k: v for k, v in hp_parsed.items() if k in ['C', 'kernel', 'gamma', 'degree', 'coef0']}
        filtered['probability'] = hp_parsed.get('probability', True)
        return SVC(**filtered)
    return RandomForestClassifier(**{k: v for k, v in hp_parsed.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap', 'random_state']})


# These are the "Search Ranges" the AI will explore
PARAM_GRIDS = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10]
    },
    'decision_tree': {
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 11, 15, 21, 25],
        'weights': ['uniform', 'distance']
    },
    'adaboost': {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0]
    },
    'xgboost': {
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'max_iter': [50, 100, 200, 300]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'svr': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    },
    'logistic_regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2'] # Keeping l2 for stability across solvers
    },
    'linear_regression': {
        'fit_intercept': [True, False]
    }
}

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
        hp = config.get('hyperparameters')
        filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')
        df = pd.read_csv(filepath)
        target = config.get('target')
        algorithm = config.get('algorithm', 'random_forest')
        hp_parsed = parse_params(algorithm, hp)

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

        # --- NEW: DATA HEALING LOGIC ---
        healing_config = config.get('preprocessing', {}).get('healing', {})
        
        # 1. Outlier Handling (IQR Method)
        if healing_config.get('outliers') == 'remove':
            for col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Filtering rows
                X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
                y = y[X.index] # Keep y in sync

        # 2. Imbalance Handling (SMOTE)
        if not is_regression and healing_config.get('imbalance') == 'smote':
            # Check if we have enough samples to run SMOTE
            if y.value_counts().min() > 1:
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaling = config.get('preprocessing', {}).get('scaling', 'none')
        if scaling != 'none':
            scaler = StandardScaler() if scaling == 'standard' else MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = build_model(algorithm, is_regression, hp_parsed)
        
        # Check if Auto-Tune is requested
        is_auto_tune = config.get('auto_tune', False)
        best_params = hp_parsed
        cv_score = None

        if is_auto_tune and algorithm in PARAM_GRIDS:
            # 1. Handle Small Dataset Edge Case (Adjust Folds Safely)
            if is_regression:
                n_samples = X_train.shape[0]
                n_folds = min(5, n_samples) if n_samples >= 2 else 2
                cv_strategy = KFold(n_splits=max(2, n_folds), shuffle=True, random_state=42)
            else:
                min_class_count = int(pd.Series(y_train).value_counts().min()) if len(y_train) else 0
                if min_class_count < 2:
                    # Not enough samples per class for CV search; fall back to a normal fit.
                    model.fit(X_train, y_train)
                    search = None
                else:
                    n_folds = min(5, min_class_count)
                    cv_strategy = StratifiedKFold(n_splits=max(2, n_folds), shuffle=True, random_state=42)
            
            if 'search' not in locals() or search is not None:
                # 2. Setup Randomized Search
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=PARAM_GRIDS[algorithm],
                    n_iter=int(config.get('search_intensity', 10)),
                    cv=cv_strategy,
                    scoring='r2' if is_regression else 'accuracy',
                    n_jobs=-1, # Use all CPU cores for speed
                    random_state=42
                )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                # Record the best params found to send back to UI
                best_params = search.best_params_
                cv_score = float(search.best_score_)
        else:
            # Standard Manual Fit
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

        # --- ADD TRAIN/TEST HEALTH COMPARISON ---
        results_data = {'metrics': metrics}

        # 1. Calculate scores for both buckets
        if is_regression:
            train_score = model.score(X_train, y_train)  # Training R2
            test_score = results_data['metrics']['r2']   # Test R2 (already calculated)
            metric_name = "R² Score"
        else:
            train_score = model.score(X_train, y_train)  # Training Accuracy
            test_score = results_data['metrics']['accuracy'] # Test Accuracy
            metric_name = "Accuracy"

        # 2. Logic to detect health status
        gap = train_score - test_score
        health_status = "Stable"
        health_msg = "The model generalizes well and is ready for real-world use."
        color_code = "green"

        if gap > 0.15:  # If the gap is more than 15%, it's overfitting
            health_status = "Overfitting"
            health_msg = f"High {metric_name} on training but low on testing. The model is memorizing noise. Try increasing 'Min Samples to Split' or reducing 'Max Depth'."
            color_code = "red"
        elif test_score < 0.5:  # If both scores are bad
            health_status = "Underfitting"
            health_msg = f"The model is too simple to find patterns. Try increasing 'Estimators' or choosing a more complex algorithm."
            color_code = "yellow"

        # 3. Add to your results JSON
        results_data['model_health'] = {
            'status': health_status,
            'message': health_msg,
            'train_score': round(train_score, 4),
            'gap': round(gap, 4),
            'color': color_code
        }
            
        return jsonify({
            'status': 'success', 
            'metrics': metrics,
            'best_params': best_params,
            'cv_score': cv_score,
            'is_auto_tuned': is_auto_tune,
            'model_health': results_data['model_health']
        })
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


# ------------------------------------------------------------------
# --- 7. CATCH-ALL: SERVE FRONTEND (React SPA) ---
# ------------------------------------------------------------------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the React frontend for all non-API routes (SPA catch-all)."""
    if path and osp.exists(osp.join(app.static_folder, path)):
        return app.send_static_file(path)
    # For SPA, serve index.html for all non-file routes
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


if __name__ == '__main__':
    app.run(debug=True, port=5000)