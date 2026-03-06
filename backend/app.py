from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib 

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

# Algorithms
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 1. UPLOAD ROUTE ---
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
            
            # --- ADVANCED: AI Missing Values Logic ---
            missing_counts = df.isnull().sum()
            max_missing = int(missing_counts.max())
            
            missing_percentage = (max_missing / rows) * 100 if rows > 0 else 0
            
            if missing_percentage == 0 or missing_percentage < 5:
                recommendation = "drop"
            else:
                worst_column = missing_counts.idxmax()
                col_type = df[worst_column].dtype
                
                if col_type == 'object':
                    recommendation = "mode" 
                else:
                    skewness = df[worst_column].skew()
                    if abs(skewness) > 1:
                        recommendation = "median" 
                    else:
                        recommendation = "mean"   
            # -----------------------------------------
            
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


# --- 2. TRAINING ROUTE ---
@app.route('/train', methods=['POST'])
def train_model():
    try:
        config = request.json
        filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Dataset not found. Please upload first.'}), 400
            
        df = pd.read_csv(filepath)
        
        target = config.get('target')
        if target not in df.columns:
            return jsonify({'error': f'Target column "{target}" not found'}), 400

        missing_strategy = config.get('preprocessing', {}).get('missing_value_strategy', 'drop')
        
        if missing_strategy == 'drop':
            df = df.dropna()
        else:
            sklearn_strategy = 'most_frequent' if missing_strategy == 'mode' else missing_strategy
            
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns

            if sklearn_strategy in ['mean', 'median']:
                if len(num_cols) > 0:
                    num_imputer = SimpleImputer(strategy=sklearn_strategy)
                    df[num_cols] = num_imputer.fit_transform(df[num_cols])
                if len(cat_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent') 
                    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            elif sklearn_strategy == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        X = df.drop(columns=[target])
        y = df[target]
        
        X = X.select_dtypes(include=['number'])
        
        algorithm = config.get('algorithm', 'random_forest_classifier')
        
        # --- AUTO-DETECT TASK TYPE ---
        if 'regressor' in algorithm or 'regression' in algorithm and algorithm != 'logistic_regression':
            task_type = 'regression'
        elif algorithm == 'svr':
            task_type = 'regression'
        else:
            task_type = 'classification'
            
        if task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        split_ratio = config.get('preprocessing', {}).get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

        scaling = config.get('preprocessing', {}).get('scaling', 'none')
        if scaling == 'standard':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = None
        
        if algorithm == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
        elif algorithm == 'random_forest_classifier':
            model = RandomForestClassifier(n_estimators=50)
        elif algorithm == 'svm':
            model = SVC()
        elif algorithm == 'linear_regression':
            model = LinearRegression()
        elif algorithm == 'random_forest_regressor':
            model = RandomForestRegressor(n_estimators=50)
        elif algorithm == 'svr':
            model = SVR()
        else:
            return jsonify({'error': 'Algorithm not supported'}), 400
            
        # --- TRAIN AND SAVE MODEL ---
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Save the brain!
        model_path = os.path.join(UPLOAD_FOLDER, 'trained_model.pkl')
        joblib.dump(model, model_path)
        # ----------------------------
        
        metrics = {}
        
        # --- FEATURE IMPORTANCE (Moved outside so it runs for both!) ---
        feature_importance = []
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, col in enumerate(X.columns):
                    feature_importance.append({"feature": col, "importance": float(importances[i])})
            elif hasattr(model, 'coef_'):
                importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                for i, col in enumerate(X.columns):
                    feature_importance.append({"feature": col, "importance": float(abs(importances[i]))})
            
            feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
            metrics['feature_importance'] = feature_importance
        except Exception as e:
            print(f"Could not calculate feature importance: {e}")
            metrics['feature_importance'] = None
        # -------------------------------------------------------------
        
        if task_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, predictions))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            metrics['confusion_matrix'] = cm.tolist() 
            
            if 'le' in locals():
                metrics['classes'] = [str(c) for c in le.classes_]
            else:
                metrics['classes'] = [str(c) for c in pd.Series(y).unique()]
                
        else:
            # Safe Edge Case: Regression (Fixed Scikit-Learn 1.4 crash)
            mse = mean_squared_error(y_test, predictions)
            metrics['rmse'] = float(mse ** 0.5) 
            metrics['r2'] = float(r2_score(y_test, predictions))
            
        return jsonify({'status': 'success', 'metrics': metrics})
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        return jsonify({'error': str(e)}), 500

# --- 3. DOWNLOAD ROUTE ---
@app.route('/download', methods=['GET'])
def download_model():
    try:
        model_path = os.path.join(UPLOAD_FOLDER, 'trained_model.pkl')
        if os.path.exists(model_path):
            # Send the file to the user's browser as an attachment!
            return send_file(model_path, as_attachment=True, download_name="ml_studio_model.pkl")
        else:
            return jsonify({'error': 'No trained model found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)