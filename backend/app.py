from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 1. UPLOAD ROUTE (UPDATED) ---
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
            
            # NEW: Calculate Missing Values
            missing_values = df.isnull().sum().to_dict()
            
            # Create a safe preview
            preview = df.head(5).where(pd.notnull(df), None).to_dict(orient='records')
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': file.filename,
                'columns': df.shape[1],
                'rows': df.shape[0],
                'column_names': df.columns.tolist(),
                'dtypes': {k: str(v) for k, v in df.dtypes.items()},
                'missing_values': missing_values, # <--- SENDING THIS TO REACT
                'preview': preview
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# --- 2. TRAINING ROUTE (Unchanged) ---
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

        # Handle Missing Values
        missing_strategy = config.get('preprocessing', {}).get('missing_value_strategy', 'drop')
        if missing_strategy == 'drop':
            df = df.dropna()
        else:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                imputer = SimpleImputer(strategy=missing_strategy)
                df[num_cols] = imputer.fit_transform(df[num_cols])

        X = df.drop(columns=[target])
        y = df[target]
        
        X = X.select_dtypes(include=['number'])
        
        task_type = config.get('task_type', 'classification')
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
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=50)
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, predictions)
        else:
            metrics['rmse'] = mean_squared_error(y_test, predictions, squared=False)
            metrics['r2'] = r2_score(y_test, predictions)

        return jsonify({'status': 'success', 'metrics': metrics})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)