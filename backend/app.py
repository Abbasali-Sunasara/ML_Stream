from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Algorithms
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 1. UPLOAD ROUTE (UPDATED WITH AI LOGIC) ---
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
            
            # Keep your existing logic for the React Data Preview
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
                    recommendation = "mode" # Text data gets Mode
                else:
                    # Numeric data: Check for outliers using skewness
                    skewness = df[worst_column].skew()
                    if abs(skewness) > 1:
                        recommendation = "median" # High outliers get Median
                    else:
                        recommendation = "mean"   # Normal distribution gets Mean
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
                'missing_recommendation': recommendation # <--- Sending the AI choice to React!
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# --- 2. TRAINING ROUTE ---
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

        # --- 1. HANDLE MISSING VALUES (Fully Implemented) ---
        missing_strategy = config.get('preprocessing', {}).get('missing_value_strategy', 'drop')
        
        if missing_strategy == 'drop':
            df = df.dropna()
        else:
            # Scikit-Learn calls 'mode' -> 'most_frequent'
            sklearn_strategy = 'most_frequent' if missing_strategy == 'mode' else missing_strategy
            
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns

            # Apply Imputer
            if sklearn_strategy in ['mean', 'median']:
                if len(num_cols) > 0:
                    num_imputer = SimpleImputer(strategy=sklearn_strategy)
                    df[num_cols] = num_imputer.fit_transform(df[num_cols])
                if len(cat_cols) > 0:
                    # Text columns can't have a 'mean', so fallback to mode for text
                    cat_imputer = SimpleImputer(strategy='most_frequent') 
                    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            elif sklearn_strategy == 'most_frequent':
                # Mode works on both numbers and text
                imputer = SimpleImputer(strategy='most_frequent')
                df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # --- 2. SPLIT FEATURES AND TARGET ---
        X = df.drop(columns=[target])
        y = df[target]
        
        # For Phase 5, we only train on numerical columns
        X = X.select_dtypes(include=['number'])
        
        # --- 3. ENCODE TARGET FOR CLASSIFICATION ---
        task_type = config.get('task_type', 'classification')
        if task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # --- 4. TRAIN / TEST SPLIT ---
        split_ratio = config.get('preprocessing', {}).get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

        # --- 5. SCALING ---
        scaling = config.get('preprocessing', {}).get('scaling', 'none')
        if scaling == 'standard':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # --- 6. DYNAMIC ALGORITHM SELECTION ---
        algorithm = config.get('algorithm', 'random_forest_classifier') # Read what React sent!
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
            
        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # --- 7. CALCULATE METRICS ---
        metrics = {}
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, predictions)
        else:
            metrics['rmse'] = mean_squared_error(y_test, predictions, squared=False)
            metrics['r2'] = r2_score(y_test, predictions)

        return jsonify({'status': 'success', 'metrics': metrics})

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Prints the exact error line in your Mac terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)