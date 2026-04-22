"""
Flask Web Application
Simple web interface for network traffic classification
"""

import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from src.predict import predict_from_csv

app = Flask(__name__)

METRIC_DROP_COLS = [
    "difficulty",
    "num_outbound_cmds",
    "is_host_login",
    "su_attempted",
    "urgent",
    "land",
    "num_access_files",
    "num_shells",
    "root_shell",
    "num_file_creations",
    "num_failed_logins",
    "num_root",
]

MODEL_DASHBOARD_SOURCES = {
    "xgboost": {
        "display_name": "XGBoost",
        "color": "#2f80ed",
        "model_paths": [
            os.path.join("model", "xgboost", "xgboost_regularised.pkl"),
            os.path.join("model", "xgboost.pkl"),
        ],
        "prediction_paths": {
            "acc_test_plus": [
                os.path.join("data", "test", "xgboost", "KDDTest+_predictions.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest+_predictions_xgboost.csv"),
            ],
            "acc_test_21": [
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions_xgboost.csv"),
            ],
            "attack_recall_21": [
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions_xgboost.csv"),
            ],
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "color": "#4c9b1d",
        "model_paths": [
            os.path.join("model", "random_forest.pkl"),
            os.path.join("model", "randomforest.pkl"),
        ],
        "prediction_paths": {
            "acc_test_plus": [
                os.path.join("data", "test", "random_forest", "KDDTest+_predictions.csv"),
            ],
            "acc_test_21": [
                os.path.join("data", "test", "random_forest", "KDDTest-21_predictions.csv"),
            ],
            "attack_recall_21": [
                os.path.join("data", "test", "random_forest", "KDDTest-21_predictions.csv"),
            ],
        },
    },
    "svm": {
        "display_name": "SVM",
        "color": "#a26411",
        "model_paths": [
            os.path.join("model", "svm.pkl"),
        ],
        "prediction_paths": {
            "acc_test_plus": [
                os.path.join("data", "test", "svm", "KDDTest+_predictions.csv"),
            ],
            "acc_test_21": [
                os.path.join("data", "test", "svm", "KDDTest-21_predictions.csv"),
            ],
            "attack_recall_21": [
                os.path.join("data", "test", "svm", "KDDTest-21_predictions.csv"),
            ],
        },
    },
}

# Configure upload folder
UPLOAD_FOLDER = os.path.join('data', 'test')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _first_existing_path(path_list):
    for path in path_list:
        if os.path.exists(path):
            return path
    return None


def _find_column(df, candidates):
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _to_binary_label(value):
    text = str(value).strip().lower()
    return 0 if text == "normal" else 1


def _compute_prediction_file_metrics(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    label_col = _find_column(df, ["label", "actual_label", "class"])
    pred_col = _find_column(df, ["prediction", "pred", "predicted_label"])

    if label_col is None or pred_col is None or df.empty:
        return {}

    y_true = df[label_col].apply(_to_binary_label)
    y_pred = df[pred_col].apply(_to_binary_label)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0) * 100.0
    recall = recall_score(y_true, y_pred, zero_division=0) * 100.0
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0) * 100.0

    return {
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "specificity": round(float(specificity), 2),
        "false_positive_rate": round(float(fpr), 2),
        "false_negative_rate": round(float(fnr), 2),
        "f1_score": round(float(f1), 2),
        "f2_score": round(float(f2), 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_negative": int(tn + fp),
        "support_positive": int(tp + fn),
    }


def _compute_train_val_gap(model_path):
    train_candidates = [
        os.path.join("data", "train", "KDDTrain+.csv"),
        os.path.join("data", "train", "KDDTrain+_20Percent.csv"),
    ]
    train_path = _first_existing_path(train_candidates)

    if train_path is None or not os.path.exists(model_path):
        return None

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(train_path)

        if "label" not in df.columns:
            return None

        df = df.drop(columns=[col for col in METRIC_DROP_COLS if col in df.columns])
        y = df["label"].apply(_to_binary_label)
        X = df.drop(columns=["label"])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        gap = abs(train_acc - val_acc) * 100.0
        return round(float(gap), 2)
    except Exception:
        return None


def _build_dashboard_payload():
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "models": {},
    }

    for key, config in MODEL_DASHBOARD_SOURCES.items():
        metric_values = {
            "acc_test_plus": None,
            "acc_test_21": None,
            "attack_recall_21": None,
            "train_val_gap": None,
            "confusion": {
                "tn": None,
                "fp": None,
                "fn": None,
                "tp": None,
                "precision": None,
                "recall": None,
                "specificity": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "f1_score": None,
                "f2_score": None,
                "accuracy": None,
            },
        }

        for metric_key, file_candidates in config["prediction_paths"].items():
            metric_file = _first_existing_path(file_candidates)
            if metric_file is None:
                continue

            calculated = _compute_prediction_file_metrics(metric_file)
            if metric_key in ["acc_test_plus", "acc_test_21"]:
                metric_values[metric_key] = calculated["accuracy"]
            elif metric_key == "attack_recall_21":
                metric_values[metric_key] = calculated.get("recall")

            if metric_key == "acc_test_21" and calculated:
                metric_values["confusion"] = {
                    "tn": calculated.get("tn"),
                    "fp": calculated.get("fp"),
                    "fn": calculated.get("fn"),
                    "tp": calculated.get("tp"),
                    "precision": calculated.get("precision"),
                    "recall": calculated.get("recall"),
                    "specificity": calculated.get("specificity"),
                    "false_positive_rate": calculated.get("false_positive_rate"),
                    "false_negative_rate": calculated.get("false_negative_rate"),
                    "f1_score": calculated.get("f1_score"),
                    "f2_score": calculated.get("f2_score"),
                    "accuracy": calculated.get("accuracy"),
                }

        model_path = _first_existing_path(config["model_paths"])
        if model_path is not None:
            metric_values["train_val_gap"] = _compute_train_val_gap(model_path)

        payload["models"][key] = {
            "display_name": config["display_name"],
            "color": config["color"],
            "metrics": metric_values,
        }

    return payload


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html', predictions=None, error=None)


@app.route('/api/dashboard-metrics', methods=['GET'])
def dashboard_metrics():
    """
    Return live dashboard metrics for all models.
    """
    return jsonify(_build_dashboard_payload())


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle CSV upload and make predictions
    """
    error = None
    predictions = None
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            error = "No file uploaded"
            return render_template('index.html', predictions=None, error=error)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            error = "No file selected"
            return render_template('index.html', predictions=None, error=error)
        
        # Check if file is CSV
        if not file.filename.endswith('.csv'):
            error = "Please upload a CSV file"
            return render_template('index.html', predictions=None, error=error)
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_input.csv')
        file.save(file_path)
        
        # Make predictions
        result_df = predict_from_csv(file_path)
        
        # Prepare predictions for display (limit to first 100 rows for display)
        display_limit = 100
        predictions = []
        
        for idx, row in result_df.head(display_limit).iterrows():
            pred_result = {
                'row_number': idx + 1,
                'prediction': row['Prediction'],
                'features': {}
            }
            
            # Add first few features for context (exclude the prediction column)
            feature_count = 0
            for col in result_df.columns:
                if col != 'Prediction' and feature_count < 5:
                    pred_result['features'][col] = row[col]
                    feature_count += 1
            
            predictions.append(pred_result)
        
        # Calculate summary statistics
        total = len(result_df)
        threat_count = (result_df['Prediction'] == 'Threat').sum()
        harmless_count = (result_df['Prediction'] == 'Harmless').sum()
        
        summary = {
            'total': total,
            'threat': threat_count,
            'harmless': harmless_count,
            'threat_percentage': f"{(threat_count/total*100):.1f}",
            'harmless_percentage': f"{(harmless_count/total*100):.1f}",
            'showing': min(display_limit, total)
        }
        
        return render_template('index.html', predictions=predictions, summary=summary, error=None)
    
    except FileNotFoundError as e:
        error = "Model not found. Please train the model first by running: python src/train_model.py"
        return render_template('index.html', predictions=None, error=error)
    
    except Exception as e:
        error = f"Error processing file: {str(e)}"
        return render_template('index.html', predictions=None, error=error)


if __name__ == '__main__':
    # Check if model exists
    model_path = os.path.join('model', 'intrusion_model.pkl')
    if not os.path.exists(model_path):
        print("\n" + "="*60)
        print("WARNING: Model not found!")
        print("="*60)
        print("Please train the model first by running:")
        print("  python src/train_model.py")
        print("="*60 + "\n")
    
    # Run the Flask app
    print("\n" + "="*60)
    print("CYBERSENTINEL - Network Traffic Classifier")
    print("="*60)
    print("Starting web server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
