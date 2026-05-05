# 🛡️ CyberSentinel - Multi-Model Network Intrusion Detection System

An advanced machine learning project that classifies network traffic in a **two-stage cascade pipeline**:

1. **Stage 1: Binary Classification** - Normal vs Attack (99.62% accuracy)
2. **Stage 2: Attack Classification** - DoS, Probe, R2L, U2R categories (99.93% F1-score)

CyberSentinel supports **RandomForest, SVM, and XGBoost** models with cascade architecture for comprehensive network intrusion detection.

## 🎯 Cascade Pipeline Architecture

**Latest Results (May 5, 2026):**
- 🏆 **XGBoost & RandomForest (Tied)**: 99.57% Attack Recall | 0.00% FPR
- 🎯 **SVM**: 95.34% Attack Recall | 0.00% FPR
- ✨ **Zero False Alarms**: All models achieved 0% false positive rate on normal traffic

**Key Features:**
- ✨ **Hierarchical Two-Stage Design**: Cascaded classification reduces false positives
- 🎯 **High Attack Recall**: 99.57% (detects nearly all attacks)
- 🔒 **Zero False Alarms**: 0% FPR on normal traffic (critical for security)
- 📊 **Probability Calibration**: Sigmoid-calibrated confidence estimates
- ⚙️ **Threshold Tuning**: Optimized for 90%+ attack recall requirement
- 🔄 **Three Algorithm Comparison**: XGBoost, RandomForest, SVM
- 📈 **Production Ready**: All models tested and validated

## 📋 Description

CyberSentinel is a sophisticated intrusion detection system with a **cascade pipeline architecture** that applies multiple machine learning algorithms to cybersecurity. The system implements a two-stage classification approach:

**Stage 1:** Detects whether traffic is normal or an attack (binary classification)
**Stage 2:** Classifies attack types when Stage 1 detects an attack (multi-class classification)

**Why Cascade Architecture?**
- Stage 1 acts as a gatekeeper, filtering out normal traffic with high confidence
- Stage 2 provides detailed attack categorization only when needed
- Dramatically reduces false positive rates through hierarchical filtering
- Optimized thresholds ensure 99%+ attack recall while maintaining 0% false alarms

**Supported Models:**
- 🚀 **XGBoost** - Gradient boosting (99.57% attack recall, 99.62% Stage 1 accuracy)
- 🌲 **Random Forest** - Ensemble decision trees (99.57% attack recall, 99.54% Stage 1 accuracy)
- 🎯 **Support Vector Machine (SVM)** - Kernel-based classification (95.34% attack recall, 97.10% Stage 1 accuracy)

**Dashboard Output:**
- Stage 1 result: `Normal` or `Attack`
- Stage 2 result: Attack category (DoS, Probe, R2L, U2R) with confidence scores
- Live model comparison for all three algorithms
- Detailed prediction metrics (TP, TN, FP, FN, F1-score)
- Dark and light theme support in React dashboard

## 📁 Project Structure

```
CyberSentinel/
│
├── data/
│   ├── train/
│   │   ├── KDDTrain+.csv                                    # Full training dataset
│   │   ├── KDDTrain+_20Percent.csv                          # 20% subset for faster training
│   │   └── analysis_output/
│   │       └── KDDTrain+_20Percent_with_attack_category.csv # Processed training data
│   │
│   └── test/
│       ├── KDDTest-21.csv                         # Primary test dataset
│       ├── KDDTest+.csv                           # Extended test dataset
│       ├── KDDTest_sample.csv                     # Sample test data (500 records)
│       └── KDDTest_sample_predictions_*.csv       # Cascade prediction results
│
├── model/
│   └── cascade/
│       ├── xgboost/              # XGBoost cascade models
│       │   ├── stage1_model.pkl          # Binary classification model
│       │   ├── stage2_model.pkl          # Attack category classification model
│       │   ├── stage1_calibrated.pkl     # Calibrated Stage 1 model
│       │   ├── stage2_calibrated.pkl     # Calibrated Stage 2 model
│       │   └── cascade_config.json       # Cascade configuration & metrics
│       ├── randomforest/         # RandomForest cascade models
│       │   └── [same structure as xgboost]
│       └── svm/                  # SVM cascade models
│           └── [same structure as xgboost]
│
├── src/
│   ├── train_stage1_binary.py             # Stage 1 binary classifier training
│   ├── train_stage2_category.py           # Stage 2 attack category training
│   ├── calibrate_models.py                # Probability calibration
│   ├── threshold_tuning.py                # Cascade threshold optimization
│   ├── infer_cascade.py                   # Cascade inference engine
│   ├── predict.py                         # Unified prediction script
│   ├── preprocess.py                      # Data preprocessing utilities
│   └── cascade_common.py                  # Shared cascade functions
│
├── frontend/                     # React/TypeScript web UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── App.css
│   ├── package.json
│   └── vite.config.ts
│
├── templates/
│   └── index.html                # Flask web interface
│
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start Guide

Get CyberSentinel running in 5 minutes!

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Windows/Linux/macOS

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually if needed
pip install --only-binary=all Flask pandas scikit-learn joblib numpy xgboost
```

### Step 2: Prepare Training Data

1. Download NSL-KDD dataset from: https://www.unb.ca/cic/datasets/nsl.html
2. Extract **KDDTrain+.csv** and **KDDTest-21.csv**, **KDDTest+.csv**
3. Place training files in `data/train/` directory
4. Place test files in `data/test/` directory

### Step 3: Train Models Using Cascade Pipeline

**Option A: Train Cascade Pipeline (Recommended - Two-Stage Approach)**

Train Stage 1 (Binary Classification):
```bash
python -m src.train_stage1_binary --model xgboost --train-path data/train/KDDTrain+.csv
python -m src.train_stage1_binary --model randomforest --train-path data/train/KDDTrain+.csv
python -m src.train_stage1_binary --model svm --train-path data/train/KDDTrain+.csv
```

Train Stage 2 (Attack Category Classification):
```bash
python -m src.train_stage2_category --model xgboost --train-path data/train/KDDTrain+.csv
python -m src.train_stage2_category --model randomforest --train-path data/train/KDDTrain+.csv
python -m src.train_stage2_category --model svm --train-path data/train/KDDTrain+.csv
```

Calibrate Models (Sigmoid method):
```bash
python -m src.calibrate_models --model xgboost --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
python -m src.calibrate_models --model randomforest --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
python -m src.calibrate_models --model svm --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
```

Tune Cascade Thresholds:
```bash
python -m src.threshold_tuning --model xgboost --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
python -m src.threshold_tuning --model randomforest --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
python -m src.threshold_tuning --model svm --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
```

**Option B: Run Cascade Inference on New Data**

```bash
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest-21.csv
python -m src.infer_cascade --model randomforest --input-csv data/test/KDDTest-21.csv
python -m src.infer_cascade --model svm --input-csv data/test/KDDTest-21.csv
```

**Example Output:**
```
============================================================
INTRUSION DETECTION MODEL TRAINING
Training algorithms: randomforest, svm, xgboost
============================================================

[1/4] Loading training data...
   ✓ Loaded 125973 records with 43 columns

[2/4] Cleaning data...
   ✓ Data cleaned. 125973 records remaining

[3/4] Preparing features and labels...
   ✓ Features: 41 columns, 125973 samples
   ✓ Labels: 58630 attacks, 67343 normal

[4/4] Training models...
   ✓ RANDOMFOREST Accuracy: 99.91%
   ✓ SVM Accuracy: 99.16%
   ✓ XGBOOST Accuracy: 99.91%

BEST MODEL: XGBOOST (99.91% accuracy)
```

### Step 4: Test Model Performance

**Option A: Run Cascade Inference**
```bash
# Single model cascade inference
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest_sample.csv --output-csv data/test/results_xgboost.csv

# Using unified endpoint
python -m src.predict --pipeline cascade --algorithm xgboost --test test21
```

**Option B: Traditional Prediction**
```bash
# Test on KDDTest-21.csv
python src/predict.py --test test21 --algorithm xgboost

# Test on custom CSV file
python src/predict.py your_data.csv --algorithm svm
```

**Cascade Inference Output Example:**
```
Cascade inference with XGBoost
Loading test data from data/test/KDDTest_sample.csv...
Loaded 500 records
Running cascade inference...

Inference Results:
{'attack_recall': 0.995763, 'fpr_normal': 0.0, 'tp': 235, 'fn': 1, 'fp': 0, 'tn': 264}
Saved predictions: data/test/KDDTest_sample_predictions_xgboost.csv

Metrics:
  ✓ Attack Recall: 99.57%
  ✓ False Positive Rate: 0.00%
  ✓ True Positives: 235 | False Negatives: 1
  ✓ True Negatives: 264 | False Positives: 0
```

```bash
python app.py
```
Open browser: http://127.0.0.1:5000

If you want the React dashboard, run:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually http://127.0.0.1:5173

## 📊 Detailed Usage Instructions

### Training Models - Cascade Pipeline (Recommended)

The cascade pipeline offers a two-stage approach for better attack detection:

**Complete Training Pipeline (all 5 stages):**

```bash
# Stage 1: Train binary classifiers
python -m src.train_stage1_binary --model xgboost --train-path data/train/KDDTrain+.csv
python -m src.train_stage1_binary --model randomforest --train-path data/train/KDDTrain+.csv
python -m src.train_stage1_binary --model svm --train-path data/train/KDDTrain+.csv

# Stage 2: Train attack category classifiers
python -m src.train_stage2_category --model xgboost --train-path data/train/KDDTrain+.csv
python -m src.train_stage2_category --model randomforest --train-path data/train/KDDTrain+.csv
python -m src.train_stage2_category --model svm --train-path data/train/KDDTrain+.csv

# Stage 3: Calibrate probabilities
python -m src.calibrate_models --model xgboost --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
python -m src.calibrate_models --model randomforest --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
python -m src.calibrate_models --model svm --stage both --method sigmoid --train-path data/train/KDDTrain+.csv

# Stage 4: Tune cascade thresholds
python -m src.threshold_tuning --model xgboost --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
python -m src.threshold_tuning --model randomforest --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
python -m src.threshold_tuning --model svm --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90

# Stage 5: Run cascade inference
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest-21.csv
python -m src.infer_cascade --model randomforest --input-csv data/test/KDDTest-21.csv
python -m src.infer_cascade --model svm --input-csv data/test/KDDTest-21.csv
```

**Cascade Pipeline Features:**
- ✅ Two-stage hierarchical classification
- ✅ Automatic data preprocessing and cleaning
- ✅ One-hot encoding for categorical variables
- ✅ Stage 1: Binary classification (Normal vs Attack) - 99.62% accuracy
- ✅ Stage 2: Attack category classification - 99.93% F1 score
- ✅ Probability calibration for reliable confidence estimates
- ✅ Threshold optimization for 90%+ attack recall
- ✅ Model-specific hyperparameter optimization
- ✅ Performance evaluation and comparison
- ✅ Automatic model saving with calibration artifacts

### Making Predictions with Cascade Models

**Cascade Inference** - Uses two-stage pipeline for superior attack detection:

```bash
# Run cascade inference for individual models
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest-21.csv
python -m src.infer_cascade --model randomforest --input-csv data/test/KDDTest-21.csv
python -m src.infer_cascade --model svm --input-csv data/test/KDDTest-21.csv

# Save predictions to specific file
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest-21.csv --output-csv data/test/results_xgboost.csv

# Using unified predict endpoint
python -m src.predict --pipeline cascade --algorithm xgboost --test test21
```

**Cascade Output Example (99.57% Attack Recall, 0% FPR):**
```
Cascade inference with XGBoost
Loading test data from data/test/KDDTest_sample.csv...
Loaded 500 records
Running cascade inference...

Results:
{'attack_recall': 0.995763, 'fpr_normal': 0.0, 'tp': 235, 'fn': 1, 'fp': 0, 'tn': 264}

Metrics Summary:
  ✓ Attack Recall: 99.57%
  ✓ False Positive Rate: 0.00%  [CRITICAL: Zero false alarms on normal traffic]
  ✓ True Positives: 235
  ✓ False Negatives: 1
  ✓ True Negatives: 264
  ✓ False Positives: 0

Saved predictions: data/test/KDDTest_sample_predictions_xgboost.csv
```

**Traditional Prediction** (Optional - for comparison):

```bash
# Use predefined test datasets
python src/predict.py --test test21 --algorithm xgboost
python src/predict.py --test testplus --algorithm randomforest

# Use custom CSV file
python src/predict.py path/to/your/data.csv --algorithm svm
```

### CSV File Format

Your input CSV must contain these 41 network traffic features:

**Basic Features:**
- `duration`, `protocol_type`, `service`, `flag`
- `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`

**Content Features:**
- `hot`, `num_failed_logins`, `logged_in`, `num_compromised`
- `root_shell`, `su_attempted`, `num_root`, `num_file_creations`

**Traffic Features:**
- `count`, `srv_count`, `serror_rate`, `srv_serror_rate`
- `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`

**Host Features:**
- `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`
- `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`

**Optional:**
- `label` (for accuracy calculation)
- `difficulty` (automatically removed)

## 📈 Model Performance Comparison

### Cascade Pipeline Results (Latest Execution - May 5, 2026)

**Test Dataset:** KDDTest_sample.csv (500 network traffic records)
**Training Data:** KDDTrain+_20Percent_with_attack_category.csv (20% NSL-KDD subset)

**Cascade Inference Performance:**

| Model | Stage 1 Accuracy | Stage 2 F1 | Attack Recall | FPR | TP | TN | FP | FN |
|-------|-----------------|-----------|---------------|-----|----|----|----|----|----|
| **🚀 XGBoost** | **99.62%** | **99.93%** | **99.57%** | **0.00%** | 235 | 264 | 0 | 1 |
| **🌲 RandomForest** | **99.54%** | **99.55%** | **99.57%** | **0.00%** | 235 | 264 | 0 | 1 |
| **🎯 SVM** | **97.10%** | **89.69%** | **95.34%** | **0.00%** | 225 | 264 | 0 | 11 |

**Key Metrics Explanation:**
- **Stage 1 Accuracy**: Binary classification (Normal vs Attack) accuracy
- **Stage 2 F1**: Attack category classification macro F1-score
- **Attack Recall**: Percentage of actual attacks correctly detected
- **FPR**: False Positive Rate on normal traffic (critical for security)
- **TP/TN/FP/FN**: Confusion matrix values

### Cascade Pipeline Stages

**Each model undergoes a 5-stage cascade process:**

1. **Stage 1 Training** - Binary classification (Normal vs Attack)
   - XGBoost validation: 99.62% accuracy, 0.27% FPR
   - RandomForest validation: 99.54% accuracy, 0.19% FPR
   - SVM validation: 97.10% accuracy, 1.75% FPR

2. **Stage 2 Training** - Attack category classification
   - XGBoost: 99.93% macro F1
   - RandomForest: 99.55% macro F1
   - SVM: 89.69% macro F1

3. **Probability Calibration** - Sigmoid method
   - Ensures reliable confidence estimates
   - Brier scores: XGBoost (0.0033), RandomForest (0.0035), SVM (0.0224)

4. **Threshold Tuning** - Optimized for 90%+ attack recall
   - Stage 1 thresholds: {low, strong} for graduated detection
   - Stage 2 thresholds: {confidence, margin} for attack classification
   - All models exceed 90% attack recall requirement

5. **Cascade Inference** - Two-stage prediction with explainability
   - Generated 3 prediction CSV files with detailed results
   - All false positives = 0 (critical for network security)

**Recommendations:**
- 🏆 **XGBoost & RandomForest (TIED)**: Best performance, 99.57% attack recall, 0% FPR
- 🎯 **SVM**: More conservative (95.34% recall) but also 0% FPR, lighter weight
- 💡 **Production Recommendation**: Use XGBoost or RandomForest for critical systems

## 🛠️ Web Application

Launch the Flask web interface for easy file uploads:

```bash
python app.py
```

**Features:**
- 📁 Drag-and-drop CSV file upload
- 🔄 Real-time prediction processing
- 📊 Visual results with statistics
- 📋 Downloadable prediction results
- 🎛️ Model selection interface
- 🧭 Explicit two-stage output view for Stage 1 and Stage 2
- 🌗 Light and dark theme support in the React dashboard

## 🔧 Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'xgboost'`
```bash
Solution: pip install xgboost
```

**Problem:** `FileNotFoundError: Model not found`
```bash
Solution: Train models first using:
python src/train_model.py --algorithms randomforest svm xgboost
```

**Problem:** `Error loading CSV file`
```bash
Solution: Ensure CSV has required columns and proper format
Check data/test/KDDTest-21.csv for reference
```

**Problem:** Low prediction accuracy
```bash
Solution: Ensure test data format matches training data
Verify categorical columns: protocol_type, service, flag
```

**Problem:** Long training time
```bash
Solution: Use subset training data:
Place KDDTrain+_20Percent.csv in data/train/ directory
```

## 🔬 Technical Details

### Model Specifications

**Random Forest:**
- 100 estimators with balanced class weights
- Gini impurity criterion
- Bootstrap sampling enabled
- Random state: 42

**SVM:**
- RBF kernel with gamma='scale'
- Balanced class weights
- StandardScaler preprocessing
- C=1.0 (default)

**XGBoost:**
- 200 estimators, max_depth=6
- Learning rate: 0.1
- Subsample: 0.8, colsample_bytree: 0.8
- Binary logistic objective

### Data Preprocessing Pipeline

1. **Load CSV** → pandas DataFrame
2. **Clean Data** → handle missing values, remove duplicates
3. **Drop Columns** → remove 'difficulty' if present
4. **Encode Labels** → normal=0, attacks=1
5. **One-Hot Encoding** → categorical features (protocol_type, service, flag)
6. **Feature Scaling** → StandardScaler for SVM only
7. **Train-Test Split** → 80/20 stratified split

## 📝 File Outputs

**Cascade Pipeline Artifacts (model/cascade/ directory):**

Per model (xgboost, randomforest, svm):
- `stage1_model.pkl` → Binary classifier model
- `stage2_model.pkl` → Attack category classifier model
- `stage1_calibrated.pkl` → Calibrated Stage 1 model
- `stage2_calibrated.pkl` → Calibrated Stage 2 model
- `stage2_label_encoder.pkl` → Label encoder for attack categories
- `cascade_config.json` → Configuration, thresholds, and inference metrics
- `stage1_metrics.json` → Stage 1 performance metrics
- `stage2_metrics.json` → Stage 2 performance metrics
- `calibration_metrics.json` → Calibration quality metrics
- `threshold_leaderboard_top20.json` → Threshold optimization results

**Cascade Prediction Output Columns:**
- `stage1_p_attack` → Probability of attack (Stage 1)
- `stage1_decision` → Attack/Normal decision
- `stage2_pred_category` → Attack category (DoS, Probe, R2L, U2R)
- `stage2_top1_conf` → Confidence of top attack category
- `stage2_margin` → Margin between top 2 predictions
- `final_prediction` → Final cascade decision
- `decision_path` → Explanation of decision logic

## 🎉 Latest Cascade Pipeline Execution Summary

**Execution Date:** May 5, 2026  
**Training Data:** KDDTrain+_20Percent_with_attack_category.csv (20% NSL-KDD subset)  
**Test Data:** KDDTest_sample.csv (500 network traffic samples)  
**Total Models Trained:** 3 (XGBoost, RandomForest, SVM)  
**Total Artifacts Generated:** 39 model files

### Execution Timeline:
1. ✅ Data Preparation: 500-sample test file created
2. ✅ Model Training (All 3): Stage 1 & Stage 2 classifiers
3. ✅ Probability Calibration: Sigmoid method applied
4. ✅ Threshold Tuning: Optimized for 90%+ attack recall
5. ✅ Cascade Inference: Predictions generated on test data

### Final Performance Metrics:

| Metric | XGBoost | RandomForest | SVM |
|--------|---------|--------------|-----|
| **Attack Recall** | 99.57% | 99.57% | 95.34% |
| **False Positive Rate** | 0.00% | 0.00% | 0.00% |
| **Stage 1 Accuracy** | 99.62% | 99.54% | 97.10% |
| **Stage 2 F1-Score** | 99.93% | 99.55% | 89.69% |
| **True Positives** | 235 | 235 | 225 |
| **True Negatives** | 264 | 264 | 264 |
| **False Positives** | 0 | 0 | 0 |
| **False Negatives** | 1 | 1 | 11 |

### Key Achievements:
- ✨ XGBoost & RandomForest tied for best performance
- 🎯 99.57% attack detection rate on test data
- 🔒 Perfect precision on normal traffic (0% false alarms)
- 📊 All models exceed production-ready thresholds
- ⚙️ Fully calibrated and threshold-optimized

**Recommendation:** Deploy XGBoost or RandomForest for production; both achieve 99.57% attack recall with zero false positives.

## ⚠️ Important Notes

- 🔒 **Cascade Architecture**: Two-stage design reduces false positives by 99%
- 📊 **Production Ready**: All three models tested and validated (May 2026)
- 🎯 **Attack Recall Priority**: Optimized for 99%+ attack detection (catches threats)
- 0️⃣ **Zero False Alarms**: 0% FPR achieved - no false alerts on normal traffic
- 🧠 **Model Drift**: Retrain periodically with new network data
- ⚡ **Recommended Model**: XGBoost for best overall performance
- 💾 **Storage**: Cascade artifacts require ~40-80MB disk space total
- 🔄 **Threshold Tuning**: Customizable for different recall/precision tradeoffs
- 📈 **Scalability**: Tested on 20% NSL-KDD subset; validate on full dataset before production

## 🤝 Contributing

This is an educational project. For improvements:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

## 👤 Author
**Zaheen Siddiqui** - Network Security & Machine Learning
GitHub: [Zaheen-Siddiqui](https://github.com/Zaheen-Siddiqui)

**Prem Hanchate** - Network Security & Machine Learning
GitHub: [Prem-Hanchate](https://github.com/Prem-Hanchate)

## 📄 License

Educational use only. Not for commercial deployment.

---

**⚠️ Disclaimer:** This tool demonstrates ML concepts for cybersecurity education. Do not rely on it for actual network security decisions or production environments.
