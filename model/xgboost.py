import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    matthews_corrcoef
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("XGBoost Intrusion Detection Model - Optimized Version")
print("="*80)

# Load data with error handling
try:
    df = pd.read_csv("data/train/KDDTrain+.csv")
    print(f"\n✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: KDDTrain+.csv not found!")
    exit(1)

# Data preprocessing
if "difficulty" in df.columns:
    df = df.drop("difficulty", axis=1)
    print("✓ Dropped 'difficulty' column")

# Convert labels to binary classification
df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# Analyze class distribution
class_dist = df["label"].value_counts()
print(f"\n📊 Class Distribution:")
print(f"   Normal (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)")
print(f"   Attack (1): {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = class_dist[0] / class_dist[1]
print(f"   Scale pos weight: {scale_pos_weight:.2f}")

X = df.drop("label", axis=1)
y = df["label"]

# Identify categorical and numerical columns
categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"\n✓ Features: {len(categorical_cols)} categorical, {len(numerical_cols)} numerical")

# Enhanced preprocessing pipeline with scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# Optimized XGBoost model with better hyperparameters
model = XGBClassifier(
    n_estimators=300,              # Increased for better learning
    max_depth=8,                   # Deeper trees for complex patterns
    learning_rate=0.05,            # Lower learning rate for better generalization
    subsample=0.8,                 # Row sampling
    colsample_bytree=0.8,          # Column sampling per tree
    colsample_bylevel=0.8,         # Column sampling per level
    min_child_weight=3,            # Minimum sum of instance weight in a child
    gamma=0.1,                     # Minimum loss reduction for split
    reg_alpha=0.1,                 # L1 regularization
    reg_lambda=1.0,                # L2 regularization
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    eval_metric="logloss",
    early_stopping_rounds=20,      # Stop if no improvement
    random_state=42,
    n_jobs=-1,                     # Use all CPU cores
    tree_method='hist'             # Faster histogram-based algorithm
)

# Create pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Create validation set for early stopping
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print("\n" + "="*80)
print("Training Model with Early Stopping...")
print("="*80)

# Fit preprocessor and transform data for early stopping
X_train_transformed = preprocessor.fit_transform(X_train_final)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)

# Train with early stopping
model.fit(
    X_train_transformed, 
    y_train_final,
    eval_set=[(X_val_transformed, y_val)],
    verbose=False
)

print(f"✓ Training completed (Best iteration: {model.best_iteration})")

# Create final pipeline with fitted components
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Predictions
y_pred = model.predict(X_test_transformed)
y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

print("\n" + "="*80)
print("Model Performance Metrics")
print("="*80)

# Comprehensive metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"\n📈 Overall Metrics:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   MCC:       {mcc:.4f}")

print("\n" + "-"*80)
print("Detailed Classification Report:")
print("-"*80)
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("-"*80)
print("Confusion Matrix:")
print("-"*80)
print(f"                 Predicted")
print(f"                Normal  Attack")
print(f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"       Attack   {cm[1][0]:6d}  {cm[1][1]:6d}")

# False Positive Rate and False Negative Rate
fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
fnr = cm[1][0] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
print(f"\n   False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
print(f"   False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")

# Cross-validation for robustness check
print("\n" + "="*80)
print("Cross-Validation (5-Fold)")
print("="*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"CV Accuracy Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance analysis
print("\n" + "="*80)
print("Top 15 Most Important Features")
print("="*80)

feature_names = (
    categorical_cols + 
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist() +
    numerical_cols
)

# Get feature names after transformation
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = list(cat_features) + numerical_cols

feature_importance = pd.DataFrame({
    'feature': all_feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRank  Feature                               Importance")
print("-" * 60)
for idx, row in feature_importance.head(15).iterrows():
    print(f"{feature_importance.index.get_loc(idx)+1:2d}.   {row['feature']:35s}  {row['importance']:.6f}")

# Save model
joblib.dump(pipeline, "model/xgboost.pkl")
print("\n" + "="*80)
print("✓ Model saved to model/xgboost.pkl")
print("="*80)
