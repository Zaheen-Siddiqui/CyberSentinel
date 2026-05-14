"""
CyberSentinel PowerBI Data Preparation Script

This script consolidates all prediction CSVs and metrics into a single reporting dataset
suitable for Power BI visualization.

Usage:
    python prepare_powerbi_data.py

Output:
    cyberssentinel_powerbi_combined.csv — ready for Power BI
"""

import pandas as pd
import json
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Use the repository root (parent of this powerBi folder) so data/test is found
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "train" / "analysis_output"
CASCADE_DIR = PROJECT_ROOT / "model" / "cascade" / "xgboost"

OUTPUT_CSV = Path(__file__).parent / "cyberssentinel_powerbi_combined.csv"

# Attack category mapping (from cascade_common.py)
ATTACK_TO_CATEGORY = {
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "mailbomb": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_cascade_metrics(model_name="xgboost"):
    """
    Load cascade metrics from JSON files (if they exist)
    """
    metrics = {}
    
    stage1_json = CASCADE_DIR / "stage1_metrics.json"
    stage2_json = CASCADE_DIR / "stage2_metrics.json"
    config_json = CASCADE_DIR / "cascade_config.json"
    
    if stage1_json.exists():
        with open(stage1_json) as f:
            metrics["stage1"] = json.load(f)
    
    if stage2_json.exists():
        with open(stage2_json) as f:
            metrics["stage2"] = json.load(f)
    
    if config_json.exists():
        with open(config_json) as f:
            metrics["config"] = json.load(f)
    
    return metrics

def convert_label_to_binary(label):
    """Convert label to binary: 0=Normal, 1=Attack"""
    label_str = str(label).strip().lower()
    return 0 if label_str == "normal" else 1

def get_attack_category(label):
    """Map attack label to category"""
    label_str = str(label).strip().lower()
    if label_str == "normal":
        return "NORMAL"
    return ATTACK_TO_CATEGORY.get(label_str, "Unknown")

def add_metrics_columns(df):
    """
    Add pre-calculated metrics columns to the dataframe
    These help Power BI create consistent measures across datasets
    """
    
    # Ensure we have required columns
    if "actual_label" not in df.columns:
        df["actual_label"] = ""
    if "prediction_numeric" not in df.columns:
        df["prediction_numeric"] = None
    if "correct_prediction" not in df.columns:
        df["correct_prediction"] = 0

    # Ensure correct_prediction is numeric 0/1 for DAX filters
    if "correct_prediction" in df.columns:
        df["correct_prediction"] = (
            df["correct_prediction"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    
    # Convert actual label to binary
    df["actual_binary"] = df["actual_label"].apply(convert_label_to_binary)
    
    # Map to attack categories
    df["AttackCategory"] = df["actual_label"].apply(get_attack_category)
    
    # Ensure prediction_numeric is numeric
    df["prediction_numeric"] = pd.to_numeric(df["prediction_numeric"], errors="coerce").fillna(0).astype(int)
    
    # Recalculate metrics if labels are present
    if "actual_label" in df.columns and len(df) > 0:
        # Accuracy
        df["Accuracy"] = (df["actual_binary"] == df["prediction_numeric"]).astype(float)
        
        # TP, FP, TN, FN
        tp = ((df["actual_binary"] == 1) & (df["prediction_numeric"] == 1)).astype(int)
        fp = ((df["actual_binary"] == 0) & (df["prediction_numeric"] == 1)).astype(int)
        tn = ((df["actual_binary"] == 0) & (df["prediction_numeric"] == 0)).astype(int)
        fn = ((df["actual_binary"] == 1) & (df["prediction_numeric"] == 0)).astype(int)
        
        # Precision: TP / (TP + FP)
        df["Precision"] = tp.astype(float) / (tp + fp).replace(0, 1)
        df["Precision"] = df["Precision"].fillna(0)
        
        # Recall: TP / (TP + FN)
        df["Recall"] = tp.astype(float) / (tp + fn).replace(0, 1)
        df["Recall"] = df["Recall"].fillna(0)
        
        # F1: 2 * (Precision * Recall) / (Precision + Recall)
        denom = (df["Precision"] + df["Recall"]).replace(0, 1)
        df["F1Score"] = 2 * (df["Precision"] * df["Recall"]) / denom
        df["F1Score"] = df["F1Score"].fillna(0)
        
        # False Positive Rate: FP / (FP + TN)
        df["FalsePositiveRate"] = fp.astype(float) / (fp + tn).replace(0, 1)
        df["FalsePositiveRate"] = df["FalsePositiveRate"].fillna(0)
        
        # Attack Recall (sensitivity for attacks): TP / (TP + FN)
        df["AttackRecall"] = tp.astype(float) / (tp + fn).replace(0, 1)
        df["AttackRecall"] = df["AttackRecall"].fillna(0)
    
    return df

def prepare_training_data_summary():
    """
    Create summary rows from training data analysis CSVs
    (for informational purposes in the dashboard)
    """
    summary_rows = []
    
    training_summary_file = TRAIN_DATA_DIR / "KDDTrain+_20Percent_label_distribution.csv"
    
    if training_summary_file.exists():
        train_dist = pd.read_csv(training_summary_file)
        # Create summary row for training data
        # Note: This is just metadata, won't match the feature columns
        summary_rows.append({
            "dataset": "KDDTrain+_20Percent",
            "model_used": "metadata",
            "actual_label": "training_summary",
            "prediction_numeric": 0,
        })
    
    return summary_rows

def normalize_decision_path(decision_path_str):
    """
    Normalize decision_path from cascade CSV to match DAX expectations
    Examples:
    - "stage1_normal" → "Stage1_Normal"
    - "stage2_attack_strong_stage1" → "Stage2_Attack_Strong_Stage1"
    - "stage2_reverted_normal" → "Stage2_Reverted_Normal"
    """
    if pd.isna(decision_path_str):
        return None
    
    dp = str(decision_path_str).strip().lower()
    
    # Map cascade decision paths to DAX-compatible names
    mapping = {
        "stage1_normal": "Stage1_Normal",
        "attack_path": "Stage2_Attack_Path",
        "stage2_attack_confident": "Stage2_Attack_Confident",
        "stage2_attack_strong_stage1": "Stage2_Attack_Strong_Stage1",
        "stage2_reverted_normal": "Stage2_Reverted_Normal",
    }
    
    # Try direct mapping first
    if dp in mapping:
        return mapping[dp]
    
    # If not found, convert to title case with underscores
    return "_".join(word.capitalize() for word in dp.split("_"))

def merge_cascade_decision_paths():
    """
    Load decision_path from cascade CSVs and create a mapping.
    Returns a dict: {(dataset_name, model_name, row_index): decision_path}
    """
    cascade_map = {}
    
    cascade_files = [
        ("KDDTest+_cascade_randomforest.csv", "KDDTest+", "randomforest"),
        ("KDDTest+_cascade_svm.csv", "KDDTest+", "svm"),
        ("KDDTest+_cascade_xgboost.csv", "KDDTest+", "xgboost"),
        ("KDDTest-21_cascade_randomforest.csv", "KDDTest-21", "randomforest"),
        ("KDDTest-21_cascade_svm.csv", "KDDTest-21", "svm"),
        ("KDDTest-21_cascade_xgboost.csv", "KDDTest-21", "xgboost"),
    ]
    
    for filename, dataset_name, model_name in cascade_files:
        filepath = TEST_DATA_DIR / filename
        
        if filepath.exists():
            print(f"   Loading cascade for {model_name} on {dataset_name}...")
            cascade_df = pd.read_csv(filepath, usecols=["decision_path"])
            
            # Normalize decision paths and store with key (dataset, model, row_idx)
            for idx, row in cascade_df.iterrows():
                key = (dataset_name, model_name, idx)
                cascade_map[key] = normalize_decision_path(row["decision_path"])
        else:
            print(f"   ⚠️ Cascade file not found: {filename}")
    
    return cascade_map

def merge_all_predictions():
    """
    Merge all prediction CSVs into a single dataframe and attach cascade DecisionPath
    """
    
    all_predictions = []
    row_counter = {}  # Track row index for each (dataset, model) pair
    
    # Define prediction files to load
    prediction_files = [
        ("KDDTest+_predictions_randomforest.csv", "KDDTest+", "randomforest"),
        ("KDDTest+_predictions_svm.csv", "KDDTest+", "svm"),
        ("KDDTest+_predictions_xgboost.csv", "KDDTest+", "xgboost"),
        ("KDDTest-21_predictions_randomforest.csv", "KDDTest-21", "randomforest"),
        ("KDDTest-21_predictions_svm.csv", "KDDTest-21", "svm"),
        ("KDDTest-21_predictions_xgboost.csv", "KDDTest-21", "xgboost"),
    ]
    
    # Load cascade decision paths first
    print("   Loading cascade decision paths...")
    cascade_map = merge_cascade_decision_paths()
    
    for filename, dataset_name, model_name in prediction_files:
        filepath = TEST_DATA_DIR / filename
        
        if filepath.exists():
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath)
            
            # Add metadata columns
            df["dataset"] = dataset_name
            df["model_used"] = model_name
            
            # Add DecisionPath from cascade map
            decision_paths = []
            row_counter[(dataset_name, model_name)] = 0
            
            for idx in range(len(df)):
                key = (dataset_name, model_name, idx)
                cascade_path = cascade_map.get(key)
                decision_paths.append(cascade_path)
            
            df["DecisionPath"] = decision_paths
            
            all_predictions.append(df)
        else:
            print(f"⚠️ File not found: {filename}")
    
    if not all_predictions:
        raise FileNotFoundError("No prediction CSV files found. Please ensure prediction files exist in data/test/")
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True, sort=False)
    
    return combined_df

def main():
    """
    Main data preparation pipeline
    """
    
    print("=" * 70)
    print("CyberSentinel Power BI Data Preparation")
    print("=" * 70)
    
    # Step 1: Merge all predictions
    print("\n[1/3] Merging all prediction CSVs and cascade decision paths...")
    df = merge_all_predictions()
    print(f"   ✓ Combined {len(df)} records from all models and datasets")
    print(f"   ✓ Added DecisionPath from cascade inference")
    
    # Step 2: Add metrics columns
    print("\n[2/3] Computing metrics columns...")
    df = add_metrics_columns(df)
    print(f"   ✓ Added Accuracy, Precision, Recall, F1, FPR, AttackRecall")
    
    # Step 3: Ensure correct column order and types
    print("\n[3/3] Finalizing dataset...")
    
    # Define column order (prioritize important columns)
    base_columns = [
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate"
    ]
    
    metadata_columns = [
        "actual_label", "actual_binary", "AttackCategory",
        "prediction_numeric", "correct_prediction",
        "model_used", "dataset", "DecisionPath"
    ]
    
    metrics_columns = [
        "Accuracy", "Precision", "Recall", "F1Score",
        "FalsePositiveRate", "AttackRecall"
    ]
    
    # Reorder columns (only include those that exist)
    existing_columns = [col for col in base_columns + metadata_columns + metrics_columns if col in df.columns]
    
    df = df[existing_columns]
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✓ Saved to: {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Records:         {len(df):,}")
    print(f"Models:                {df['model_used'].nunique()} ({', '.join(df['model_used'].unique())})")
    print(f"Datasets:              {df['dataset'].nunique()} ({', '.join(df['dataset'].unique())})")
    print(f"Average Accuracy:      {df['Accuracy'].mean():.2%}")
    print(f"Average FPR:           {df['FalsePositiveRate'].mean():.2%}")

    actual_attacks = (df["actual_binary"] == 1).sum()
    actual_normal = (df["actual_binary"] == 0).sum()
    true_positives = ((df["actual_binary"] == 1) & (df["prediction_numeric"] == 1)).sum()
    avg_attack_recall = (true_positives / actual_attacks) if actual_attacks > 0 else 0

    print(f"Actual Attacks:        {actual_attacks:,}")
    print(f"Actual Normal:         {actual_normal:,}")
    print(f"Attack Rate:           {actual_attacks / len(df):.2%}")
    print(f"Attack Recall (TP/Att): {avg_attack_recall:.2%}")
    
    print("\n" + "=" * 70)
    print("✅ Power BI data preparation complete!")
    print("=" * 70)
    print(f"\nNext Steps:")
    print(f"1. Open Power BI Desktop")
    print(f"2. Follow the PowerBI_Build_Guide.md")
    print(f"3. Load the CSV: {OUTPUT_CSV}")
    print(f"4. Apply the theme: CyberSentinel_Theme.json")
    print(f"5. Create measures from: DAX_Measures.txt")

if __name__ == "__main__":
    main()
