"""
Cascade Two-Stage Combined Data Preparation

This script consolidates ONLY cascade predictions and computes final metrics
for the two-stage cascade pipeline (Stage1 Binary → Stage2 Category).

Usage:
    python prepare_cascade_combined.py

Output:
    cyberssentinel_cascade_combined.csv — cascade predictions + final metrics
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
OUTPUT_CSV = Path(__file__).parent / "cyberssentinel_cascade_combined.csv"

# Attack category mapping
ATTACK_TO_CATEGORY = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

def convert_label_to_binary(label):
    """0=Normal, 1=Attack"""
    return 0 if str(label).strip().lower() == "normal" else 1

def get_attack_category(label):
    """Map attack label to category"""
    label_str = str(label).strip().lower()
    return "NORMAL" if label_str == "normal" else ATTACK_TO_CATEGORY.get(label_str, "Unknown")

def compute_metrics(df):
    """Compute TP, FP, TN, FN, Precision, Recall, F1, FPR, AttackRecall"""
    
    # Binary metrics
    df["actual_binary"] = df["label"].apply(convert_label_to_binary)
    df["cascade_binary"] = (df["final_prediction"] != "normal").astype(int)
    df["AttackCategory"] = df["label"].apply(get_attack_category)
    
    # Confusion matrix per row
    tp = ((df["actual_binary"] == 1) & (df["cascade_binary"] == 1)).astype(int)
    fp = ((df["actual_binary"] == 0) & (df["cascade_binary"] == 1)).astype(int)
    tn = ((df["actual_binary"] == 0) & (df["cascade_binary"] == 0)).astype(int)
    fn = ((df["actual_binary"] == 1) & (df["cascade_binary"] == 0)).astype(int)
    
    # Accuracy (binary)
    df["Accuracy"] = (df["actual_binary"] == df["cascade_binary"]).astype(float)
    
    # Precision: TP / (TP + FP)
    df["Precision"] = tp.astype(float) / (tp + fp).replace(0, 1)
    df["Precision"] = df["Precision"].fillna(0)
    
    # Recall: TP / (TP + FN)
    df["Recall"] = tp.astype(float) / (tp + fn).replace(0, 1)
    df["Recall"] = df["Recall"].fillna(0)
    
    # F1
    denom = (df["Precision"] + df["Recall"]).replace(0, 1)
    df["F1Score"] = 2 * (df["Precision"] * df["Recall"]) / denom
    df["F1Score"] = df["F1Score"].fillna(0)
    
    # False Positive Rate: FP / (FP + TN)
    df["FalsePositiveRate"] = fp.astype(float) / (fp + tn).replace(0, 1)
    df["FalsePositiveRate"] = df["FalsePositiveRate"].fillna(0)
    
    # Attack Recall: TP / (TP + FN)
    df["AttackRecall"] = tp.astype(float) / (tp + fn).replace(0, 1)
    df["AttackRecall"] = df["AttackRecall"].fillna(0)
    
    return df

def merge_cascade_predictions():
    """
    Load cascade prediction CSVs for all models and datasets
    """
    all_cascade = []
    
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
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath)
            
            # Add metadata
            df["dataset"] = dataset_name
            df["model_used"] = model_name
            df["prediction_numeric"] = (df["final_prediction"] != "normal").astype(int)
            
            all_cascade.append(df)
        else:
            print(f"⚠️ Not found: {filename}")
    
    if not all_cascade:
        raise FileNotFoundError("No cascade CSV files found.")
    
    combined = pd.concat(all_cascade, ignore_index=True, sort=False)
    return combined

def main():
    print("=" * 70)
    print("CyberSentinel CASCADE Two-Stage Combined Data Preparation")
    print("=" * 70)
    
    # Step 1: Load cascade predictions
    print("\n[1/3] Loading cascade predictions (final output only)...")
    df = merge_cascade_predictions()
    print(f"   ✓ Combined {len(df)} cascade predictions")
    
    # Step 2: Compute metrics
    print("\n[2/3] Computing cascade metrics...")
    df = compute_metrics(df)
    print(f"   ✓ Computed Accuracy, Precision, Recall, F1, FPR, AttackRecall")
    
    # Step 3: Select and reorder columns
    print("\n[3/3] Finalizing dataset...")
    
    # Select key columns for cascade analysis
    cascade_columns = [
        "label", "actual_binary", "AttackCategory",
        "stage1_decision", "stage1_p_attack",
        "stage2_pred_category", "stage2_top1_conf", "stage2_margin",
        "final_prediction", "prediction_numeric", "decision_path",
        "model_used", "dataset",
        "Accuracy", "Precision", "Recall", "F1Score",
        "FalsePositiveRate", "AttackRecall"
    ]
    
    existing_cols = [col for col in cascade_columns if col in df.columns]
    df = df[existing_cols]
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Summary stats
    print(f"\n{'='*70}")
    print("CASCADE DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total Records:          {len(df):,}")
    print(f"Models:                 {df['model_used'].nunique()} ({', '.join(df['model_used'].unique())})")
    print(f"Datasets:               {df['dataset'].nunique()} ({', '.join(df['dataset'].unique())})")
    print(f"\nByModel Performance:")
    
    for model in sorted(df['model_used'].unique()):
        model_df = df[df['model_used'] == model]
        acc = model_df['Accuracy'].mean()
        prec = model_df['Precision'].mean()
        rec = model_df['Recall'].mean()
        recall_att = model_df['AttackRecall'].mean()
        fpr = model_df['FalsePositiveRate'].mean()
        f1 = model_df['F1Score'].mean()
        
        print(f"\n  {model.upper()}:")
        print(f"    Accuracy:        {acc:.2%}")
        print(f"    Precision:       {prec:.2%}")
        print(f"    Recall:          {rec:.2%}")
        print(f"    Attack Recall:   {recall_att:.2%}")
        print(f"    FPR:             {fpr:.2%}")
        print(f"    F1 Score:        {f1:.2%}")
    
    print(f"\n{'='*70}")
    print("✅ Cascade combined data prepared!")
    print(f"   Output: {OUTPUT_CSV}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
