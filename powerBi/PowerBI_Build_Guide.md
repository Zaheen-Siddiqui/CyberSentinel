# CyberSentinel Power BI Dashboard - Cascade-Only Build Guide

This guide is for your current requirement: final two-stage cascade analytics only.

It uses:
- `cyberssentinel_cascade_combined.csv` (final inference-level rows)
- `CascadeModelSummary.csv` (stage-wise validation summary)
- `CASCADE_DAX_Measures.txt` (cascade measures)

## 1) Files To Use

Load only these files from `powerBi/`:
1. `cyberssentinel_cascade_combined.csv`
2. `CascadeModelSummary.csv`
3. `CASCADE_DAX_Measures.txt`
4. `CyberSentinel_Theme.json`

Do not use `DAX_Measures.txt` for this cascade-only dashboard.

## 2) Critical Field Mapping (Use This Exactly)

If you cannot find fields in visuals, use this mapping:

| Old/Wrong Name | Correct Field In CSV |
|---|---|
| `ModelAlgorithm` | `model_used` |
| `Dataset` (capitalized transformed col) | `dataset` |
| `PredictionLabel` | `final_prediction` |
| `ActualLabel` | `label` |
| `DecisionPath` | `decision_path` |
| `IsCorrect` | create measure or use `Accuracy` |
| `Year`, `Month` | not available in this dataset |
| `AQI_Category`, `LabelCategory` | not available in this project |

Available columns in `cyberssentinel_cascade_combined.csv`:
- `label`, `actual_binary`, `AttackCategory`
- `stage1_decision`, `stage1_p_attack`
- `stage2_pred_category`, `stage2_top1_conf`, `stage2_margin`
- `final_prediction`, `prediction_numeric`, `decision_path`
- `model_used`, `dataset`
- `Accuracy`, `Precision`, `Recall`, `F1Score`, `FalsePositiveRate`, `AttackRecall`

## 3) Load Data (Fresh)

1. Open Power BI Desktop.
2. Remove old queries/tables that were based on non-cascade flow.
3. Home -> Get data -> Text/CSV -> load `cyberssentinel_cascade_combined.csv`.
4. Rename this table exactly to `cyberssentinel_cascade_combined`.
5. Home -> Get data -> Text/CSV -> load `CascadeModelSummary.csv`.
6. Rename this table exactly to `CascadeModelSummary`.
7. Ensure numeric columns are numeric types:
   - In `cyberssentinel_cascade_combined`: `Accuracy`, `Precision`, `Recall`, `F1Score`, `FalsePositiveRate`, `AttackRecall`, `stage1_p_attack`, `stage2_top1_conf`, `stage2_margin`
   - In `CascadeModelSummary`: `Validation Accuracy`, `Validation F1 Score`, `Macro F1`, `False Positive Rate (Normal)`, `Attack Recall`, `TN`, `FP`, `FN`, `TP`

## 4) Create Measures

1. Create `_Measures` table (Home -> Enter Data -> name `_Measures` -> Load).
2. Open `CASCADE_DAX_Measures.txt`.
3. Add all measures one by one (Modeling -> New measure).
4. If any measure errors, verify table names are exactly:
   - `cyberssentinel_cascade_combined`
   - `CascadeModelSummary`

## 5) Build Visuals With Existing Fields Only

## Page 1 - Final Cascade Overview

KPI cards:
- `Total Records Tested`
- `Actual Attacks`
- `Avg Cascade Accuracy`
- `Cascade Precision`
- `Cascade Attack Recall`
- `Cascade FPR`

Pie chart:
- Legend: `final_prediction`
- Values: Count of `label`

Clustered bar chart:
- Axis: `model_used`
- Values: `Avg Cascade Accuracy`, `Cascade Attack Recall`

Slicers:
- `dataset`
- `model_used`

## Page 2 - Stage 1 Decision Flow

Stacked column:
- Axis: `decision_path`
- Values: Count of `label`
- Legend: `dataset` or `model_used`

Table:
- Columns: `stage1_decision`, `stage1_p_attack`, `decision_path`, `final_prediction`, `label`

Cards:
- `Stage1 Normal Path`
- `Stage1 Attack Path`
- `% Stage1 to Stage2`
- `Stage2 Reverted Normal`

## Page 3 - Model Comparison (Final Cascade)

Matrix (Rows = `model_used`):
- `Avg Cascade Accuracy`
- `Cascade Precision`
- `Cascade Recall`
- `Cascade F1 Score`
- `Cascade Attack Recall`
- `Cascade FPR`

Optional cards:
- `RF Accuracy`, `RF Precision`, `RF Attack Recall`
- `SVM Accuracy`, `SVM Precision`, `SVM Attack Recall`
- `XGBoost Accuracy`, `XGBoost Precision`, `XGBoost Attack Recall`

## Page 4 - Stage Validation Summary

Use `CascadeModelSummary` table.

KPI cards:
- `Stage1 Binary Accuracy`
- `Stage1 Attack Detection Rate`
- `Stage2 Category Accuracy`
- `Cascade FPR`

Table (Rows: `Model`, Filter: `Stage` = `Stage 2`):
- `Macro F1`
- `Remarks`

Clustered chart (Stage 1 only):
- Axis: `Model`
- Values: `Attack Recall`, `False Positive Rate (Normal)`
- Visual filter: `Stage = Stage 1`

## Page 5 - Attack Category Analysis

Bar chart:
- Axis: `AttackCategory`
- Values: Count of `label`

Matrix:
- Rows: `AttackCategory`
- Columns: `final_prediction`
- Values: Count of `label`

Cards:
- `Unique Attack Types`
- `Most Common Attack`
- `Rarest Attack`

## 6) Why Your Earlier Fields Were Missing

Those fields came from an older transformed schema or unrelated template sections.
Your current cascade CSV does not contain:
- `ModelAlgorithm`
- `PredictionLabel`
- `Year`
- `AQI_Category`
- `LabelCategory`

Use the mapping section above to replace them everywhere.

## 7) Quick Fix Checklist (If You Still Get "Field Not Found")

1. Confirm you loaded `cyberssentinel_cascade_combined.csv`, not `cyberssentinel_powerbi_combined.csv`.
2. Confirm table name is exactly `cyberssentinel_cascade_combined`.
3. Confirm `CASCADE_DAX_Measures.txt` is used, not `DAX_Measures.txt`.
4. In Fields pane, verify `decision_path` exists before creating decision-path visuals.
5. Refresh after any query rename or type conversion.

If you share the exact missing field name from Power BI, I can map it immediately to the correct one.
