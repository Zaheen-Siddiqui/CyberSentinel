# CyberSentinel Power BI Dashboard — Complete Build Guide

This guide turns your prediction CSVs and training data into a polished Power BI dashboard in about **20 minutes**. Follow each step in order.

---

## What You'll Build

A 5-page Power BI dashboard with:
- **Page 1** – Executive Overview (KPIs, detection accuracy, threat distribution)
- **Page 2** – Data Preprocessing (class balance, attack category skew, preprocessing pipeline)
- **Page 3** – Model Comparison (RandomForest vs SVM vs XGBoost accuracy, metrics)
- **Page 4** – Cascade Classification (Stage 1 binary detection, Stage 2 attack-type breakdown)
- **Page 5** – Attack Type Deep Dive (attack category distribution, misclassification heatmap)

---

## Prerequisites

- **Power BI Desktop** (free) — download from https://powerbi.microsoft.com/desktop/
- The 5 files included in this package:
  1. `cyberssentinel_powerbi_combined.csv` — the consolidated reporting dataset
  2. `CyberSentinel_Theme.json` — security-focused color theme (green=normal, red=threat)
  3. `PowerQuery_M_Code.txt` — data prep and transformation code
  4. `DAX_Measures.txt` — all KPI and classification metrics measures
  5. `PowerBI_Build_Guide.md` — this file

---

## STEP 1 — Apply the Theme

1. Open Power BI Desktop → blank report
2. **View** tab (top ribbon) → **Themes** dropdown → **Browse for themes**
3. Select `CyberSentinel_Theme.json` → Open
4. The theme applies instantly — your visuals will use the correct threat/normal color palette

---

## STEP 2 — Load the Data with Power Query

1. **Home** tab → **Get data** → **Blank query**
2. In the Power Query Editor that opens, click **Advanced Editor** (Home tab)
3. Delete everything inside, paste the entire contents of `PowerQuery_M_Code.txt`
4. **Update line 9** — change the file path to where your CSV lives:
   ```
   Source = Csv.Document(File.Contents("C:\Users\YourName\CyberSentinel\cyberssentinel_powerbi_combined.csv")...
   ```
5. Click **Done**
6. Right-click the query in the left pane → **Rename** → call it `cyberssentinel`
7. Click **Close & Apply** (top-left)

You now have a table with all original columns PLUS computed columns: Year, Month, MonthName, Quarter, Dataset, ModelAlgorithm, IsCorrect, and metrics for categorization.

---

## STEP 3 — Set Sort Order Properties

This makes your charts order things correctly (algorithm names, datasets, attack categories).

1. In the **Data view** (left sidebar, table icon), click on `ModelAlgorithm` column
2. **Column tools** tab → **Sort by column** → select `ModelAlgorithmSort`
3. Click on `Dataset` column → **Sort by column** → select `DatasetSort`
4. Click on `AttackCategory` column → **Sort by column** → select `CategorySort` (if that column exists)
5. Click on `PredictionLabel` column → **Sort by column** → select `PredictionSort`

---

## STEP 4 — Create the Measures Table

Keeps your Fields pane organized and measures separate from raw data.

1. **Home** tab → **Enter data** → name the table `_Measures` → leave the row blank → **Load**
2. In the Fields pane, you'll see `_Measures` with one column. Hide that column (right-click → Hide).
3. Click `_Measures` to select it as your active table
4. Open `DAX_Measures.txt`. For each measure block:
   - **Modeling** tab → **New measure**
   - Replace the formula bar text with the entire measure (name + `=` + formula)
   - Press Enter
5. Repeat for all ~35 measures. (Takes ~5–7 minutes, but they're permanent and reusable across all pages.)

---

## STEP 5 — Build Page 1: Executive Overview

Rename **Page 1** at the bottom to **Overview**.

### KPI Cards (7 cards across the top)

For each card: **Visualizations** pane → **Card** visual → drag the measure into "Fields"

| Card # | Measure | Title (set in Format → Title) |
|---|---|---|
| 1 | `Total Records Tested` | Total Test Records |
| 2 | `Total Attacks Predicted` | Attacks Detected |
| 3 | `Avg Test Accuracy` | Average Accuracy |
| 4 | `Avg Attack Recall` | Attack Recall Rate |
| 5 | `Avg FPR` | False Positive Rate |
| 6 | `Best Model` | Best Performing Model |
| 7 | `Unique Attack Types` | Attack Categories |

For each card: Format → **Callout value** → set conditional formatting:
- Green for good metrics (Accuracy, Recall, Total Records)
- Red for bad metrics (FPR)

### Pie Chart: Normal vs Attack Distribution

- **Visualization**: Pie chart
- **Legend**: `PredictionLabel` (or "prediction_numeric" if using raw predictions)
- **Values**: `Count of Records`
- **Format**: Data labels show percentages

### Clustered Bar Chart: Accuracy by Model (on selected dataset)

- **Visualization**: Clustered bar chart
- **Axis**: `ModelAlgorithm`
- **Values**: `Avg Accuracy`, `Avg Attack Recall`
- **Legend**: Shows both metrics side-by-side

---

## STEP 6 — Build Page 2: Data Preprocessing

Rename **Page 2** to **Preprocessing**.

### KPI Cards (Top row)

| Measure | Title |
|---|---|
| `Total Records` | Total Records |
| `Normal Count` | Normal Samples |
| `Attack Count` | Attack Samples |
| `Attack Rate %` | Attack Rate |
| `Imbalance Ratio` | Class Imbalance |

### Stacked Bar Chart: Training Data Label Distribution

- **Visualization**: Stacked column chart
- **X-axis**: `Dataset`
- **Y-axis**: `Count of Records`  
- **Legend**: `AQI_Category` or `LabelCategory` (Good/Moderate/Unhealthy)
- Shows raw label distribution before binary conversion

### Bar Chart: Attack Category Frequency (Top 10)

- **Visualization**: Horizontal bar chart
- **Category**: `AttackCategory`
- **Values**: `Count of Attack Records`
- **Sort**: Descending by count
- Shows that Neptune, Ipsweep dominate; other attacks are rare

### Text Box: Preprocessing Pipeline

Create a text box listing the preprocessing steps:
```
PREPROCESSING PIPELINE
1. Load raw KDD data
2. Remove duplicates
3. Impute missing values (median for numeric, mode for categorical)
4. Encode categorical features (protocol_type, service, flag → one-hot)
5. Convert labels to binary (Normal=0, Attack=1)
6. Balance dataset (undersample majority class)
7. Split: 80% train, 20% validation
```

---

## STEP 7 — Build Page 3: Model Comparison

Rename **Page 3** to **Model Comparison**.

### KPI Row: By-Model Metrics

Create 3 sets of 4 KPI cards (one set per model: RandomForest, SVM, XGBoost):
- Model Accuracy
- Attack Recall
- False Positive Rate
- Total Predictions

Use **Slicers** on this page:
- **Dataset**: Dropdown to switch between Test-21 and Test+
- **Model**: Dropdown to highlight a specific model

### Clustered Column Chart: Accuracy by Model & Dataset

- **Visualization**: Clustered column chart
- **X-axis**: `ModelAlgorithm`
- **Y-axis**: `Avg Accuracy`
- **Legend**: `Dataset`
- Shows which model generalizes best across both test sets

### Matrix/Table: Detailed Model Metrics

- **Visualization**: Table or Matrix
- **Rows**: `ModelAlgorithm`
- **Values**: 
  - `Avg Accuracy`
  - `Avg Precision`
  - `Avg Recall`
  - `Avg F1_Score`
  - `Avg Attack Recall`
  - `Avg FPR`

Conditional formatting: Green for high values (Accuracy, Recall), Red for high FPR.

### Slicer (Right side)

- **Slicer**: `Dataset` — allows filtering between KDDTest+ and KDDTest-21
- All visuals above update when you change the slicer

---

## STEP 8 — Build Page 4: Cascade Classification

Rename **Page 4** to **Cascade Analysis** (focuses on XGBoost two-stage flow).

### KPI Cards (Top row)

These show cascade-specific metrics from the stage1 and stage2 calibration:

| Measure | Title |
|---|---|
| `Stage1 Binary Accuracy` | Stage 1 Accuracy |
| `Stage1 Attack Detection Rate` | Stage 1 Recall |
| `Stage2 Category Accuracy` | Stage 2 Accuracy |
| `Cascade FPR` | Cascade FPR |

### Flow Diagram: Decision Path (Text Box or Custom Visual)

Create a visual explanation:
```
STAGE 1 (Binary: Normal vs Attack)
├─ Input: Network traffic features
├─ Output: P(Attack) probability
├─ Threshold T_low: If P < T_low → Predict NORMAL (confident)
├─ Threshold T_strong: If P ≥ T_strong → Proceed to Stage 2
└─ Else: Use Stage 2 confidence to decide

STAGE 2 (Multi-class: Attack Category)
├─ Input: Same features
├─ Output: P(Category) for 4 categories (DoS, Probe, R2L, U2R)
├─ Decision: Max confidence, margin, threshold
└─ Final: Specific attack type (Neptune, Ipsweep, etc.)
```

### Stacked Column Chart: Stage 1 Decision Distribution

- **Visualization**: Stacked column chart
- **X-axis**: `DecisionPath` (Stage1_Normal, Stage2_Attack_Strong, Stage2_Confident, etc.)
- **Y-axis**: `Count of Records`
- **Legend**: `PredictionLabel` (Normal, Attack)

Shows how many records follow each decision path through the cascade.

### Table: Stage 2 Attack Category Breakdown

- **Visualization**: Table
- **Rows**: `Stage2PredCategory` (attack type)
- **Values**:
  - `Count of Records`
  - `% of Attacks Classified` (custom measure: count / total attacks)
  - `Confidence` (avg Stage2_TopConfidence)

---

## STEP 9 — Build Page 5: Attack Type Deep Dive

Rename **Page 5** to **Attack Analysis**.

### KPI Cards (Top row)

| Measure | Title |
|---|---|
| `Unique Attack Types` | Total Attack Types |
| `Most Common Attack` | Most Frequent Attack |
| `Rarest Attack Type` | Rarest Attack |
| `Category Accuracy` | Attack Classification Accuracy |

### Horizontal Bar Chart: Attack Type Frequency (Top 15)

- **Visualization**: Horizontal bar chart
- **Category**: `AttackCategory` (Neptune, Ipsweep, Satan, etc.)
- **Values**: `Count of Attack Records`
- **Sort**: Descending
- Shows the heavy imbalance (Neptune ~33% of attacks)

### Clustered Bar Chart: Correct vs Incorrect by Attack Type (Top 10)

- **Visualization**: Clustered bar chart
- **Category**: Top 10 attack types by frequency
- **Series 1**: Count where `IsCorrect = TRUE`
- **Series 2**: Count where `IsCorrect = FALSE`
- Shows which attack types are hardest to classify

### Heatmap: Attack Type Confusion (Actual vs Predicted)

- **Visualization**: Matrix or custom heatmap
- **Rows**: `ActualLabel` (true attack type)
- **Columns**: `PredictionLabel` (predicted attack type)
- **Values**: `Count of Records` (color intensity)
- **Format**: Conditional formatting (red=high misclassification)

This is the most powerful visual: shows if the model confuses Neptune with Ipsweep, etc.

---

## STEP 10 — Add Interactivity (Optional but Recommended)

### Add Slicers to Page 1 (Executive Overview)

- **Slicer 1**: `ModelAlgorithm` (RandomForest, SVM, XGBoost)
- **Slicer 2**: `Dataset` (KDDTest+, KDDTest-21)
- **Slicer 3**: `Year` (if your data spans multiple years)

All cards and charts update dynamically.

### Drillthrough Detail Page

1. **Insert** → **New page** → name it `Details`
2. Add a **Table** visual with all columns:
   - Original features (protocol_type, service, src_bytes, dst_bytes, etc.)
   - actual_label, prediction_numeric, correct_prediction
   - model_used, dataset
3. On Page 1, right-click any visual → **Drillthrough** → **Add drillthrough field** → `protocol_type` or `ModelAlgorithm`
4. Users can now right-click a bar in Page 1 and drill to the details table filtered to that model/dataset

---

## STEP 11 — Format and Polish

### Colors & Conditional Formatting

- **Normal/Harmless**: Green (#2ECC71)
- **Attack/Threat**: Red (#E74C3C)
- **Neutral/Metrics**: Blue or Gray

All your visuals are already themed by `CyberSentinel_Theme.json`.

### Add Report Title

- **Insert** → **Text box** at the top
- Type: `CyberSentinel — Network Intrusion Detection Dashboard`
- Font: 24pt, bold, color: dark gray

### Publish to Power BI Service (Optional)

1. **File** → **Publish** → sign in with your Microsoft account
2. Select a workspace
3. Your dashboard is now live and can be shared with your team

---

## QUICK REFERENCE — What Each Measure Does

| Measure | Purpose |
|---|---|
| `Total Records Tested` | Count of all test predictions |
| `Total Attacks Predicted` | Sum of predictions = "attack" or 1 |
| `Total Normal Predicted` | Sum of predictions = "normal" or 0 |
| `Avg Test Accuracy` | % of correct predictions |
| `Avg Attack Recall` | % of actual attacks correctly detected |
| `Avg Precision` | % of predicted attacks that were actually attacks |
| `Avg FPR` | % of normal traffic incorrectly flagged as attacks |
| `Best Model` | Algorithm with highest accuracy |
| `Unique Attack Types` | Count of distinct attack categories |
| `Attack Rate %` | Total attacks / total records |
| `Stage1 Binary Accuracy` | Cascade Stage 1 (Normal vs Attack) accuracy |
| `Stage2 Category Accuracy` | Cascade Stage 2 (Attack type) accuracy |
| `Cascade FPR` | False positive rate of final cascade predictions |

---

## TROUBLESHOOTING

**Q: Slicer doesn't update visuals**  
A: Make sure the slicer field is in the same table or has a relationship. Check **Model view** (Modeling tab) → ensure relationships exist.

**Q: DAX measure shows error**  
A: Check that table names and column names exactly match your CSV columns. DAX is case-sensitive.

**Q: CSV data doesn't load**  
A: Verify the file path in Step 2 line 9. Use full absolute path, not relative.

**Q: Colors don't match the theme**  
A: Re-apply the theme in Step 1. Clear your browser cache if using Power BI Service.

---

## NEXT STEPS

1. **Share the report**: File → Export → PDF (for executives) or Publish (for interactive sharing)
2. **Add more pages**: Duplicate Page 3 to compare different time periods or regions (if your data includes location)
3. **Set up alerts**: Power BI Service → set alerts on key metrics (e.g., "notify if FPR > 10%")
4. **Schedule refresh**: If using Power BI Service, set the CSV to auto-refresh daily

---

**Total build time: 15–25 minutes**  
**Result: A professional, interactive dashboard ready for stakeholder presentations**

Good luck! 🛡️
