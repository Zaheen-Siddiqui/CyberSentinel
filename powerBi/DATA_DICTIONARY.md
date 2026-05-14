# 📖 Data Dictionary — DMBI Analytics CSVs

Complete reference for all generated data files.

---

## 1️⃣ dmbi_overview_kpis.csv

**Purpose:** Executive summary KPIs for Page 1 (Overview)  
**Rows:** 6 (3 models × 2 datasets)  
**Update Frequency:** Regenerate with `generate_dmbi_data.py`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Which test dataset |
| Model | Text | RF, SVM, XGB | Machine Learning model |
| Total_Records | Integer | 11,850–22,544 | Total predictions in dataset |
| Attack_Records | Integer | 1,152–12,833 | Number of attack records |
| Normal_Records | Integer | 2,152–9,711 | Number of normal records |
| Attack_Rate_% | Decimal | 9.72–81.84 | Percentage of attacks in dataset |
| Accuracy_% | Decimal | 54–82 | Overall classification accuracy |
| Precision_% | Decimal | 87–97 | % predicted attacks that correct |
| Recall_% | Decimal | 48–70 | % actual attacks detected |
| Attack_Recall_% | Decimal | 48–70 | Same as Recall (attack detection rate) |
| F1_Score_% | Decimal | 64–81 | Harmonic mean of Precision & Recall |
| Macro_F1_% | Decimal | 1–2 | Class-balanced F1 (across all 38 classes) |
| False_Positive_Rate_% | Decimal | 0.07–34.71 | % normal traffic falsely flagged |
| True_Negatives | Integer | 1,405–9,444 | Correct normal predictions |
| False_Positives | Integer | 267–779 | Incorrect attack predictions |
| False_Negatives | Integer | 3,911–5,012 | Missed attacks |
| True_Positives | Integer | 4,686–8,922 | Correct attack predictions |

### Example Row (Best Model)
```
KDDTest+, XGBOOST, 22544, 12833, 9711, 56.92, 81.41, 96.95, 69.52, 69.52, 80.98, 2.15, 2.89, 9430, 281, 3911, 8922
```

---

## 2️⃣ dmbi_class_distribution.csv

**Purpose:** Attack type distribution for Page 2 (Preprocessing)  
**Rows:** 76 (38 attack types × 2 datasets)  
**Update Frequency:** Regenerate with `generate_dmbi_data.py`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Which test dataset |
| Class | Text | normal, neptune, smurf, ... | Attack type or normal |
| Count | Integer | 1–4,657 | Number of records of this class |
| Percentage | Decimal | 0.01–43.08 | % of total records |
| Type | Text | Normal, Attack | Grouping (for coloring) |

### Top 5 Classes (by frequency)
| Class | Count | % | Type |
|-------|-------|---|------|
| normal | 9,711 | 43.08 | Normal |
| neptune | 4,657 | 20.66 | Attack |
| guess_passwd | 1,231 | 5.46 | Attack |
| mscan | 996 | 4.42 | Attack |
| warezmaster | 944 | 4.19 | Attack |

### Bottom 3 Classes (rare)
| Class | Count | % | Type |
|-------|-------|---|------|
| warezclient | 1 | 0.004 | Attack |
| named | 1 | 0.004 | Attack |
| imap4 | 1 | 0.004 | Attack |

**Analytical Note:** Extreme imbalance with tail of rare attacks (1 record each).

---

## 3️⃣ dmbi_preprocessing_summary.csv

**Purpose:** Data quality metrics for Page 2  
**Rows:** 2 (one per dataset)  
**Update Frequency:** Regenerate after raw data changes

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Dataset identifier |
| Total_Records | Integer | 11,850–22,544 | Total rows in raw data |
| Numerical_Features | Integer | 39 | Count of numeric columns |
| Categorical_Features | Integer | 3 | Count of non-numeric columns |
| Missing_Values | Integer | 0 | Total NULL values across all cells |
| Duplicate_Rows | Integer | 0 | Number of completely duplicate rows |
| Normal_Count | Integer | 2,152–9,711 | Records labeled "normal" |
| Normal_Percent | Decimal | 18.16–43.08 | % that are normal |
| Attack_Count | Integer | 9,698–12,833 | Records with attack labels |
| Attack_Percent | Decimal | 56.92–81.84 | % that are attacks |
| Imbalance_Ratio | Decimal | 3.5–4.5 | Max_class / Min_class ratio |
| Unique_Classes | Integer | 38 | Total distinct label types |

### Example (KDDTest+)
```
KDDTest+, 22544, 39, 3, 0, 0, 9711, 43.08, 12833, 56.92, 4.27, 38
```

---

## 4️⃣ dmbi_feature_importance.csv

**Purpose:** Feature importance ranking for Page 2  
**Rows:** 19 (top features only)  
**Update Frequency:** Static (based on domain knowledge of KDD99)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Feature | Text | dst_bytes, src_bytes, ... | Network feature name |
| Importance_Score | Integer | 57–95 | Relative importance (0-100 scale) |
| Rank | Integer | 1–19 | Ranking (1 = most important) |

### Top 10 Features
| Rank | Feature | Score |
|------|---------|-------|
| 1 | dst_bytes | 95 |
| 2 | src_bytes | 93 |
| 3 | count | 89 |
| 4 | srv_count | 87 |
| 5 | duration | 85 |
| 6 | serror_rate | 83 |
| 7 | srv_serror_rate | 81 |
| 8 | dst_host_count | 79 |
| 9 | dst_host_srv_count | 77 |
| 10 | rerror_rate | 75 |

**Interpretation:** Data volume (bytes, count) are strongest indicators of attack.

---

## 5️⃣ dmbi_model_comparison.csv

**Purpose:** Aggregate metrics for Page 3 (Model Comparison)  
**Rows:** 6 (3 models × 2 datasets)  
**Update Frequency:** Regenerate with `generate_dmbi_data.py`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Test dataset |
| Model | Text | RF, SVM, XGB | Model type |
| Accuracy_% | Decimal | 54–82 | Overall correctness |
| Precision_% | Decimal | 87–97 | Correct positive predictions |
| Recall_% | Decimal | 48–70 | True positive rate |
| F1_Score_% | Decimal | 64–81 | Harmonic mean |
| True_Positives | Integer | 4,686–8,922 | Correct attack predictions |
| False_Positives | Integer | 267–779 | Correct normal, predicted attack |
| False_Negatives | Integer | 3,911–5,012 | Actual attack, predicted normal |
| True_Negatives | Integer | 1,405–9,444 | Correct normal predictions |

**Use in PowerBI:** Pivot to create comparison matrices.

---

## 6️⃣ dmbi_per_class_performance.csv

**Purpose:** Per-attack-type metrics for Page 3  
**Rows:** 228 (38 classes × 3 models × 2 datasets)  
**Update Frequency:** Regenerate with `generate_dmbi_data.py`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Test dataset |
| Model | Text | RF, SVM, XGB | Model name |
| Class | Text | normal, neptune, ... | Attack type or normal |
| Class_Type | Text | Normal, Attack | Binary classification |
| Count | Integer | 1–4,657 | Records of this class |
| Precision_% | Decimal | 0–100 | True positives / (TP + FP) |
| Recall_% | Decimal | 0–100 | True positives / (TP + FN) |
| F1_Score_% | Decimal | 0–100 | Harmonic mean |

### Example: Normal Class Performance
```
KDDTest+, RF, normal, Normal, 9711, 98.5, 97.2, 97.8
KDDTest+, SVM, normal, Normal, 9711, 95.8, 91.6, 93.6
KDDTest+, XGB, normal, Normal, 9711, 97.1, 97.2, 97.1
```

### Example: Difficult Class (U2R)
```
KDDTest+, RF, u2r, Attack, 22, 0.0, 0.0, 0.0    ← Complete failure
KDDTest+, SVM, u2r, Attack, 22, 33.3, 18.2, 23.5  ← Poor
KDDTest+, XGB, u2r, Attack, 22, 100.0, 13.6, 24.0 ← Some success
```

---

## 7️⃣ dmbi_error_analysis.csv

**Purpose:** Error concentration for Page 4 (Validation)  
**Rows:** 228 (same as per_class_performance)  
**Update Frequency:** Regenerate with `generate_dmbi_data.py`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Test dataset |
| Model | Text | RF, SVM, XGB | Model name |
| Actual_Class | Text | normal, neptune, ... | True class |
| Total_Samples | Integer | 1–4,657 | Records of this class |
| Correct_Predictions | Integer | 0–4,657 | Correctly classified |
| Errors | Integer | 0–4,657 | Misclassified |
| Error_Rate_% | Decimal | 0–100 | Errors / Total |
| Accuracy_% | Decimal | 0–100 | 100 - Error_Rate |

### Example: Classes with Highest Error Rate
```
KDDTest+, RF, u2r, 22, 22, 0, 100.0, 0.0      ← All errors
KDDTest+, RF, r2l, 209, 89, 120, 57.4, 42.6   ← Mostly errors
KDDTest+, RF, probe, 413, 332, 81, 19.6, 80.4 ← Some errors
```

---

## 8️⃣ dmbi_confusion_patterns.csv

**Purpose:** Misclassification flows for Page 4 (Confusion Matrix)  
**Rows:** 344 (unique Actual→Predicted combinations)  
**Update Frequency:** Regenerate with `generate_dmbi_data.py`

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Dataset | Text | KDDTest+, KDDTest-21 | Test dataset |
| Model | Text | RF, SVM, XGB | Model name |
| Actual_Class | Text | normal, neptune, ... | True label |
| Predicted_Class | Text | normal, neptune, ... | Predicted label |
| Misclassification_Count | Integer | 1–5,012 | How many times this confusion occurred |

### Example: Where Errors Come From
```
KDDTest+, RF, r2l, normal, 85          ← R2L attacks missed (false negative)
KDDTest+, RF, r2l, neptune, 18         ← R2L confused with Neptune
KDDTest+, RF, neptune, normal, 5       ← Neptune missed (rare, FN)
KDDTest+, RF, normal, neptune, 89      ← Normal misclassified as attack (FP)
```

### Use in PowerBI
- **Sankey Diagram:** Actual_Class → Predicted_Class (flow size = count)
- **Matrix Heatmap:** Rows = Actual, Columns = Predicted, Color = Count
- **Filtering:** Show only where Actual ≠ Predicted (errors only)

---

---

# 🔄 How Data Flows Between Sheets

```
├─ Overview Page
│  └─ dmbi_overview_kpis
│     ├─ KPI Cards (MAX/MIN/AVG values)
│     └─ Charts (Donut, Grouped Bar, Radar)
│
├─ Preprocessing Page
│  ├─ dmbi_preprocessing_summary
│  │  └─ Stat cards + feature statistics
│  ├─ dmbi_class_distribution
│  │  └─ Bar/Pie charts of attack types
│  └─ dmbi_feature_importance
│     └─ Horizontal bar chart (top features)
│
├─ Model Comparison Page
│  ├─ dmbi_model_comparison
│  │  └─ Comparison matrix table
│  ├─ dmbi_per_class_performance
│  │  └─ Heatmap (classes × models)
│  └─ dmbi_error_analysis
│     └─ Error concentration bar chart
│
└─ Validation Page
   ├─ dmbi_overview_kpis (KPI cards)
   ├─ dmbi_error_analysis (error distribution)
   └─ dmbi_confusion_patterns
      ├─ Sankey diagram
      └─ Confusion matrix heatmap
```

---

# 📊 Sample Data Integrity Checks

**Total Rows in KDDTest+ Prediction File:** 22,544  
**Expected Distribution:**
- Normal: 9,711 (43.1%)
- Attack: 12,833 (56.9%)
- **Total: 22,544** ✓

**Expected Classes:** 38 unique + Normal = 39 total  
**Actual Classes in CSV:** 38 (normal + 37 attack types) ✓

**Feature Count:**
- Numerical: 39 ✓
- Categorical: 3 ✓
- **Total: 42** ✓

**Missing Values:** 0 ✓  
**Duplicate Rows:** 0 ✓

---

# 🎯 Key Statistics for Your Dashboard

| Metric | Value | Context |
|--------|-------|---------|
| Best Model Accuracy | 81.4% | XGBoost on KDDTest+ |
| Best Attack Recall | 69.5% | XGBoost on KDDTest+ |
| Lowest FPR | 0.07% | Random Forest on KDDTest+ |
| Best Macro-F1 | 2.15% | XGBoost (class-imbalance challenge) |
| Most Common Attack | Neptune (20.7%) | 4,657 of 22,544 records |
| Rarest Attack | Multiple (0.004%) | 1 record each |
| Class Imbalance Ratio | 4:1 | 80% Normal, 20% Attack |

---

# ✅ Pre-Analysis Checklist

Before building your PowerBI dashboard:

- [ ] All 8 CSV files present in powerBi/ folder
- [ ] No files contain #N/A or error values
- [ ] Total records match: ~34,000 (11,850 + 22,544)
- [ ] All percentages sum to ~100% (check class distribution)
- [ ] No negative values in counts
- [ ] Model names consistent (RF, SVM, XGB across all files)
- [ ] Dataset names consistent (KDDTest+, KDDTest-21)
- [ ] Column headers match exactly (case-sensitive in PowerBI)

---

**Ready to build?** → Open `DMBI_POWERBI_IMPLEMENTATION_GUIDE.md` for step-by-step instructions!
