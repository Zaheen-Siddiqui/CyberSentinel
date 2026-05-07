# CyberSentinel Power BI Dashboard Package

## 📦 What's Included

This package contains everything you need to build a professional Power BI dashboard for the CyberSentinel intrusion detection system. It follows the exact structure of the Mumbai AQI example project.

### Files in This Package

| File | Purpose |
|------|---------|
| **PowerBI_Build_Guide.md** | Complete step-by-step guide to building all 5 dashboard pages |
| **DAX_Measures.txt** | All ~35 calculated measures (KPIs, metrics, aggregations) |
| **PowerQuery_M_Code.txt** | Data transformation code for Power Query editor |
| **CyberSentinel_Theme.json** | Custom color theme (Green=Normal, Red=Attack) |
| **cyberssentinel_powerbi_combined.csv** | Consolidated reporting dataset (103,182 records) |
| **prepare_powerbi_data.py** | Script to generate/update the CSV from prediction files |

---

## 🚀 Quick Start (5 Steps)

### Step 1: Install Power BI Desktop
Download from: https://powerbi.microsoft.com/desktop/

### Step 2: Apply the Theme
1. Open Power BI Desktop → blank report
2. **View** → **Themes** → **Browse for themes** → Select `CyberSentinel_Theme.json`

### Step 3: Load the Data
1. **Home** → **Get data** → **Blank query**
2. **Advanced Editor** → Delete everything → Paste contents of `PowerQuery_M_Code.txt`
3. Update the file path on line 9 to point to `cyberssentinel_powerbi_combined.csv`
4. **Done** → **Close & Apply**

### Step 4: Create Measures Table
1. **Home** → **Enter data** → Name it `_Measures` → Load
2. **Modeling** → **New measure** for each measure in `DAX_Measures.txt` (~35 measures)
3. Takes ~5–7 minutes, but only done once

### Step 5: Build Your Pages
Follow the page-by-page instructions in `PowerBI_Build_Guide.md`:
- **Page 1**: Executive Overview
- **Page 2**: Data Preprocessing
- **Page 3**: Model Comparison
- **Page 4**: Cascade Analysis (XGBoost)
- **Page 5**: Attack Type Deep Dive

**Total time: 20–30 minutes**

---

## 📊 Dashboard Overview

### Page 1 — Executive Overview
KPIs: Total records, attacks detected, accuracy, attack recall, FPR
- 7 KPI cards at the top
- Pie chart: Normal vs Attack distribution
- Clustered bar chart: Model accuracy comparison

### Page 2 — Data Preprocessing
Class distribution and data quality
- Label distribution (Normal vs Attack)
- Attack category frequency (showing Neptune dominance)
- Imbalance ratio and preprocessing pipeline explanation

### Page 3 — Model Comparison
Side-by-side comparison of RandomForest, SVM, XGBoost
- Accuracy, Precision, Recall, F1, FPR for each model
- Performance across both test datasets (Test+ and Test-21)
- Slicer to filter by dataset or model

### Page 4 — Cascade Classification
Two-stage classification flow (XGBoost)
- Stage 1: Binary Normal vs Attack detection
- Stage 2: Attack-type classification (DoS, Probe, R2L, U2R)
- Decision path breakdown showing how records flow through stages

### Page 5 — Attack Type Deep Dive
Which attacks are detected, which are missed
- Top 15 attack types by frequency
- Confusion matrix: Actual attack type vs Predicted
- Misclassification heatmap showing which attacks are confused

---

## 📈 Key Metrics in the Dashboard

| Metric | Definition | Color |
|--------|-----------|-------|
| **Accuracy** | % of correct predictions | Green (high) |
| **Attack Recall** | % of actual attacks correctly detected | Green (high) |
| **Precision** | % of predicted attacks that were correct | Green (high) |
| **False Positive Rate** | % of normal traffic flagged as attack | Red (low is good) |
| **F1 Score** | Harmonic mean of Precision & Recall | Blue |
| **Attack Rate** | % of records that are attacks | Gray (informational) |

---

## 🛠️ Updating the Data

If you re-train your models or make new predictions:

1. Run: `python prepare_powerbi_data.py`
   - This reads all CSVs from `data/test/` and combines them
   - Outputs: `cyberssentinel_powerbi_combined.csv`

2. In Power BI:
   - **Home** → **Refresh** (top-left)
   - Or: **File** → **Options & settings** → **Data source settings** → Change path to CSV

3. Your dashboard updates instantly

---

## 🎨 Color Scheme

The `CyberSentinel_Theme.json` applies these colors:

| Color | Meaning | Hex |
|-------|---------|-----|
| 🟢 Green | Normal, Safe, Good Metrics | #2ECC71 |
| 🔴 Red | Attack, Threat, Bad Metrics | #E74C3C |
| 🔵 Blue | Neutral, Data Metrics | #3498DB |
| 🟡 Orange | Warning, Attention | #F39C12 |
| ⚫ Gray | Background, Grid | #95A5A6 |

---

## 📋 CSV Columns (What You'll See in Power BI)

### Network Features (41 original KDD columns)
- `duration`, `protocol_type`, `service`, `flag`
- `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`
- `hot`, `num_failed_logins`, `logged_in`, `num_compromised`
- ... and 31 more

### Classification Columns
- `actual_label` — Original label (normal, neptune, ipsweep, etc.)
- `actual_binary` — Binary label (0=Normal, 1=Attack)
- `prediction_numeric` — Model prediction (0 or 1)
- `correct_prediction` — Whether prediction was correct (0 or 1)

### Metadata Columns
- `model_used` — Algorithm (randomforest, svm, xgboost)
- `dataset` — Test set (KDDTest+, KDDTest-21)
- `AttackCategory` — Attack type (DoS, Probe, R2L, U2R, NORMAL)

### Metric Columns (Pre-calculated)
- `Accuracy` — Is this record's prediction correct?
- `Precision`, `Recall`, `F1Score` — Per-record (helper for aggregate measures)
- `FalsePositiveRate` — Binary FPR
- `AttackRecall` — Binary recall (attack detection rate)

---

## ❓ FAQ

**Q: Do I need all the prediction CSVs?**  
A: Yes, the script reads all 6 files (2 datasets × 3 models). If some are missing, the script will warn you but skip them.

**Q: Can I use just one model instead of all three?**  
A: Yes! Just comment out the prediction files you don't want in `prepare_powerbi_data.py` lines 134–144.

**Q: How do I add more pages?**  
A: Use the same measures and data in new pages. For example, duplicate Page 3 to compare different time periods if your data spans multiple months/years.

**Q: Can I export this as a PDF?**  
A: Yes! **File** → **Export** → **Export to PDF** (in Power BI Desktop, this exports all pages)

**Q: How do I share this with my team?**  
A: Upload to Power BI Service: **File** → **Publish** → Select a workspace → Share the link

**Q: What if my CSV file is very large?**  
A: Power BI can handle datasets up to 1 GB. The current CSV is ~30 MB, so plenty of room.

---

## 📚 Reference

- **Power BI Documentation**: https://docs.microsoft.com/power-bi/
- **DAX Functions**: https://dax.guide/
- **Power Query M Reference**: https://docs.microsoft.com/en-us/powerquery-m/power-query-m-function-reference
- **CyberSentinel Repository**: [Your project root]

---

## 🎓 Learning Resources

If you're new to Power BI:
1. **Visuals**: Start with the guide's Page 1 (simplest page)
2. **Measures**: Measures are formulas—think of them like Excel functions
3. **Slicers**: These are filter dropdowns that update all visuals instantly
4. **Drill-through**: Right-click a chart element to drill to detail pages

---

## ✅ Checklist Before Building

- [ ] Power BI Desktop installed
- [ ] `cyberssentinel_powerbi_combined.csv` exists in the project root
- [ ] `CyberSentinel_Theme.json` exists
- [ ] `PowerQuery_M_Code.txt` has the correct file path (line 9)
- [ ] `DAX_Measures.txt` is ready to copy/paste
- [ ] `PowerBI_Build_Guide.md` is open for reference

---

## 🤝 Support

If something doesn't work:
1. Check the **TROUBLESHOOTING** section in `PowerBI_Build_Guide.md`
2. Verify file paths and column names (DAX is case-sensitive)
3. Make sure your CSV has all required columns
4. Clear Power BI cache and restart if visuals don't update

---

## 📝 License & Attribution

This Power BI package is designed for the CyberSentinel intrusion detection project.  
Based on the structure and approach of the Mumbai AQI dashboard.

---

**Happy dashboarding! 🛡️📊**

Last Updated: 2026-05-04
