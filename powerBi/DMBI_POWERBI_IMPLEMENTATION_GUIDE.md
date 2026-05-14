# DMBI Analytics Dashboard — PowerBI Implementation Guide

**Focus:** Data insights, mining analytics, validation trends  
**Tone:** Professional business intelligence report  
**Audience:** Academic evaluators, data analysts  

---

## 📋 Dashboard Structure (4 Pages)

| Page | Purpose | Key Visuals | Analytical Focus |
|------|---------|-------------|------------------|
| **1. Overview** | Executive analytics summary | KPI cards, pie chart, bar chart | Best metrics, dataset context |
| **2. Preprocessing** | Data understanding & quality | Distribution charts, correlation heatmap, feature analysis | Data transformation, imbalance, feature relationships |
| **3. Model Comparison** | Comparative ML analytics | Grouped bars, scatter plot, heatmaps | Performance tradeoffs, class-level metrics |
| **4. Validation Summary** | Reliability & error analysis | Error distribution, confusion matrix, threshold sensitivity | Prediction stability, misclassification patterns |

---

## 📁 Data Files Required

All files must be in the `powerBi/` folder:

```
dmbi_overview_kpis.csv              → Page 1: Executive KPIs
dmbi_class_distribution.csv          → Page 2: Class distribution
dmbi_preprocessing_summary.csv       → Page 2: Data quality
dmbi_feature_importance.csv          → Page 2: Feature analysis
dmbi_model_comparison.csv            → Page 3: Model metrics
dmbi_per_class_performance.csv       → Page 3: Per-class analytics
dmbi_error_analysis.csv              → Page 4: Error concentration
dmbi_confusion_patterns.csv          → Page 4: Misclassifications
```

---

# 🎨 THEME & FORMATTING

## Color Palette

**Professional DMBI Theme:**
- Primary: `#0078D4` (Azure Blue) — for primary metrics
- Success: `#107C10` (Green) — for strong performance
- Warning: `#FFB900` (Amber) — for moderate performance
- Danger: `#D83B01` (Red) — for weak/concerning metrics
- Neutral: `#8A8886` (Gray) — for neutral information
- Background: `#F3F2F1` (Light Gray) — professional appearance

**Text Formatting:**
- Headers: Bold, 14pt+
- Values: 20-28pt (KPI cards)
- Labels: 11-12pt, sans-serif
- Tooltip text: Clear, concise

---

# PAGE 1: OVERVIEW

**Question:** *"What are the major analytical findings from the experiment?"*

## Layout

```
┌─────────────────────────────────────────────────────────┐
│  [Best Accuracy]  [Best Recall]  [Best Macro F1]       │
│   [Best Precision]  [Lowest FPR]  [Total Records]      │
└─────────────────────────────────────────────────────────┘
                         
┌──────────────────────┐  ┌──────────────────┐  ┌────────┐
│ Dataset Composition  │  │ Performance      │  │ Key    │
│ (Pie/Donut)        │  │ Snapshot         │  │Insights│
│                     │  │ (Grouped Bars)   │  │(Text)  │
└──────────────────────┘  └──────────────────┘  └────────┘

┌──────────────────────────────────────────────────────────┐
│ Model Performance Profiles (Line/Radar Chart)           │
└──────────────────────────────────────────────────────────┘
```

### TOP SECTION: KPI Cards (3 rows × 2-3 cards)

**Create 6 KPI Cards:**

| KPI | Measure | Format | Color |
|-----|---------|--------|-------|
| **Best Accuracy** | MAX of dmbi_overview_kpis[Accuracy_%] | % (rounded) | Green if >95% else Amber |
| **Best Attack Recall** | MAX of dmbi_overview_kpis[Attack_Recall_%] | % (rounded) | Green if >95% else Red |
| **Best Macro F1** | MAX of dmbi_overview_kpis[Macro_F1_%] | % (rounded) | Green if >90% else Amber |
| **Lowest FPR** | MIN of dmbi_overview_kpis[False_Positive_Rate_%] | % (1 decimal) | Green if <1% else Orange |
| **Total Records** | SUM of dmbi_overview_kpis[Total_Records] | Formatted number | Gray |
| **Attack Classes** | DISTINCTCOUNT of dmbi_class_distribution[Class] | Count | Gray |

**Formatting:**
- Card size: Large (3×3 tile)
- Font size: 28pt for value, 12pt for label
- Background: White with subtle shadow

---

### LEFT SECTION: Dataset Composition

**Visual Type:** Donut Chart

**Data Source:** `dmbi_class_distribution`

**Configuration:**
- **Values:** Sum of dmbi_class_distribution[Count]
- **Legend:** dmbi_class_distribution[Class]
- **Colors:** Normal = Green (#107C10), Attacks = Red (#D83B01)
- **Data Labels:** Both count and percentage

**Analytical Value:**
> Shows immediate class imbalance. Normal traffic dominates (~80%), attacks are minority (~20%). This context is critical for understanding why F1 and Macro-F1 matter more than accuracy alone.

---

### CENTER SECTION: Performance Snapshot

**Visual Type:** Grouped Column Chart

**Data Source:** `dmbi_model_comparison`

**Configuration:**
- **X-axis:** Model (RF, SVM, XGB)
- **Y-axis:** Value (0-100%)
- **Legend:** Accuracy, Precision, Recall, F1
- **Data labels:** Show value on bars

**Metric Colors:**
- Accuracy: #0078D4
- Precision: #107C10
- Recall: #FFB900
- F1: #D83B01

**Analytical Value:**
> At-a-glance model performance comparison. Shows which metrics each model excels at. Highlights the accuracy-recall tradeoff.

---

### RIGHT SECTION: Key Analytical Insights

**Visual Type:** Text Box / Shape with Insights

**Content Example:**
```
📊 ANALYTICAL FINDINGS

1. Tree-based models (RF, XGB) show superior 
   recall consistency across both datasets.
   
2. Precision-recall tradeoff varies significantly:
   - RF: Balanced performance (Recall 99.7%, Precision 97%)
   - SVM: Lower recall (96.6%), higher precision (97%)
   
3. False-positive rate well-controlled:
   All models maintain FPR <0.1% (critical for ops)
   
4. Macro-F1 exposes class imbalance impact:
   Best Macro-F1: 92%, showing minority class
   challenge in attack-type classification.
   
5. Minority attacks (U2R, R2L) remain difficult
   to classify reliably across all models.
```

**Formatting:**
- Font: 11pt, left-aligned
- Background: Light Gray (#F3F2F1)
- Border: Subtle
- Icons/symbols: Use for visual hierarchy

---

### BOTTOM SECTION: Model Performance Profiles

**Visual Type:** Radar Chart

**Data Source:** `dmbi_model_comparison` (pivoted)

**Configuration:**
- **Axes:** Accuracy, Precision, Recall, Macro F1, FPR (inverted)
- **Series:** One per model (RF, SVM, XGB)
- **Colors:** RF = Blue, SVM = Teal, XGB = Green

**Analytical Value:**
> Visual "fingerprint" of each model. Shows at-a-glance strengths and weaknesses. Radar shape reveals specializations.

---

---

# PAGE 2: PREPROCESSING

**Question:** *"How was the dataset transformed into an analyzable structure?"*

## Layout

```
┌──────────────────────────────────────────────────────┐
│ Raw Data Analysis (Stats Cards)                     │
└──────────────────────────────────────────────────────┘

┌────────────────────────────────┐  ┌───────────────────┐
│ Class Distribution             │  │ Feature Analysis  │
│ (Horizontal Bar Chart)        │  │ (Column Chart)    │
└────────────────────────────────┘  └───────────────────┘

┌──────────────────────────────────────────────────────┐
│ Feature Importance Ranking (Bar Chart)              │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Imbalance & Transformation Process (Info Box)       │
└──────────────────────────────────────────────────────┘
```

### TOP SECTION: Raw Data Analysis Stats

**Visual Type:** 5 Text Box Cards

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Total Records** | SUM(dmbi_preprocessing_summary[Total_Records]) | Dataset size for context |
| **Numerical Features** | AVERAGE(dmbi_preprocessing_summary[Numerical_Features]) | Dimensionality indicator |
| **Missing Values** | SUM(dmbi_preprocessing_summary[Missing_Values]) | Data quality check |
| **Duplicate Rows** | SUM(dmbi_preprocessing_summary[Duplicate_Rows]) | Preprocessing rigor |
| **Imbalance Ratio** | AVERAGE(dmbi_preprocessing_summary[Imbalance_Ratio]) | Severity of class imbalance |

**Layout:** 5 cards in a row, each 3×1 tile

---

### LEFT SECTION: Class Distribution

**Visual Type:** Horizontal Stacked Bar Chart

**Data Source:** `dmbi_class_distribution`

**Configuration:**
- **Axis:** dmbi_class_distribution[Dataset]
- **Values:** SUM of Count
- **Legend:** dmbi_class_distribution[Class] (top 10 attack types + normal)
- **Sorting:** Descending by count

**Color Scheme:**
- Normal: Green (#107C10)
- Each attack type: Different red/orange shade

**Data Labels:** Show percentages

**Analytical Value:**
> Reveals severe imbalance: Normal traffic dominates (>80%), specific attacks (Neptune, Smurf) account for majority of attacks. Minority classes (U2R, R2L) are extremely rare, explaining classification difficulty.

---

### RIGHT SECTION: Feature Statistics

**Visual Type:** Column Chart

**Data Source:** `dmbi_preprocessing_summary` (transposed)

**Configuration:**
- **Categories:** Dataset (KDDTest+, KDDTest-21)
- **Values:** [Numerical_Features, Categorical_Features, Missing_Values]
- **Data labels:** Show value

**Analytical Value:**
> Shows consistency across datasets: 39 numerical features, 3 categorical features, no missing values. This indicates clean preprocessing.

---

### CENTER SECTION: Feature Importance

**Visual Type:** Horizontal Bar Chart (Top 10)

**Data Source:** `dmbi_feature_importance`

**Configuration:**
- **Values:** SUM of Importance_Score
- **Legend:** dmbi_feature_importance[Feature]
- **Sort:** Descending by score
- **Data labels:** Show rank + score

**Top Features Expected:**
- dst_bytes, src_bytes (data volume)
- count, srv_count (connection statistics)
- serror_rate, srv_serror_rate (error metrics)

**Analytical Value:**
> Shows which features drive the ML model decisions. Data flow characteristics (bytes, counts) dominate, while error rates are strong indicators of attack presence.

---

### BOTTOM SECTION: Imbalance & Transformation Process

**Visual Type:** Info Box / Process Diagram (Text/Shapes)

**Content:**
```
DATA TRANSFORMATION PIPELINE

Raw Data (34,394 records)
    ↓
[Encoding] Categorical → Numerical (protocol, service, flag)
    ↓
[Scaling] Normalize 0-1 range for numerical features
    ↓
[Label Transformation] Attack types → Binary (Normal/Attack)
    ↓
[Feature Selection] 39 most relevant features retained
    ↓
Analytical Dataset (34,394 × 42 features)

KEY CHALLENGE: Class Imbalance
    Normal:  27,566 (80.3%)  [Majority class]
    Attack:  6,828  (19.7%)  [Minority class]
    
Imbalance Ratio: 4:1
Impact: Accuracy alone insufficient → Use Recall, F1, Macro-F1

Attack Subtypes (show 5 rarest):
    U2R:      52 (0.2%)   — Hardest to detect
    R2L:     209 (0.6%)   — High false-negative rate
    Probe:   413 (1.2%)   — Moderate difficulty
    Smurf: 2,807 (8.2%)   — Dominant attack type
    Neptune: 3,235 (9.4%) — Most common
```

**Formatting:**
- Monospace for diagram
- Color-code the imbalance severity (Red for extreme imbalance)
- Include a small note: "Minority classes require special attention"

---

---

# PAGE 3: MODEL COMPARISON

**Question:** *"Which model performs best under different evaluation criteria?"*

## Layout

```
┌──────────────────────────────────────────────────────┐
│ Slicers: [Model] [Metric] [Dataset]                 │
└──────────────────────────────────────────────────────┘

┌────────────────────────────┐  ┌─────────────────────┐
│ Comparison Matrix          │  │ Precision-Recall    │
│ (Table)                    │  │ Tradeoff (Scatter)  │
└────────────────────────────┘  └─────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Per-Class Performance Heatmap                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Error Concentration Analysis                        │
└─────────────────────────────────────────────────────┘
```

### TOP SECTION: Interactive Slicers

**Create 3 Slicers:**

1. **Model Slicer** (Buttons or Dropdown)
   - Options: RF, SVM, XGB
   - Default: All

2. **Metric Slicer** (Dropdown)
   - Options: Accuracy, Precision, Recall, F1, Macro F1, FPR
   - Default: All

3. **Dataset Slicer** (Buttons)
   - Options: KDDTest+, KDDTest-21, Both
   - Default: Both

**Filtering:** All visuals on this page should respond to these slicers

---

### LEFT SECTION: Comparison Matrix

**Visual Type:** Conditional Formatted Table

**Data Source:** `dmbi_model_comparison`

**Columns:**
| Column | Source | Format |
|--------|--------|--------|
| Model | Model | Text |
| Dataset | Dataset | Text |
| Accuracy | Accuracy_% | % (conditional color) |
| Precision | Precision_% | % (conditional color) |
| Recall | Recall_% | % (conditional color) |
| F1 Score | F1_Score_% | % (conditional color) |
| TP | True_Positives | Integer |
| FP | False_Positives | Integer |
| FN | False_Negatives | Integer |
| TN | True_Negatives | Integer |

**Conditional Formatting Rules:**
- Accuracy >98%: Dark Green
- Accuracy 95-98%: Light Green
- Accuracy <95%: Orange/Red
- Similar for Recall, FPR (inverted: lower FPR = green)

**Analytical Value:**
> Side-by-side comparison reveals each model's strengths:
> - RF: Best accuracy (99.8%), excellent recall (99.7%)
> - XGB: Competitive with RF, slightly better F1
> - SVM: Lower recall (96.6%), slightly higher precision

---

### RIGHT SECTION: Precision-Recall Tradeoff

**Visual Type:** Scatter Plot (Bubble Chart)

**Data Source:** `dmbi_model_comparison`

**Configuration:**
- **X-axis:** Recall_% (horizontal)
- **Y-axis:** Precision_% (vertical)
- **Bubble size:** F1_Score_% (larger = better F1)
- **Bubble color:** Model (RF=Blue, SVM=Teal, XGB=Green)
- **Legend:** Model
- **Data labels:** Model name + Metric combo

**Analytical Insight:**
> Shows the fundamental tradeoff: high recall often means lower precision. Best performers (RF, XGB) cluster in the high-recall, high-precision region. This scatter illustrates why Macro-F1 matters: it balances precision and recall.

**Expected Pattern:**
```
    100% ┌────────────────────────────┐
  P      │  RF • XGB (high recall,    │
  r      │  high precision ideal)     │
  e  95% │                            │
  c      │                       • SVM│
  i      │                            │
  s  90% └────────────────────────────┘
    85%     95%      97%       99%   101%
             ← Recall % →
```

---

### CENTER SECTION: Per-Class Performance Heatmap

**Visual Type:** Heatmap (Matrix Visual)

**Data Source:** `dmbi_per_class_performance`

**Configuration:**
- **Rows:** Class (Normal, Neptune, Smurf, Probe, R2L, U2R, etc.)
- **Columns:** Model (RF, SVM, XGB)
- **Values:** F1_Score_% (color-coded)
- **Values to display:** Also show Recall_% as secondary metric

**Color Scale:**
- 90-100%: Dark Green
- 80-90%: Light Green
- 70-80%: Yellow
- 50-70%: Orange
- <50%: Red

**Sorting:** By overall F1 score

**Analytical Value:**
> Reveals which models excel at which attack types:
> - Normal traffic: All models perform well (>99%)
> - Neptune/Smurf (common): All models >95%
> - Probe: Slight difficulty (80-95%)
> - R2L: Significant challenge (40-70%)
> - U2R: Extreme difficulty (0-40%)

---

### BOTTOM SECTION: Error Concentration Analysis

**Visual Type:** Stacked Column Chart + Table

**Data Source:** `dmbi_error_analysis`

**Configuration:**
- **X-axis:** Class (sorted by error count descending)
- **Y-axis:** Error_Rate_% (0-100%)
- **Color:** Gradient Red (higher = more problematic)
- **Data labels:** Show error count + rate

**Table below chart:**
| Class | Total Samples | Errors | Error Rate | Model |
|-------|---------------|--------|-----------|-------|
| ... | ... | ... | ... | ... |

**Analytical Value:**
> Shows which attack classes generate most errors. U2R and R2L have highest error rates (>50%), while common attacks (Neptune, Smurf) have <5% error rate. This explains why the model needs specialist attention for rare attacks.

---

---

# PAGE 4: VALIDATION SUMMARY

**Question:** *"How reliable and stable are the predictions?"*

## Layout

```
┌──────────────────────────────────────────────────────┐
│ Validation KPIs (Accuracy, Recall, FPR, Macro-F1)  │
└──────────────────────────────────────────────────────┘

┌────────────────────────────┐  ┌─────────────────────┐
│ Misclassification Pattern   │  │ Error by Class      │
│ (Sankey / Alluvial)        │  │ (Pie/Bar)          │
└────────────────────────────┘  └─────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Confusion Matrix Heatmap (XGBoost as reference)    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Reliability Assessment (Text Summary)               │
└─────────────────────────────────────────────────────┘
```

### TOP SECTION: Validation KPI Cards

**Create 4 Cards:**

| KPI | Measure | Interpretation |
|-----|---------|-----------------|
| **Validation Accuracy** | AVERAGE(dmbi_overview_kpis[Accuracy_%]) | Overall correct predictions |
| **Validation Recall** | AVERAGE(dmbi_overview_kpis[Attack_Recall_%]) | Attack detection reliability |
| **Validation FPR** | AVERAGE(dmbi_overview_kpis[False_Positive_Rate_%]) | False alarm rate (lower=better) |
| **Validation Macro-F1** | AVERAGE(dmbi_overview_kpis[Macro_F1_%]) | Class-balanced reliability |

**Color Coding:**
- Green: >95% (for positive metrics) or <1% (for FPR)
- Amber: 90-95%
- Red: <90%

---

### LEFT SECTION: Misclassification Patterns

**Visual Type:** Sankey Diagram (if available) OR Grouped Bar Chart

**Data Source:** `dmbi_confusion_patterns`

**Configuration (if Sankey):**
- **Source:** Actual_Class
- **Target:** Predicted_Class
- **Value:** Misclassification_Count
- **Show top 10 misclassification flows**

**Configuration (if Bar Chart):**
- **Categories:** Actual_Class
- **Values:** Count by Predicted_Class
- **Stacked or grouped**

**Analytical Value:**
> Shows how errors flow: which classes are confused with which. Example:
> - R2L often confused with Normal (missed attacks)
> - U2R sometimes confused with Probe
> - Neptune almost never confused (robust detection)

---

### RIGHT SECTION: Error Distribution by Model

**Visual Type:** Pie or Doughnut Chart

**Data Source:** `dmbi_error_analysis`

**Configuration:**
- **Legend:** Model (RF, SVM, XGB)
- **Values:** SUM(Errors)

**Analytical Value:**
> Shows which model has most/fewest errors. All models should be similar; if one is significantly worse, flag for investigation.

---

### CENTER SECTION: Confusion Matrix Heatmap

**Visual Type:** Matrix/Heatmap

**Data Source:** `dmbi_confusion_patterns` (aggregated by actual vs predicted)

**Configuration:**
- **Rows:** Actual_Class
- **Columns:** Predicted_Class
- **Values:** Misclassification_Count (color-coded)
- **Show diagonal separately** (correct predictions)

**Color Scale:**
- White/Light: Few misclassifications
- Red: Many misclassifications

**Analytical Value:**
> Visual identification of problematic class pairs:
> - Diagonal should be brightest (correct classifications)
> - Off-diagonal shows confusion patterns
> - Highlights which attacks are hardest to distinguish

**Example Pattern (binary level):**
```
                 Predicted
         Normal    Attack
Actual
Normal     ■■■       ▢       (Few false positives ✓)
Attack     ▢▢        ■■■     (Few false negatives ✓)
```

---

### BOTTOM SECTION: Reliability Assessment

**Visual Type:** Info Box / Text Summary

**Content:**

```
🔍 VALIDATION RELIABILITY ASSESSMENT

OVERALL RELIABILITY: ★★★★★ STRONG
   - Accuracy: 99.1% (Excellent)
   - Recall: 99.7% (Can detect 99.7% of attacks)
   - FPR: 0.07% (Minimal false alarms)
   - Macro-F1: 92.3% (Balanced across classes)

STRENGTHS:
   ✓ Tree-based models (RF, XGB) demonstrate excellent 
     recall (>99%), critical for attack detection
   ✓ False-positive rate well-controlled (<0.1%), 
     reducing operational alert fatigue
   ✓ Consistent performance across datasets 
     (KDDTest+ and KDDTest-21)
   ✓ Normal traffic distinguished reliably (99%+ accuracy)

CHALLENGES:
   ⚠ Minority attack classes difficult:
      - U2R: 15-30% accuracy (too unreliable)
      - R2L: 45-60% accuracy (borderline usable)
      - Probe: 80-90% accuracy (acceptable)
   
   ⚠ Macro-F1 gap (92.3%) vs Accuracy (99.1%) shows 
     models perform poorly on rare classes

IMPLICATIONS:
   → System is reliable for detecting attacks (general level)
   → System reliable for common attacks (Neptune, Smurf)
   → System needs improvement for attack-type classification
   → Majority-class bias means normal traffic easily classified
   → Recommend using cascade approach for attack subcategories

RECOMMENDATION:
   Deploy for attack detection (Stage 1: Normal vs Attack)
   with confidence. For attack-type classification (Stage 2),
   use Macro-F1 as primary metric and focus on improving 
   minority class recall.
```

**Formatting:**
- Font: 11pt, monospace for alignment
- Use checkmarks (✓) and warnings (⚠) for visual hierarchy
- Color code: Green text for strengths, Orange for challenges, Blue for implications

---

---

# ⚙️ POWERBI SETUP INSTRUCTIONS

## Step 1: Data Loading

1. Open **Power BI Desktop** (blank report)
2. **Home** → **Get Data** → **Folder**
3. Select the `powerBi/` folder
4. Select all 8 CSV files (or load them individually)
5. **Load** → **Power Query Editor**

### Power Query Transformations

For each table:
1. Change column types:
   - Percentage columns → Decimal Number
   - Count/Count columns → Whole Number
   - Category columns → Text

2. Rename tables for clarity:
   - `dmbi_overview_kpis` → `Overview_KPIs`
   - `dmbi_class_distribution` → `Class_Distribution`
   - `dmbi_preprocessing_summary` → `Preprocessing`
   - `dmbi_model_comparison` → `Model_Comparison`
   - `dmbi_per_class_performance` → `PerClass_Performance`
   - `dmbi_error_analysis` → `Error_Analysis`
   - `dmbi_confusion_patterns` → `Confusion_Patterns`
   - `dmbi_feature_importance` → `Features`

3. **Close & Apply**

---

## Step 2: Create Dimension Tables

Create two helper tables for consistent filtering:

### Model Dimension
**New Table** via "Enter Data":
```
Model
RF
SVM
XGB
```

### Dataset Dimension
**New Table** via "Enter Data":
```
Dataset
KDDTest+
KDDTest-21
```

### Metric Dimension
**New Table** via "Enter Data":
```
Metric
Accuracy_%
Precision_%
Recall_%
F1_Score_%
Macro_F1_%
False_Positive_Rate_%
```

---

## Step 3: Create Relationships

**Modeling** → **Manage Relationships**

Create relationships:
- `Model_Comparison[Model]` ← **M:1** → `Model[Model]`
- `Model_Comparison[Dataset]` ← **M:1** → `Dataset[Dataset]`
- Similar for other fact tables

---

## Step 4: Create Calculated Columns (if needed)

**Modeling** → **New Column** (for each):

```dax
// If you need to invert FPR for radar chart
FPR_Inverted = 100 - [False_Positive_Rate_%]

// Performance category
Performance_Level = 
IF([Accuracy_%] >= 98, "Excellent",
IF([Accuracy_%] >= 95, "Very Good",
IF([Accuracy_%] >= 90, "Good", "Fair")))

// Class type
Class_Type = 
IF([Class] = "normal", "Normal", "Attack")
```

---

## Step 5: Build Page 1 — OVERVIEW

### Add KPI Cards

1. **Home** → **New Page** → Rename to "Overview"
2. **Insert** → **KPI Card** (for each KPI)
3. Configure each card:

**Best Accuracy Card:**
```
Value: MAX('Overview_KPIs'[Accuracy_%])
Trend axis: Blank
Target goal: 95
```

**Best Recall Card:**
```
Value: MAX('Overview_KPIs'[Attack_Recall_%])
Trend axis: Blank
Target goal: 95
```

Repeat for Lowest FPR, Best Macro F1, Total Records, Attack Classes.

### Add Donut Chart (Dataset Composition)

1. **Insert** → **Donut Chart**
2. **Values:** SUM('Class_Distribution'[Count])
3. **Legend:** 'Class_Distribution'[Class]
4. **Data labels:** Category and percentage
5. **Colors:** Normal = Green, others = Red shades
6. **Title:** "Dataset Composition (Normal vs Attack)"

### Add Grouped Column Chart (Performance Snapshot)

1. **Insert** → **Grouped Column Chart**
2. **X-axis:** 'Model_Comparison'[Model]
3. **Y-axis:** 'Model_Comparison'[Accuracy_%], [Precision_%], [Recall_%], [F1_Score_%]
4. **Sort X-axis by:** Accuracy descending
5. **Data labels:** On
6. **Legend:** Top
7. **Title:** "Model Performance Comparison"

### Add Radar Chart (Bottom)

1. **Insert** → **Radar Chart**
2. **Values:** Accuracy, Precision, Recall, F1, FPR_Inverted
3. **Legend:** Model
4. **Colors:** RF Blue, SVM Teal, XGB Green
5. **Title:** "Model Performance Profiles"

### Add Text Box (Insights)

1. **Insert** → **Text Box**
2. Paste the insights summary from earlier
3. Format: 11pt, gray background

---

## Step 6: Build Page 2 — PREPROCESSING

1. **Home** → **New Page** → Rename to "Preprocessing"

### Add Stats Cards (Top)

1. **Insert** → **Card** (5 cards)
2. For each:
   - Card 1: SUM('Preprocessing'[Total_Records])
   - Card 2: AVERAGE('Preprocessing'[Numerical_Features])
   - Card 3: SUM('Preprocessing'[Missing_Values])
   - Card 4: SUM('Preprocessing'[Duplicate_Rows])
   - Card 5: AVERAGE('Preprocessing'[Imbalance_Ratio])

### Add Horizontal Stacked Bar (Class Distribution)

1. **Insert** → **Stacked Horizontal Bar Chart**
2. **Axis:** 'Class_Distribution'[Dataset]
3. **Values:** SUM('Class_Distribution'[Count])
4. **Legend:** 'Class_Distribution'[Class] (limit to top 10)
5. **Data labels:** Percentage and value
6. **Colors:** Normal = Green (#107C10), others = Red spectrum
7. **Title:** "Class Distribution Analysis"

### Add Column Chart (Feature Statistics)

1. **Insert** → **Clustered Column Chart**
2. **X-axis:** 'Preprocessing'[Dataset]
3. **Y-axis:** Numerical_Features, Categorical_Features, Missing_Values
4. **Data labels:** On
5. **Title:** "Dataset Features Overview"

### Add Horizontal Bar (Feature Importance)

1. **Insert** → **Horizontal Bar Chart**
2. **Values:** SUM('Features'[Importance_Score]) — TOP 10
3. **Axis:** 'Features'[Feature]
4. **Data labels:** Show value
5. **Sort:** Descending by score
6. **Title:** "Top 10 Feature Importance"

### Add Text Box (Transformation Pipeline)

1. **Insert** → **Text Box**
2. Paste the transformation pipeline diagram
3. Format as info box with light gray background

---

## Step 7: Build Page 3 — MODEL COMPARISON

1. **Home** → **New Page** → Rename to "Model Comparison"

### Add Slicers (Top)

1. **Insert** → **Slicer** (3 slicers for Model, Dataset, Metric)
2. **Field:** 'Model'[Model] / 'Dataset'[Dataset] / 'Metric'[Metric]
3. **Style:** Dropdown or Buttons
4. **Sync slicers** across visuals

### Add Comparison Matrix Table

1. **Insert** → **Matrix**
2. **Rows:** 'Model_Comparison'[Model]
3. **Columns:** 'Model_Comparison'[Dataset]
4. **Values:** Accuracy_%, Precision_%, Recall_%, F1_Score_%, TP, FP, FN, TN
5. **Conditional formatting:** Color by value (Green=High, Red=Low)
6. **Title:** "Model Performance Comparison Matrix"

### Add Scatter Plot (Precision-Recall Tradeoff)

1. **Insert** → **Scatter Chart**
2. **X-axis:** 'Model_Comparison'[Recall_%]
3. **Y-axis:** 'Model_Comparison'[Precision_%]
4. **Size:** 'Model_Comparison'[F1_Score_%]
5. **Legend:** 'Model_Comparison'[Model]
6. **Data labels:** Model name
7. **Reference lines:** 95% on both axes
8. **Title:** "Precision-Recall Tradeoff"

### Add Heatmap (Per-Class Performance)

1. **Insert** → **Matrix**
2. **Rows:** 'PerClass_Performance'[Class]
3. **Columns:** 'PerClass_Performance'[Model]
4. **Values:** F1_Score_% (also show Recall_%)
5. **Conditional formatting:** Color by value (Green-Yellow-Red)
6. **Sort:** By F1 descending
7. **Title:** "Per-Class Performance Heatmap"

### Add Error Concentration Chart

1. **Insert** → **Clustered Column Chart**
2. **X-axis:** 'Error_Analysis'[Actual_Class] (sorted by Error count)
3. **Y-axis:** 'Error_Analysis'[Error_Rate_%]
4. **Legend:** 'Error_Analysis'[Model]
5. **Data labels:** Value
6. **Color:** Gradient Red (higher error = darker red)
7. **Title:** "Error Concentration by Class"

---

## Step 8: Build Page 4 — VALIDATION SUMMARY

1. **Home** → **New Page** → Rename to "Validation Summary"

### Add Validation KPI Cards (Top)

1. **Insert** → **KPI Cards** (4 cards)
   - Validation Accuracy: AVG('Overview_KPIs'[Accuracy_%])
   - Validation Recall: AVG('Overview_KPIs'[Attack_Recall_%])
   - Validation FPR: AVG('Overview_KPIs'[False_Positive_Rate_%])
   - Validation Macro-F1: AVG('Overview_KPIs'[Macro_F1_%])

### Add Sankey/Alluvial Chart (Misclassification)

1. **Insert** → **Sankey** (if available)
   - **Source:** 'Confusion_Patterns'[Actual_Class]
   - **Target:** 'Confusion_Patterns'[Predicted_Class]
   - **Value:** 'Confusion_Patterns'[Misclassification_Count]
   - Filter to top 10 flows
   - **Title:** "Misclassification Flow"

*Alternative (if Sankey unavailable):* Use Grouped Bar Chart

### Add Error Distribution Pie Chart

1. **Insert** → **Pie Chart**
2. **Legend:** 'Error_Analysis'[Model]
3. **Values:** SUM('Error_Analysis'[Errors])
4. **Data labels:** Count and percentage
5. **Title:** "Error Distribution by Model"

### Add Confusion Matrix Heatmap

1. **Insert** → **Matrix**
2. **Rows:** 'Confusion_Patterns'[Actual_Class]
3. **Columns:** 'Confusion_Patterns'[Predicted_Class]
4. **Values:** SUM('Confusion_Patterns'[Misclassification_Count])
5. **Conditional formatting:** Color intensity (light = few, dark = many)
6. **Sort:** By error count
7. **Title:** "Confusion Matrix - Error Patterns"

### Add Reliability Assessment Text Box

1. **Insert** → **Text Box**
2. Paste the reliability assessment summary
3. Format with checkmarks and color-coded sections

---

## Step 9: Apply Theme

1. **View** → **Themes** → **Browse for themes**
2. If using custom theme:
   - Upload `CyberSentinel_Theme.json` (if available)
   - Or manually set colors:
     - Primary: #0078D4 (Azure)
     - Success: #107C10 (Green)
     - Warning: #FFB900 (Amber)
     - Danger: #D83B01 (Red)

---

## Step 10: Finalize & Publish

1. **File** → **Save As** → Save locally with version
2. **Home** → **Publish** (to Power BI Service) — if sharing needed
3. Set refresh schedule if data updates

---

---

# 📊 VISUAL BEST PRACTICES FOR DMBI EVALUATION

## What Evaluators Look For

| Criterion | Why It Matters | Examples |
|-----------|---------------|----------|
| **Data-Driven Insights** | Shows analytical thinking, not just pretty charts | "TreeBased models show 99% recall vs SVM 96% — suggesting ensemble advantage" |
| **Proper Metrics Selection** | Understands why Accuracy alone insufficient | Emphasize Recall (attack detection) and Macro-F1 (class balance) |
| **Clear Storytelling** | Dashboard guides through analysis logically | Each page answers a research question |
| **Honest Error Discussion** | Acknowledges limitations, not hiding them | "U2R class <30% accuracy — rare, difficult to generalize" |
| **Professional Appearance** | Looks like business BI, not student project | Consistent colors, clear labels, no clutter |
| **Proper Contextual Information** | Explains "so what?" not just "what is?" | Why 80:20 imbalance matters for F1-score interpretation |

---

## Anti-Patterns to Avoid

| ❌ Bad | ✅ Good |
|-------|--------|
| "Our system implementation" | "Comparative analysis of three approaches" |
| Charts showing code/files | Charts showing metrics and insights |
| Deployment architecture diagram | Data transformation pipeline diagram |
| "How we built it" narrative | "What we learned" narrative |
| Hacker/SOC theme | Professional analytics theme |
| Buried error metrics | Prominent discussion of limitations |
| Accuracy only | Multiple metrics (Precision, Recall, F1, Macro-F1) |
| One model highlighted | Fair side-by-side comparison |

---

# 📝 ANALYTICAL WRITING STYLE FOR DASHBOARD

### Example: How to Frame Findings

**❌ Technical:**
> "The XGBoost model with cascade architecture achieved 99.89% accuracy on the test set."

**✅ Analytical:**
> "Gradient-boosted ensemble methods demonstrated superior recall (99.85%) for attack detection while maintaining controlled false-positive rates (<0.1%), indicating stronger generalization on unseen threat patterns compared to linear SVM models."

---

**❌ Implementation-focused:**
> "We applied feature scaling, encoded categorical variables, and tuned hyperparameters."

**✅ Insight-focused:**
> "Network traffic volume metrics (bytes transferred) and connection frequency (count) emerged as primary attack indicators (89-95% importance), while error rates and protocol-specific patterns provided secondary signals for attack-type classification."

---

**❌ Avoids difficulty:**
> "The model achieved 92% accuracy on all classes."

**✅ Honest assessment:**
> "While majority-class performance exceeded 99% (Normal traffic, DoS, Smurf), minority-class recall suffered significantly (U2R: 15-30%, R2L: 45-60%), revealing the fundamental challenge of imbalanced intrusion datasets where rare attacks naturally receive less training signal."

---

---

# ✅ FINAL CHECKLIST

Before submission:

- [ ] All 8 CSV files generated and validated
- [ ] Power BI report loads without errors
- [ ] 4 pages created with correct structure
- [ ] All slicers functional and linked to visuals
- [ ] Color scheme consistent and professional
- [ ] Data labels clear and readable
- [ ] No implementation details visible (architecture, code, files)
- [ ] Strong emphasis on data insights and analysis
- [ ] All metrics properly calculated and correct
- [ ] Text boxes include honest assessments of limitations
- [ ] Theme applied consistently
- [ ] Tested interactivity across all slicers
- [ ] Saved and backed up

---

**Dashboard Status:** Ready for Academic DMBI Evaluation

**Estimated Build Time:** 45-60 minutes (for first-time builders)

**Total Analytical Content:** 4 pages × 5-7 visuals each = ~28 analytical insights
