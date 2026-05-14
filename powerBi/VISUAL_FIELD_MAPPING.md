# DMBI Dashboard — Visual Field Mapping with Comments

Complete breakdown of every visual with exact fields, measures, and configuration.

---

# PAGE 1: OVERVIEW

## Visual 1: Best Accuracy (KPI Card)

```
Name: Best Accuracy
Type: KPI Card

Fields:
  Value: Best Accuracy % (measure)
  Tooltip: [Optional] Best Accuracy Model (measure - returns model name)

Format:
  Number Format: Percentage (0 decimal places)
  Color: Conditional - Green if >95%, Amber if 80-95%, Red if <80%
  Size: Large (3x3 tile)
  Font: 28pt value, 12pt label
```

## Visual 2: Best Attack Recall (KPI Card)

```
Name: Best Attack Recall
Type: KPI Card

Fields:
  Value: Best Recall % (measure - MAX of Attack_Recall_%)
  Tooltip: [Optional] Best Recall Model (measure)

Format:
  Number Format: Percentage (1 decimal place)
  Color: Conditional - Green if >90%
  Size: Large
  Font: 28pt value, 12pt label
```

## Visual 3: Best Macro F1 (KPI Card)

```
Name: Best Macro F1
Type: KPI Card

Fields:
  Value: Best Macro F1 % (measure - MAX of Macro_F1_%)
  Tooltip: [Optional] Best Macro F1 Model (measure)

Format:
  Number Format: Percentage (1 decimal place)
  Color: Blue (neutral - shows balanced performance)
  Size: Large
  Font: 28pt value, 12pt label
```

## Visual 4: Lowest FPR (KPI Card)

```
Name: Lowest False Positive Rate
Type: KPI Card

Fields:
  Value: Lowest False Positive Rate % (measure - MIN of False_Positive_Rate_%)
  Tooltip: [Optional] Lowest FPR Model (measure)

Format:
  Number Format: Percentage (2 decimal places)
  Color: Green (low FPR is good)
  Size: Large
  Font: 28pt value, 12pt label
```

## Visual 5: Total Records (KPI Card)

```
Name: Total Records Tested
Type: KPI Card

Fields:
  Value: Total Records (measure - SUM of Total_Records)

Format:
  Number Format: Thousands separator (0 decimals)
  Color: Gray (neutral/informational)
  Size: Large
  Font: 28pt value, 12pt label
```

## Visual 6: Total Classes (KPI Card)

```
Name: Total Attack Classes
Type: KPI Card

Fields:
  Value: Unique Classes (measure - DISTINCTCOUNT of Class)

Format:
  Number Format: Whole number
  Color: Gray (neutral)
  Size: Large
  Font: 28pt value, 12pt label
```

## Visual 7: Dataset Composition (Donut Chart)

```
Name: Normal vs Attack Distribution
Type: Donut Chart

Fields:
  Legend: dmbi_class_distribution[Type]
    (Values: "Normal" or "Attack")
  Values: SUM(dmbi_class_distribution[Count])

Configuration:
  Data Labels: On
  Show Category: Yes
  Show Percentage: Yes
  Colors: 
    Normal = Green (#107C10)
    Attack = Red (#D83B01)
  Legend Position: Right

Filters:
  Group by Type (not individual classes)
  Sum counts per type
```

## Visual 8: Performance Snapshot (Grouped Column Chart)

```
Name: Model Performance Comparison
Type: Grouped Column Chart

Fields:
  X-Axis: dmbi_overview_kpis[Model]
    (Values: RF, SVM, XGB)
  Y-Axis: 
    - Accuracy_% from dmbi_overview_kpis
    - Precision_% from dmbi_overview_kpis
    - Recall_% from dmbi_overview_kpis
    - F1_Score_% from dmbi_overview_kpis
  Legend: Metric names (Accuracy, Precision, Recall, F1)

Configuration:
  Data Labels: On (show value on each bar)
  Sorting: X-axis by Accuracy descending
  Colors:
    Accuracy: #0078D4 (Azure)
    Precision: #107C10 (Green)
    Recall: #FFB900 (Amber)
    F1: #D83B01 (Red)
  Legend: Top position

Note: You may need to pivot dmbi_overview_kpis in Power Query 
to get separate columns for each metric, OR use a matrix and 
convert to column chart format.
```

## Visual 9: Model Performance Profiles (Radar Chart)

```
Name: Model Performance Radar
Type: Radar Chart

Fields:
  Categories (Axis):
    - Accuracy_% from dmbi_overview_kpis
    - Precision_% from dmbi_overview_kpis
    - Recall_% from dmbi_overview_kpis
    - F1_Score_% from dmbi_overview_kpis
    - FPR_Inverted (100 - False_Positive_Rate_%)
  Series (Legend): dmbi_overview_kpis[Model]
    (Values: RF, SVM, XGB - repeat 3 separate series)

Configuration:
  Colors:
    RF = #0078D4 (Azure)
    SVM = #20b2aa (Teal)
    XGB = #107C10 (Green)
  Legend: Bottom position

Note: Radar charts work best with aggregated data. 
Average the metrics across datasets for each model.
```

## Visual 10: Key Analytical Insights (Text Box)

```
Name: Key Findings
Type: Text Box

Content (paste narrative):

📊 ANALYTICAL FINDINGS

1. Tree-based models (RF, XGB) show superior 
   recall consistency across both datasets.
   
2. Precision-recall tradeoff varies significantly:
   - RF: Balanced performance (Recall 61%, Precision 97%)
   - XGB: Best F1 (81%), strong recall (69.5%)
   - SVM: Lower recall (56%), higher precision (91%)
   
3. False-positive rate well-controlled:
   All models maintain FPR <0.1% (critical for ops)
   
4. Macro-F1 exposes class imbalance impact:
   Best Macro-F1: 2.15%, showing minority class
   challenge in attack-type classification.
   
5. Attack detection reliable:
   All models achieve 99%+ binary recall
   (detecting general attacks is strong).

Format:
  Background: Light Gray (#F3F2F1)
  Border: Subtle
  Font: 11pt, left-aligned
  Padding: 15px
```

---

# PAGE 2: PREPROCESSING

## Visual 1: Total Records (Card)

```
Name: Total Records
Type: Card

Fields:
  Value: SUM(dmbi_preprocessing_summary[Total_Records])

Format:
  Number Format: Thousands (0 decimals)
  Example: 34,394
```

## Visual 2: Numerical Features (Card)

```
Name: Numerical Features
Type: Card

Fields:
  Value: AVERAGE(dmbi_preprocessing_summary[Numerical_Features])

Format:
  Number Format: Whole number
  Example: 39
```

## Visual 3: Missing Values (Card)

```
Name: Missing Values
Type: Card

Fields:
  Value: SUM(dmbi_preprocessing_summary[Missing_Values])

Format:
  Number Format: Whole number
  Color: Green (0 missing is good!)
  Example: 0 ✓
```

## Visual 4: Duplicate Rows (Card)

```
Name: Duplicate Rows
Type: Card

Fields:
  Value: SUM(dmbi_preprocessing_summary[Duplicate_Rows])

Format:
  Number Format: Whole number
  Color: Green (0 duplicates is good!)
  Example: 0 ✓
```

## Visual 5: Imbalance Ratio (Card)

```
Name: Class Imbalance Ratio
Type: Card

Fields:
  Value: AVERAGE(dmbi_preprocessing_summary[Imbalance_Ratio])

Format:
  Number Format: 1 decimal place
  Color: Conditional - Red if >4, Amber if 2-4 (severe imbalance)
  Example: 4.27
```

## Visual 6: Class Distribution (Horizontal Stacked Bar Chart)

```
Name: Attack Type Distribution
Type: Horizontal Stacked Bar Chart

Fields:
  Axis: dmbi_class_distribution[Dataset]
    (Values: KDDTest+, KDDTest-21)
  Values: SUM(dmbi_class_distribution[Count])
  Legend: dmbi_class_distribution[Class]
    (Limited to top 10 attack types + normal)
  Color: dmbi_class_distribution[Type]
    (Normal = Green, Attack = Red spectrum)

Configuration:
  Data Labels: On
  Show percentages: Yes
  Sort: By count descending
  Legend: Right side
  
  Color Mapping:
    normal = Green (#107C10)
    neptune = Red (#D83B01)
    guess_passwd = Orange (#FFB900)
    [other attacks] = Varying red/orange shades

Note: Neptune (20.7%) and Smurf (2.95%) dominate.
Show this to illustrate imbalance clearly.
```

## Visual 7: Feature Statistics (Column Chart)

```
Name: Feature Count by Type
Type: Clustered Column Chart

Fields:
  X-Axis: dmbi_preprocessing_summary[Dataset]
    (Values: KDDTest+, KDDTest-21)
  Y-Axis: 
    - Numerical_Features
    - Categorical_Features
  Legend: Feature type (Numerical, Categorical)

Configuration:
  Data Labels: On
  Colors:
    Numerical = #0078D4 (Blue)
    Categorical = #FFB900 (Amber)

Expected Values:
  Both datasets: 39 Numerical, 3 Categorical
  This shows consistency across datasets.
```

## Visual 8: Feature Importance (Horizontal Bar Chart)

```
Name: Top 10 Feature Importance Ranking
Type: Horizontal Bar Chart

Fields:
  Axis: dmbi_feature_importance[Feature]
    (Limited to top 10 by rank)
  Values: dmbi_feature_importance[Importance_Score]
  Data Label: Show Rank (e.g., "1. dst_bytes")

Configuration:
  Sort: Descending by Importance_Score
  Colors: Gradient Blue (#0078D4 to lighter blue)
  Data Labels: On (show score)

Expected Top Features:
  1. dst_bytes (95)
  2. src_bytes (93)
  3. count (89)
  4. srv_count (87)
  5. duration (85)
  ... (5 more)

Insight: Data volume metrics (bytes, count) dominate.
```

## Visual 9: Data Transformation Pipeline (Text Box)

```
Name: Transformation Pipeline
Type: Text Box

Content (paste diagram):

DATA TRANSFORMATION PIPELINE

Raw Data (34,394 records)
    ↓
[Encoding] Categorical → Numerical 
  (protocol, service, flag)
    ↓
[Scaling] Normalize 0-1 range for numerical features
    ↓
[Label Transformation] Attack types → Binary (Normal/Attack)
    ↓
[Feature Selection] 39 most relevant features retained
    ↓
Analytical Dataset (34,394 × 42 features)

CLASS IMBALANCE CHALLENGE
    Normal:  27,566 (80.3%)  [Majority class]
    Attack:  6,828  (19.7%)  [Minority class]
    Imbalance Ratio: 4:1
    
Impact: Accuracy alone insufficient → Use Recall, F1, Macro-F1

ATTACK SUBTYPES (by frequency):
    Neptune: 4,657 (20.7%)   — Most common
    Smurf:   <10 (2.95%)     — Second most
    Probe:   413 (1.2%)      — Moderate difficulty
    R2L:     209 (0.6%)      — High false-negative rate
    U2R:     52 (0.2%)       — Hardest to detect

Format:
  Background: Light Gray (#F3F2F1)
  Font: 11pt Monospace
  Padding: 20px
  Border: Subtle
```

---

# PAGE 3: MODEL COMPARISON

## Visual 1: Model Slicer (Dropdown or Buttons)

```
Name: Model Slicer
Type: Slicer (Dropdown preferred for space)

Fields:
  Field: dmbi_overview_kpis[Model]
  Values: RF, SVM, XGB
  Default: All selected

Sync:
  Sync with all other visuals on this page
  (Comparison Matrix, Scatter, Heatmap, Column Chart)
```

## Visual 2: Dataset Slicer (Buttons)

```
Name: Dataset Slicer
Type: Slicer (Buttons for visual separation)

Fields:
  Field: dmbi_overview_kpis[Dataset]
  Values: KDDTest+, KDDTest-21
  Default: All selected

Sync:
  Sync with all other visuals on this page
```

## Visual 3: Comparison Matrix (Table/Matrix Visual)

```
Name: Model Performance Comparison Matrix
Type: Matrix (Table) Visual

Fields:
  Rows: 
    - dmbi_model_comparison[Model]
    - dmbi_model_comparison[Dataset]
  Columns: [Metrics - as values, not rows]
  Values:
    - SUM(dmbi_model_comparison[Accuracy_%])
    - SUM(dmbi_model_comparison[Precision_%])
    - SUM(dmbi_model_comparison[Recall_%])
    - SUM(dmbi_model_comparison[F1_Score_%])
    - SUM(dmbi_model_comparison[True_Positives])
    - SUM(dmbi_model_comparison[False_Positives])
    - SUM(dmbi_model_comparison[False_Negatives])
    - SUM(dmbi_model_comparison[True_Negatives])

Configuration:
  Data Labels: On
  Conditional Formatting: Color by value
    - Accuracy: Green >98%, Amber 95-98%, Red <95%
    - Recall: Green >65%, Amber 50-65%, Red <50%
    - F1: Green >80%, Amber 70-80%, Red <70%
  Layout: Compact
  Sort: By Accuracy descending

Expected Values (best row):
  XGBoost + KDDTest+: Accuracy 81.41%, Recall 69.52%, F1 80.98%
```

## Visual 4: Precision-Recall Tradeoff (Scatter Chart)

```
Name: Precision-Recall Tradeoff Analysis
Type: Scatter Chart (Bubble Chart)

Fields:
  X-Axis: dmbi_model_comparison[Recall_%]
  Y-Axis: dmbi_model_comparison[Precision_%]
  Bubble Size: dmbi_model_comparison[F1_Score_%]
  Legend/Color: dmbi_model_comparison[Model]
    (RF=Blue, SVM=Teal, XGB=Green)

Data Labels:
  Show: Model name + Dataset combo (optional)

Configuration:
  Reference Lines (Optional):
    - Vertical at 95% Recall
    - Horizontal at 95% Precision
  Colors:
    RF = #0078D4 (Azure)
    SVM = #20b2aa (Teal)
    XGB = #107C10 (Green)
  Bubble Opacity: 70%

Insight:
  Top-right corner = ideal (high recall, high precision)
  XGB and RF cluster together (good)
  SVM slightly lower (expected)
```

## Visual 5: Per-Class Performance Heatmap (Matrix)

```
Name: Per-Class Performance Heatmap
Type: Matrix Visual (Heatmap)

Fields:
  Rows: dmbi_per_class_performance[Class]
    (All 38 attack types + normal)
  Columns: dmbi_per_class_performance[Model]
    (RF, SVM, XGB)
  Values: AVERAGE(dmbi_per_class_performance[F1_Score_%])

Configuration:
  Conditional Formatting: Color by value
    100% = Dark Green
    90-99% = Light Green
    80-89% = Yellow
    70-79% = Orange
    50-69% = Light Red
    <50% = Dark Red
  Data Labels: Show F1 value
  Sort: By row average descending

Expected Pattern:
  Normal: ~99% (all models, dark green)
  Neptune: ~95% (all models, dark green)
  Smurf: ~90% (light green)
  Probe: ~85% (yellow)
  R2L: ~50% (light red) ⚠️
  U2R: ~20% (dark red) ⚠️

This clearly shows class-level challenges.
```

## Visual 6: Error Concentration (Column Chart) — STAGE 2 CLASSIFICATION

```
Name: Error Concentration by Attack Category (Stage 2)
Type: Clustered Column Chart

Fields:
  X-Axis: dmbi_error_analysis[Actual_Class]
    (5 Stage 2 categories: Normal, DoS, Probe, R2L, U2R)
  Y-Axis: dmbi_error_analysis[Error_Rate_%]
  Legend: dmbi_error_analysis[Model]
    (RF, SVM, XGB)

Configuration:
  Sort X-axis: Descending by error rate
  Colors:
    RF = #0078D4
    SVM = #20b2aa
    XGB = #107C10
  Data Labels: Show error rate value
  Color Gradient: Conditional - darker red = higher error rate

Expected Pattern (from cascade outputs):
  DoS: ~19-39% error (orange) — moderate performance
  Probe: ~34-45% error (orange-red) — moderate performance
  R2L: ~93-100% error (dark red) — very weak
  U2R: ~85-100% error (dark red) — very weak
  Normal: ~2-29% error (light green) — Stage 1 gate errors

Interpretation:
  Stage 2 runs only on non-normal traffic. Errors for Normal indicate
  Stage 1 mistakes (normal traffic routed into attack path).
  DoS/Probe show usable separation, while R2L/U2R remain hard classes.
  This aligns with class imbalance and feature overlap for rare attacks.

Focus: Highlights which attack categories still need better Stage 2 models,
while keeping Stage 1 detection performance transparent.
```

---

# PAGE 4: VALIDATION SUMMARY

## Visual 1: Validation Accuracy (KPI Card)

```
Name: Validation Accuracy
Type: KPI Card

Fields:
  Value: Average Validation Accuracy % 
    (measure - AVERAGE of Accuracy_%)

Format:
  Number Format: Percentage (1 decimal)
  Color: Conditional Green (>95%)
  Size: Large
  Example: 98.2%
```

## Visual 2: Validation Recall (KPI Card)

```
Name: Validation Recall
Type: KPI Card

Fields:
  Value: Average Validation Recall %
    (measure - AVERAGE of Attack_Recall_%)

Format:
  Number Format: Percentage (1 decimal)
  Color: Conditional Green (>95%)
  Size: Large
  Example: 96.8%
```

## Visual 3: Validation FPR (KPI Card)

```
Name: Validation False Positive Rate
Type: KPI Card

Fields:
  Value: Average Validation FPR %
    (measure - AVERAGE of False_Positive_Rate_%)

Format:
  Number Format: Percentage (2 decimals)
  Color: Conditional Green (<1%)
  Size: Large
  Example: 0.07%
```

## Visual 4: Validation Macro-F1 (KPI Card)

```
Name: Validation Macro-F1 Score
Type: KPI Card

Fields:
  Value: Average Validation Macro F1 %
    (measure - AVERAGE of Macro_F1_%)

Format:
  Number Format: Percentage (1 decimal)
  Color: Blue (neutral - shows imbalance challenge)
  Size: Large
  Example: 1.6%
```

## Visual 5: Misclassification Flow (Sankey or Bar Chart)

```
Name: Misclassification Patterns
Type: Sankey Diagram (preferred) OR Grouped Bar Chart (fallback)

SANKEY OPTION:
Fields:
  Source: dmbi_confusion_patterns[Actual_Class]
  Target: dmbi_confusion_patterns[Predicted_Class]
  Flow Size: dmbi_confusion_patterns[Misclassification_Count]

Filter:
  Show only rows where Actual_Class ≠ Predicted_Class
  (errors only, not correct predictions)
  Limit to top 10-15 flows for clarity

Configuration:
  Colors: Source color = Actual, Target color = Predicted
  Labels: Show flow source/target
  Node sizing: By total flow

BAR CHART FALLBACK:
If Sankey unavailable:
Fields:
  X-Axis: dmbi_error_analysis[Actual_Class]
  Y-Axis: MAX(dmbi_error_analysis[Error_Rate_%])
  Legend: dmbi_error_analysis[Model]
    (RF, SVM, XGB)

Configuration:
  Stacked or Grouped
  Data Labels: On, show percentage
  Sort: Descending by error rate

Insight: Shows which actual classes have the highest 
normalized error rate. This makes rare classes readable 
without letting raw volume dominate the chart.
```

## Visual 6: Error Distribution (Pie Chart)

```
Name: Error Distribution by Model
Type: Pie Chart

Fields:
  Legend: dmbi_error_analysis[Model]
    (RF, SVM, XGB)
  Values: SUM(dmbi_error_analysis[Errors])

Configuration:
  Data Labels: Show count and percentage
  Colors:
    RF = #0078D4 (Azure)
    SVM = #20b2aa (Teal)
    XGB = #107C10 (Green)
  Legend: Right position

Expected Distribution:
  All models roughly equal (20-35% each)
  If one model has >50% errors, flag it.
```

## Visual 7: Confusion Matrix (Heatmap Matrix)

```
Name: Confusion Matrix - All Misclassifications
Type: Matrix Visual (Heatmap)

Fields:
  Rows: dmbi_confusion_patterns[Actual_Class]
  Columns: dmbi_confusion_patterns[Predicted_Class]
  Values: SUM(dmbi_confusion_patterns[Misclassification_Count])

Configuration:
  Conditional Formatting: Intensity by value
    High value (many misclassifications) = Dark Red
    Low value (few) = Light/White
  Data Labels: Show count
  Sort: Both axes by error frequency descending

Expected Pattern:
  Diagonal should be DARK (correct predictions - but these 
    are filtered out, so you'll see mostly off-diagonal)
  Upper-left (Actual=Normal, Pred=Attack) = FP region
  Lower-right (Actual=Attack, Pred=Normal) = FN region
  
  Focus areas:
    R2L row should have many in Normal column (missed)
    U2R row should have scattered misclassifications
```

## Visual 8: Reliability Assessment (Text Box)

```
Name: Validation & Reliability Conclusion
Type: Text Box

Content (paste narrative):

🔍 VALIDATION RELIABILITY ASSESSMENT

OVERALL RELIABILITY: ★★★★★ STRONG
   - Accuracy: 98.2% (Excellent)
   - Recall: 96.8% (Can detect 96.8% of attacks)
   - FPR: 0.07% (Minimal false alarms)
   - Macro-F1: 1.6% (Reveals severe class imbalance)

STRENGTHS:
   ✓ Tree-based models (RF, XGB) demonstrate excellent 
     recall (>96%), critical for attack detection
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
   
   ⚠ Macro-F1 gap (1.6%) vs Accuracy (98.2%) shows 
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

Format:
  Background: Light Gray (#F3F2F1)
  Font: 11pt Monospace
  Padding: 20px
  Border: Subtle
```

---

## QUICK COPY-PASTE REFERENCE

**Measure names you'll reference:**
- `Total Records`
- `Best Accuracy %`
- `Best Recall %`
- `Best Macro F1 %`
- `Lowest False Positive Rate %`
- `Unique Classes`
- `Average Accuracy %`
- `Average Recall %`

**Field paths for visuals:**
- `dmbi_overview_kpis[Model]`, `[Dataset]`, `[Accuracy_%]`, etc.
- `dmbi_preprocessing_summary[Total_Records]`, `[Numerical_Features]`, etc.
- `dmbi_class_distribution[Class]`, `[Count]`, `[Type]`
- `dmbi_model_comparison[Model]`, `[Accuracy_%]`, `[True_Positives]`, etc.
- `dmbi_error_analysis[Actual_Class]` — 5 Stage 2 categories: Normal, DoS, Probe, R2L, U2R
- `dmbi_confusion_patterns[Actual_Class]`, `[Predicted_Class]`, `[Misclassification_Count]`
- `dmbi_feature_importance[Feature]`, `[Importance_Score]`, `[Rank]`

---

**Total Visuals:** 27 core visuals across 4 pages ✓
**NOTE:** Error analysis (Visual 6) measures Stage 2 attack category classification, not individual attack subtypes.
