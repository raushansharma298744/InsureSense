# InsureSense - Phase 2: Exploratory Data Analysis Report

**Dataset**: Vehicle Insurance Cross-Sell Prediction  
**Samples**: 50,000  
**Target**: Response (binary: 0=No Purchase, 1=Purchase)  
**Date**: 2026-11-06

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 50,000 |
| Features (excl. ID) | 11 |
| Missing Values | 0 |
| Duplicate Rows | 4 |
| Duplicate IDs | 4 |

---

## 2. Target Variable Analysis

### Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| 0 (No Purchase) | 43,861 | 87.72% |
| 1 (Purchase) | 6,139 | 12.28% |

**Imbalance Ratio**: 7.14:1 (Negative:Positive)

### Implications
- Severe class imbalance requires specialized handling
- Accuracy alone is misleading (naive baseline: 87.7%)
- Must optimize for: **Precision, Recall, F1, ROC-AUC, PR-AUC**
- Business priority: High Recall balanced with acceptable Precision

---

## 3. Feature Analysis

### 3.1 Numerical Features

| Feature | Range | Mean | Median | Std | Skew | Key Finding |
|---------|-------|------|--------|-----|------|-------------|
| **Age** | 20-85 | 38.9 | 36 | 15.5 | Right | Purchase mean age: 43.4 vs 38.3 (+13.4%) |
| **Annual_Premium** | 2,630-540,165 | 30,651 | 31,748 | 16,856 | **Heavy Right** | Log transform essential; outliers: 2.66% |
| **Vintage** | 10-299 | 154.5 | 155 | 83.7 | ~Uniform | Minimal difference by target |
| **Region_Code** | 0-52 | 26.4 | 28 | 13.2 | - | 53 unique; categorical despite numeric |
| **Policy_Sales_Channel** | 1-163 | 111.7 | 125 | 54.3 | - | 126 unique; strong predictor (-20.6% diff) |

### 3.2 Categorical Features

| Feature | Levels | Distribution | Purchase Rate Range |
|---------|--------|--------------|---------------------|
| **Gender** | 2 | Male: 54%, Female: 46% | Male: 13.96%, Female: 10.31% |
| **Driving_License** | 2 | Yes: 99.8%, No: 0.2% | Near-constant feature |
| **Previously_Insured** | 2 | No: 54.4%, Yes: 45.6% | **Critical**: No=22.5%, Yes=0.07% |
| **Vehicle_Age** | 3 | 1-2yr: 53%, <1yr: 43%, >2yr: 4% | >2yr: 28.6%, 1-2yr: 17.5%, <1yr: 4.3% |
| **Vehicle_Damage** | 2 | Yes: 50.7%, No: 49.3% | **Strongest**: Yes=23.6%, No=0.6% |

---

## 4. Key Predictive Insights

### Strongest Predictors (by target correlation & purchase rate difference)

1. **Vehicle_Damage** (Δ = 23.0 pp): Vehicle damage history is the single strongest predictor
2. **Previously_Insured** (Δ = 22.4 pp, corr = -0.34): Negative correlation - previously insured customers rarely buy again
3. **Vehicle_Age** (Δ = 24.3 pp): Older vehicles (>2 years) have 28.6% purchase rate
4. **Policy_Sales_Channel** (Δ = 20.6 pp, corr = -0.14): High cardinality, significant variation
5. **Region_Code** (corr = +0.005): 53 regions, purchase rates 2.5% - 19.1%
6. **Age** (Δ = 13.4 pp, corr = +0.11): Peak purchase age 30-50
7. **Gender** (Δ = 3.7 pp): Male higher than Female
8. **Annual_Premium** (corr = +0.024): Weak positive correlation
9. **Vintage** (corr = +0.001): Negligible

### Critical Interactions

| Segment Combination | Purchase Rate |
|---------------------|---------------|
| Vehicle_Age >2yr + Vehicle_Damage=Yes | **28.6%** |
| Vehicle_Age 1-2yr + Vehicle_Damage=Yes | **26.7%** |
| Previously_Insured=0 + Vehicle_Damage=Yes | **24.9%** |
| Previously_Insured=1 + Vehicle_Damage=No | **0.04%** |
| Vehicle_Age <1yr + Vehicle_Damage=No | **0.34%** |

---

## 5. Outlier Analysis

| Feature | Outliers | Percentage | Treatment |
|---------|----------|------------|-----------|
| Age | 0 | 0% | None needed |
| **Annual_Premium** | **1,329** | **2.66%** | **Log transform + RobustScaler** |
| Vintage | 0 | 0% | None needed |

---

## 6. Preprocessing Recommendations

### Features to Drop
- `id` - No predictive value

### Encoding Strategy
| Feature | Strategy | Rationale |
|---------|----------|-----------|
| Gender | Binary (0/1) | 2 levels |
| Driving_License | Keep as-is | Near-constant (99.8% = 1) |
| Previously_Insured | Keep as-is | Binary |
| Vehicle_Damage | Binary (0/1) | 2 levels |
| Vehicle_Age | Ordinal (0,1,2) | Natural order: <1yr < 1-2yr < >2yr |
| Region_Code | **Target Encoding** | 53 levels, high cardinality |
| Policy_Sales_Channel | **Target Encoding** | 126 levels, very high cardinality |

### Scaling Strategy
| Feature | Strategy | Rationale |
|---------|----------|-----------|
| Annual_Premium | **Log → RobustScaler** | Heavy right skew, outliers |
| Age | RobustScaler | Some outliers at extremes |
| Vintage | StandardScaler | ~Uniform distribution |
| Region_Code (encoded) | StandardScaler | After target encoding |
| Policy_Sales_Channel (encoded) | StandardScaler | After target encoding |

---

## 7. Class Imbalance Handling Plan

1. **Algorithm-level**: `class_weight='balanced'` for sklearn models, `scale_pos_weight` for XGBoost
2. **Resampling**: Evaluate SMOTE/ADASYN but not default
3. **Threshold Tuning**: Optimize decision threshold on validation set for business objectives
4. **Metrics**: Focus on PR-AUC, F1, Precision@K, Recall@K over Accuracy

---

## 8. Visualizations Generated

All saved to `docs/`:
1. `eda_numerical_distributions.png` - Histograms of numerical features
2. `eda_categorical_distributions.png` - Bar charts of categorical features
3. `eda_numerical_by_target.png` - Overlayed distributions by target class
4. `eda_categorical_by_target.png` - Purchase rates by category
5. `eda_correlation_matrix.png` - Correlation heatmap
6. `eda_boxplots_by_target.png` - Box plots by target

---

## 9. Next Steps (Phase 3)

Proceed to **Phase 3: ML Preprocessing & Feature Engineering Pipeline** with:

1. **Data Validation Module** - Schema validation, type checking, range validation
2. **Train/Test Split** - Stratified 80/20 with fixed random state
3. **Preprocessing Pipeline** - ColumnTransformer with all encoders/scalers
4. **Feature Engineering** - Interaction terms, binned features
5. **Pipeline Serialization** - Save as single reproducible artifact

---

*Report generated: 2026-11-06*  
*Phase 2 Status: ✅ COMPLETE*