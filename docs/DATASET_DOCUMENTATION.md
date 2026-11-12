# InsureSense - Dataset Documentation

## 1. Business Problem

### Problem Statement
Insurance companies invest significantly in acquiring new customers but often underutilize their existing customer base for cross-selling additional products. The **Vehicle Insurance Cross-Sell Prediction** problem aims to identify which existing health insurance customers are likely to purchase an additional vehicle insurance policy.

### Business Value
- **Revenue Maximization**: Targeted cross-selling campaigns increase conversion rates and revenue per customer
- **Resource Optimization**: Marketing budgets focused on high-propensity customers reduce waste
- **Customer Retention**: Personalized product offerings strengthen relationships and improve retention
- **Competitive Advantage**: Data-driven customer insights enable proactive engagement strategies

### ML Problem Type
**Binary Classification** - Predict whether a customer will purchase vehicle insurance (Response = 1) or not (Response = 0).

---

## 2. Dataset Overview

### Source
- **Dataset**: Vehicle Insurance Cross-Sell Prediction
- **Source**: Publicly available on Kaggle / GitHub (nigamshivanshu/Vehicle_Insurance_Cross_Sell_Prediction)
- **File**: `insurance_dataset.csv` (renamed to `train.csv` for consistency)
- **Location**: `ml/data/raw/train.csv`

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Samples | 50,000 |
| Features | 11 (excluding ID) |
| Target Variable | Response (binary) |
| Missing Values | None |
| Class Distribution | 87.7% No Purchase (0), 12.3% Purchase (1) |
| Class Imbalance Ratio | ~7.1:1 (Negative:Positive) |

---

## 3. Data Dictionary

### Feature Descriptions

| Feature | Type | Description | Values/Range | Business Meaning |
|---------|------|-------------|--------------|------------------|
| `id` | Integer | Unique customer identifier | 3 - 381,105 | Primary key, no predictive value |
| `Gender` | Categorical | Customer gender | Male, Female | Demographic segmentation |
| `Age` | Integer | Customer age in years | 20 - 85 | Risk profile indicator |
| `Driving_License` | Binary | Valid driving license | 0 (No), 1 (Yes) | Eligibility indicator (99.8% have license) |
| `Region_Code` | Categorical (numeric) | Geographic region code | 52 unique codes (0.0 - 52.0) | Regional risk/behavior patterns |
| `Previously_Insured` | Binary | Previous vehicle insurance | 0 (No), 1 (Yes) | Prior experience with vehicle insurance |
| `Vehicle_Age` | Ordinal Categorical | Age of customer's vehicle | `< 1 Year`, `1-2 Year`, `> 2 Years` | Vehicle risk profile |
| `Vehicle_Damage` | Binary | History of vehicle damage | No, Yes | Claims history proxy |
| `Annual_Premium` | Continuous | Annual premium paid (health) | 2,630 - 540,165 | Customer value/loyalty indicator |
| `Policy_Sales_Channel` | Categorical (numeric) | Sales channel code | 126 unique codes (0.0 - 163.0) | Acquisition channel behavior |
| `Vintage` | Integer | Days as customer | 10 - 299 | Customer tenure/loyalty |
| `Response` | Binary (TARGET) | Purchased vehicle insurance | 0 (No), 1 (Yes) | **Prediction target** |

---

## 4. Target Variable Analysis

### Response Distribution
```
Response = 0 (No Purchase): 43,861 samples (87.72%)
Response = 1 (Purchase):     6,139 samples (12.28%)
```

### Class Imbalance Implications
- **Severe imbalance** (~7:1 ratio) requires special handling
- Accuracy alone is misleading (naive model achieves 87.7% by predicting all 0)
- Must optimize for: **Precision, Recall, F1, ROC-AUC, PR-AUC**
- Business priority: **Recall** (identify as many actual buyers as possible) balanced with **Precision** (avoid wasting resources on false positives)

---

## 5. Feature Analysis Summary

### Categorical Features (Require Encoding)
1. **Gender** (2 levels): Binary encoding sufficient
2. **Vehicle_Age** (3 levels): Ordinal encoding preserves order (`< 1 Year` < `1-2 Year` < `> 2 Years`)
3. **Vehicle_Damage** (2 levels): Binary encoding
4. **Region_Code** (52 levels): High cardinality - target encoding or frequency encoding recommended
5. **Policy_Sales_Channel** (126 levels): Very high cardinality - target encoding or embedding recommended

### Numerical Features (May Need Scaling)
1. **Age** (20-85): StandardScaler or RobustScaler (outliers possible at extremes)
2. **Annual_Premium** (2,630-540,165): **Heavy right skew** - log transform + RobustScaler essential
3. **Vintage** (10-299): StandardScaler appropriate
4. **Driving_License** (0/1): Binary, no scaling needed
5. **Previously_Insured** (0/1): Binary, no scaling needed

### Features to Drop
- **id**: No predictive value, pure identifier

---

## 6. Preprocessing Strategy (Planned)

### Phase 3 Pipeline Design
```
Raw Data
    → Validation (schema, types, ranges, missing)
    → Train/Test Split (stratified, 80/20, NO leakage)
    → Preprocessing Pipeline:
        ├── Categorical: OneHot (low cardinality) / Target Encoding (high cardinality)
        ├── Numerical: Log transform (Annual_Premium) → RobustScaler
        ├── Ordinal: Vehicle_Age mapping (0, 1, 2)
        └── Feature Engineering: Interaction terms, ratios
    → Feature Selection (if justified)
    → Model Training (CV on train only)
    → Final Evaluation (untouched test set)
```

### Data Leakage Prevention
- **NO** preprocessing fit on full dataset before train/test split
- **NO** target encoding fit on test data
- **ALL** transformations fitted on training data only, applied to test
- Pipeline saved as single artifact (preprocessing + model)

---

## 7. ML Approach (Planned for Phases 3-5)

### Algorithms to Compare (5 Required)
1. **Logistic Regression** - Baseline, interpretable, calibrated probabilities
2. **Decision Tree** - Non-linear baseline, feature importance
3. **Random Forest** - Ensemble, handles non-linearity, robust
4. **Gradient Boosting** - Strong performance, handles imbalance
5. **XGBoost** - State-of-the-art for tabular, built-in imbalance handling

### Evaluation Metrics (Required)
- Accuracy (monitor only)
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC (critical for imbalance)
- Confusion Matrix
- Precision@K, Recall@K (business-aligned)

### Cross-Validation Strategy
- **Stratified K-Fold (k=5)** on training data only
- **GridSearchCV** for hyperparameter optimization
- **Nested CV** for unbiased performance estimation

### Class Imbalance Handling
- **scale_pos_weight** (XGBoost)
- **class_weight='balanced'** (sklearn models)
- **SMOTE** / **ADASYN** (evaluated but not default)
- **Threshold tuning** on validation set for business objectives

---

## 8. Phase 1 Completion Checklist

- [x] Business problem defined and documented
- [x] Dataset identified, downloaded, and verified
- [x] Data dictionary created with all 12 columns
- [x] Target variable identified (`Response`: binary 0/1)
- [x] Dataset availability and format confirmed (CSV, 50K rows, no missing values)
- [x] Binary classification problem characterized (12.3% positive class)
- [x] Initial project structure created per specification
- [x] Dataset copied to `ml/data/raw/train.csv`
- [x] Class imbalance quantified (7.1:1 ratio)
- [x] Preprocessing strategy outlined
- [x] ML approach and algorithms defined

---

## 9. Next Steps (Phase 2)

Proceed to **Phase 2: Exploratory Data Analysis** with:
1. Univariate analysis of all features
2. Bivariate analysis (feature vs target)
3. Correlation analysis
4. Outlier detection
5. Feature distribution visualizations
6. Target analysis by key segments
7. EDA report generation

---

*Document Version: 1.0*  
*Created: 2026-11-06*  
*Phase: 1 - Problem Understanding & Dataset Acquisition*
