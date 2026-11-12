import pandas as pd
import numpy as np

df = pd.read_csv('ml/data/raw/train.csv')

print("=" * 80)
print("INSURESENSE - PHASE 2: EXPLORATORY DATA ANALYSIS REPORT")
print("=" * 80)

# ============ DATASET OVERVIEW ============
print("\n1. DATASET OVERVIEW")
print("-" * 40)
print(f"Total samples: {len(df):,}")
print(f"Total features: {len(df.columns) - 1} (excluding ID)")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate IDs: {df['id'].duplicated().sum()}")

# ============ TARGET DISTRIBUTION ============
print("\n2. TARGET VARIABLE (Response)")
print("-" * 40)
target_counts = df['Response'].value_counts()
target_pct = df['Response'].value_counts(normalize=True) * 100
for cls in [0, 1]:
    print(f"  Response={cls}: {target_counts[cls]:,} ({target_pct[cls]:.2f}%)")
imbalance_ratio = target_counts[0] / target_counts[1]
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

# ============ NUMERICAL STATISTICS ============
print("\n3. NUMERICAL FEATURES - DESCRIPTIVE STATISTICS")
print("-" * 40)
num_features = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
print(df[num_features].describe().to_string())

# ============ CATEGORICAL VALUE COUNTS ============
print("\n\n4. CATEGORICAL FEATURES - VALUE COUNTS")
print("-" * 40)
cat_features = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']
for feat in cat_features:
    print(f"\n  {feat}:")
    counts = df[feat].value_counts()
    pcts = df[feat].value_counts(normalize=True) * 100
    for val in counts.index:
        print(f"    {val}: {counts[val]:,} ({pcts[val]:.2f}%)")

# ============ NUMERICAL BY TARGET ============
print("\n\n5. NUMERICAL FEATURES BY TARGET")
print("-" * 40)
for feat in num_features:
    grp0 = df[df['Response'] == 0][feat]
    grp1 = df[df['Response'] == 1][feat]
    print(f"\n  {feat}:")
    print(f"    No Purchase (0): mean={grp0.mean():.2f}, median={grp0.median():.2f}, std={grp0.std():.2f}")
    print(f"    Purchase (1):    mean={grp1.mean():.2f}, median={grp1.median():.2f}, std={grp1.std():.2f}")
    diff_pct = ((grp1.mean() - grp0.mean()) / grp0.mean()) * 100
    print(f"    Difference: {diff_pct:+.2f}%")

# ============ CATEGORICAL PURCHASE RATES ============
print("\n\n6. PURCHASE RATES BY CATEGORICAL FEATURES")
print("-" * 40)
for feat in cat_features:
    print(f"\n  {feat}:")
    grp = df.groupby(feat)['Response'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'Total', 'sum': 'Purchases', 'mean': 'Rate'})
    grp['Rate'] = grp['Rate'] * 100
    print(grp.round(2).to_string())

# ============ CORRELATION WITH TARGET ============
print("\n\n7. CORRELATION WITH TARGET")
print("-" * 40)
num_cols = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel', 'Driving_License', 'Previously_Insured', 'Response']
target_corr = df[num_cols].corr()['Response'].drop('Response').sort_values(key=abs, ascending=False)
for feat, corr in target_corr.items():
    print(f"  {feat:25s}: {corr:+.4f}")

# ============ OUTLIER DETECTION ============
print("\n\n8. OUTLIER DETECTION (IQR Method)")
print("-" * 40)
for feat in ['Age', 'Annual_Premium', 'Vintage']:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[feat] < lower) | (df[feat] > upper)]
    print(f"\n  {feat}:")
    print(f"    Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"    Bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"    Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    if len(outliers) > 0:
        print(f"    Range: [{outliers[feat].min():.2f}, {outliers[feat].max():.2f}]")

# ============ HIGH CARDINALITY ANALYSIS ============
print("\n\n9. HIGH CARDINALITY FEATURES")
print("-" * 40)

print(f"\n  Region_Code:")
print(f"    Unique values: {df['Region_Code'].nunique()}")
region_stats = df.groupby('Region_Code')['Response'].agg(['count', 'mean']).rename(columns={'count': 'Total', 'mean': 'Rate'})
region_stats = region_stats[region_stats['Total'] >= 100].sort_values('Rate', ascending=False)
print(f"    Top 5 by purchase rate (min 100 samples):")
print(region_stats.head(5).round(4).to_string())
print(f"    Bottom 5 by purchase rate (min 100 samples):")
print(region_stats.tail(5).round(4).to_string())

print(f"\n  Policy_Sales_Channel:")
print(f"    Unique values: {df['Policy_Sales_Channel'].nunique()}")
channel_stats = df.groupby('Policy_Sales_Channel')['Response'].agg(['count', 'mean']).rename(columns={'count': 'Total', 'mean': 'Rate'})
channel_stats = channel_stats[channel_stats['Total'] >= 50].sort_values('Rate', ascending=False)
print(f"    Top 5 by purchase rate (min 50 samples):")
print(channel_stats.head(5).round(4).to_string())
print(f"    Bottom 5 by purchase rate (min 50 samples):")
print(channel_stats.tail(5).round(4).to_string())

# ============ SEGMENT ANALYSIS ============
print("\n\n10. KEY SEGMENT COMBINATIONS")
print("-" * 40)

print("\n  Vehicle_Age x Vehicle_Damage:")
seg = df.groupby(['Vehicle_Age', 'Vehicle_Damage'])['Response'].mean().unstack() * 100
print(seg.round(2).to_string())

print("\n  Previously_Insured x Vehicle_Damage:")
seg = df.groupby(['Previously_Insured', 'Vehicle_Damage'])['Response'].mean().unstack() * 100
print(seg.round(2).to_string())

print("\n  Gender x Previously_Insured:")
seg = df.groupby(['Gender', 'Previously_Insured'])['Response'].mean().unstack() * 100
print(seg.round(2).to_string())

# ============ BINNED ANALYSIS ============
print("\n\n11. BINNED FEATURE ANALYSIS")
print("-" * 40)

df['Age_Bin'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70, 85], labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70+'])
age_bin_stats = df.groupby('Age_Bin')['Response'].agg(['count', 'mean']).rename(columns={'count': 'Total', 'mean': 'Rate'})
age_bin_stats['Rate'] = age_bin_stats['Rate'] * 100
print("\n  Age Bins:")
print(age_bin_stats.round(2).to_string())

df['Premium_Bin'] = pd.qcut(df['Annual_Premium'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
premium_stats = df.groupby('Premium_Bin')['Response'].agg(['count', 'mean']).rename(columns={'count': 'Total', 'mean': 'Rate'})
premium_stats['Rate'] = premium_stats['Rate'] * 100
print("\n  Annual_Premium Quintiles:")
print(premium_stats.round(2).to_string())

df['Vintage_Bin'] = pd.qcut(df['Vintage'], q=5, labels=['Very New', 'New', 'Medium', 'Old', 'Very Old'])
vintage_stats = df.groupby('Vintage_Bin')['Response'].agg(['count', 'mean']).rename(columns={'count': 'Total', 'mean': 'Rate'})
vintage_stats['Rate'] = vintage_stats['Rate'] * 100
print("\n  Vintage Quintiles:")
print(vintage_stats.round(2).to_string())

print("\n" + "=" * 80)
print("EDA COMPLETE - Visualizations saved to docs/")
print("=" * 80)