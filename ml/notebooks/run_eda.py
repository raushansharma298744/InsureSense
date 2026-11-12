import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

df = pd.read_csv('ml/data/raw/train.csv')

# ============ NUMERICAL DISTRIBUTIONS ============
num_features = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, feat in enumerate(num_features):
    ax = axes[idx]
    df[feat].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
    ax.set_title(f'{feat} Distribution')
    ax.set_xlabel(feat)
    ax.set_ylabel('Frequency')
axes[-1].remove()
plt.tight_layout()
plt.savefig('docs/eda_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ CATEGORICAL DISTRIBUTIONS ============
cat_features = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, feat in enumerate(cat_features):
    ax = axes[idx]
    counts = df[feat].value_counts()
    counts.plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
    ax.set_title(f'{feat} Distribution')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f'{v:,}', ha='center', va='bottom')
axes[-1].remove()
plt.tight_layout()
plt.savefig('docs/eda_categorical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ NUMERICAL BY TARGET ============
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, feat in enumerate(num_features):
    ax = axes[idx]
    for target_val in [0, 1]:
        subset = df[df['Response'] == target_val][feat]
        label = 'Purchase (1)' if target_val == 1 else 'No Purchase (0)'
        ax.hist(subset, bins=50, alpha=0.5, label=label, density=True, edgecolor='black')
    ax.set_title(f'{feat} by Target')
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')
    ax.legend()
axes[-1].remove()
plt.tight_layout()
plt.savefig('docs/eda_numerical_by_target.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ CATEGORICAL BY TARGET ============
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, feat in enumerate(cat_features):
    ax = axes[idx]
    purchase_rate = df.groupby(feat)['Response'].mean().sort_values()
    purchase_rate.plot(kind='barh', ax=ax, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_title(f'Purchase Rate by {feat}')
    ax.set_xlabel('Purchase Rate')
    ax.set_xlim(0, purchase_rate.max() * 1.3)
    for i, v in enumerate(purchase_rate.values):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center')
axes[-1].remove()
plt.tight_layout()
plt.savefig('docs/eda_categorical_by_target.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ CORRELATION MATRIX ============
num_cols = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel', 'Driving_License', 'Previously_Insured', 'Response']
corr_matrix = df[num_cols].corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix (Numerical Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/eda_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ BOXPLOTS BY TARGET ============
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, feat in enumerate(['Age', 'Annual_Premium', 'Vintage']):
    df.boxplot(column=feat, by='Response', ax=axes[idx])
    axes[idx].set_title(f'{feat} by Target')
    axes[idx].set_xlabel('Response')
    axes[idx].set_ylabel(feat)
plt.suptitle('')
plt.tight_layout()
plt.savefig('docs/eda_boxplots_by_target.png', dpi=150, bbox_inches='tight')
plt.close()

print("All EDA visualizations saved to docs/")