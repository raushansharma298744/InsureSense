"""
Preprocessing Pipeline for InsureSense ML Pipeline
Handles encoding, scaling, and feature engineering with ColumnTransformer
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding for high cardinality categorical features.
    Uses smoothing to prevent overfitting on rare categories.
    """

    def __init__(self, smoothing: float = 10.0, min_samples_leaf: int = 10):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings_ = {}
        self.global_mean_ = None
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TargetEncoder':
        self.global_mean_ = y.mean()
        self.feature_names_ = X.columns.tolist()

        for col in X.columns:
            # Compute target mean per category
            stats = pd.concat([X[col], y], axis=1).groupby(col)[y.name].agg(['mean', 'count'])
            stats.columns = ['target_mean', 'count']

            # Apply smoothing: (count * cat_mean + smoothing * global_mean) / (count + smoothing)
            smoothed = (stats['count'] * stats['target_mean'] + self.smoothing * self.global_mean_) / \
                       (stats['count'] + self.smoothing)

            # For categories with very few samples, use global mean
            smoothed[stats['count'] < self.min_samples_leaf] = self.global_mean_

            self.encodings_[col] = smoothed.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in X.columns:
            if col in self.encodings_:
                # Map categories, unseen categories get global mean
                X_encoded[col] = X[col].map(self.encodings_[col]).fillna(self.global_mean_)
            else:
                X_encoded[col] = self.global_mean_
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering: create interaction features and binned features.
    Applied BEFORE preprocessing on raw features.
    """

    def __init__(self, create_interactions: bool = True, create_bins: bool = True):
        self.create_interactions = create_interactions
        self.create_bins = create_bins
        self.age_bins_ = [20, 30, 40, 50, 60, 70, 85]
        self.age_labels_ = ['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        self.premium_quantiles_ = None
        self.vintage_quantiles_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineer':
        if self.create_bins:
            # Compute quantiles for binning (on training data only)
            if 'Annual_Premium' in X.columns:
                self.premium_quantiles_ = X['Annual_Premium'].quantile([0.2, 0.4, 0.6, 0.8]).values
            if 'Vintage' in X.columns:
                self.vintage_quantiles_ = X['Vintage'].quantile([0.2, 0.4, 0.6, 0.8]).values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()

        if self.create_interactions:
            # Vehicle Age ordinal encoding for interaction
            vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
            if 'Vehicle_Age' in X_new.columns:
                X_new['Vehicle_Age_encoded'] = X_new['Vehicle_Age'].map(vehicle_age_map).fillna(0)

            # Vehicle Damage binary encoding for interaction
            if 'Vehicle_Damage' in X_new.columns:
                X_new['Vehicle_Damage_encoded'] = (X_new['Vehicle_Damage'] == 'Yes').astype(int)

            # Key interaction features identified in EDA
            if 'Vehicle_Age_encoded' in X_new.columns and 'Vehicle_Damage_encoded' in X_new.columns:
                X_new['Vehicle_Age_Damage'] = (
                    X_new['Vehicle_Age_encoded'] * X_new['Vehicle_Damage_encoded']
                ).astype(float)

            if 'Previously_Insured' in X_new.columns and 'Vehicle_Damage_encoded' in X_new.columns:
                X_new['PrevInsured_Damage'] = (
                    X_new['Previously_Insured'] * X_new['Vehicle_Damage_encoded']
                ).astype(float)

            if 'Age' in X_new.columns and 'Annual_Premium' in X_new.columns:
                X_new['Age_Premium_Ratio'] = X_new['Age'] / (X_new['Annual_Premium'] + 1)

            if 'Vintage' in X_new.columns and 'Age' in X_new.columns:
                X_new['Vintage_Age_Ratio'] = X_new['Vintage'] / (X_new['Age'] + 1)

        if self.create_bins:
            # Age bins (fixed boundaries)
            if 'Age' in X_new.columns:
                X_new['Age_Bin'] = pd.cut(
                    X_new['Age'], bins=self.age_bins_, labels=self.age_labels_,
                    include_lowest=True
                ).astype(str)

            # Premium quantile bins (from training data)
            if 'Annual_Premium' in X_new.columns and self.premium_quantiles_ is not None:
                premium_bins = [-np.inf] + list(self.premium_quantiles_) + [np.inf]
                premium_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
                X_new['Premium_Bin'] = pd.cut(
                    X_new['Annual_Premium'], bins=premium_bins, labels=premium_labels
                ).astype(str)

            # Vintage quantile bins (from training data)
            if 'Vintage' in X_new.columns and self.vintage_quantiles_ is not None:
                vintage_bins = [-np.inf] + list(self.vintage_quantiles_) + [np.inf]
                vintage_labels = ['Very_New', 'New', 'Medium', 'Old', 'Very_Old']
                X_new['Vintage_Bin'] = pd.cut(
                    X_new['Vintage'], bins=vintage_bins, labels=vintage_labels
                ).astype(str)

        return X_new

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        new_features = []
        if self.create_interactions:
            new_features.extend([
                'Vehicle_Age_Damage', 'PrevInsured_Damage',
                'Age_Premium_Ratio', 'Vintage_Age_Ratio',
                'Vehicle_Age_encoded', 'Vehicle_Damage_encoded'
            ])
        if self.create_bins:
            new_features.extend(['Age_Bin', 'Premium_Bin', 'Vintage_Bin'])
        return base + new_features


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log transformation for skewed features"""

    def __init__(self, columns: List[str], offset: float = 1.0):
        self.columns = columns
        self.offset = offset

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'LogTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for col in self.columns:
            if col in X_new.columns:
                X_new[col] = np.log1p(X_new[col] / self.offset)
        return X_new

    def get_feature_names_out(self, input_features=None):
        return input_features


class ColumnNamePreserver(BaseEstimator, TransformerMixin):
    """
    Preserves column names through sklearn transformers that don't support get_feature_names_out
    """

    def __init__(self, transformer, feature_names: List[str]):
        self.transformer = transformer
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'ColumnNamePreserver':
        self.transformer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = self.transformer.transform(X)
        if isinstance(result, np.ndarray):
            return pd.DataFrame(result, columns=self.feature_names, index=X.index)
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


def build_preprocessing_pipeline(
    target_encode_cols: List[str] = None,
    onehot_cols: List[str] = None,
    ordinal_cols: List[str] = None,
    binary_cols: List[str] = None,
    scale_cols: List[str] = None,
    log_cols: List[str] = None
) -> ColumnTransformer:
    """
    Build the complete preprocessing ColumnTransformer.
    This runs AFTER feature engineering on the engineered features.

    Args:
        target_encode_cols: High cardinality categorical columns for target encoding
        onehot_cols: Low cardinality categorical columns for one-hot encoding
        ordinal_cols: Ordinal categorical columns
        binary_cols: Binary columns (keep as-is)
        scale_cols: Numerical columns to scale
        log_cols: Columns to log-transform before scaling

    Returns:
        ColumnTransformer with all preprocessing steps
    """

    if target_encode_cols is None:
        target_encode_cols = ['Region_Code', 'Policy_Sales_Channel']
    if onehot_cols is None:
        onehot_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Age_Bin', 'Premium_Bin', 'Vintage_Bin']
    if ordinal_cols is None:
        ordinal_cols = []
    if binary_cols is None:
        binary_cols = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage_encoded']
    if scale_cols is None:
        scale_cols = ['Age', 'Annual_Premium', 'Vintage', 'Vehicle_Age_encoded',
                      'Age_Premium_Ratio', 'Vintage_Age_Ratio', 'Vehicle_Age_Damage', 'PrevInsured_Damage']
    if log_cols is None:
        log_cols = ['Annual_Premium']

    transformers = []

    # Target encoding for high cardinality features
    if target_encode_cols:
        transformers.append((
            'target_encoder',
            Pipeline([
                ('target_enc', TargetEncoder(smoothing=10.0, min_samples_leaf=10)),
                ('scaler', StandardScaler())
            ]),
            target_encode_cols
        ))

    # One-hot encoding for low cardinality categorical (including binned features)
    if onehot_cols:
        transformers.append((
            'onehot',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]),
            onehot_cols
        ))

    # Ordinal encoding (if any)
    if ordinal_cols:
        transformers.append((
            'ordinal',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]),
            ordinal_cols
        ))

    # Binary columns - pass through (already 0/1)
    if binary_cols:
        transformers.append((
            'binary',
            'passthrough',
            binary_cols
        ))

    # Numerical scaling with optional log transform
    if scale_cols:
        num_pipeline_steps = []
        if log_cols:
            num_pipeline_steps.append(('log', LogTransformer(log_cols)))
        num_pipeline_steps.append(('scaler', RobustScaler()))

        transformers.append((
            'numerical',
            Pipeline(num_pipeline_steps),
            scale_cols
        ))

    # Build ColumnTransformer
    ct = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not specified (like 'id', original categorical)
        verbose_feature_names_out=False
    )

    return ct


def build_full_pipeline(
    target_encode_cols: List[str] = None,
    onehot_cols: List[str] = None,
    binary_cols: List[str] = None,
    scale_cols: List[str] = None,
    log_cols: List[str] = None
) -> Pipeline:
    """
    Build the complete pipeline: feature engineering -> preprocessing

    Returns a Pipeline that can be fit on (X, y) and transform X
    """
    # Feature engineering first (on raw features)
    feature_engineer = FeatureEngineer(create_interactions=True, create_bins=True)

    # Preprocessing second (on engineered features)
    preprocessing = build_preprocessing_pipeline(
        target_encode_cols=target_encode_cols,
        onehot_cols=onehot_cols,
        binary_cols=binary_cols,
        scale_cols=scale_cols,
        log_cols=log_cols
    )

    full_pipeline = Pipeline([
        ('feature_engineering', feature_engineer),
        ('preprocessing', preprocessing),
    ])

    return full_pipeline


def get_feature_names(pipeline: Pipeline, input_features: List[str]) -> List[str]:
    """Get output feature names from a fitted pipeline"""
    try:
        # Get names from preprocessing step (which is the last step)
        preproc = pipeline.named_steps['preprocessing']
        if hasattr(preproc, 'get_feature_names_out'):
            final_features = preproc.get_feature_names_out(input_features)
        else:
            final_features = input_features

        return list(final_features)
    except Exception as e:
        print(f"Error getting feature names: {e}")
        return []


def save_pipeline(pipeline: Pipeline, filepath: str):
    """Save pipeline to disk"""
    joblib.dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")


def load_pipeline(filepath: str) -> Pipeline:
    """Load pipeline from disk"""
    pipeline = joblib.load(filepath)
    print(f"Pipeline loaded from {filepath}")
    return pipeline


if __name__ == "__main__":
    # Test the pipeline
    df = pd.read_csv('../../data/raw/train.csv')
    X = df.drop(['id', 'Response'], axis=1)
    y = df['Response']

    print("Building full pipeline...")
    pipeline = build_full_pipeline()

    print("Fitting pipeline...")
    pipeline.fit(X, y)

    print("Transforming...")
    X_transformed = pipeline.transform(X)
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")

    # Get feature names
    feature_names = get_feature_names(pipeline, X.columns.tolist())
    print(f"\nFeature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")

    # Save pipeline
    save_pipeline(pipeline, '../../artifacts/preprocessing_pipeline.pkl')