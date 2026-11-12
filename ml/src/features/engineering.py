"""
Feature Engineering Module for InsureSense ML Pipeline
Additional feature engineering utilities and transformers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection wrapper supporting multiple methods
    """

    def __init__(
        self,
        method: str = 'mutual_info',
        k: int = 20,
        threshold: float = None
    ):
        self.method = method
        self.k = k
        self.threshold = threshold
        self.selector_ = None
        self.selected_features_ = None
        self.scores_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        if self.method == 'mutual_info':
            self.selector_ = SelectKBest(score_func=mutual_info_classif, k=self.k)
        elif self.method == 'f_classif':
            self.selector_ = SelectKBest(score_func=f_classif, k=self.k)
        elif self.method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            self.selector_ = VarianceThreshold(threshold=self.threshold or 0.01)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.selector_.fit(X, y)
        self.selected_features_ = X.columns[self.selector_.get_support()].tolist()

        if hasattr(self.selector_, 'scores_'):
            self.scores_ = dict(zip(X.columns, self.selector_.scores_))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector_ is None:
            raise ValueError("FeatureSelector not fitted yet")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate interaction features based on domain knowledge
    """

    def __init__(self, interactions: List[Tuple[str, str]] = None):
        self.interactions = interactions or [
            ('Vehicle_Age_encoded', 'Vehicle_Damage_encoded'),
            ('Previously_Insured', 'Vehicle_Damage_encoded'),
            ('Age', 'Annual_Premium'),
            ('Vintage', 'Age'),
        ]
        self.interaction_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'InteractionFeatureGenerator':
        # Check which interaction columns exist
        self.valid_interactions_ = []
        self.interaction_names_ = []
        for col1, col2 in self.interactions:
            if col1 in X.columns and col2 in X.columns:
                self.valid_interactions_.append((col1, col2))
                self.interaction_names_.append(f"{col1}_x_{col2}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = X.copy()
        for (col1, col2), name in zip(self.valid_interactions_, self.interaction_names_):
            X_new[name] = X_new[col1] * X_new[col2]
        return X_new

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + self.interaction_names_


class PolynomialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate polynomial features for numerical columns
    """

    def __init__(self, columns: List[str] = None, degree: int = 2, include_bias: bool = False):
        self.columns = columns or ['Age', 'Annual_Premium', 'Vintage']
        self.degree = degree
        self.include_bias = include_bias
        self.poly_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'PolynomialFeatureGenerator':
        from sklearn.preprocessing import PolynomialFeatures
        available_cols = [c for c in self.columns if c in X.columns]
        self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        self.poly_.fit(X[available_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.poly_ is None:
            raise ValueError("PolynomialFeatureGenerator not fitted yet")

        available_cols = [c for c in self.columns if c in X.columns]
        poly_features = self.poly_.transform(X[available_cols])
        feature_names = self.poly_.get_feature_names_out(available_cols)

        X_new = X.copy()
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)

        # Drop original columns that were expanded, keep only new polynomial features
        original_in_poly = [c for c in available_cols if c in feature_names]
        new_features = [c for c in feature_names if c not in original_in_poly]

        X_new = pd.concat([X_new.drop(columns=available_cols), poly_df[new_features]], axis=1)
        return X_new

    def get_feature_names_out(self, input_features=None):
        base = [c for c in (input_features or []) if c not in self.columns]
        if self.poly_ is not None:
            available_cols = [c for c in self.columns if c in (input_features or [])]
            poly_names = self.poly_.get_feature_names_out(available_cols)
            new_poly = [c for c in poly_names if c not in available_cols]
            return base + list(new_poly)
        return base


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific features based on insurance domain knowledge

    Args:
        df: Input dataframe with original features

    Returns:
        DataFrame with additional engineered features
    """
    df_new = df.copy()

    # 1. Risk indicators
    df_new['has_license'] = df_new['Driving_License'].astype(int)
    df_new['was_insured'] = df_new['Previously_Insured'].astype(int)

    # 2. Vehicle risk score (combining age and damage)
    vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    df_new['vehicle_age_ordinal'] = df_new['Vehicle_Age'].map(vehicle_age_map)
    df_new['vehicle_damage_binary'] = (df_new['Vehicle_Damage'] == 'Yes').astype(int)
    df_new['vehicle_risk_score'] = df_new['vehicle_age_ordinal'] * df_new['vehicle_damage_binary']

    # 3. Customer value indicators
    df_new['premium_per_year_customer'] = df_new['Annual_Premium'] / (df_new['Vintage'] / 365.25 + 1)
    df_new['premium_per_age'] = df_new['Annual_Premium'] / (df_new['Age'] + 1)

    # 4. Tenure segments
    df_new['is_new_customer'] = (df_new['Vintage'] < 90).astype(int)
    df_new['is_loyal_customer'] = (df_new['Vintage'] > 200).astype(int)

    # 5. Age segments
    df_new['is_young'] = (df_new['Age'] < 30).astype(int)
    df_new['is_middle_age'] = ((df_new['Age'] >= 30) & (df_new['Age'] < 55)).astype(int)
    df_new['is_senior'] = (df_new['Age'] >= 55).astype(int)

    # 6. Gender encoding
    df_new['gender_encoded'] = (df_new['Gender'] == 'Male').astype(int)

    # 7. Region/Channel frequency encoding (will be computed from training data)
    # Placeholder - actual encoding done in preprocessing pipeline

    return df_new


def get_feature_groups() -> Dict[str, List[str]]:
    """Get feature groups for different preprocessing strategies"""
    return {
        'target_encode': ['Region_Code', 'Policy_Sales_Channel'],
        'onehot': ['Gender', 'Vehicle_Age', 'Vehicle_Damage'],
        'binary': ['Driving_License', 'Previously_Insured'],
        'numerical_scale': ['Age', 'Annual_Premium', 'Vintage'],
        'log_transform': ['Annual_Premium'],
        'drop': ['id'],
        'target': ['Response']
    }


if __name__ == "__main__":
    df = pd.read_csv('../../data/raw/train.csv')
    print("Original shape:", df.shape)

    # Test domain features
    df_feat = create_domain_features(df)
    print("With domain features:", df_feat.shape)
    print("New columns:", [c for c in df_feat.columns if c not in df.columns])

    # Test interaction generator
    import sys
    sys.path.append('../preprocessing')
    from pipeline import build_preprocessing_pipeline
    preproc = build_preprocessing_pipeline()
    X = df.drop(['id', 'Response'], axis=1)
    y = df['Response']
    preproc.fit(X, y)
    X_trans = preproc.transform(X)
    print(f"\nAfter preprocessing: {X_trans.shape}")

    # Test feature selector
    selector = FeatureSelector(method='mutual_info', k=15)
    selector.fit(X_trans, y)
    print(f"\nSelected features ({len(selector.selected_features_)}):")
    for feat in selector.selected_features_:
        print(f"  {feat}")