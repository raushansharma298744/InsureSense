"""
Data Validation Module for InsureSense ML Pipeline
Validates schema, types, ranges, and data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict


class DataValidator:
    """
    Validates the vehicle insurance cross-sell dataset.
    Checks schema, data types, value ranges, missing values, and duplicates.
    """

    EXPECTED_COLUMNS = [
        'id', 'Gender', 'Age', 'Driving_License', 'Region_Code',
        'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
        'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response'
    ]

    CATEGORICAL_VALUES = {
        'Gender': ['Male', 'Female'],
        'Vehicle_Age': ['< 1 Year', '1-2 Year', '> 2 Years'],
        'Vehicle_Damage': ['No', 'Yes'],
    }

    BINARY_COLUMNS = ['Driving_License', 'Previously_Insured', 'Response']

    NUMERIC_RANGES = {
        'Age': (18, 100),
        'Annual_Premium': (0, 1000000),
        'Vintage': (0, 1000),
        'Region_Code': (0, 100),
        'Policy_Sales_Channel': (0, 200),
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.errors = []
        self.warnings = []
        self.stats = {}

    def validate(self) -> ValidationResult:
        """Run all validation checks"""
        self._check_schema()
        self._check_missing_values()
        self._check_duplicates()
        self._check_data_types()
        self._check_categorical_values()
        self._check_binary_values()
        self._check_numeric_ranges()
        self._check_target_distribution()
        self._compute_stats()

        is_valid = len(self.errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            stats=self.stats
        )

    def _check_schema(self):
        """Validate column names and count"""
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
        extra_cols = set(self.df.columns) - set(self.EXPECTED_COLUMNS)

        if missing_cols:
            self.errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            self.warnings.append(f"Unexpected columns: {extra_cols}")

        if len(self.df.columns) != len(self.EXPECTED_COLUMNS):
            self.warnings.append(f"Column count mismatch: expected {len(self.EXPECTED_COLUMNS)}, got {len(self.df.columns)}")

    def _check_missing_values(self):
        """Check for missing/null values"""
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]

        if len(missing_cols) > 0:
            for col, count in missing_cols.items():
                pct = count / len(self.df) * 100
                self.errors.append(f"Column '{col}' has {count} missing values ({pct:.2f}%)")

    def _check_duplicates(self):
        """Check for duplicate rows and IDs"""
        dup_rows = self.df.duplicated().sum()
        if dup_rows > 0:
            self.warnings.append(f"Found {dup_rows} duplicate rows")

        dup_ids = self.df['id'].duplicated().sum()
        if dup_ids > 0:
            self.warnings.append(f"Found {dup_ids} duplicate IDs")

    def _check_data_types(self):
        """Validate data types"""
        expected_types = {
            'id': 'int64',
            'Gender': 'object',
            'Age': 'int64',
            'Driving_License': 'int64',
            'Region_Code': 'float64',
            'Previously_Insured': 'int64',
            'Vehicle_Age': 'object',
            'Vehicle_Damage': 'object',
            'Annual_Premium': 'float64',
            'Policy_Sales_Channel': 'float64',
            'Vintage': 'int64',
            'Response': 'int64',
        }

        for col, expected in expected_types.items():
            if col in self.df.columns:
                actual = str(self.df[col].dtype)
                if actual != expected:
                    self.warnings.append(f"Column '{col}': expected {expected}, got {actual}")

    def _check_categorical_values(self):
        """Validate categorical column values"""
        for col, allowed_values in self.CATEGORICAL_VALUES.items():
            if col in self.df.columns:
                unique_vals = set(self.df[col].unique())
                allowed_set = set(allowed_values)
                unexpected = unique_vals - allowed_set
                if unexpected:
                    self.errors.append(f"Column '{col}' has unexpected values: {unexpected}")

    def _check_binary_values(self):
        """Validate binary columns contain only 0/1"""
        for col in self.BINARY_COLUMNS:
            if col in self.df.columns:
                unique_vals = set(self.df[col].unique())
                allowed = {0, 1}
                unexpected = unique_vals - allowed
                if unexpected:
                    self.errors.append(f"Binary column '{col}' has unexpected values: {unexpected}")

    def _check_numeric_ranges(self):
        """Validate numeric columns are within expected ranges"""
        for col, (min_val, max_val) in self.NUMERIC_RANGES.items():
            if col in self.df.columns:
                actual_min = self.df[col].min()
                actual_max = self.df[col].max()
                if actual_min < min_val or actual_max > max_val:
                    self.warnings.append(
                        f"Column '{col}' out of expected range [{min_val}, {max_val}]: "
                        f"actual [{actual_min}, {actual_max}]"
                    )

    def _check_target_distribution(self):
        """Check target class distribution"""
        if 'Response' in self.df.columns:
            dist = self.df['Response'].value_counts(normalize=True)
            minority_pct = dist.min() * 100
            if minority_pct < 5:
                self.warnings.append(f"Severe class imbalance: minority class only {minority_pct:.2f}%")
            elif minority_pct < 20:
                self.warnings.append(f"Class imbalance detected: minority class {minority_pct:.2f}%")

    def _compute_stats(self):
        """Compute summary statistics"""
        self.stats = {
            'n_samples': len(self.df),
            'n_features': len(self.df.columns) - 1,
            'missing_total': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_ids': self.df['id'].duplicated().sum(),
            'target_distribution': self.df['Response'].value_counts().to_dict() if 'Response' in self.df.columns else {},
            'categorical_cardinality': {
                col: self.df[col].nunique()
                for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Region_Code', 'Policy_Sales_Channel']
                if col in self.df.columns
            }
        }


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Convenience function to validate a dataset"""
    validator = DataValidator(df)
    return validator.validate()


if __name__ == "__main__":
    df = pd.read_csv('../../data/raw/train.csv')
    result = validate_dataset(df)

    print("=" * 60)
    print("DATA VALIDATION RESULT")
    print("=" * 60)
    print(f"Valid: {result.is_valid}")
    print(f"\nErrors ({len(result.errors)}):")
    for err in result.errors:
        print(f"  - {err}")
    print(f"\nWarnings ({len(result.warnings)}):")
    for warn in result.warnings:
        print(f"  - {warn}")
    print(f"\nStats:")
    for key, val in result.stats.items():
        print(f"  {key}: {val}")