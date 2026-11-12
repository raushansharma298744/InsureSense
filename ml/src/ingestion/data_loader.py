"""
Data Ingestion Module for InsureSense ML Pipeline
Handles loading, splitting, and saving of datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataIngestion:
    """
    Handles data ingestion: loading raw data, train/test split, and saving processed data
    """

    def __init__(
        self,
        raw_data_path: str = '../../data/raw/train.csv',
        processed_dir: str = '../../data/processed',
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path(processed_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset from CSV"""
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded raw data: {df.shape}")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Response',
        drop_cols: list = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets with stratification

        Returns:
            X_train, X_test, y_train, y_test
        """
        if drop_cols is None:
            drop_cols = ['id']

        X = df.drop(columns=drop_cols + [target_col])
        y = df[target_col]

        stratify_param = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

        return X_train, X_test, y_train, y_test

    def save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ):
        """Save train/test splits to processed directory"""
        # Save features
        X_train.to_csv(self.processed_dir / 'X_train.csv', index=False)
        X_test.to_csv(self.processed_dir / 'X_test.csv', index=False)

        # Save targets
        y_train.to_csv(self.processed_dir / 'y_train.csv', index=False)
        y_test.to_csv(self.processed_dir / 'y_test.csv', index=False)

        print(f"Saved splits to {self.processed_dir}")

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load previously saved train/test splits"""
        X_train = pd.read_csv(self.processed_dir / 'X_train.csv')
        X_test = pd.read_csv(self.processed_dir / 'X_test.csv')
        y_train = pd.read_csv(self.processed_dir / 'y_train.csv').squeeze()
        y_test = pd.read_csv(self.processed_dir / 'y_test.csv').squeeze()

        print(f"Loaded splits - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def run(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run complete ingestion pipeline"""
        if df is None:
            df = self.load_raw_data()

        X_train, X_test, y_train, y_test = self.split_data(df)
        self.save_splits(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test


def load_processed_data(processed_dir: str = '../../data/processed') -> Tuple:
    """Convenience function to load processed data"""
    ingestion = DataIngestion(processed_dir=processed_dir)
    return ingestion.load_splits()


if __name__ == "__main__":
    ingestion = DataIngestion()
    X_train, X_test, y_train, y_test = ingestion.run()

    print("\nIngestion complete!")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")