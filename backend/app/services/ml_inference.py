"""
ML Inference Service for InsureSense Backend
Loads model and preprocessing pipeline for predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class MLInferenceService:
    """
    Service for making ML predictions.
    Loads model and preprocessing pipeline once at startup.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.model = None
            self.pipeline = None
            self.feature_names = []
            self.model_metadata = {}
            self._loaded = False
            MLInferenceService._initialized = True

    def _ensure_loaded(self):
        """Lazy load artifacts on first use"""
        if self._loaded:
            return

        # Locate project root containing 'ml' folder (supports both host and Docker)
        if Path("/app/ml").exists():
            project_root = Path("/app")
        else:
            project_root = Path(__file__).resolve().parent
            for _ in range(5):
                if (project_root / "ml").exists():
                    break
                project_root = project_root.parent
        ml_preprocessing_path = project_root / "ml" / "src" / "preprocessing"
        sys.path.insert(0, str(ml_preprocessing_path))

        import __main__
        from pipeline import FeatureEngineer, TargetEncoder, LogTransformer
        # Inject into __main__ so joblib unpickler finds classes pickled from notebook/script __main__
        __main__.FeatureEngineer = FeatureEngineer
        __main__.TargetEncoder = TargetEncoder
        __main__.LogTransformer = LogTransformer

        # Load model
        model_path = project_root / settings.model_path
        pipeline_path = project_root / settings.pipeline_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")

        try:
            artifact = joblib.load(model_path)
            self.model = artifact['model']
            self.feature_names = artifact.get('feature_names', [])
            self.model_metadata = {
                'model_type': artifact.get('model_type', 'Unknown'),
                'feature_count': len(self.feature_names) if self.feature_names else 0
            }
            logger.info(f"Model loaded: {self.model_metadata.get('model_type')}")

            self.pipeline = joblib.load(pipeline_path)
            logger.info("Preprocessing pipeline loaded")
            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}")
            raise

    def preprocess_input(self, input_data: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data using the fitted pipeline"""
        self._ensure_loaded()

        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be dict, list of dicts, or DataFrame")

        # Standardize column names (rename lowercase inputs to pipeline's Title Case)
        col_rename = {
            'gender': 'Gender',
            'age': 'Age',
            'driving_license': 'Driving_License',
            'region_code': 'Region_Code',
            'previously_insured': 'Previously_Insured',
            'vehicle_age': 'Vehicle_Age',
            'vehicle_damage': 'Vehicle_Damage',
            'annual_premium': 'Annual_Premium',
            'policy_sales_channel': 'Policy_Sales_Channel',
            'vintage': 'Vintage'
        }
        df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

        # Validate required columns
        required_cols = [
            'Gender', 'Age', 'Driving_License', 'Region_Code',
            'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
            'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
        ]

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Apply preprocessing pipeline
        X_transformed = self.pipeline.transform(df)
        return X_transformed

    def predict(
        self,
        input_data: Union[Dict, List[Dict], pd.DataFrame],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Make prediction on input data"""
        X_transformed = self.preprocess_input(input_data)

        # Get prediction probabilities
        y_pred_proba = self.model.predict_proba(X_transformed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Determine propensity category
        propensity_categories = []
        for prob in y_pred_proba:
            if prob >= 0.7:
                category = "high"
            elif prob >= 0.3:
                category = "medium"
            else:
                category = "low"
            propensity_categories.append(category)

        # Format results
        is_single = len(y_pred) == 1
        if is_single:
            return {
                'prediction': int(y_pred[0]),
                'probability': float(y_pred_proba[0]),
                'propensity_category': propensity_categories[0],
                'threshold_used': threshold
            }
        else:
            return {
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'propensity_categories': propensity_categories,
                'threshold_used': threshold
            }

    def predict_batch(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Make batch predictions on a DataFrame"""
        X_transformed = self.preprocess_input(df)

        y_pred_proba = self.model.predict_proba(X_transformed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        result_df = df.copy()
        result_df['predicted_response'] = y_pred
        result_df['purchase_probability'] = y_pred_proba

        def categorize(prob):
            if prob >= 0.7:
                return 'high'
            elif prob >= 0.3:
                return 'medium'
            else:
                return 'low'

        result_df['propensity_category'] = result_df['purchase_probability'].apply(categorize)
        return result_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        self._ensure_loaded()
        return {
            'model_type': self.model_metadata.get('model_type'),
            'feature_count': self.model_metadata.get('feature_count'),
            'feature_names': self.feature_names,
            'thresholds': {
                'high': 0.7,
                'medium': 0.3,
                'low': 0.0
            }
        }


# Lazy singleton - only creates instance when first accessed
_ml_service_instance = None


def get_ml_service() -> MLInferenceService:
    """Dependency for FastAPI to get ML service (lazy initialization)"""
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLInferenceService()
    return _ml_service_instance