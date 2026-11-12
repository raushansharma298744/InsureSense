"""
Inference Module for InsureSense ML Pipeline
Loads trained model and preprocessing pipeline for predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src/preprocessing to path for custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../preprocessing'))
from pipeline import FeatureEngineer, TargetEncoder, LogTransformer


class ModelInference:
    """
    Handles model inference: loading artifacts, preprocessing input, making predictions
    """

    def __init__(self, artifacts_dir: str = None):
        if artifacts_dir is None:
            # Default to project artifacts directory
            # Go up from inference -> src -> ml -> project_root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ml_dir = os.path.dirname(os.path.dirname(current_dir))
            artifacts_dir = os.path.join(ml_dir, 'artifacts')

        self.artifacts_dir = artifacts_dir
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.model_metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessing pipeline"""
        model_path = os.path.join(self.artifacts_dir, 'best_model.pkl')
        pipeline_path = os.path.join(self.artifacts_dir, 'preprocessing_pipeline.pkl')

        # Load model
        if os.path.exists(model_path):
            artifact = joblib.load(model_path)
            self.model = artifact['model']
            self.feature_names = artifact.get('feature_names', [])
            self.model_metadata = {
                'model_type': artifact.get('model_type', 'Unknown'),
                'feature_count': len(self.feature_names) if self.feature_names else 0
            }
            print(f"Model loaded: {self.model_metadata.get('model_type')}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load preprocessing pipeline
        if os.path.exists(pipeline_path):
            self.pipeline = joblib.load(pipeline_path)
            print("Preprocessing pipeline loaded")
        else:
            raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")

    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame, List[Dict]]) -> pd.DataFrame:
        """
        Preprocess input data using the fitted pipeline

        Args:
            input_data: Single record (dict), multiple records (list of dicts), or DataFrame

        Returns:
            Transformed features ready for model prediction
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be dict, list of dicts, or DataFrame")

        # Ensure all required columns are present
        required_cols = ['Gender', 'Age', 'Driving_License', 'Region_Code',
                        'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
                        'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Apply preprocessing pipeline
        X_transformed = self.pipeline.transform(df)

        return X_transformed

    def predict(self, input_data: Union[Dict, pd.DataFrame, List[Dict]], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make prediction on input data

        Args:
            input_data: Input features
            threshold: Classification threshold

        Returns:
            Dictionary with prediction, probability, and propensity category
        """
        X_transformed = self.preprocess_input(input_data)

        # Get prediction probabilities
        y_pred_proba = self.model.predict_proba(X_transformed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Determine propensity category
        propensity_categories = []
        for prob in y_pred_proba:
            if prob >= 0.7:
                category = 'High'
            elif prob >= 0.3:
                category = 'Medium'
            else:
                category = 'Low'
            propensity_categories.append(category)

        # Format results
        if len(y_pred) == 1:
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
        """
        Make batch predictions on a DataFrame

        Args:
            df: DataFrame with input features
            threshold: Classification threshold

        Returns:
            DataFrame with original data plus predictions
        """
        X_transformed = self.preprocess_input(df)

        y_pred_proba = self.model.predict_proba(X_transformed)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['predicted_response'] = y_pred
        result_df['purchase_probability'] = y_pred_proba

        # Add propensity category
        def categorize(prob):
            if prob >= 0.7:
                return 'High'
            elif prob >= 0.3:
                return 'Medium'
            else:
                return 'Low'

        result_df['propensity_category'] = result_df['purchase_probability'].apply(categorize)

        return result_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            'model_type': self.model_metadata.get('model_type'),
            'feature_count': self.model_metadata.get('feature_count'),
            'feature_names': self.feature_names,
            'artifacts_dir': self.artifacts_dir
        }


def create_sample_input() -> Dict:
    """Create a sample input for testing"""
    return {
        'Gender': 'Male',
        'Age': 35,
        'Driving_License': 1,
        'Region_Code': 28.0,
        'Previously_Insured': 0,
        'Vehicle_Age': '1-2 Year',
        'Vehicle_Damage': 'Yes',
        'Annual_Premium': 35000.0,
        'Policy_Sales_Channel': 152.0,
        'Vintage': 150
    }


if __name__ == "__main__":
    # Test inference
    inference = ModelInference()

    # Test single prediction
    sample = create_sample_input()
    result = inference.predict(sample)
    print("Single prediction:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Propensity: {result['propensity_category']}")

    # Test batch prediction
    batch_df = pd.DataFrame([create_sample_input() for _ in range(5)])
    batch_results = inference.predict_batch(batch_df)
    print("\nBatch predictions:")
    print(batch_results[['Gender', 'Age', 'Vehicle_Age', 'Vehicle_Damage', 'predicted_response', 'purchase_probability', 'propensity_category']].to_string())

    # Model info
    print("\nModel info:")
    info = inference.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")