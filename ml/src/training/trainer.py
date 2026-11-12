"""
Model Training Module for InsureSense ML Pipeline
Handles model training, cross-validation, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, make_scorer
)
from sklearn.base import clone
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Model Definitions
# ============================================================

def get_models(class_weight: str = 'balanced') -> Dict[str, Any]:
    """Get dictionary of models to train"""
    return {
        'logistic_regression': LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'decision_tree': DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=42
        ),
        'xgboost': XGBClassifier(
            scale_pos_weight=7.14,  # Approximate imbalance ratio
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
    }


def get_param_grids() -> Dict[str, Dict]:
    """Get hyperparameter grids for each model - REDUCED FOR TESTING"""
    return {
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs'],
            'penalty': ['l2']
        },
        'decision_tree': {
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [5, 10],
        },
        'gradient_boosting': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3, 5],
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1],
            'max_depth': [3, 5],
        },
    }


# ============================================================
# Custom Scorers
# ============================================================

def precision_at_k(y_true, y_pred_proba, k=100):
    """Precision at top K predictions"""
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    return precision_score(y_true.iloc[top_k_idx], np.ones(k))


def recall_at_k(y_true, y_pred_proba, k=100):
    """Recall at top K predictions"""
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    y_pred_k = np.zeros_like(y_true)
    y_pred_k[top_k_idx] = 1
    return recall_score(y_true, y_pred_k)


# ============================================================
# Training Functions
# ============================================================

def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: Dict = None
) -> Dict[str, float]:
    """Evaluate model using cross-validation"""
    if scoring is None:
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision'
        }

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1, return_train_score=False)

    return {f'{metric}_mean': np.mean(scores) for metric, scores in results.items() if metric.startswith('test_')}


def train_with_cv(
    model: Any,
    param_grid: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'f1',
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """Train model with GridSearchCV"""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict = None,
    param_grids: Dict = None,
    cv: int = 5,
    scoring: str = 'f1'
) -> Dict[str, Dict]:
    """Train all models with hyperparameter tuning"""
    if models is None:
        models = get_models()
    if param_grids is None:
        param_grids = get_param_grids()

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        param_grid = param_grids.get(name, {})

        if param_grid:
            best_model, cv_info = train_with_cv(model, param_grid, X, y, cv=cv, scoring=scoring)
            print(f"Best params: {cv_info['best_params']}")
            print(f"Best CV {scoring}: {cv_info['best_score']:.4f}")
        else:
            best_model = clone(model)
            best_model.fit(X, y)
            cv_info = {'best_params': {}, 'best_score': None}

        # Evaluate on full CV
        cv_scores = evaluate_model(best_model, X, y, cv=cv)
        print(f"CV Scores: {cv_scores}")

        results[name] = {
            'model': best_model,
            'cv_scores': cv_scores,
            'cv_info': cv_info
        }

    return results


def compare_models(results: Dict[str, Dict], metric: str = 'test_f1_mean') -> pd.DataFrame:
    """Compare models based on a metric"""
    comparison = []
    for name, info in results.items():
        comparison.append({
            'model': name,
            metric: info['cv_scores'].get(metric, 0),
            'accuracy_mean': info['cv_scores'].get('test_accuracy_mean', 0),
            'precision_mean': info['cv_scores'].get('test_precision_mean', 0),
            'recall_mean': info['cv_scores'].get('test_recall_mean', 0),
            'f1_mean': info['cv_scores'].get('test_f1_mean', 0),
            'roc_auc_mean': info['cv_scores'].get('test_roc_auc_mean', 0),
            'pr_auc_mean': info['cv_scores'].get('test_pr_auc_mean', 0),
        })

    df = pd.DataFrame(comparison).sort_values(metric, ascending=False)
    return df


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    # Handle length mismatch
    min_len = min(len(importance), len(feature_names))
    return pd.DataFrame({
        'feature': feature_names[:min_len],
        'importance': importance[:min_len]
    }).sort_values('importance', ascending=False)


# ============================================================
# MLflow Integration
# ============================================================

def log_model_to_mlflow(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None,
    metrics: Dict = None,
    artifacts: Dict = None,
    mlflow_uri: str = "http://localhost:5000"
):
    """Log model, parameters, metrics, and artifacts to MLflow"""
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)

            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"InsureSense-{model_name}"
            )

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            print(f"Logged {model_name} to MLflow")
    except Exception as e:
        print(f"MLflow logging failed: {e}")


# ============================================================
# Main Training Pipeline
# ============================================================

def run_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "insuresense-training"
) -> Dict[str, Any]:
    """
    Run complete training pipeline with all models

    Returns:
        Dictionary with best model, all results, and comparison
    """
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        mlflow.set_experiment(experiment_name)
        mlflow_available = True
    except Exception as e:
        print(f"MLflow not available: {e}")
        mlflow_available = False

    # Train all models
    print("Starting model training...")
    results = train_all_models(X_train, y_train, cv=5, scoring='f1')

    # Compare models
    comparison = compare_models(results, 'f1_mean')
    print("\n" + "="*80)
    print("MODEL COMPARISON (sorted by F1)")
    print("="*80)
    print(comparison.to_string(index=False))

    # Select best model
    best_model_name = comparison.iloc[0]['model']
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name}")

    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_pr_auc': average_precision_score(y_test, y_pred_proba),
    }

    print("\nTest Set Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Feature importance
    importance_df = get_feature_importance(best_model, feature_names)
    if not importance_df.empty:
        print(f"\nTop 20 Feature Importances:")
        print(importance_df.head(20).to_string(index=False))

    # Log to MLflow
    if mlflow_available:
        try:
            log_model_to_mlflow(
                model=best_model,
                model_name=best_model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=results[best_model_name]['cv_info']['best_params'],
                metrics={**results[best_model_name]['cv_scores'], **test_metrics},
                mlflow_uri=mlflow_uri
            )
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'all_results': results,
        'comparison': comparison,
        'test_metrics': test_metrics,
        'feature_importance': importance_df
    }


def save_model_artifact(model: Any, filepath: str, feature_names: List[str] = None):
    """Save model with metadata"""
    artifact = {
        'model': model,
        'feature_names': feature_names,
        'model_type': type(model).__name__
    }
    joblib.dump(artifact, filepath)
    print(f"Model artifact saved to {filepath}")


if __name__ == "__main__":
    # Test training with processed data
    import sys
    sys.path.append('../ingestion')
    sys.path.append('../preprocessing')
    from data_loader import load_processed_data
    from pipeline import build_full_pipeline, get_feature_names

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    print("Building pipeline...")
    pipeline = build_full_pipeline()

    print("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("Transforming data...")
    X_train_trans = pipeline.transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    feature_names = get_feature_names(pipeline, X_train.columns.tolist())
    print(f"Features: {len(feature_names)}")

    # Run training
    results = run_training_pipeline(
        X_train_trans, y_train,
        X_test_trans, y_test,
        feature_names
    )

    # Save best model
    save_model_artifact(
        results['best_model'],
        '../../artifacts/best_model.pkl',
        feature_names
    )

    print("\nTraining complete!")