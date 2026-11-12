"""
Model Evaluation Module for InsureSense ML Pipeline
Comprehensive evaluation with multiple metrics and visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def evaluate_model_comprehensive(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation

    Args:
        model: Fitted model with predict and predict_proba methods
        X_test: Test features
        y_test: Test targets
        threshold: Classification threshold

    Returns:
        Dictionary with all evaluation metrics
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
    })

    # Precision@K and Recall@K for different K values
    for k in [50, 100, 200, 500, 1000]:
        if k <= len(y_test):
            top_k_idx = np.argsort(y_pred_proba)[-k:]
            y_pred_k = np.zeros_like(y_test)
            y_pred_k[top_k_idx] = 1
            metrics[f'precision_at_{k}'] = precision_score(y_test.iloc[top_k_idx], np.ones(k), zero_division=0)
            metrics[f'recall_at_{k}'] = recall_score(y_test, y_pred_k, zero_division=0)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def evaluate_thresholds(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Evaluate model at different thresholds

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test targets
        thresholds: List of thresholds to evaluate

    Returns:
        DataFrame with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })

    return pd.DataFrame(results)


def find_optimal_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal threshold based on a metric

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test targets
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Optimal threshold and metrics at that threshold
    """
    thresholds_df = evaluate_thresholds(model, X_test, y_test)
    optimal_idx = thresholds_df[metric].idxmax()
    optimal_threshold = thresholds_df.loc[optimal_idx, 'threshold']
    optimal_metrics = thresholds_df.loc[optimal_idx].to_dict()

    return optimal_threshold, optimal_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None
) -> plt.Figure:
    """Plot confusion matrix"""
    if class_names is None:
        class_names = ['No Purchase (0)', 'Purchase (1)']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    save_path: str = None
) -> plt.Figure:
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_precision_recall_curve(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    save_path: str = None
) -> plt.Figure:
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_threshold_analysis(
    thresholds_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """Plot metrics vs threshold"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds_df['threshold'], thresholds_df['precision'], label='Precision', marker='o')
    ax.plot(thresholds_df['threshold'], thresholds_df['recall'], label='Recall', marker='s')
    ax.plot(thresholds_df['threshold'], thresholds_df['f1'], label='F1', marker='^')
    ax.plot(thresholds_df['threshold'], thresholds_df['accuracy'], label='Accuracy', marker='d')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Classification Threshold')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str = None
) -> plt.Figure:
    """Plot feature importance"""
    if importance_df.empty:
        return None

    top_features = importance_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_distribution(
    y_pred_proba: np.ndarray,
    y_test: pd.Series,
    save_path: str = None
) -> plt.Figure:
    """Plot prediction probability distribution by true class"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cls in [0, 1]:
        mask = y_test == cls
        label = 'Purchase (1)' if cls == 1 else 'No Purchase (0)'
        ax.hist(y_pred_proba[mask], bins=50, alpha=0.5, label=label, density=True, edgecolor='black')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Probability Distribution by True Class')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_evaluation_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str] = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Generate complete evaluation report with plots

    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test targets
        feature_names: Feature names for importance plot
        output_dir: Directory to save plots

    Returns:
        Dictionary with all evaluation results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Comprehensive evaluation
    eval_results = evaluate_model_comprehensive(model, X_test, y_test)

    # Threshold analysis
    thresholds_df = evaluate_thresholds(model, X_test, y_test)
    optimal_threshold, optimal_metrics = find_optimal_threshold(model, X_test, y_test, 'f1')

    # Generate plots
    print("Generating plots...")
    try:
        plot_confusion_matrix(
            eval_results['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        print("  confusion_matrix.png saved")
    except Exception as e:
        print(f"  confusion_matrix.png FAILED: {e}")

    try:
        plot_roc_curve(
            y_test, eval_results['y_pred_proba'],
            save_path=os.path.join(output_dir, 'roc_curve.png')
        )
        print("  roc_curve.png saved")
    except Exception as e:
        print(f"  roc_curve.png FAILED: {e}")

    try:
        plot_precision_recall_curve(
            y_test, eval_results['y_pred_proba'],
            save_path=os.path.join(output_dir, 'pr_curve.png')
        )
        print("  pr_curve.png saved")
    except Exception as e:
        print(f"  pr_curve.png FAILED: {e}")

    try:
        plot_threshold_analysis(
            thresholds_df,
            save_path=os.path.join(output_dir, 'threshold_analysis.png')
        )
        print("  threshold_analysis.png saved")
    except Exception as e:
        print(f"  threshold_analysis.png FAILED: {e}")

    try:
        plot_prediction_distribution(
            eval_results['y_pred_proba'], y_test,
            save_path=os.path.join(output_dir, 'prediction_distribution.png')
        )
        print("  prediction_distribution.png saved")
    except Exception as e:
        print(f"  prediction_distribution.png FAILED: {e}")

    if feature_names and hasattr(model, 'feature_importances_'):
        try:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            plot_feature_importance(
                importance_df,
                save_path=os.path.join(output_dir, 'feature_importance.png')
            )
            print("  feature_importance.png saved")
        except Exception as e:
            print(f"  feature_importance.png FAILED: {e}")

    # Print summary
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\nOptimal Threshold (F1): {optimal_threshold:.3f}")
    print(f"Metrics at Optimal Threshold:")
    for k, v in optimal_metrics.items():
        if k != 'threshold':
            print(f"  {k}: {v:.4f}")

    print(f"\nDefault Threshold (0.5) Metrics:")
    for k, v in eval_results['metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nConfusion Matrix:")
    print(eval_results['confusion_matrix'])

    return {
        **eval_results,
        'thresholds_df': thresholds_df,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics
    }


if __name__ == "__main__":
    # Test evaluation
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../ingestion'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../preprocessing'))
    from data_loader import load_processed_data
    from pipeline import build_full_pipeline
    import joblib

    # Load model
    artifact = joblib.load(os.path.join(os.path.dirname(__file__), '../../artifacts/best_model.pkl'))
    model = artifact['model']
    feature_names = artifact.get('feature_names', [])

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Build and fit pipeline
    pipeline = build_full_pipeline()
    pipeline.fit(X_train, y_train)
    X_test_trans = pipeline.transform(X_test)

    # Evaluate
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(project_root, 'docs', 'evaluation')
    print(f"Project root: {project_root}")
    print(f"Output dir: {output_dir}")
    generate_evaluation_report(model, X_test_trans, y_test, feature_names, output_dir)