import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from config import (
    REPORTS_DIR,
    FIGURE_DPI,
    THRESHOLDS_TO_EVALUATE,
    COST_FALSE_POSITIVE,
    COST_FALSE_NEGATIVE
)


logger = logging.getLogger('heart_disease_prediction')


def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   model_name: str = "Model") -> Dict[str, float]:
    logger.info(f"Evaluating {model_name}...")
    
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0)
    }
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y, y_pred_proba)
    
    # Log metrics
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Prec:  {metrics['average_precision']:.4f}")
    
    return metrics


def evaluate_all_models(models: Dict[str, Any], 
                       X_test: np.ndarray, y_test: np.ndarray,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[np.ndarray] = None) -> pd.DataFrame:
    logger.info("=" * 80)
    logger.info("Evaluating All Models")
    logger.info("=" * 80)
    
    results = {}
    
    for name, model in models.items():
        # Test set evaluation
        test_metrics = evaluate_model(model, X_test, y_test, f"{name} (test)")
        results[name] = test_metrics
        
        # Validation set evaluation (if provided)
        if X_val is not None and y_val is not None:
            val_metrics = evaluate_model(model, X_val, y_val, f"{name} (val)")
            # Add validation metrics with '_val' suffix
            for metric, value in val_metrics.items():
                results[name][f'{metric}_val'] = value
    
    # Create DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    logger.info("=" * 80)
    logger.info("Model Comparison (sorted by ROC-AUC):")
    logger.info(f"\n{results_df.to_string()}")
    logger.info("=" * 80)
    
    return results_df


def plot_confusion_matrices(models: Dict[str, Any], 
                           X_test: np.ndarray, y_test: np.ndarray,
                           save_path: Optional[Path] = None) -> None:
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=True, square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_xticklabels(['No Disease', 'Disease'])
        ax.set_yticklabels(['No Disease', 'Disease'])
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved confusion matrices to {save_path}")
    
    plt.close()


def plot_roc_curves(models: Dict[str, Any],
                    X_test: np.ndarray, y_test: np.ndarray,
                    save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    
    # Plot ROC curve for each model
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
    
    plt.close()


def plot_precision_recall_curves(models: Dict[str, Any],
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate baseline (no skill)
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='k', linestyle='--', 
               label=f'Baseline (AP = {baseline:.3f})', linewidth=2)
    
    # Plot PR curve for each model
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            
            ax.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved PR curves to {save_path}")
    
    plt.close()


def analyze_threshold_impact(model, X_test: np.ndarray, y_test: np.ndarray,
                            thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    if not hasattr(model, 'predict_proba'):
        logger.warning("Model doesn't support predict_proba")
        return pd.DataFrame()
    
    if thresholds is None:
        thresholds = THRESHOLDS_TO_EVALUATE
    
    logger.info("Analyzing threshold impact...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = []
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_test, y_pred_threshold),
            'precision': precision_score(y_test, y_pred_threshold, zero_division=0),
            'recall': recall_score(y_test, y_pred_threshold, zero_division=0),
            'f1': f1_score(y_test, y_pred_threshold, zero_division=0),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        # Calculate cost (weighted by false positive/negative costs)
        total_cost = (fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE)
        metrics['total_cost'] = total_cost
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"\nThreshold Analysis:\n{results_df.to_string()}")
    
    # Find optimal threshold by cost
    optimal_idx = results_df['total_cost'].idxmin()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    logger.info(f"\nOptimal threshold by cost: {optimal_threshold}")
    
    return results_df


def analyze_errors(model, X_test: np.ndarray, y_test: np.ndarray,
                  feature_names: Optional[List[str]] = None,
                  save_path: Optional[Path] = None) -> pd.DataFrame:
    logger.info("Performing error analysis...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Identify errors
    errors = y_pred != y_test
    false_positives = (y_pred == 1) & (y_test == 0)
    false_negatives = (y_pred == 0) & (y_test == 1)
    
    logger.info(f"Total errors: {errors.sum()} ({errors.sum()/len(y_test)*100:.2f}%)")
    logger.info(f"  False Positives: {false_positives.sum()}")
    logger.info(f"  False Negatives: {false_negatives.sum()}")
    
    # Create error DataFrame
    error_data = {
        'true_label': y_test,
        'predicted_label': y_pred,
        'error': errors.astype(int),
        'error_type': np.where(false_positives, 'FP',
                              np.where(false_negatives, 'FN', 'Correct'))
    }
    
    if y_pred_proba is not None:
        error_data['prediction_confidence'] = y_pred_proba
    
    error_df = pd.DataFrame(error_data)
    
    # Add features if available
    if feature_names is not None and len(feature_names) == X_test.shape[1]:
        for idx, feat_name in enumerate(feature_names):
            error_df[feat_name] = X_test[:, idx]
    
    if save_path:
        error_df.to_csv(save_path, index=False)
        logger.info(f"Saved error analysis to {save_path}")
    
    return error_df


def plot_model_comparison(results_df: pd.DataFrame,
                         metric: str = 'roc_auc',
                         save_path: Optional[Path] = None) -> None:
    if metric not in results_df.columns:
        logger.warning(f"Metric '{metric}' not found in results")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by metric
    sorted_df = results_df.sort_values(metric, ascending=True)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(sorted_df)), sorted_df[metric])
    
    # Color bars (baseline vs ML models)
    for idx, (model_name, _) in enumerate(sorted_df.iterrows()):
        if 'dummy' in model_name:
            bars[idx].set_color('#e74c3c')  # Red for baselines
        else:
            bars[idx].set_color('#3498db')  # Blue for ML models
    
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df.index)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f'Model Comparison by {metric.upper()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for idx, value in enumerate(sorted_df[metric]):
        ax.text(value + 0.01, idx, f'{value:.3f}', 
               va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved model comparison to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    from utils import setup_logging, set_seeds
    from data_loader import prepare_data
    from preprocessing import preprocess_data
    from models import train_all_models
    
    setup_logging('INFO')
    set_seeds(42)
    
    # Prepare data
    X, y = prepare_data()
    processed_data = preprocess_data(X, y)
    
    # Train models
    training_results = train_all_models(
        processed_data['X_train'],
        processed_data['y_train'],
        include_baselines=True,
        tune_hyperparams=False,
        include_ensembles=False
    )
    
    # Evaluate
    results_df = evaluate_all_models(
        training_results['models'],
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    print(f"\nTop 3 models:")
    print(results_df.head(3))
