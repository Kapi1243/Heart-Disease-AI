import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
import shap
from sklearn.inspection import permutation_importance

from config import (
    REPORTS_DIR,
    FIGURE_DPI,
    SHAP_MAX_DISPLAY,
    SHAP_SAMPLE_SIZE,
    FEATURE_IMPORTANCE_TOP_N
)


logger = logging.getLogger('heart_disease_prediction')

# Suppress SHAP's verbose logging
logging.getLogger('shap').setLevel(logging.WARNING)


def get_feature_importance(model, feature_names: List[str],
                          model_name: str = "Model") -> pd.DataFrame:
    logger.info(f"Extracting feature importance from {model_name}...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"{model_name} doesn't have feature_importances_ or coef_")
        return pd.DataFrame()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    logger.info(f"Top 5 features for {model_name}:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame,
                           model_name: str = "Model",
                           top_n: int = FEATURE_IMPORTANCE_TOP_N,
                           save_path: Optional[Path] = None) -> None:
    if importance_df.empty:
        logger.warning("Empty importance DataFrame")
        return
    
    # Select top N features
    plot_df = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart
    ax.barh(range(len(plot_df)), plot_df['importance'], color='steelblue')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features - {model_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add value labels
    for idx, value in enumerate(plot_df['importance']):
        ax.text(value + max(plot_df['importance']) * 0.01, idx, 
               f'{value:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.close()


def calculate_permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str],
                                    n_repeats: int = 10,
                                    random_state: int = 42) -> pd.DataFrame:
    logger.info("Calculating permutation importance...")
    
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    logger.info("Top 5 features by permutation importance:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance_mean']:.4f} (+/- {row['importance_std']:.4f})")
    
    return importance_df


def explain_with_shap(model, X: np.ndarray, 
                     feature_names: List[str],
                     model_name: str = "Model",
                     sample_size: int = SHAP_SAMPLE_SIZE,
                     save_path: Optional[Path] = None) -> shap.Explanation:
    logger.info(f"Generating SHAP explanations for {model_name}...")
    
    # Sample data if too large
    if len(X) > sample_size:
        logger.info(f"Sampling {sample_size} instances for SHAP calculation")
        sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    
    # Convert to DataFrame for better SHAP visualization
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    
    try:
        # Try TreeExplainer first (faster for tree models)
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_df)
            
            # Handle binary classification output format
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
        else:
            # Fall back to KernelExplainer for other models
            logger.info("Using KernelExplainer (this may take longer)...")
            
            # Use a background dataset (sample)
            background_size = min(100, len(X_sample))
            background = X_sample_df.iloc[:background_size]
            
            explainer = shap.KernelExplainer(
                model.predict_proba, 
                background
            )
            shap_values = explainer.shap_values(X_sample_df)
            
            # For binary classification, get positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            data=X_sample_df.values,
            feature_names=feature_names
        )
        
        logger.info("SHAP explanations generated successfully")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        return None


def plot_shap_summary(shap_values: shap.Explanation,
                     model_name: str = "Model",
                     max_display: int = SHAP_MAX_DISPLAY,
                     save_path: Optional[Path] = None) -> None:
    if shap_values is None:
        logger.warning("No SHAP values to plot")
        return
    
    logger.info(f"Creating SHAP summary plot for {model_name}...")
    
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values.values,
        shap_values.data,
        feature_names=shap_values.feature_names,
        max_display=max_display,
        show=False
    )
    
    plt.title(f'SHAP Summary - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    plt.close()


def plot_shap_waterfall(shap_values: shap.Explanation,
                       instance_idx: int = 0,
                       model_name: str = "Model",
                       save_path: Optional[Path] = None) -> None:
    if shap_values is None:
        logger.warning("No SHAP values to plot")
        return
    
    logger.info(f"Creating SHAP waterfall plot for instance {instance_idx}...")
    
    plt.figure(figsize=(10, 8))
    
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[instance_idx],
            base_values=shap_values.base_values,
            data=shap_values.data[instance_idx],
            feature_names=shap_values.feature_names
        ),
        show=False
    )
    
    plt.title(f'SHAP Waterfall - {model_name} (Instance {instance_idx})', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved SHAP waterfall plot to {save_path}")
    
    plt.close()


def explain_model(model, X: np.ndarray, feature_names: List[str],
                 model_name: str = "Model",
                 output_dir: Optional[Path] = None) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = REPORTS_DIR / "explainability"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"Explaining {model_name}")
    logger.info("=" * 80)
    
    results = {}
    
    # 1. Feature importance (if available)
    importance_df = get_feature_importance(model, feature_names, model_name)
    if not importance_df.empty:
        results['feature_importance'] = importance_df
        plot_feature_importance(
            importance_df, 
            model_name,
            save_path=output_dir / f"{model_name}_feature_importance.png"
        )
    
    # 2. SHAP explanations
    shap_values = explain_with_shap(
        model, X, feature_names, model_name
    )
    if shap_values is not None:
        results['shap_values'] = shap_values
        
        # Plot SHAP summary
        plot_shap_summary(
            shap_values,
            model_name,
            save_path=output_dir / f"{model_name}_shap_summary.png"
        )
        
        # Plot SHAP waterfall for first few instances
        for i in range(min(3, len(shap_values.data))):
            plot_shap_waterfall(
                shap_values,
                instance_idx=i,
                model_name=model_name,
                save_path=output_dir / f"{model_name}_shap_waterfall_{i}.png"
            )
    
    logger.info(f"Explanations saved to {output_dir}")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    from utils import setup_logging, set_seeds
    from data_loader import prepare_data
    from preprocessing import preprocess_data
    from models import create_ml_models, train_model
    
    setup_logging('INFO')
    set_seeds(42)
    
    # Prepare data
    X, y = prepare_data()
    processed_data = preprocess_data(X, y)
    
    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model, _ = train_model(
        rf_model,
        processed_data['X_train'],
        processed_data['y_train'],
        "Random Forest"
    )
    
    # Explain model
    explain_model(
        rf_model,
        processed_data['X_test'],
        processed_data['feature_names'],
        "Random_Forest"
    )
