import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from scipy import stats

from config import (
    TARGET_COLUMN, 
    NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES,
    REPORTS_DIR,
    FIGURE_DPI
)


logger = logging.getLogger('heart_disease_prediction')
sns.set_style("whitegrid")


def plot_target_distribution(y: pd.Series, save_path: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    value_counts = y.value_counts()
    axes[0].bar(value_counts.index, value_counts.values, color=['#3498db', '#e74c3c'])
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution (Count)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Heart Disease', 'Heart Disease'])
    
    # Add counts on bars
    for i, v in enumerate(value_counts.values):
        axes[0].text(i, v + value_counts.max() * 0.01, f'{v:,}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    percentages = value_counts / len(y) * 100
    axes[1].pie(percentages, labels=['No Heart Disease', 'Heart Disease'], 
                autopct='%1.1f%%', colors=['#3498db', '#e74c3c'],
                startangle=90)
    axes[1].set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved target distribution plot to {save_path}")
    
    plt.close()


def plot_numerical_distributions(X: pd.DataFrame, y: pd.Series, 
                                 features: Optional[List[str]] = None,
                                 save_path: Optional[Path] = None) -> None:
    if features is None:
        features = [col for col in NUMERICAL_FEATURES if col in X.columns]
    
    if not features:
        logger.warning("No numerical features found")
        return
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    df = pd.concat([X[features], y], axis=1)
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # KDE plot by class
        for class_val in [0, 1]:
            class_data = df[df[TARGET_COLUMN] == class_val][feature].dropna()
            if len(class_data) > 0:
                class_data.plot(kind='kde', ax=ax, 
                              label=f'Class {class_val}',
                              alpha=0.7)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved numerical distributions to {save_path}")
    
    plt.close()


def plot_categorical_distributions(X: pd.DataFrame, y: pd.Series,
                                   features: Optional[List[str]] = None,
                                   save_path: Optional[Path] = None) -> None:
    if features is None:
        features = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    if not features:
        logger.warning("No categorical features found")
        return
    
    # Limit to top features for readability
    features = features[:12]
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    df = pd.concat([X[features], y], axis=1)
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Create grouped bar chart
        cross_tab = pd.crosstab(df[feature], df[TARGET_COLUMN], normalize='index') * 100
        cross_tab.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], alpha=0.8)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{feature} by Heart Disease')
        ax.legend(['No Disease', 'Has Disease'], loc='best')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved categorical distributions to {save_path}")
    
    plt.close()


def plot_correlation_matrix(X: pd.DataFrame, 
                           features: Optional[List[str]] = None,
                           save_path: Optional[Path] = None) -> None:
    if features is None:
        features = [col for col in NUMERICAL_FEATURES if col in X.columns]
    
    if len(features) < 2:
        logger.warning("Need at least 2 numerical features for correlation")
        return
    
    # Calculate correlation
    corr = X[features].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved correlation matrix to {save_path}")
    
    plt.close()


def plot_missing_values(X: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    missing = X.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        logger.info("No missing values to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    missing_pct = (missing / len(X) * 100)
    missing_pct.plot(kind='barh', ax=ax, color='coral')
    
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_ylabel('Feature')
    ax.set_title('Missing Values by Feature')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved missing values plot to {save_path}")
    
    plt.close()


def plot_outliers_boxplots(X: pd.DataFrame, 
                           features: Optional[List[str]] = None,
                           save_path: Optional[Path] = None) -> None:
    if features is None:
        features = [col for col in NUMERICAL_FEATURES if col in X.columns]
    
    if not features:
        logger.warning("No numerical features for outlier analysis")
        return
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        X[feature].dropna().plot(kind='box', ax=ax, vert=True)
        ax.set_ylabel(feature)
        ax.set_title(f'Boxplot: {feature}')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        logger.info(f"Saved outliers boxplots to {save_path}")
    
    plt.close()


def generate_statistics_summary(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    stats_list = []
    
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]
    
    for col in numerical_cols:
        col_stats = {
            'Feature': col,
            'Type': 'Numerical',
            'Count': X[col].count(),
            'Missing': X[col].isnull().sum(),
            'Missing %': f"{X[col].isnull().sum() / len(X) * 100:.2f}",
            'Mean': f"{X[col].mean():.2f}",
            'Std': f"{X[col].std():.2f}",
            'Min': f"{X[col].min():.2f}",
            'Max': f"{X[col].max():.2f}",
            'Unique': X[col].nunique()
        }
        stats_list.append(col_stats)
    
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    for col in categorical_cols:
        col_stats = {
            'Feature': col,
            'Type': 'Categorical',
            'Count': X[col].count(),
            'Missing': X[col].isnull().sum(),
            'Missing %': f"{X[col].isnull().sum() / len(X) * 100:.2f}",
            'Mean': '-',
            'Std': '-',
            'Min': '-',
            'Max': '-',
            'Unique': X[col].nunique()
        }
        stats_list.append(col_stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    logger.info(f"\nStatistics Summary:\n{stats_df.to_string()}")
    
    return stats_df


def perform_eda(X: pd.DataFrame, y: pd.Series, output_dir: Optional[Path] = None) -> None:
    if output_dir is None:
        output_dir = REPORTS_DIR / "eda"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Performing Exploratory Data Analysis")
    logger.info("=" * 80)
    
    # Generate statistics
    stats_df = generate_statistics_summary(X, y)
    stats_df.to_csv(output_dir / "statistics_summary.csv", index=False)
    
    # Plot target distribution
    logger.info("Creating target distribution plot...")
    plot_target_distribution(y, save_path=output_dir / "target_distribution.png")
    
    # Plot numerical distributions
    logger.info("Creating numerical distributions...")
    plot_numerical_distributions(X, y, save_path=output_dir / "numerical_distributions.png")
    
    # Plot categorical distributions
    logger.info("Creating categorical distributions...")
    plot_categorical_distributions(X, y, save_path=output_dir / "categorical_distributions.png")
    
    # Plot correlation matrix
    logger.info("Creating correlation matrix...")
    plot_correlation_matrix(X, save_path=output_dir / "correlation_matrix.png")
    
    # Plot missing values
    logger.info("Creating missing values plot...")
    plot_missing_values(X, save_path=output_dir / "missing_values.png")
    
    # Plot outliers
    logger.info("Creating outlier boxplots...")
    plot_outliers_boxplots(X, save_path=output_dir / "outliers_boxplots.png")
    
    logger.info(f"EDA complete. Plots saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    from utils import setup_logging
    from data_loader import prepare_data
    
    setup_logging('INFO')
    X, y = prepare_data()
    perform_eda(X, y)
