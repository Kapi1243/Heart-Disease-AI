# Standard library imports
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from datetime import datetime


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    # Set up basic logging configuration with timestamp, module name, level, and message
    logging.basicConfig(
        level=getattr(logging, log_level),  # Convert string level to logging constant
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Create a named logger for this project
    logger = logging.getLogger('heart_disease_prediction')
    return logger


def save_pickle(obj: Any, filepath: Path) -> None:
    """Save any Python object to disk using pickle"""
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Write object in binary mode
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: Path) -> Any:
    """Load a pickled object from disk"""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logging.info(f"Loaded pickle from {filepath}")
    return obj


def save_json(data: Dict, filepath: Path) -> None:
    """Save dictionary to JSON file with nice formatting"""
    # Create parent directories if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Write with indentation for readability, convert non-serializable types to strings
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    logging.info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded JSON from {filepath}")
    return data


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)  # NumPy random number generator
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python's built-in hash randomization
    logging.info(f"Set random seed to {seed}")


def get_timestamp() -> str:
    """Get current timestamp as a formatted string for file naming"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_directory_structure(base_path: Path) -> None:
    """Create standard project folder structure"""
    # Define all needed directories
    directories = [
        base_path / "data",
        base_path / "models",
        base_path / "reports",
        base_path / "reports" / "figures",
        base_path / "logs"
    ]
    
    # Create each directory, skip if already exists
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created directory structure at {base_path}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate balanced class weights for imbalanced datasets"""
    # Count occurrences of each class
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    # Weight inversely proportional to class frequency
    weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
    logging.info(f"Calculated class weights: {weights}")
    return weights


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print comprehensive DataFrame information for debugging"""
    logger = logging.getLogger('heart_disease_prediction')
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} Information")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {df.shape}")  # Rows x columns
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")  # Convert bytes to MB
    logger.info(f"\nColumn types:\n{df.dtypes.value_counts()}")  # Count of each data type
    logger.info(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")  # Only show columns with missing data
    logger.info(f"\nDuplicate rows: {df.duplicated().sum()}")
    logger.info(f"{'='*50}\n")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Convert metric floats to formatted strings for display"""
    return {k: f"{v:.{precision}f}" for k, v in metrics.items()}


def get_feature_names_from_transformer(transformer, feature_names: List[str]) -> List[str]:
    """Extract feature names after preprocessing transformations (e.g., one-hot encoding)"""
    output_features = []
    
    # Loop through each step in the column transformer
    for name, fitted_transformer, features in transformer.transformers_:
        if name == 'remainder':  # Skip unused columns
            continue
            
        # If transformer has method to get output names (like OneHotEncoder)
        if hasattr(fitted_transformer, 'get_feature_names_out'):
            feature_names_out = fitted_transformer.get_feature_names_out(features)
            output_features.extend(feature_names_out)
        else:  # Otherwise use original names (like StandardScaler)
            output_features.extend(features)
    
    return output_features


def create_results_summary(results: Dict[str, Dict[str, float]], 
                          sort_by: str = 'roc_auc') -> pd.DataFrame:
    """Convert nested results dictionary to sorted DataFrame"""
    # Transpose so models are rows and metrics are columns
    df = pd.DataFrame(results).T
    # Sort by specified metric (default: ROC-AUC, higher is better)
    df = df.sort_values(by=sort_by, ascending=False)
    return df


def calculate_confidence_intervals(scores: np.ndarray, 
                                   confidence: float = 0.95) -> tuple:
    """Calculate mean and confidence intervals for cross-validation scores"""
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    
    # Use normal distribution to estimate confidence interval
    from scipy import stats
    margin = stats.norm.ppf((1 + confidence) / 2) * std / np.sqrt(n)
    
    # Return: (mean, lower_bound, upper_bound)
    return mean, mean - margin, mean + margin
