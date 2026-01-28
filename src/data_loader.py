# Standard imports
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Local imports - config has all our paths and column names
from config import DATA_FILE, TARGET_COLUMN
from utils import print_dataframe_info

# Get logger for this module
logger = logging.getLogger('heart_disease_prediction')


def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load CSV data file into pandas DataFrame"""
    # Use default path from config if none provided
    if filepath is None:
        filepath = DATA_FILE
    
    # Check file exists before trying to load
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)  # Read CSV into DataFrame
    
    # Log basic info about what we loaded
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    print_dataframe_info(df, "Raw Data")  # Print detailed DataFrame info
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run quality checks on the loaded data"""
    logger.info("Validating data...")
    
    # Make sure our target variable exists in the data
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    
    # Check for completely empty columns (all NaN)
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        raise ValueError(f"Completely empty columns found: {empty_cols}")
    
    # Log what types of columns we have (numeric, object, etc.)
    logger.info(f"Data types:\n{df.dtypes.value_counts()}")
    
    # Check for duplicate rows - warn but don't fail
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates:,} duplicate rows ({n_duplicates/len(df)*100:.2f}%)")
    
    logger.info("Data validation complete")


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Convert target variable to numeric 0/1 encoding"""
    logger.info(f"Encoding target column '{target_col}'")
    
    # Check what values we have in the target
    unique_values = df[target_col].unique()
    logger.info(f"Unique target values: {unique_values}")
    
    # If target is text (e.g., 'Yes'/'No'), convert to numbers
    if df[target_col].dtype == 'object':
        mapping = {'Yes': 1, 'No': 0}  # 1 = disease present, 0 = no disease
        df[target_col] = df[target_col].map(mapping)
        logger.info(f"Mapped {mapping}")
    
    # Make sure we only have 0s and 1s (binary classification)
    if not df[target_col].isin([0, 1]).all():
        logger.warning("Target column contains values other than 0 and 1")
    
    return df


def get_class_distribution(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict:
    """Calculate and log class balance statistics"""
    # Count how many of each class (0=no disease, 1=disease)
    value_counts = df[target_col].value_counts()
    
    # Build dictionary with counts, percentages, and imbalance ratio
    distribution = {
        'counts': value_counts.to_dict(),  # Raw counts of each class
        'percentages': (value_counts / len(df) * 100).to_dict(),  # As percentages
        'imbalance_ratio': value_counts.max() / value_counts.min()  # How imbalanced (e.g., 10:1)
    }
    
    # Log the stats
    logger.info(f"\nClass Distribution:")
    logger.info(f"  Counts: {distribution['counts']}")
    logger.info(f"  Percentages: {distribution['percentages']}")
    logger.info(f"  Imbalance ratio: {distribution['imbalance_ratio']:.2f}:1")
    
    return distribution


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    
    if n_before > n_after:
        logger.info(f"Removed {n_before - n_after:,} duplicate rows")
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'report') -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if len(missing_df) > 0:
        logger.info(f"\nMissing Values:\n{missing_df}")
        
        if strategy == 'drop':
            n_before = len(df)
            df = df.dropna()
            logger.info(f"Dropped {n_before - len(df):,} rows with missing values")
        
        elif strategy == 'mark':
            # Create missing indicators for columns with missing data
            for col in missing_df.index:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
            logger.info(f"Created missing indicators for {len(missing_df)} columns")
    else:
        logger.info("No missing values found")
    
    return df


def prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("=" * 80)
    logger.info("Starting data preparation pipeline")
    logger.info("=" * 80)
    
    # Load data
    df = load_data()
    
    # Validate
    validate_data(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Encode target
    df = encode_target(df)
    
    # Get class distribution
    class_dist = get_class_distribution(df)
    
    # Handle missing values (report only for now, preprocessing will handle)
    df = handle_missing_values(df, strategy='report')
    
    # Split features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    logger.info(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    logger.info("=" * 80)
    
    return X, y


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging('INFO')
    X, y = prepare_data()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns:\n{list(X.columns)}")
