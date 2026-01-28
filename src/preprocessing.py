# Standard imports
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# scikit-learn imports for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# SMOTE for handling class imbalance (over-sampling minority class)
from imblearn.over_sampling import SMOTE

# Import our configuration settings
from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    USE_SMOTE,
    SMOTE_SAMPLING_STRATEGY,
    SMOTE_K_NEIGHBORS
)
from utils import save_pickle

# Get logger for this module
logger = logging.getLogger('heart_disease_prediction')


def create_preprocessor(numerical_features: list, 
                        categorical_features: list) -> ColumnTransformer:
    """Create preprocessing pipeline for numerical and categorical features"""
    
    # Pipeline for numerical features: fill missing values with mean, then standardize (mean=0, std=1)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Replace NaN with column mean
        ('scaler', StandardScaler())  # Standardize to z-scores (important for distance-based models)
    ])
    
    # Pipeline for categorical features: fill missing with mode, then convert to dummy variables
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace NaN with most common value
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))  
        # drop='first' prevents multicollinearity by dropping one category per feature
        # handle_unknown='ignore' prevents errors if new categories appear in test data
    ])
    
    # Combine both transformers into one object that applies the right transform to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),  # Apply numerical pipeline to these columns
            ('cat', categorical_transformer, categorical_features)  # Apply categorical pipeline to these columns
        ],
        remainder='drop'  # Drop any columns we didn't specify (safety measure)
    )
    
    logger.info(f"Created preprocessor with {len(numerical_features)} numerical "
                f"and {len(categorical_features)} categorical features")
    
    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer,
                                          numerical_features: list,
                                          categorical_features: list) -> list:
    feature_names = []
    
    # Numerical features keep their names
    feature_names.extend(numerical_features)
    
    # Get categorical feature names from OneHotEncoder
    try:
        cat_transformer = preprocessor.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    except Exception as e:
        logger.warning(f"Could not extract categorical feature names: {e}")
        feature_names.extend([f"cat_{i}" for i in range(len(categorical_features))])
    
    return feature_names


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = TEST_SIZE,
               validation_size: float = VALIDATION_SIZE,
               random_state: int = RANDOM_STATE,
               stratify: bool = True) -> Tuple:
    """Split data into train/val/test sets - MUST happen BEFORE preprocessing to prevent data leakage"""
    logger.info(f"Splitting data with test_size={test_size}, validation_size={validation_size}")
    
    # If stratify=True, maintain class proportions in each split (important for imbalanced data)
    stratify_split = y if stratify else None
    
    # First split: separate out the test set (20% by default)
    # Test set should NEVER be touched during training/tuning
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size,  # 0.2 = 20% for test
        random_state=random_state,  # Fixed seed for reproducibility
        stratify=stratify_split  # Keep class balance
    )
    
    # Second split: divide remaining data into train and validation
    # Validation is used for hyperparameter tuning, train is used for fitting models
    stratify_split_val = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=validation_size,  # As fraction of temp (not original)
        random_state=random_state,
        stratify=stratify_split_val
    )
    
    # Log how many samples ended up in each set
    logger.info(f"Split sizes:")
    logger.info(f"  Train: {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"  Val:   {X_val.shape[0]:,} ({X_val.shape[0]/len(X)*100:.1f}%)")
    logger.info(f"  Test:  {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Verify class balance was maintained in each split
    logger.info(f"\nClass distribution in splits:")
    logger.info(f"  Train: {y_train.value_counts().to_dict()}")
    logger.info(f"  Val:   {y_val.value_counts().to_dict()}")
    logger.info(f"  Test:  {y_test.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                sampling_strategy: str = SMOTE_SAMPLING_STRATEGY,
                k_neighbors: int = SMOTE_K_NEIGHBORS,
                random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes - CRITICAL: Only apply to training data AFTER split!"""
    logger.info("Applying SMOTE to training data...")
    
    # Log original class counts
    original_counts = pd.Series(y_train).value_counts()
    logger.info(f"Original class distribution: {original_counts.to_dict()}")
    
    # SMOTE creates synthetic minority class samples by interpolating between existing samples
    # This helps models learn from balanced data instead of just predicting majority class
    smote = SMOTE(
        sampling_strategy=sampling_strategy,  # 'auto' = balance to 50/50
        k_neighbors=k_neighbors,  # Number of nearest neighbors to use for interpolation
        random_state=random_state,  # For reproducibility
        n_jobs=-1  # Use all CPU cores
    )
    
    # Generate synthetic samples (only for minority class)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Log new class counts
    new_counts = pd.Series(y_resampled).value_counts()
    logger.info(f"After SMOTE class distribution: {new_counts.to_dict()}")
    logger.info(f"SMOTE increased training size from {len(X_train):,} to {len(X_resampled):,}")
    
    # IMPORTANT: We NEVER apply SMOTE to validation or test sets
    # Those must remain as real, unmodified data to evaluate true performance
    
    return X_resampled, y_resampled


def preprocess_data(X: pd.DataFrame, y: pd.Series,
                    use_smote: bool = USE_SMOTE,
                    numerical_features: Optional[list] = None,
                    categorical_features: Optional[list] = None) -> dict:
    logger.info("=" * 80)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 80)
    
    # Determine features to use
    if numerical_features is None:
        numerical_features = [col for col in NUMERICAL_FEATURES if col in X.columns]
    if categorical_features is None:
        categorical_features = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    
    all_features = numerical_features + categorical_features
    logger.info(f"Using {len(all_features)} features: "
                f"{len(numerical_features)} numerical, {len(categorical_features)} categorical")
    
    # Filter X to only include selected features
    X_filtered = X[all_features].copy()
    
    # Step 1: Split data FIRST (crucial to avoid data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_filtered, y)
    
    # Step 2: Create and fit preprocessor on TRAINING data only
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    
    logger.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Step 3: Transform validation and test data using fitted preprocessor
    logger.info("Transforming validation and test data...")
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = get_feature_names_after_preprocessing(
        preprocessor, numerical_features, categorical_features
    )
    
    logger.info(f"After preprocessing: {X_train_processed.shape[1]} features")
    
    # Step 4: Apply SMOTE to TRAINING data only (AFTER split and preprocessing)
    if use_smote:
        X_train_processed, y_train = apply_smote(X_train_processed, y_train.values)
    else:
        logger.info("SMOTE disabled, using original training data")
    
    # Package results
    result = {
        'X_train': X_train_processed,
        'X_val': X_val_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    
    logger.info("=" * 80)
    logger.info("Preprocessing complete")
    logger.info(f"  Training set: {X_train_processed.shape}")
    logger.info(f"  Validation set: {X_val_processed.shape}")
    logger.info(f"  Test set: {X_test_processed.shape}")
    logger.info("=" * 80)
    
    return result


if __name__ == "__main__":
    from utils import setup_logging
    from data_loader import prepare_data
    
    setup_logging('INFO')
    X, y = prepare_data()
    processed_data = preprocess_data(X, y)
    
    print(f"\nProcessed shapes:")
    print(f"X_train: {processed_data['X_train'].shape}")
    print(f"X_val: {processed_data['X_val'].shape}")
    print(f"X_test: {processed_data['X_test'].shape}")
    print(f"\nFeature names ({len(processed_data['feature_names'])}):")
    print(processed_data['feature_names'][:10], "...")
