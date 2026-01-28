import os
from pathlib import Path

# ========================
# Project Paths
# ========================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data files
DATA_FILE = DATA_DIR / "CVD_cleaned.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.pkl"

# Model files
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
ALL_MODELS_FILE = MODELS_DIR / "all_models.pkl"

# Report files
EDA_REPORT = REPORTS_DIR / "eda_report.html"
MODEL_COMPARISON_PLOT = REPORTS_DIR / "model_comparison.png"
CONFUSION_MATRICES_PLOT = REPORTS_DIR / "confusion_matrices.png"
ROC_CURVES_PLOT = REPORTS_DIR / "roc_curves.png"
FEATURE_IMPORTANCE_PLOT = REPORTS_DIR / "feature_importance.png"
SHAP_SUMMARY_PLOT = REPORTS_DIR / "shap_summary.png"
ERROR_ANALYSIS_PLOT = REPORTS_DIR / "error_analysis.png"

# ========================
# Data Configuration
# ========================
# Target variable
TARGET_COLUMN = "Heart_Disease"

# Feature categories
NUMERICAL_FEATURES = [
    "BMI",
    "Height_(cm)",
    "Weight_(kg)",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption"
]

CATEGORICAL_FEATURES = [
    "General_Health",
    "Checkup",
    "Exercise",
    "Skin_Cancer",
    "Other_Cancer",
    "Depression",
    "Diabetes",
    "Arthritis",
    "Sex",
    "Age_Category",
    "Smoking_History"
]

# Feature subset (original 5 features for comparison)
ORIGINAL_FEATURES = {
    'numerical': ['BMI'],
    'categorical': ['Age_Category', 'Smoking_History']
}

# ========================
# Model Configuration
# ========================
# Random seed for reproducibility
RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15  # From training set

# Cross-validation
CV_FOLDS = 5

# Class balancing
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 'auto'
SMOTE_K_NEIGHBORS = 5

# ========================
# Model Hyperparameters
# ========================

# Logistic Regression
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

LOGISTIC_REGRESSION_GRID = {
    'C': [0.1, 1.0, 10.0],  # Already found 0.1 is best, keep nearby values
    'penalty': ['l2'],
    'solver': ['lbfgs']  # Faster and already optimal
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

RANDOM_FOREST_GRID = {
    'n_estimators': [100, 200],  # 100+ usually sufficient, 200 for extra boost
    'max_depth': [20, 30],  # Deep trees for complex patterns
    'min_samples_split': [5],  # Prevent overfitting on large dataset
    'min_samples_leaf': [2],  # Small leaf = better fit
    'max_features': ['sqrt']  # Standard best practice
}

# XGBoost
XGBOOST_PARAMS = {
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

XGBOOST_GRID = {
    'n_estimators': [100, 200],  # More trees = better performance
    'max_depth': [5, 7],  # Balanced depth for generalization
    'learning_rate': [0.05, 0.1],  # Mid-range for speed + accuracy
    'subsample': [0.8],  # Standard value prevents overfitting
    'colsample_bytree': [0.8],  # Standard value
    'min_child_weight': [3]  # Regularization for large dataset
}

# Ensemble methods
VOTING_CLASSIFIER_VOTING = 'soft'  # 'hard' or 'soft'
STACKING_FINAL_ESTIMATOR = 'logistic'  # 'logistic' or 'random_forest'

# ========================
# Training Configuration
# ========================
# GridSearchCV settings
GRID_SEARCH_CV = 3  # Inner CV folds for hyperparameter tuning
GRID_SEARCH_SCORING = 'roc_auc'
GRID_SEARCH_N_JOBS = -1
GRID_SEARCH_VERBOSE = 3  # 0=silent, 1=summary, 2=one line per fit, 3=detailed progress

# Model selection
MODELS_TO_TRAIN = [
    'dummy_stratified',
    'dummy_most_frequent',
    'logistic_regression',
    'random_forest',
    'xgboost',
    'voting',
    'stacking'
]

# ========================
# Evaluation Configuration
# ========================
# Metrics to compute
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision'
]

# Classification thresholds to evaluate
THRESHOLDS_TO_EVALUATE = [0.3, 0.4, 0.5, 0.6, 0.7]

# ========================
# Explainability Configuration
# ========================
# SHAP settings
SHAP_MAX_DISPLAY = 20
SHAP_SAMPLE_SIZE = 1000  # Sample size for SHAP calculation (performance)

# Feature importance settings
FEATURE_IMPORTANCE_TOP_N = 15

# ========================
# Visualization Configuration
# ========================
# Plot settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Color palettes
COLOR_PALETTE = 'Set2'
CONFUSION_MATRIX_CMAP = 'Blues'

# ========================
# Logging Configuration
# ========================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ========================
# Feature Engineering
# ========================
# Whether to create interaction features
CREATE_INTERACTIONS = False
INTERACTION_DEGREE = 2

# Whether to create polynomial features
CREATE_POLYNOMIALS = False
POLYNOMIAL_DEGREE = 2

# ========================
# Clinical Configuration
# ========================
# Cost-benefit weights 
COST_FALSE_POSITIVE = 1  # Cost of unnecessary further testing
COST_FALSE_NEGATIVE = 10  # Cost of missing a heart disease case

# Decision threshold based on cost
DEFAULT_THRESHOLD = 0.5
