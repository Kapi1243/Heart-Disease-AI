# Standard imports
import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Any

# scikit-learn model imports
from sklearn.dummy import DummyClassifier  # Baseline models (random guessing)
from sklearn.linear_model import LogisticRegression  # Simple linear classifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier  # Tree-based ensembles
from sklearn.model_selection import GridSearchCV, cross_validate  # Hyperparameter tuning
from xgboost import XGBClassifier  # Gradient boosting (usually best performer)

# Import all our configuration settings
from config import (
    RANDOM_STATE,
    CV_FOLDS,
    GRID_SEARCH_CV,
    GRID_SEARCH_SCORING,
    GRID_SEARCH_N_JOBS,
    GRID_SEARCH_VERBOSE,
    LOGISTIC_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_GRID,
    RANDOM_FOREST_PARAMS,
    RANDOM_FOREST_GRID,
    XGBOOST_PARAMS,
    XGBOOST_GRID,
    VOTING_CLASSIFIER_VOTING,
    EVALUATION_METRICS
)
from utils import save_pickle

# Get logger for this module
logger = logging.getLogger('heart_disease_prediction')


def create_baseline_models() -> Dict[str, Any]:
    """Create simple baseline models to establish performance floor"""
    # These are "dumb" models that help us know if our ML models are actually learning
    # If our ML models don't beat these, something is wrong!
    models = {
        # Predicts classes randomly based on training set class distribution (e.g., 80% no disease, 20% disease)
        'dummy_stratified': DummyClassifier(
            strategy='stratified',
            random_state=RANDOM_STATE
        ),
        # Always predicts the most common class (majority class baseline)
        'dummy_most_frequent': DummyClassifier(
            strategy='most_frequent',
            random_state=RANDOM_STATE
        ),
        # Predicts classes completely randomly with 50/50 probability
        'dummy_uniform': DummyClassifier(
            strategy='uniform',
            random_state=RANDOM_STATE
        )
    }
    
    logger.info(f"Created {len(models)} baseline models")
    return models


def create_ml_models() -> Dict[str, Any]:
    """Create ML models with default parameters (before tuning)"""
    models = {
        # Simple but effective: learns linear decision boundary
        # Fast to train, interpretable coefficients
        'logistic_regression': LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        
        # Ensemble of decision trees: usually very strong performer
        # Can capture non-linear patterns, resistant to overfitting
        'random_forest': RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        
        # Gradient boosting: often the best for tabular data
        # Builds trees sequentially, each fixing errors of previous
        'xgboost': XGBClassifier(**XGBOOST_PARAMS)
    }
    
    logger.info(f"Created {len(models)} ML models")
    return models


def tune_model(model, param_grid: Dict, X_train: np.ndarray, y_train: np.ndarray,
               cv: int = GRID_SEARCH_CV,
               scoring: str = GRID_SEARCH_SCORING) -> Tuple[Any, Dict]:
    """Find best hyperparameters using grid search with cross-validation"""
    logger.info(f"Tuning {model.__class__.__name__} with GridSearchCV...")
    # Calculate total combinations we'll test (e.g., 3 C values × 2 solvers = 6 combinations)
    logger.info(f"  Parameter grid size: {np.prod([len(v) for v in param_grid.values()])}")
    
    start_time = time.time()  # Track how long tuning takes
    
    # GridSearchCV: tries every combination of parameters, uses cross-validation to score each
    grid_search = GridSearchCV(
        estimator=model,  # The model to tune
        param_grid=param_grid,  # Dictionary of parameters to try
        cv=cv,  # Number of cross-validation folds (default: 3)
        scoring=scoring,  # Metric to optimize (default: 'roc_auc')
        n_jobs=GRID_SEARCH_N_JOBS,  # Use all CPU cores (-1)
        verbose=GRID_SEARCH_VERBOSE,  # How much progress info to print (0-3)
        return_train_score=True  # Also track training scores (helps detect overfitting)
    )
    
    # This is where the magic happens - trains cv × grid_size models
    # e.g., 3 CV folds × 288 param combos = 864 model fits for RandomForest
    grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    # Log the best combination found
    logger.info(f"  Best score: {grid_search.best_score_:.4f}")  # Best CV score achieved
    logger.info(f"  Best params: {grid_search.best_params_}")  # Parameters that got that score
    logger.info(f"  Time elapsed: {elapsed:.2f}s")
    
    # Return the model retrained with best parameters on full training set
    return grid_search.best_estimator_, grid_search.best_params_


def get_tuned_models(X_train: np.ndarray, y_train: np.ndarray,
                     tune_hyperparams: bool = True) -> Dict[str, Any]:
    """Get ML models - either tuned via grid search or with default parameters"""
    if not tune_hyperparams:
        logger.info("Hyperparameter tuning disabled, using default parameters")
        return create_ml_models()
    
    logger.info("=" * 80)
    logger.info("Hyperparameter Tuning")
    logger.info("=" * 80)
    
    tuned_models = {}
    
    # Tune Logistic Regression (fast - usually 10-30 seconds)
    lr_model, lr_params = tune_model(
        LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        LOGISTIC_REGRESSION_GRID,  # Parameter combinations from config.py
        X_train, y_train
    )
    tuned_models['logistic_regression'] = lr_model
    
    # Tune Random Forest (slower - can take 5-20 minutes depending on grid size)
    rf_model, rf_params = tune_model(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        RANDOM_FOREST_GRID,
        X_train, y_train
    )
    tuned_models['random_forest'] = rf_model
    
    # Tune XGBoost
    # Tune XGBoost (skip if compatibility issues)
    try:
        xgb_model, xgb_params = tune_model(
            XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'),
            XGBOOST_GRID,
            X_train, y_train
        )
        tuned_models['xgboost'] = xgb_model
    except (AttributeError, TypeError) as e:
        logger.warning(f"Skipping XGBoost tuning due to compatibility issue: {e}")
        logger.info("Using default XGBoost parameters instead")
        tuned_models['xgboost'] = XGBClassifier(**XGBOOST_PARAMS)
        tuned_models['xgboost'].fit(X_train, y_train)
    
    logger.info("=" * 80)
    
    return tuned_models


def create_ensemble_models(base_models: Dict[str, Any]) -> Dict[str, Any]:
    ensemble_models = {}
    
    # Create list of estimators for ensembles
    estimators = [
        ('lr', base_models.get('logistic_regression')),
        ('rf', base_models.get('random_forest')),
        ('xgb', base_models.get('xgboost'))
    ]
    
    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=VOTING_CLASSIFIER_VOTING,
        n_jobs=-1
    )
    ensemble_models['voting'] = voting_clf
    
    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1
    )
    ensemble_models['stacking'] = stacking_clf
    
    logger.info(f"Created {len(ensemble_models)} ensemble models")
    
    return ensemble_models


def train_model(model, X_train: np.ndarray, y_train: np.ndarray,
                model_name: str = "Model") -> Tuple[Any, float]:
    logger.info(f"Training {model_name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"  Training time: {training_time:.2f}s")
    
    return model, training_time


def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                         cv: int = CV_FOLDS,
                         scoring: Optional[List[str]] = None) -> Dict:
    if scoring is None:
        scoring = EVALUATION_METRICS
    
    logger.info(f"Performing {cv}-fold cross-validation...")
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results_summary = {}
    for metric in scoring:
        test_key = f'test_{metric}'
        if test_key in cv_results:
            mean_score = cv_results[test_key].mean()
            std_score = cv_results[test_key].std()
            results_summary[metric] = {
                'mean': mean_score,
                'std': std_score,
                'scores': cv_results[test_key]
            }
            logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return results_summary


def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                     include_baselines: bool = True,
                     tune_hyperparams: bool = False,
                     include_ensembles: bool = True) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("Training All Models")
    logger.info("=" * 80)
    
    all_models = {}
    training_times = {}
    
    # Train baseline models
    if include_baselines:
        logger.info("\n--- Baseline Models ---")
        baseline_models = create_baseline_models()
        for name, model in baseline_models.items():
            trained_model, train_time = train_model(model, X_train, y_train, name)
            all_models[name] = trained_model
            training_times[name] = train_time
    
    # Get (tuned) ML models
    logger.info("\n--- Machine Learning Models ---")
    ml_models = get_tuned_models(X_train, y_train, tune_hyperparams)
    
    # Train ML models
    for name, model in ml_models.items():
        if tune_hyperparams:
            # Model is already trained during tuning
            all_models[name] = model
            training_times[name] = 0.0  # Time was logged during tuning
        else:
            trained_model, train_time = train_model(model, X_train, y_train, name)
            all_models[name] = trained_model
            training_times[name] = train_time
    
    # Train ensemble models
    if include_ensembles and len(ml_models) >= 2:
        logger.info("\n--- Ensemble Models ---")
        ensemble_models = create_ensemble_models(ml_models)
        for name, model in ensemble_models.items():
            trained_model, train_time = train_model(model, X_train, y_train, name)
            all_models[name] = trained_model
            training_times[name] = train_time
    
    logger.info("=" * 80)
    logger.info(f"Total models trained: {len(all_models)}")
    logger.info("=" * 80)
    
    return {
        'models': all_models,
        'training_times': training_times
    }


if __name__ == "__main__":
    from utils import setup_logging, set_seeds
    from data_loader import prepare_data
    from preprocessing import preprocess_data
    
    setup_logging('INFO')
    set_seeds(RANDOM_STATE)
    
    # Prepare data
    X, y = prepare_data()
    processed_data = preprocess_data(X, y)
    
    # Train models
    results = train_all_models(
        processed_data['X_train'],
        processed_data['y_train'],
        include_baselines=True,
        tune_hyperparams=False,  # Set to True to enable tuning
        include_ensembles=True
    )
    
    print(f"\nTrained {len(results['models'])} models:")
    for name in results['models'].keys():
        print(f"  - {name}")
