import logging
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import modules
from config import (
    PROJECT_ROOT,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    BEST_MODEL_FILE,
    PREPROCESSOR_FILE,
    ALL_MODELS_FILE
)
from utils import (
    setup_logging,
    set_seeds,
    save_pickle,
    save_json,
    get_timestamp,
    create_results_summary
)
from data_loader import prepare_data
from eda import perform_eda
from preprocessing import preprocess_data
from models import train_all_models
from evaluation import (
    evaluate_all_models,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_model_comparison,
    analyze_threshold_impact,
    analyze_errors
)
from explainability import explain_model


logger = logging.getLogger('heart_disease_prediction')


def run_eda_phase(X, y, skip_eda=False):
    """Run exploratory data analysis phase."""
    if skip_eda:
        logger.info("Skipping EDA phase")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Exploratory Data Analysis")
    logger.info("=" * 80)
    
    perform_eda(X, y)


def run_preprocessing_phase(X, y):
    """Run data preprocessing phase."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Data Preprocessing")
    logger.info("=" * 80)
    
    processed_data = preprocess_data(X, y)
    
    # Save preprocessor
    save_pickle(processed_data['preprocessor'], PREPROCESSOR_FILE)
    
    return processed_data


def run_training_phase(processed_data, tune_hyperparams=False):
    """Run model training phase."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: Model Training")
    logger.info("=" * 80)
    
    training_results = train_all_models(
        processed_data['X_train'],
        processed_data['y_train'],
        include_baselines=True,
        tune_hyperparams=tune_hyperparams,
        include_ensembles=False
    )
    
    # Save all models
    save_pickle(training_results['models'], ALL_MODELS_FILE)
    
    return training_results


def run_evaluation_phase(training_results, processed_data):
    """Run model evaluation phase."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: Model Evaluation")
    logger.info("=" * 80)
    
    # Evaluate all models
    results_df = evaluate_all_models(
        training_results['models'],
        processed_data['X_test'],
        processed_data['y_test'],
        processed_data['X_val'],
        processed_data['y_val']
    )
    
    # Save results
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv")
    
    # Create visualizations
    logger.info("\nGenerating evaluation visualizations...")
    
    # Filter out baseline models for some visualizations
    ml_models = {k: v for k, v in training_results['models'].items() 
                 if not k.startswith('dummy')}
    
    # Confusion matrices
    plot_confusion_matrices(
        ml_models,
        processed_data['X_test'],
        processed_data['y_test'],
        save_path=REPORTS_DIR / "confusion_matrices.png"
    )
    
    # ROC curves
    plot_roc_curves(
        ml_models,
        processed_data['X_test'],
        processed_data['y_test'],
        save_path=REPORTS_DIR / "roc_curves.png"
    )
    
    # PR curves
    plot_precision_recall_curves(
        ml_models,
        processed_data['X_test'],
        processed_data['y_test'],
        save_path=REPORTS_DIR / "pr_curves.png"
    )
    
    # Model comparison
    plot_model_comparison(
        results_df,
        metric='roc_auc',
        save_path=REPORTS_DIR / "model_comparison_roc_auc.png"
    )
    
    plot_model_comparison(
        results_df,
        metric='f1',
        save_path=REPORTS_DIR / "model_comparison_f1.png"
    )
    
    # Find best model
    best_model_name = results_df['roc_auc'].idxmax()
    best_model = training_results['models'][best_model_name]
    
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"  ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
    logger.info(f"  F1 Score: {results_df.loc[best_model_name, 'f1']:.4f}")
    
    # Save best model
    save_pickle(best_model, BEST_MODEL_FILE)
    
    return results_df, best_model_name, best_model


def run_error_analysis_phase(best_model, processed_data):
    """Run error analysis phase."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: Error Analysis")
    logger.info("=" * 80)
    
    # Analyze errors
    error_df = analyze_errors(
        best_model,
        processed_data['X_test'],
        processed_data['y_test'],
        feature_names=processed_data['feature_names'],
        save_path=REPORTS_DIR / "error_analysis.csv"
    )
    
    # Threshold analysis
    threshold_df = analyze_threshold_impact(
        best_model,
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    
    return error_df, threshold_df


def run_explainability_phase(best_model, best_model_name, processed_data):
    """Run model explainability phase."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: Model Explainability")
    logger.info("=" * 80)
    
    # Explain best model
    explanation_results = explain_model(
        best_model,
        processed_data['X_test'],
        processed_data['feature_names'],
        model_name=best_model_name
    )
    
    return explanation_results


def generate_final_report(results_df, training_results, processed_data,
                         best_model_name, timestamp):
    """Generate final summary report."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 7: Final Report Generation")
    logger.info("=" * 80)
    
    report = {
        'timestamp': timestamp,
        'dataset_info': {
            'total_samples': len(processed_data['y_train']) + 
                           len(processed_data['y_val']) + 
                           len(processed_data['y_test']),
            'train_samples': len(processed_data['y_train']),
            'val_samples': len(processed_data['y_val']),
            'test_samples': len(processed_data['y_test']),
            'n_features': len(processed_data['feature_names']),
            'numerical_features': processed_data['numerical_features'],
            'categorical_features': processed_data['categorical_features']
        },
        'models_trained': list(training_results['models'].keys()),
        'training_times': training_results['training_times'],
        'best_model': best_model_name,
        'best_model_metrics': results_df.loc[best_model_name].to_dict(),
        'all_results': results_df.to_dict(),
        'config': {
            'random_state': RANDOM_STATE,
            'use_smote': True,
            'hyperparameter_tuning': False
        }
    }
    
    # Save report
    report_path = REPORTS_DIR / f"training_report_{timestamp}.json"
    save_json(report, report_path)
    
    logger.info(f"Final report saved to {report_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total samples: {report['dataset_info']['total_samples']:,}")
    logger.info(f"Features used: {report['dataset_info']['n_features']}")
    logger.info(f"Models trained: {len(report['models_trained'])}")
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"  Accuracy:  {results_df.loc[best_model_name, 'accuracy']:.4f}")
    logger.info(f"  Precision: {results_df.loc[best_model_name, 'precision']:.4f}")
    logger.info(f"  Recall:    {results_df.loc[best_model_name, 'recall']:.4f}")
    logger.info(f"  F1 Score:  {results_df.loc[best_model_name, 'f1']:.4f}")
    logger.info(f"  ROC-AUC:   {results_df.loc[best_model_name, 'roc_auc']:.4f}")
    logger.info("=" * 80)
    
    return report


def main(args):
    """Main pipeline execution."""
    # Setup
    timestamp = get_timestamp()
    setup_logging(args.log_level)
    set_seeds(RANDOM_STATE)
    
    logger.info("=" * 80)
    logger.info("HEART DISEASE PREDICTION - ML PIPELINE")
    logger.info(f"Started at: {timestamp}")
    logger.info("=" * 80)
    
    try:
        # Phase 1: Data Loading
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 0: Data Loading")
        logger.info("=" * 80)
        X, y = prepare_data()
        
        # Phase 1: EDA
        run_eda_phase(X, y, skip_eda=args.skip_eda)
        
        # Phase 2: Preprocessing
        processed_data = run_preprocessing_phase(X, y)
        
        # Phase 3: Training
        # Auto-enable tuning for better accuracy if not explicitly disabled
        tune = args.tune_hyperparams or args.auto_tune
        training_results = run_training_phase(
            processed_data,
            tune_hyperparams=tune
        )
        
        # Phase 4: Evaluation
        results_df, best_model_name, best_model = run_evaluation_phase(
            training_results,
            processed_data
        )
        
        # Phase 5: Error Analysis
        if not args.skip_error_analysis:
            error_df, threshold_df = run_error_analysis_phase(
                best_model,
                processed_data
            )
        
        # Phase 6: Explainability
        if not args.skip_explainability:
            explanation_results = run_explainability_phase(
                best_model,
                best_model_name,
                processed_data
            )
        
        # Phase 7: Final Report
        report = generate_final_report(
            results_df,
            training_results,
            processed_data,
            best_model_name,
            timestamp
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {MODELS_DIR}")
        logger.info(f"Reports saved to: {REPORTS_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heart Disease Prediction ML Pipeline"
    )
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip exploratory data analysis'
    )
    parser.add_argument(
        '--tune-hyperparams',
        action='store_true',
        help='Enable hyperparameter tuning (slower but better results)'
    )
    parser.add_argument(
        '--auto-tune',
        action='store_true',
        help='Auto-enable tuning for optimal accuracy'
    )
    parser.add_argument(
        '--skip-error-analysis',
        action='store_true',
        help='Skip error analysis phase'
    )
    parser.add_argument(
        '--skip-explainability',
        action='store_true',
        help='Skip model explainability phase'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    main(args)
