Heart Disease Risk Prediction 🫀

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reproducible, leakage-aware machine learning pipeline for predicting cardiovascular disease risk with a focus on correctness, explainability, and decision trade-offs — not complexity.

## Project Evolution

This project started as a university assignment focused on exploratory data analysis ([see initial notebook](notebooks/01_exploratory_analysis.ipynb)). 

The production pipeline was subsequently engineered to address real-world ML concerns: data leakage, class imbalance, reproducibility, and explainability.

See [TRANSFORMATION_SUMMARY.md](TRANSFORMATION_SUMMARY.md) for details on improvements made.

## Overview

This repository implements an **end-to-end ML workflow** on the Cardiovascular Diseases Risk Prediction dataset (≈309k samples, 18 input features + 1 target). The goal is to build a **trustworthy, interpretable system** designed with regulated/high-stakes contexts in mind.

### Key Principles

**No data leakage** — Train/validation/test splits before preprocessing  
**Meaningful baselines** — Dummy classifiers establish performance floor  
**Class imbalance handling** — SMOTE applied only to training data  
**Explainability** — Global & local SHAP interpretations  
**Reproducibility** — Fixed seeds, versioned artifacts, JSON snapshots  
**Threshold optimization** — Cost-aware decision point selection

## Quick Start

```bash
# Clone repository
git clone https://github.com/kacper-kowalski/Heart-Disease-AI.git
cd Heart-Disease-AI

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python src/main.py --tune-hyperparams

# View results in reports/
```

**Note:** If optional dependencies (e.g., XGBoost) are unavailable, the pipeline skips those models gracefully.

## Installation

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `pandas`, `numpy`, `scikit-learn`
- `xgboost` (optional)
- `shap` (explainability)
- `imbalanced-learn` (SMOTE)
- `matplotlib`, `seaborn` (visualization)

## Models & Techniques

| Component | Details |
|-----------|---------|
| **Baseline** | Dummy (most frequent, stratified) |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Resampling** | SMOTE (training only) |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| **Explainability** | SHAP (global + local), feature importance |
| **Optimization** | Threshold-cost analysis for operating point selection |

## Project Structure

```
Heart-Disease-AI/
├── src/
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration & hyperparameters
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py     # Scaling, encoding, SMOTE
│   ├── models.py            # Model definitions
│   ├── evaluation.py        # Metrics & threshold analysis
│   ├── explainability.py    # SHAP interpretations
│   └── utils.py             # Helper functions
├── data/
│   └── CVD_cleaned.csv      # ≈309k samples, 18 input features + 1 target
├── models/
│   ├── best_model.pkl       # Trained model (versioned per run)
│   └── preprocessor.pkl     # Fitted ColumnTransformer
├── reports/
│   ├── training_report_*.json      # Run metadata & hyperparameters
│   ├── model_comparison.csv        # Cross-model metrics
│   ├── threshold_analysis.csv      # Cost-based threshold evaluation
│   ├── error_analysis.csv          # Misclassification patterns
│   ├── eda/                        # Exploratory data analysis
│   └── explainability/             # SHAP plots, feature importance
├── notebooks/                      # Jupyter notebooks (optional)
├── tests/                          # Unit tests
├── requirements.txt
├── QUICKSTART.md            # Quick reference guide
└── README.md
```

## Methodology

### Data & Splitting Strategy

- **Target:** Binary cardiovascular disease indicator
- **Imbalance handling:** SMOTE applied to **training data only** (prevents leakage into validation/test)
- **Split ratio:** 80% train+validation / 20% test; then split remaining 80% into 68% train / 17% validation
- **Split timing:** Before any preprocessing (leakage prevention)

### Preprocessing Pipeline

| Step | Method | Fitted On |
|------|--------|-----------|
| Numerical scaling | StandardScaler | Training data only |
| Categorical encoding | One-hot encoding | Training data only |
| Implementation | ColumnTransformer | Training data only |

### Model Selection & Tuning

1. **Baseline models** establish performance floor
2. **Hyperparameter tuning** via cross-validation on training split
3. **Final evaluation** on held-out test set only
4. **No threshold assumptions** — operating point chosen via cost analysis

### Threshold Optimization

Rather than defaulting to 0.5 classification threshold, the pipeline evaluates operating points under configurable cost assumptions (e.g., penalty for false negatives vs. false positives). This reflects real-world trade-offs in clinical decision-making.

## Results

### 🔍 Model Performance (Default Threshold = 0.5)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|-------|---------|--------|--------|-----------|-----|
| **Logistic Regression** | 0.836 | 0.311 | 0.793 | 0.207 | 0.328 |
| **Random Forest** | 0.821 | 0.268 | 0.196 | 0.343 | 0.250 |
| **XGBoost** | 0.829 | 0.287 | 0.064 | 0.418 | 0.111 |
| **Dummy (Stratified)** | 0.504 | 0.081 | 0.508 | 0.082 | 0.141 |

### Interpretation

- **Severe class imbalance** (~8–9% positive class): Accuracy is misleading; focus on ROC-AUC and PR-AUC instead
- **Logistic Regression** achieves highest recall and PR-AUC, making it suitable as a **screening model** (identifies most positive cases)
- **Tree-based models** are more conservative at the default 0.5 threshold, trading recall for precision
- **ROC-AUC scores** (all >0.50) show models rank risk reasonably well, but **operating-point selection dominates real-world usefulness**

### Threshold Sensitivity

Performance changes substantially under alternative decision thresholds:
- The pipeline includes **cost-based threshold optimization** to explore false-positive vs. false-negative trade-offs
- See `threshold_analysis.csv` in reports for candidate operating points under different cost assumptions

### Explainability

- **Global SHAP:** Identifies consistently important cardiovascular risk factors across the dataset
- **Local SHAP:** Supports per-sample explanations for individual predictions
- **Stability analysis:** Cross-fold feature importance checks highlight brittle relationships vs. genuine signal

### Reproducibility Note

Exact metrics vary slightly with random seed and tuning flags. Full per-run results, plots, and confusion matrices are saved under `reports/`.

## Reproducibility

Runs are reproducible given the same environment (Python + pinned dependencies) and fixed seeds:

```bash
# Run with fixed seed (default)
python src/main.py --tune-hyperparams --seed 42

# All artifacts saved:
# - models/best_model.pkl
# - models/preprocessor.pkl
# - reports/training_report_*.json (hyperparams, metrics, metadata)
```

**Reproducibility features:**
- Fixed random seeds across all stages (train/test split, model initialization, SMOTE)
- Preprocessing and model objects persisted as pickled artifacts
- Complete run metadata (hyperparameters, thresholds, fold splits) stored as JSON

## Limitations & Future Work

- **SMOTE placement:** Currently applied once before inner CV; next iteration will move resampling inside CV folds using `imblearn.Pipeline`
- **Clinical interpretation:** Dataset represents self-reported and observational data — not suitable for clinical decision-making without expert validation
- **Fairness analysis:** No subgroup performance analysis or demographic parity checks yet
- **Multiclass extension:** Currently binary classification; future work may extend to disease severity levels

## Why This Project?

Most student ML projects optimize accuracy in a notebook. This project was built to see **what breaks ML systems in practice:**

Data leakage (train/test contamination, preprocessing order)  
Class imbalance (naive baselines, misleading metrics)  
Misinterpreted metrics (accuracy ≠ usefulness)  
Unstable explanations (overfitting to noise)  

That focus on **correctness over complexity** guided every design decision.

## Author

**Kacper Kowalski**  
BSc Data Science & Artificial Intelligence  
[GitHub](https://github.com/kacper-kowalski) | [LinkedIn](https://linkedin.com/in/kacper-kowalski)
