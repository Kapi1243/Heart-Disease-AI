# Quick Start Guide - Heart Disease Prediction

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements_new.txt
```

### Step 2: Run the Pipeline
```bash
cd src
python main.py
```

That's it! The pipeline will:
1. ✅ Load and validate data
2. ✅ Perform EDA (15+ visualizations)
3. ✅ Preprocess features (no data leakage!)
4. ✅ Train 7+ models (baselines, ML, ensembles)
5. ✅ Evaluate with 6 metrics
6. ✅ Generate SHAP explanations
7. ✅ Create comprehensive reports

---

## ⚙️ Command Options

### Fast Mode (Skip EDA)
```bash
python main.py --skip-eda
```

### Best Performance (Hyperparameter Tuning)
```bash
python main.py --tune-hyperparams
# Warning: Takes 10-30 minutes depending on hardware
```

### Custom Logging
```bash
python main.py --log-level DEBUG
```

---

## 📊 Where to Find Results

After running the pipeline, check:

### Models
- `models/best_model.pkl` - Top performing model
- `models/preprocessor.pkl` - Fitted preprocessor
- `models/all_models.pkl` - All trained models

### Reports
- `reports/model_comparison.csv` - Performance metrics table
- `reports/roc_curves.png` - ROC curve comparison
- `reports/confusion_matrices.png` - All confusion matrices
- `reports/training_report_*.json` - Complete pipeline log

### EDA
- `reports/eda/target_distribution.png` - Class balance
- `reports/eda/correlation_matrix.png` - Feature correlations
- `reports/eda/numerical_distributions.png` - Feature distributions
- `reports/eda/statistics_summary.csv` - Descriptive statistics

### Explainability
- `reports/explainability/*_shap_summary.png` - SHAP importance
- `reports/explainability/*_feature_importance.png` - Feature rankings
- `reports/explainability/*_shap_waterfall_*.png` - Individual predictions

---

## 🔧 Running Individual Components

### Just EDA
```bash
cd src
python eda.py
```

### Just Train Models
```bash
python models.py
```

### Just Evaluate
```bash
python evaluation.py
```

---

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Make sure you're in the `src/` directory
```bash
cd src
python main.py
```

### Issue: Missing data file
**Solution**: Ensure `CVD_cleaned.csv` is in the `data/` folder
```bash
# Check if file exists
ls ../data/CVD_cleaned.csv
```

### Issue: Out of memory
**Solution**: Reduce SHAP sample size in `config.py`
```python
SHAP_SAMPLE_SIZE = 500  # Instead of 1000
```

---

## 📈 Expected Runtime

| Phase | Default | With Tuning |
|-------|---------|-------------|
| Data Loading | 5-10s | 5-10s |
| EDA | 30-60s | 30-60s |
| Preprocessing | 10-20s | 10-20s |
| Training | 2-5 min | 20-40 min |
| Evaluation | 30-60s | 30-60s |
| Explainability | 2-5 min | 2-5 min |
| **Total** | **~8 min** | **~45 min** |

---

## 💡 Pro Tips

1. **First Run**: Use default settings to see the full pipeline
   ```bash
   python main.py
   ```

2. **Quick Iterations**: Skip EDA and explainability
   ```bash
   python main.py --skip-eda --skip-explainability
   ```

3. **Best Results**: Enable tuning for production models
   ```bash
   python main.py --tune-hyperparams
   ```

4. **Testing**: Run unit tests before deployment
   ```bash
   cd tests
   pytest test_preprocessing.py -v
   ```

---

## 📚 Next Steps

### For Job Applications
1. ✅ Run the full pipeline with tuning
2. ✅ Review generated reports and visualizations
3. ✅ Update `README_new.md` with your actual results
4. ✅ Add your personal information (name, email, GitHub)
5. ✅ Create a GitHub repository
6. ✅ Add to your portfolio/resume

### For Further Development
1. Add more models (Neural Networks, SVM)
2. Implement AutoML (PyCaret, TPOT)
3. Create interactive dashboard (Streamlit)
4. Deploy as API (FastAPI)
5. Add Docker containerization
6. Set up CI/CD pipeline

---

## 🎯 Key Differentiators

What makes this project stand out:

✅ **No Data Leakage** - SMOTE applied correctly after split
✅ **Baseline Comparisons** - Shows ML models add value
✅ **Comprehensive EDA** - 15+ professional visualizations
✅ **Model Explainability** - SHAP analysis included
✅ **Error Analysis** - Detailed misclassification study
✅ **Production Code** - Modular, tested, documented
✅ **Reproducible** - Fixed random seeds throughout
✅ **Industry Standards** - Follows ML best practices

---

## 📞 Support

If you encounter issues:

1. Check the logs in the terminal output
2. Review `reports/training_report_*.json`
3. Verify all dependencies are installed
4. Ensure Python 3.8+ is being used

---

**Happy modeling! 🎉**
