# Heart-Disease-AI
Heart Disease prediction model made for a university project

# 🫀 Heart Disease Prediction using Machine Learning

A machine learning pipeline that predicts the risk of heart attacks from health and lifestyle data,
achieving up to 94.6% ROC-AUC using Random Forests.
Built as a part of a university project to demonstrate ML, data preprocessing, and evaluation best practices

This project predicts the likelihood of a heart attack using health and lifestyle factors. It uses:
- Logistic Regression
- Random Forest
- XGBoost
- SMOTE for class balancing

## 📊 Dataset
- Source: `heart_2022.csv` ([insert source if public](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset))
- Target: `HadHeartAttack`

## 🛠 Features Used
- Age Category
- BMI
- Physical Health (days)
- Sleep Hours
- Smoker Status

## 🔍 Techniques
- Preprocessing with `ColumnTransformer`
- One-hot encoding
- Imputation for missing values
- SMOTE for imbalanced data
- Model evaluation with precision, recall, F1, ROC-AUC

## 🧪 Results

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Random Forest        | 88.2%    | 89.9%     | 86.1%  | 87.9%    | 94.6%   |
| XGBoost              | 81.0%    | 80.4%     | 81.9%  | 81.2%    | 90.6%   |
| Logistic Regression  | 71.4%    | 68.8%     | 78.3%  | 73.2%    | 78.3%   |

## 📈 Visualizations

Confusion matrices and model comparison plots included.

## 🚀 How to Run
1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
