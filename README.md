# 🫀 Heart Disease Risk Prediction

A complete end-to-end machine learning project for predicting the risk of cardiovascular diseases. It includes preprocessing, model training using XGBoost, evaluation, and a simple Streamlit app for interaction.

---

## 📁 Project Structure

```
CodeClause_internship_heart_disease_risk_prediction/
│
├── app/
│   └── app.py                                      # Streamlit app
│
├── dataset/
│   └── heart_train.csv                             # Main dataset
│
├── model/
│   └── xgb_heart_model.pkl                         # Trained model pipeline
│
├── notebooks/
│   ├── EDA.ipynb                                   # EDA and feature importance
│   ├── model_dev.ipynb                             # Model development
│   └── voting_classification_dev.ipynb             # voting classification model development
│
├── snapshots/
│   ├── classification_report.png                   # Classification Report plot
│   ├── feature_importance.png                      # Feature importance plot
│   └── web_app_preview.png                         # Web App preview
│
├── README.md                                       # Project overview
└── requirements.txt                                # Required libraries
```

---

## 🧠 Problem Statement

Cardiovascular diseases are one of the top causes of death globally. The objective is to develop a machine learning model that predicts the likelihood of a person having heart disease using basic medical data.

---

## 📊 Dataset Overview

- **Source:** [Kaggle – Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Rows:** 70,000
- **Target Variable:** `cardio` (0 = No Disease, 1 = Disease)
- **Key Features:** age, height, weight, blood pressure, cholesterol, glucose, smoke, alcohol, physical activity

---

## ✅ Project Workflow

1. **EDA**
   - Checked class distribution
   - Boxplot and Histogram analysis
   - Identified skewness and outliers

2. **Preprocessing**
   - Outlier removal and clipping
   - Applied log transformation on skewed features
   - Scaling with `StandardScaler`

3. **Model Pipeline**
   - `SelectKBest` for feature selection
   - `PCA` for dimensionality reduction
   - `XGBClassifier` as the base model
   - Used `GridSearchCV` for hyperparameter tuning

4. **Evaluation**
   - Accuracy: ~72%
   - Balanced Precision/Recall for both classes
   - Confusion matrix and classification report used

5. **Feature Importance**
   - Visualized using bar plot (XGBoost built-in importances)

6. **Deployment**
   - Streamlit UI to take user input
   - Model predicts risk and shows probability

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
streamlit
```

Install with:

```bash
pip install -r requirements.txt
```

---

## ▶ How to Run

1. Make sure the trained pipeline is saved as `xgb_pipeline.pkl`
2. Start the Streamlit app:

```bash
streamlit run app/app.py
```

---

## 👨‍💻 Author

- **Sanjay Kumar**
- [GitHub](https://github.com/insanjay)
- [LinkedIn](https://www.linkedin.com/in/insanjay)

