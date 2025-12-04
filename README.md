# 🧠 Stroke Prediction Project

---

## 🔑 Quick Summary (for recruiters/companies)
This project demonstrates:
- **End‑to‑end ML pipeline**: cleaning → feature engineering → model training → evaluation → tuning.  
- **Handling imbalanced healthcare data** with SMOTE and weighted learning.  
- **Model comparison** across multiple algorithms.  
- **Deployment readiness** with processed dataset and tuned model.  

👉 In short: *It’s a practical stroke risk prediction system showing my ability to handle imbalanced medical datasets, apply advanced ML techniques, and deliver clinically relevant insights.*

---


## 📌 Project Overview
This project builds a **machine learning pipeline to predict stroke risk** using the Kaggle `stroke-data.csv` dataset.  
Because stroke cases are rare (~4.87% of the dataset), the workflow emphasizes **handling class imbalance** and evaluating models with metrics that matter in healthcare: **Recall** (catching high‑risk patients) and **ROC‑AUC** (overall discrimination ability).

The goal is to create a reliable tool for **early stroke risk detection** that balances predictive performance with clinical usefulness.

---

## ⚙️ Workflow Summary
1. **Data Exploration**
   - Load dataset and inspect structure (`.info()`, `.describe()`).
   - Visualize stroke class distribution with seaborn.
   - Confirm imbalance (~5% stroke cases).

2. **Data Cleaning**
   - Impute missing BMI values with median (preserve all 5110 rows).
   - Drop irrelevant `id` column.

3. **Feature Engineering**
   - Encode categorical variables:
     - Label encode binary features (`gender`, `ever_married`, `Residence_type`).
     - One‑hot encode multi‑category features (`work_type`, `smoking_status`).
   - Add interaction terms:
     - `age_hypertension` = age × hypertension
     - `age_heart_disease` = age × heart_disease
   - Save processed dataset (`processed_stroke_data.csv`).

4. **Correlation Analysis**
   - Compute correlations with stroke outcome.
   - Visualize top predictors in bar chart (`stroke_correlations.png`).

5. **Train/Test Split**
   - Stratified split (80/20) to preserve class distribution.
   - Apply **SMOTE** to oversample minority class in training set.

6. **Scaling**
   - Standardize continuous variables (`age`, `avg_glucose_level`, `bmi`, interaction terms) using `StandardScaler`.

7. **Model Training**
   - Train multiple models:
     - Logistic Regression (balanced class weights)
     - Support Vector Classifier (RBF kernel)
     - K‑Nearest Neighbors
     - Random Forest (balanced class weights)
     - XGBoost (with `scale_pos_weight`)
   - Evaluate with **Accuracy, Precision, Recall, ROC‑AUC**.

8. **Hyperparameter Tuning**
   - Use **GridSearchCV** to optimize XGBoost parameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`).
   - Select best model based on ROC‑AUC.

9. **Feature Importance**
   - Plot feature importance for best XGBoost model (`feature_importance.png`).

---

## 🧪 Models Compared
| Model | Key Notes |
|-------|-----------|
| Logistic Regression | Balanced weights, interpretable baseline |
| SVC (RBF kernel) | Non‑linear decision boundary |
| KNN | Simple distance‑based classifier |
| Random Forest | Ensemble with balanced weights |
| XGBoost | Gradient boosting, tuned with GridSearchCV |

---

## 📊 Evaluation Metrics
- **Accuracy**: overall correctness (less reliable with imbalance).  
- **Precision**: proportion of predicted strokes that are correct.  
- **Recall**: proportion of actual strokes correctly identified (critical in healthcare).  
- **ROC‑AUC**: discrimination ability across thresholds.  
- **Confusion Matrix**: visualize true vs false predictions.  

---

## 🚀 Why This Approach?
- Stroke cases are rare → accuracy alone is misleading.  
- **Recall** ensures high‑risk patients are flagged.  
- **SMOTE** balances training data for fairer learning.  
- **XGBoost with tuning** provides strong performance and feature importance insights.  
- **Feature engineering** (interaction terms) captures clinical relationships.  

---





