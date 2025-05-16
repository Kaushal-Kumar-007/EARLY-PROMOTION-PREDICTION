# 🏆 Early Promotion Eligibility Prediction

This project predicts whether an employee is likely to be promoted early in a multinational corporation using supervised machine learning techniques. The aim is to help HR departments identify high-potential employees earlier in the evaluation cycle.

---

## 📂 Project Structure

├── data/ # Raw and cleaned datasets (CSV)
├── notebooks/ # Jupyter notebooks for EDA, modeling, and evaluation
├── models/ # Trained ML model, encoders, and scalers (joblib format)
├── outputs/ # Evaluation results and plots
└── README.md # Project documentation


---

## ✅ Project Objectives

- Identify key factors influencing employee promotion.
- Build a robust machine learning model to predict promotion eligibility.
- Handle class imbalance effectively.
- Optimize model performance using metrics like **F1 Score**.

---

## 📊 Dataset Summary

The dataset contains information on employee demographics, performance, and history, including:

- `employee_id`
- `department`
- `region`
- `education`
- `gender`
- `recruitment_channel`
- `no_of_trainings`
- `age`
- `previous_year_rating`
- `length_of_service`
- `KPIs_met >80%`
- `awards_won?`
- `avg_training_score`
- `is_promoted` (Target)

---

## 🧪 Workflow Summary

### 1. Data Preprocessing
- Handling missing values.
- Label encoding and one-hot encoding for categorical features.
- Scaling of numerical features using `StandardScaler`.

### 2. Exploratory Data Analysis (EDA)
- Distribution plots and heatmaps.
- Correlation analysis.
- Feature importance based on domain knowledge and XGBoost model.

### 3. Model Building
- XGBoost Classifier selected after experimenting with multiple models.
- Hyperparameter tuning using `RandomizedSearchCV`.
- Performance evaluated using:
  - F1 Score
  - Precision / Recall
  - Confusion Matrix

### 4. Imbalance Handling
- Handled via `scale_pos_weight` in XGBoost.
- Class distribution checked and balanced accordingly.

### 5. Model Evaluation
- Detailed classification report.
- Residual and error analysis.
- Validation using stratified train-test splits.

---

## 📈 Results

- Best Model: **XGBoostClassifier**
- **F1 Score (macro avg)**: ~`0.72` (tuned)
- High performance achieved even on imbalanced data using proper techniques.

---

## 🔧 Requirements

- Python 3.8+
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
