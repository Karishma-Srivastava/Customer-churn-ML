# 📊 Telco Customer Churn Prediction

## 🚀 Project Overview

This project predicts whether a telecom customer will churn (leave the service) using machine learning.
The goal is to **identify at-risk customers early** so businesses can take preventive action.

---

## 🎯 Business Problem

Customer churn leads to revenue loss.
It is **cheaper to retain existing customers than acquire new ones**.

👉 Objective:

* Predict churn (Yes/No)
* Maximize **recall** (catch more churners)
* Maintain reasonable **precision** (avoid false alarms)

---

## 📁 Project Structure

```
churn_project-ML/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── process/
│   │   │   └── preprocess.py
│   │   └── raw/
│   │       └── customer_churn_files.csv
│   │
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate_model.py
│
├── main.py
└── README.md
```

---

## ⚙️ Workflow

1. Load raw dataset
2. Data preprocessing:

   * Handle missing values
   * Convert data types
   * Encode categorical variables
3. Train-test split
4. Handle class imbalance using **SMOTE**
5. Train model (**XGBoost**)
6. Predict probabilities
7. Apply optimized threshold
8. Evaluate model

---

## 🧠 Model Details

### 🔹 Algorithm

* XGBoost Classifier

### 🔹 Why XGBoost?

* Handles tabular data efficiently
* Captures complex patterns
* Outperforms traditional models like Logistic Regression / Random Forest

---

## ⚖️ Handling Imbalance

* Dataset is imbalanced (more non-churn than churn)
* Used **SMOTE (Synthetic Minority Oversampling Technique)**

---

## 🎯 Threshold Tuning

Instead of default 0.5:

* Optimized threshold based on **precision-recall trade-off**
* Different thresholds tested based on business needs

---

## 📊 Model Performance

| Metric            | Value |
| ----------------- | ----- |
| ROC-AUC           | ~0.83 |
| Precision (Churn) | ~0.64 |
| Recall (Churn)    | ~0.48 |

---

## 🔍 Key Insights

* High recall helps capture more churn customers
* High precision reduces unnecessary retention cost
* Trade-off depends on business objective

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost

---

## ▶️ How to Run

```bash
# install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost

# run project
python main.py
```

---

## 📌 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Feature importance analysis
* Model deployment (API using FastAPI)
* Real-time churn prediction

---

## 💡 Key Learning

* Handling imbalanced datasets
* Threshold tuning vs accuracy
* Model selection (RF → XGBoost)
* Building modular ML pipeline

---

## 👩‍💻 Author

Karishma Srivastava
