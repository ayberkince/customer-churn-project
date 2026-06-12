# Telco Customer Churn Predictor (v2.0)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badge-gradient-gradient.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-%23232F3E.svg?style=flat)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-%233B5998.svg?style=flat)](https://github.com/microsoft/LightGBM)

An interactive, multi-model machine learning pipeline and Streamlit dashboard designed to identify customers at high risk of churn and estimate the financial viability of targeted retention campaigns.

---

## 📊 Business Problem & Objectives

Customer churn is a multi-billion dollar challenge in the telecommunications industry. Acquiring a new customer is significantly more expensive than retaining an existing one. 

### Core Goals:
1. **Identify Churn Risk Early**: Predict individual customer churn probabilities based on demographics, subscribed services, contract details, and billing history.
2. **Support Decision-Making**: Quantify customer risk ratio relative to the baseline churn rate (**26.5%**).
3. **Optimize Retention Campaigns**: Provide tools for exploring "what-if" scenarios (Feature Effect Explorer) and calculating potential economic savings.

---

## 🛠️ Project Architecture & File Structure

Version 2.0 modularizes the project, separating data processing, model training, validation, and dashboard presentation.

```
customer-churn-project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw Kaggle dataset (7,043 customers)
├── src/                                      # Source package
│   ├── __init__.py
│   ├── data_preprocessing.py                # Cleaning, target replacement, binary mapping, and scaling
│   ├── evaluation.py                        # Profit evaluation curves and native model importance helpers
│   ├── model_training.py                    # Training routines for LR, RF, XGBoost, and LightGBM
│   ├── validation.py                        # Cross-validation, permutation importance, and calibration
│   └── visualization.py                     # Custom visualization helper functions
├── app.py                                   # Interactive Streamlit dashboard application
├── train_all_models.py                      # Core model training, evaluation, and serialization pipeline
├── requirements.txt                         # Python packages and version requirements
├── validation_summary.json                  # Pre-compiled cross-validation results and permutation metrics
├── scaler.pkl                               # Serialized StandardScaler fitted on training features
├── feature_names.txt                        # Aligned features matching training schema
├── churn_{lr/rf/xgb/lgb}_model.pkl          # Serialized predictive models
└── calibration_{Model Name}.png             # Exported calibration curves
```

---

## 🧪 Methodology, Preprocessing & Feature Engineering

### 1. Data Cleaning
* **Missing Value Imputation**: Handled missing `TotalCharges` values for new customers (tenure = 0) by setting them to 0.
* **Target Mapping**: Converted the target variable `Churn` (`Yes`/`No`) into binary representation (`1`/`0`).

### 2. Feature Engineering
* **`AvgMonthlySpend`**: Computed as `TotalCharges` / `tenure` (falling back to `MonthlyCharges` if tenure is 0) to represent the velocity of customer expenditure, serving as a proxy for customer lifetime value (CLV).
* **`TotalServices`**: Aggregated the number of services a customer subscribes to (Phone Service, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, and Multiple Lines).

### 3. Feature Transformation
* **Binary Encodings**: Encodings applied to `gender`, `Partner`, `Dependents`, `PhoneService`, and `PaperlessBilling`.
* **One-Hot Encoding**: Handled multi-categorical variables (`Contract`, `InternetService`, `PaymentMethod`).
* **Feature Scaling**: Scaled continuous columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `AvgMonthlySpend`) using `StandardScaler` to ensure numerical stability and performance, especially for Logistic Regression.

---

## 📈 Model Performance & Validation Summary

We train and evaluate four diverse classifiers using **5-fold Stratified Cross-Validation (CV)** and compute **Permutation Feature Importance** on the test set.

| Model | Mean CV ROC AUC | CV AUC Std Dev | Top 3 Permutation Importance Features |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **0.845** | 0.013 | 1. `tenure`<br>2. `InternetService_Fiber optic`<br>3. `Contract_Two year` |
| **Random Forest** | **0.844** | 0.011 | 1. `tenure`<br>2. `Contract_Two year`<br>3. `InternetService_Fiber optic` |
| **XGBoost** | **0.841** | 0.012 | 1. `Contract_Two year`<br>2. `tenure`<br>3. `Contract_One year` |
| **LightGBM** | **0.840** | 0.011 | 1. `Contract_Two year`<br>2. `tenure`<br>3. `Contract_One year` |

### Key Findings
* **Contract Type & Tenure**: Across all models, contract duration (specifically two-year contracts) and tenure are the primary drivers in reducing churn risk.
* **Fiber Optic Internet**: Subscribing to Fiber Optic internet service consistently shows high permutation importance, indicating a significant correlation with churn (potentially due to price or quality sensitivity).
* **Performance Consistency**: All four models demonstrate competitive predictive capabilities, achieving a mean cross-validation ROC AUC of **~0.84**.

---

## 🖥️ Streamlit Dashboard Features

The dynamic Streamlit dashboard (`app.py`) enables real-time customer assessment and model analysis:

1. **Model Selection**: Toggle between **Logistic Regression**, **Random Forest**, **XGBoost**, and **LightGBM** in the sidebar to compare model behavior on the same customer profile.
2. **Interactive Profiling**: Adjust demographics (gender, partner, senior citizen), billing parameters, contract types, and services using sliders and selectors.
3. **Real-Time Predictions**: Computes the Churn Probability and a **Risk Ratio** relative to the average churn rate. Color-coded alerts flag customers as *Low Risk* (<0.5x), *Moderate Risk* (0.5x - 1.5x), or *High Risk* (>1.5x).
4. **Global Feature Importance**: Renders a bar chart showing the top 10 features utilized by the active model.
5. **Feature Effect Explorer**: Let users explore "what-if" scenarios. Visualizes how changing a single metric (e.g. `tenure` or `MonthlyCharges`) changes the churn probability, keeping all other variables constant. Plots the current customer as a reference point.
6. **Model Trust & Validation**: Pulls live CV scores, lists the top 3 permutation features, and displays the model's calibration curve to inspect predictive reliability.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Training & Validation Pipeline
To run the full training pipeline, engineer features, perform cross-validation, and generate artifacts:
```bash
python train_all_models.py
```
This script updates the serialized `.pkl` models, calibration plots, and the `validation_summary.json` file.

### 3. Launch the Dashboard
Run the Streamlit app:
```bash
streamlit run app.py
```
Open the provided local URL (typically `http://localhost:8501`) in your browser to interact with the predictor.

---

## 💼 Business Impact & Targeting Strategy

Using a standard customer lifetime value (LTV) framework:
* **Customer Lifetime Value (LTV)**: $2,000
* **Retention Campaign Incentives**: $100 discount + $5 contact cost = $105 total cost per customer targeted.

With a **ROC AUC of 0.84**, a model-driven targeting strategy achieves significantly higher returns than random targeting:
* **Targeting Optimization**: By focusing on the top **30%** highest-risk customers, the company can capture **>70%** of prospective churners.
* **Incentive Allocation**: This optimization prevents wasted incentive spending on low-risk customers while proactively securing high-value accounts before they churn.
