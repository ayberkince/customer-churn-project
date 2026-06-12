import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import json
from src.data_preprocessing import preprocess_single_input
from src.evaluation import get_native_importance

def app_path(filename):
    """Return absolute path to a file in the same directory as this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# ---------- Load shared artifacts ----------
scaler = joblib.load('scaler.pkl')
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# ---------- Cache model loading ----------
@st.cache_resource
def load_models():
    return {
        'Logistic Regression': joblib.load('churn_lr_model.pkl'),
        'Random Forest':       joblib.load('churn_rf_model.pkl'),
        'XGBoost':             joblib.load('churn_xgb_model.pkl'),
        'LightGBM':            joblib.load('churn_lgb_model.pkl')
    }

models = load_models()

# ---------- Sidebar ----------
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox("Choose prediction model:", list(models.keys()))

# ---------- Main UI ----------
OVERALL_CHURN_RATE = 0.265
st.title("Telco Customer Churn Predictor")
st.markdown("Adjust the attributes and compare predictions across models.")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = tenure * monthly_charges
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    gender = st.selectbox("Gender", ["Female", "Male"])
with col2:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    total_services = st.number_input("Total Services", min_value=0, max_value=9, value=1)

# ---------- Build raw input ----------
raw_input = {
    'gender': 0 if gender == "Female" else 1,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1,
    'MultipleLines': 'No',
    'InternetService': internet_service,
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': contract,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'TotalServices': total_services,
    'AvgMonthlySpend': total_charges / tenure if tenure > 0 else monthly_charges
}

# Preprocess
input_df = preprocess_single_input(raw_input, feature_names, scaler)

# ---------- Predict ----------
model = models[model_choice]
prob = model.predict_proba(input_df)[0, 1]
risk_ratio = prob / OVERALL_CHURN_RATE

st.subheader(f"Prediction using **{model_choice}**")
st.metric("Churn Probability", f"{prob:.1%}")
st.metric("Risk Ratio (vs. average)", f"{risk_ratio:.2f}x")

if risk_ratio < 0.5:
    st.success("Low risk – far less likely to churn than the average customer.")
elif risk_ratio < 1.5:
    st.warning("Moderate risk – around the average.")
else:
    st.error("High risk – consider retention action.")

# ---------- Feature Importance (Global) ----------
st.subheader("Feature Importance (Global)")


@st.cache_data
def get_importance(model_name):
    m = models[model_name]
    # Use model‑intrinsic importance – always works, no SHAP crashes
    return get_native_importance(m, feature_names)

imp_df = get_importance(model_choice)
if imp_df is not None:
    top10 = imp_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10['feature'][::-1], top10['importance'][::-1], color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Top 10 Features – {model_choice}')
    st.pyplot(fig)

# ---------- Feature Effect Explorer ----------
st.subheader("Feature Effect Explorer")
st.markdown("See how changing a single feature affects the churn prediction (other features held constant).")

raw_features = [
    'tenure', 'MonthlyCharges', 'TotalServices',
    'Contract', 'InternetService', 'PaymentMethod',
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling'
]
selected_feature = st.selectbox("Choose a feature to explore:", raw_features)

def get_feature_values(feature_name, current_value):
    if feature_name in ['tenure', 'MonthlyCharges', 'TotalServices']:
        if feature_name == 'tenure':
            low, high = 0, 72
        elif feature_name == 'MonthlyCharges':
            low, high = 18, 120
        else:  # TotalServices
            low, high = 0, 9
        return np.linspace(low, high, 50).tolist()
    else:
        # Categorical features
        categories = {
            'Contract': ["Month-to-month", "One year", "Two year"],
            'InternetService': ["DSL", "Fiber optic", "No"],
            'PaymentMethod': ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            'gender': ["Female", "Male"],
            'SeniorCitizen': ["No", "Yes"],
            'Partner': ["No", "Yes"],
            'Dependents': ["No", "Yes"],
            'PaperlessBilling': ["No", "Yes"]
        }
        return categories.get(feature_name, [current_value])

feature_values = get_feature_values(selected_feature, raw_input.get(selected_feature, None))
base_input = raw_input.copy()

probs = []
for val in feature_values:
    temp = base_input.copy()
    temp[selected_feature] = val
    # Recalculate dependent fields
    if selected_feature == 'MonthlyCharges':
        temp['TotalCharges'] = temp['tenure'] * val
        temp['AvgMonthlySpend'] = val if temp['tenure'] == 0 else temp['TotalCharges'] / temp['tenure']
    elif selected_feature == 'tenure':
        temp['TotalCharges'] = val * temp['MonthlyCharges']
        temp['AvgMonthlySpend'] = temp['MonthlyCharges'] if val == 0 else temp['TotalCharges'] / val
    # Note: TotalServices needs no recalculation of other fields

    input_df_temp = preprocess_single_input(temp, feature_names, scaler)
    probs.append(model.predict_proba(input_df_temp)[0, 1])

fig2, ax2 = plt.subplots(figsize=(8, 4))
is_categorical = selected_feature not in ['tenure', 'MonthlyCharges', 'TotalServices']

if is_categorical:
    ax2.bar(feature_values, probs, color='skyblue')
    ax2.set_ylabel('Predicted Churn Probability')
else:
    ax2.plot(feature_values, probs, marker='o', color='skyblue')
    ax2.set_ylabel('Predicted Churn Probability')
    ax2.set_xlabel(selected_feature)
    # Mark current customer
    current_val = raw_input[selected_feature]
    ax2.axvline(current_val, color='red', linestyle='--', label='Current customer')
    ax2.legend()

ax2.set_title(f'Effect of {selected_feature} ({model_choice})')
st.pyplot(fig2)

# ---------- Model Trust & Validation ----------
st.subheader("Model Trust & Validation")

val_json_path = app_path('validation_summary.json')
if not os.path.exists(val_json_path):
    st.warning(f"Missing file: {val_json_path}. Run train_all_models.py first.")
else:
    try:
        with open(val_json_path, 'r') as f:
            val_summary = json.load(f)
        current_val = val_summary.get(model_choice)
        if current_val is None:
            st.warning(f"No validation data for '{model_choice}'. Retrain with the latest script.")
        else:
            cv_auc = current_val['cv_auc_mean']
            cv_std = current_val['cv_auc_std']
            st.markdown(f"**5‑fold CV AUC:** {cv_auc:.3f} ± {cv_std:.3f}")
            st.markdown("**Top 3 features by permutation importance:**")
            for item in current_val['permutation_top3']:
                st.write(f"- {item['feature']}: {item['importance_mean']:.4f}")

            # Show calibration plot
            calib_path = app_path(f'calibration_{model_choice}.png')
            if os.path.exists(calib_path):
                st.image(calib_path, caption='Calibration Plot')
            else:
                st.warning(f"Calibration plot not found: {calib_path}")
    except Exception as e:
        st.error(f"Failed to load validation data: {e}")