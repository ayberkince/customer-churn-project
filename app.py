import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# Overall churn rate from your training data (adjust if yours differs slightly)
OVERALL_CHURN_RATE = 0.265

st.title("Telco Customer Churn Predictor")
st.markdown(
    "Adjust the customer attributes below to see the predicted churn probability "
    "**and its risk relative to the average customer**."
)

# Inputs
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

# Build raw data
raw_data = {
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
    'TotalCharges': total_charges
}
input_df = pd.DataFrame([raw_data])

# Binary columns → int
binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    input_df[col] = input_df[col].astype(int)

# One‑hot encoding for multi‑category columns
multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
input_encoded = pd.get_dummies(input_df, columns=multi_cat_cols, drop_first=True)

# Align with training features
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_names]

# Ensure all columns are numeric
input_encoded = input_encoded.astype(float)

# Scale continuous columns
scaled_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_encoded[scaled_cols] = scaler.transform(input_encoded[scaled_cols])

# Predict
prob = model.predict_proba(input_encoded)[0, 1]
risk_ratio = prob / OVERALL_CHURN_RATE

# Display results
st.metric("Churn Probability", f"{prob:.1%}")
st.metric("Risk Ratio (vs. average)", f"{risk_ratio:.2f}x")

# Ratio‑based interpretation
if risk_ratio < 0.5:
    st.success("✅ Low risk – far less likely to churn than the average customer.")
elif risk_ratio < 1.5:
    st.warning("⚠️ Moderate risk – churn likelihood is around the company average.")
else:
    st.error("🔴 High risk – strongly consider a proactive retention action.")