"""
Load, clean, engineer features, and preprocess data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load CSV and fix TotalCharges."""
    df = pd.read_csv(filepath)
    # Convert TotalCharges: blank strings become NaN, then to float, fill with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create AvgMonthlySpend and TotalServices."""
    # AvgMonthlySpend – proxy for CLV
    df['AvgMonthlySpend'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    # Total services used
    service_map = {
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes'
    }
    service_count = 0
    for col, yes_val in service_map.items():
        if col in df.columns:
            service_count += (df[col] == yes_val).astype(int)
    service_count += (df['PhoneService'] == 'Yes').astype(int)
    service_count += (df['InternetService'] != 'No').astype(int)
    service_count += (df['MultipleLines'] == 'Yes').astype(int)
    df['TotalServices'] = service_count
    return df


def preprocess_features(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    scaler: StandardScaler = None,
    fit_scaler: bool = True
) -> tuple:
    """
    Encode categoricals, scale numeric columns, return X, y, feature_names, scaler.
    Works with PyArrow string types and plain objects.
    """
    # --- Target encoding (handles all string types safely) ---
    y = df[target_col].replace({'Yes': 1, 'No': 0}).astype(int)

    X = df.drop(columns=[target_col, 'customerID'], errors='ignore')

    # ---- Encode binary categoricals ----
    binary_mappings = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'SeniorCitizen': {1: 1, 0: 0}
    }
    for col, mapping in binary_mappings.items():
        if col in X.columns:
            # Use replace to handle PyArrow strings and plain objects
            X[col] = X[col].replace(mapping)

    # ---- One‑hot encode all remaining object columns ----
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # ---- Scale numeric features ----
    scaled_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']
    scaled_cols = [c for c in scaled_cols if c in X.columns]

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X[scaled_cols] = scaler.fit_transform(X[scaled_cols])
    else:
        X[scaled_cols] = scaler.transform(X[scaled_cols])

    return X, y, list(X.columns), scaler


def preprocess_single_input(
    raw_dict: dict,
    feature_names: list,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Preprocess a single customer dictionary for prediction.
    Mirrors the training preprocessing exactly.
    """
    df = pd.DataFrame([raw_dict])

    # ---- Binary encodings (use replace for robustness) ----
    binary_mappings = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'SeniorCitizen': {'Yes': 1, 'No': 0}  # just in case
    }
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # ---- One‑hot encode any remaining object columns ----
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # ---- Align with training features ----
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]  # enforce exact column order
    df = df.astype(float)

    # ---- Scale numeric features ----
    scaled_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']
    scaled_cols = [c for c in scaled_cols if c in df.columns]
    df[scaled_cols] = scaler.transform(df[scaled_cols])

    return df