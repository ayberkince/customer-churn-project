#!/usr/bin/env python
"""
Full pipeline: load data → engineer features → split → train 4 models → save artifacts.
Run this from the project root.
"""
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from src.data_preprocessing import load_raw_data, engineer_features, preprocess_features
from src.model_training import (
    train_logistic_regression, train_random_forest,
    train_xgboost, train_lightgbm
)
from sklearn.model_selection import train_test_split
import joblib
import json
from src.validation import (
    cross_validate_model,
    permutation_importance_df,
    plot_calibration_curve
)

# 1. Load & engineer
df = load_raw_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = engineer_features(df)

# 2. Preprocess
X, y, feature_names, scaler = preprocess_features(df, fit_scaler=True)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Train all models
models = {
    'lr': train_logistic_regression(X_train, y_train),
    'rf': train_random_forest(X_train, y_train),
    'xgb': train_xgboost(X_train, y_train),
    'lgb': train_lightgbm(X_train, y_train)
}

# 5. Save each model (short names for file prefixes)
for name, model in models.items():
    joblib.dump(model, f'churn_{name}_model.pkl')

# Save shared scaler and feature names once
joblib.dump(scaler, 'scaler.pkl')
with open('feature_names.txt', 'w') as f:
    for feat in feature_names:
        f.write(feat + '\n')

print("All models and shared scaler/feature_names saved.")

# Save X_test sample for eventual SHAP / dashboard
joblib.dump(X_test.iloc[:200], 'X_test_sample.pkl')

# ---------- Validation (dashboard‑compatible names) ----------
name_map = {
    'lr': 'Logistic Regression',
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'lgb': 'LightGBM'
}

validation_summary = {}
for short_name, model in models.items():
    display_name = name_map[short_name]

    # Cross‑validation
    cv_result = cross_validate_model(model, X, y)

    # Permutation importance
    perm_df = permutation_importance_df(model, X_test, y_test, feature_names)
    perm_df.to_csv(f'permutation_importance_{display_name}.csv', index=False)

    # Calibration plot
    fig = plot_calibration_curve(model, X_test, y_test, model_name=display_name)
    fig.savefig(f'calibration_{display_name}.png', bbox_inches='tight')
    plt.close(fig)

    validation_summary[display_name] = {
        'cv_auc_mean': cv_result['cv_auc_mean'],
        'cv_auc_std': cv_result['cv_auc_std'],
        'permutation_top3': perm_df.head(3)[['feature', 'importance_mean']].to_dict(orient='records')
    }

with open('validation_summary.json', 'w') as f:
    json.dump(validation_summary, f, indent=2)

print("Validation artifacts saved: permutation CSVs, calibration PNGs, validation_summary.json")