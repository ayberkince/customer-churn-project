"""
Train and save multiple models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    neg, pos = np.bincount(y_train)
    model = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def save_artifacts(model, scaler, feature_names, model_prefix, scaler_prefix='scaler', feature_prefix='feature_names'):
    """Save model, scaler, and feature list. Use separate prefixes if needed."""
    joblib.dump(model, f'{model_prefix}.pkl')
    joblib.dump(scaler, f'{scaler_prefix}.pkl')
    with open(f'{feature_prefix}.txt', 'w') as f:
        for name in feature_names:
            f.write(name + '\n')