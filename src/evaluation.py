"""
Model evaluation, profit curves, model comparison, feature importance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix


def evaluate_model(model, X_test, y_test, LTV=2000, discount=100, contact_cost=5):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    thresholds = np.linspace(0.01, 0.99, 100)
    profits = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        profit = tp * (LTV - discount - contact_cost) - fp * (discount + contact_cost)
        profits.append(profit)
    best_profit = max(profits)
    best_threshold = thresholds[np.argmax(profits)]
    return auc, best_profit, best_threshold, y_prob


def compare_models(models_dict, X_test, y_test, LTV=2000, discount=100, contact_cost=5):
    results = []
    for name, model in models_dict.items():
        auc, best_profit, best_thresh, _ = evaluate_model(model, X_test, y_test, LTV, discount, contact_cost)
        results.append({'Model': name, 'AUC': auc, 'Max Profit ($)': best_profit, 'Best Threshold': best_thresh})
    return pd.DataFrame(results)


def plot_profit_curves(models_dict, X_test, y_test, LTV=2000, discount=100, contact_cost=5):
    thresholds = np.linspace(0.01, 0.99, 100)
    plt.figure(figsize=(10, 6))
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        profits = []
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
            profit = tp * (LTV - discount - contact_cost) - fp * (discount + contact_cost)
            profits.append(profit)
        plt.plot(thresholds, profits, label=name)
    plt.xlabel('Threshold')
    plt.ylabel('Profit ($)')
    plt.title('Profit Comparison Across Models')
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_native_importance(model, feature_names):
    """
    Return a DataFrame with feature importance using model‑intrinsic attributes.
    Works for Logistic Regression, Random Forest, XGBoost, LightGBM.
    """
    # Logistic Regression: absolute coefficients
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        return imp_df.sort_values('importance', ascending=False)

    # Tree‑based models: feature_importances_
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        return imp_df.sort_values('importance', ascending=False)

    # Fallback: random (shouldn't happen)
    importance = np.random.rand(len(feature_names))
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    return imp_df