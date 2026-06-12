"""
Model validation: cross‑validation, permutation importance, calibration.
All functions return plain DataFrames or figures ready for the dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score


def cross_validate_model(model, X, y, cv_folds=5):
    """Return a dict with mean AUC and std across folds."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return {'cv_auc_mean': aucs.mean(), 'cv_auc_std': aucs.std()}


def permutation_importance_df(model, X, y, feature_names, n_repeats=10):
    """
    Compute permutation importance (decrease in AUC) and return a DataFrame.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, scoring='roc_auc'
    )
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    return df


def plot_permutation_importance(perm_df, top_n=10, title='Permutation Importance'):
    """Return a matplotlib figure for the top features."""
    top = perm_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top['feature'][::-1], top['importance_mean'][::-1], xerr=top['importance_std'][::-1],
            color='skyblue', capsize=3)
    ax.set_xlabel('Decrease in AUC')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_calibration_curve(model, X, y, model_name='Model'):
    """Return a calibration plot figure."""
    y_prob = model.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker='o', label=model_name)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration – {model_name}')
    ax.legend()
    plt.tight_layout()
    return fig