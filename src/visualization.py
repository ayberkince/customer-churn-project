"""
Reusable plotting functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_churn_rate_by_category(df, column, target='Churn', ax=None):
    """Bar chart of churn rate per category."""
    if ax is None:
        ax = plt.gca()
    positive_class = 1 if df[target].dtype != object else 'Yes'
    ct = pd.crosstab(df[column], df[target], normalize='index')
    if positive_class in ct.columns:
        ct[positive_class].plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f'Churn Rate by {column}')
    ax.set_ylabel('Churn Rate')


def plot_boxplot(df, x, y, ax=None):
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(f'{y} by {x}')


def plot_correlation_heatmap(df, cols, ax=None):
    if ax is None:
        ax = plt.gca()
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')


def plot_cumulative_gains(y_true, y_prob, ax=None):
    if ax is None:
        ax = plt.gca()
    df = pd.DataFrame({'actual': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    df['cum_churners'] = df['actual'].cumsum()
    total_churners = df['actual'].sum()
    df['cum_gain'] = df['cum_churners'] / total_churners
    df['pct_population'] = np.arange(1, len(df)+1) / len(df)
    ax.plot(df['pct_population'], df['cum_gain'], label='Model')
    ax.plot([0,1], [0,1], 'k--', label='Random')
    ax.set_xlabel('Percentage of customers targeted')
    ax.set_ylabel('Percentage of churners captured')
    ax.set_title('Cumulative Gains Curve')
    ax.legend()