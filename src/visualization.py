# visualization.py
"""
Reusable matplotlib + seaborn visualizations for the Sustainable AI app.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models import PromptData


# Make seaborn look decent by default
sns.set(style="whitegrid")


def feature_comparison_bar(
    feats_before: Dict[str, float],
    feats_after: Dict[str, float],
) -> plt.Figure:
    """
    Bar chart comparing key features before vs after optimization.
    """
    features_for_plot = ["tokens", "avg_sentence_len", "stopword_ratio", "sections"]

    df_feat = pd.DataFrame({
        "Feature": features_for_plot,
        "Before": [feats_before[f] for f in features_for_plot],
        "After": [feats_after[f] for f in features_for_plot],
    })

    df_melt = df_feat.melt(id_vars="Feature", var_name="Version", value_name="Value")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df_melt, x="Feature", y="Value", hue="Version", ax=ax)
    ax.set_title("Before vs After Feature Comparison")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.legend(title="Version")
    fig.tight_layout()
    return fig


def energy_distribution_hist(
    energy_kwh: float,
    log_path: str = "energy_logs.csv",
) -> Optional[plt.Figure]:
    """
    Histogram of historical energy usage with a red line for the current prompt.

    Returns None if there is not enough history.
    """
    if not os.path.exists(log_path):
        return None

    df = pd.read_csv(log_path)
    if "energy_kwh" not in df.columns or df.empty:
        return None

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df["energy_kwh"], bins=30, kde=True, ax=ax)
    ax.axvline(energy_kwh, color="red", linestyle="--", label="Current prompt")
    ax.set_title("Energy Usage Distribution (Historical)")
    ax.set_xlabel("Energy (kWh)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def token_breakdown_bar(prompt: PromptData) -> plt.Figure:
    """
    Bar chart showing token usage per section: Role, Context, Expectations.
    """
    section_tokens = {
        "Role": len(prompt.role.split()) if prompt.role else 0,
        "Context": len(prompt.context.split()) if prompt.context else 0,
        "Expectations": len(prompt.expectations.split()) if prompt.expectations else 0,
    }

    df_tokens = pd.DataFrame({
        "Section": list(section_tokens.keys()),
        "Tokens": list(section_tokens.values()),
    })

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df_tokens, x="Section", y="Tokens", ax=ax)
    ax.set_title("Token Usage per Section")
    ax.set_xlabel("Prompt section")
    ax.set_ylabel("Tokens")
    fig.tight_layout()
    return fig


def anomaly_score_bar(score: float, is_anomaly: bool) -> plt.Figure:
    """
    Simple horizontal bar to visualize anomaly score.

    Typically IsolationForest decision_function() is around [-0.5, 0.5].
    """
    color = "red" if is_anomaly else "green"
    label = "Anomalous" if is_anomaly else "Normal"

    fig, ax = plt.subplots(figsize=(4, 1.2))
    ax.barh([label], [score], color=color)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Anomaly score")
    ax.set_title("Energy Anomaly Indicator")
    fig.tight_layout()
    return fig
