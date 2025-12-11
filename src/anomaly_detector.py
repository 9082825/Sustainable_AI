import os
from typing import Dict

import numpy as np
import pandas as pd
import joblib

ANOMALY_MODEL_PATH = "model/anomaly_detector/iso_pipeline.pkl"
FEATURE_COLS = [
    "num_layers",
    "training_hours",
    "flops_per_hour",
    "flops_per_layer",
    "energy_kwh",
    "energy_per_hour",
    "prompt_tokens",
    "energy_per_token",
]


def load_anomaly_model(path: str = ANOMALY_MODEL_PATH):
    """Load the trained IsolationForest pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Anomaly model not found at {path}")
    return joblib.load(path)


def _build_feature_frame(record: Dict) -> pd.DataFrame:
    """Create a single-row DataFrame with all required features."""
    df = pd.DataFrame([record])

    df["flops_per_layer"] = df["flops_per_hour"] / df["num_layers"]
    df["energy_per_hour"] = df["energy_kwh"] / df["training_hours"].replace(0, np.nan)
    df["energy_per_token"] = df["energy_kwh"] / df["prompt_tokens"].clip(lower=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLS]


def flag_usage_anomaly(
    record: dict,
    pipeline=None,
) -> dict:
    """
    record: dict with at least num_layers, training_hours, flops_per_hour,
            energy_kwh, prompt_tokens.

    Returns:
    {
        "is_anomaly": bool,
        "score": float,
        "explanation": str,
    }
    """
    if pipeline is None:
        pipeline = load_anomaly_model()

    X = _build_feature_frame(record)

    # Try both decision_function / score_samples depending on estimator
    try:
        scores = pipeline.decision_function(X)
    except AttributeError:
        try:
            scores = pipeline.score_samples(X)
        except AttributeError:
            scores = np.zeros(len(X))

    try:
        labels = pipeline.predict(X)  # 1 = normal, -1 = anomaly for IsolationForest
    except Exception:
        labels = np.ones(len(X))

    score = float(scores[0])
    is_anom = bool(labels[0] == -1)

    explanation = (
        "Usage deviates from normal energy/computation patterns."
        if is_anom
        else "Usage falls within the range of normal energy/computation patterns."
    )

    return {
        "is_anomaly": is_anom,
        "score": score,
        "explanation": explanation,
    }
