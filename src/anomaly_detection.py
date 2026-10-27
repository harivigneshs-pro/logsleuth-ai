"""
Anomaly detection module for LogSleuth AI using Scikit-Learn Isolation Forest.
Provides configurable contamination, fixed random state, safe handling of edge cases,
and guaranteed higher-score-is-more-anomalous direction.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.feature_extraction import extract_features


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
    n_estimators: int = 150
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fits an IsolationForest model on extracted features and scores each log record.

    Parameters:
        df: Preprocessed log DataFrame.
        contamination: Proportion of outliers expected in the data (0.01 to 0.4).
        random_state: Seed for reproducible results.
        n_estimators: Number of isolation trees.

    Returns:
        Tuple of (scored_df, anomaly_summary_dict)
    """
    if df.empty:
        empty_df = df.copy()
        empty_df["is_anomaly"] = []
        empty_df["anomaly_score"] = []
        return empty_df, {
            "total_records": 0,
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "contamination_param": contamination,
            "model_type": "IsolationForest",
            "score_interpretation": "Higher score indicates higher anomaly severity (non-probabilistic).",
            "status": "empty_dataset"
        }

    # Bound contamination to valid interval
    clamped_contamination = float(np.clip(contamination, 0.01, 0.45))

    # Extract numerical features
    X, feat_names, feat_df = extract_features(df)

    scored_df = df.copy()

    # Edge case: Very small dataset (< 5 records)
    if len(df) < 5:
        # Fallback heuristic based on error severity indicators without crashing
        is_err = scored_df["level"].isin(["ERROR", "CRITICAL"])
        scored_df["is_anomaly"] = is_err
        scored_df["anomaly_score"] = np.where(is_err, 0.95, 0.10)
        anomaly_count = int(is_err.sum())
        return scored_df, {
            "total_records": len(df),
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round((anomaly_count / len(df)) * 100, 2),
            "contamination_param": clamped_contamination,
            "model_type": "RuleFallback_SmallDataset",
            "score_interpretation": "Dataset smaller than 5 records; evaluated via severity heuristic.",
            "suspicious_events_count": anomaly_count
        }

    # Fit IsolationForest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=clamped_contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X)

    # Scikit-learn score_samples: more negative = more abnormal
    # Invert so higher score = MORE anomalous
    raw_scores = -iso_forest.score_samples(X)
    
    # Normalize score smoothly into [0.0, 1.0] scale
    min_s = float(raw_scores.min())
    max_s = float(raw_scores.max())
    score_range = max_s - min_s
    if score_range > 1e-7:
        normalized_scores = (raw_scores - min_s) / score_range
    else:
        normalized_scores = np.full_like(raw_scores, 0.5)

    # Predictions: -1 for anomaly, 1 for normal
    preds = iso_forest.predict(X)
    is_anomaly = preds == -1

    scored_df["is_anomaly"] = is_anomaly
    scored_df["anomaly_score"] = np.round(normalized_scores, 4)
    scored_df["raw_outlier_score"] = np.round(raw_scores, 4)

    anomaly_count = int(is_anomaly.sum())
    total_count = len(df)
    anomaly_pct = round((anomaly_count / max(1, total_count)) * 100, 2)

    # Calculate threshold info
    suspicious_threshold = float(np.percentile(normalized_scores, (1 - clamped_contamination) * 100))

    summary = {
        "total_records": total_count,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "normal_count": total_count - anomaly_count,
        "contamination_param": clamped_contamination,
        "random_state": random_state,
        "model_type": "IsolationForest (scikit-learn)",
        "score_interpretation": (
            "Scores represent continuous isolation path-length anomaly metrics (0.0 to 1.0). "
            "Higher score = more anomalous. Anomaly scores are NOT probability values."
        ),
        "suspicious_threshold": round(suspicious_threshold, 4),
        "score_statistics": {
            "mean": round(float(normalized_scores.mean()), 4),
            "median": round(float(np.median(normalized_scores)), 4),
            "max": round(float(normalized_scores.max()), 4),
            "min": round(float(normalized_scores.min()), 4),
            "std": round(float(normalized_scores.std()), 4)
        },
        "features_used": feat_names
    }

    # Evaluate ground truth ONLY if valid label column was provided in original logs
    ground_truth_col = None
    for cand in ["is_incident", "ground_truth", "synthetic_incident", "true_anomaly"]:
        if cand in df.columns:
            ground_truth_col = cand
            break

    if ground_truth_col:
        try:
            y_true = df[ground_truth_col].astype(bool)
            tp = int(((y_true == True) & (is_anomaly == True)).sum())
            fp = int(((y_true == False) & (is_anomaly == True)).sum())
            fn = int(((y_true == True) & (is_anomaly == False)).sum())
            tn = int(((y_true == False) & (is_anomaly == False)).sum())
            precision = round(tp / max(1, tp + fp), 4)
            recall = round(tp / max(1, tp + fn), 4)
            f1 = round(2 * (precision * recall) / max(1e-7, precision + recall), 4)
            accuracy = round((tp + tn) / max(1, len(df)), 4)
            summary["ground_truth_metrics"] = {
                "ground_truth_column": ground_truth_col,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }
        except Exception:
            pass

    return scored_df, summary
