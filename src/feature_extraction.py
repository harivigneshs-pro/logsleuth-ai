"""
Feature extraction module for LogSleuth AI.
Converts preprocessed structured or raw text logs into a rich numerical feature matrix
for unsupervised anomaly detection.
"""


import numpy as np
import pandas as pd

SEVERITY_WEIGHTS = {
    "DEBUG": 0.0,
    "INFO": 1.0,
    "WARNING": 2.5,
    "ERROR": 4.5,
    "CRITICAL": 6.0
}


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Extracts numerical features from preprocessed log DataFrame.

    Returns:
        Tuple of (feature_matrix: np.ndarray, feature_names: List[str], features_df: pd.DataFrame)
    """
    if df.empty:
        return np.empty((0, 0)), [], pd.DataFrame()

    feat_df = pd.DataFrame(index=df.index)
    messages = df["message"].astype(str).str.lower()

    # 1. Message structural indicators
    feat_df["msg_length"] = df["message"].astype(str).str.len().fillna(0).astype(float)
    feat_df["word_count"] = messages.apply(lambda m: len(m.split())).astype(float)

    # 2. Keyword & semantic anomaly indicators
    # Error & exception keywords
    err_pattern = r'\b(?:error|exception|failed|failure|traceback|errno|fault|crash|unhandled|5\d{2})\b'
    feat_df["has_error_kw"] = messages.str.contains(err_pattern, regex=True).astype(float)

    # Warning keywords
    warn_pattern = r'\b(?:warn|warning|degraded|slow|elevated|throttl|high latency)\b'
    feat_df["has_warning_kw"] = messages.str.contains(warn_pattern, regex=True).astype(float)

    # Retry indicators
    retry_pattern = r'\b(?:retry|retrying|attempt|backoff|reconnect|re-trying)\b'
    feat_df["has_retry_kw"] = messages.str.contains(retry_pattern, regex=True).astype(float)

    # Timeout indicators
    timeout_pattern = r'\b(?:timeout|timed out|deadline exceeded|connection reset|timedout|hang)\b'
    feat_df["has_timeout_kw"] = messages.str.contains(timeout_pattern, regex=True).astype(float)

    # Critical / Fatal indicators
    crit_pattern = r'\b(?:critical|fatal|panic|emergency|oom|out of memory|segfault|killed)\b'
    feat_df["has_critical_kw"] = messages.str.contains(crit_pattern, regex=True).astype(float)

    # Database / Network friction indicators
    net_pattern = r'\b(?:pool exhausted|disconnect|refused|unreachable|socket|deadlock|lock wait)\b'
    feat_df["has_infra_kw"] = messages.str.contains(net_pattern, regex=True).astype(float)

    # 3. Normalized Severity weights
    feat_df["severity_weight"] = df["level"].map(
        lambda l: SEVERITY_WEIGHTS.get(str(l).upper(), 1.0)
    ).astype(float)
    feat_df["is_error_or_critical"] = df["level"].isin(["ERROR", "CRITICAL"]).astype(float)

    # 4. Service-level frequency and distribution features
    total_events = len(df)
    service_counts = df["service"].value_counts()
    feat_df["service_event_count"] = df["service"].map(service_counts).astype(float)
    feat_df["service_frequency_ratio"] = feat_df["service_event_count"] / max(1, total_events)

    # 5. Temporal window statistics (if valid timestamps exist)
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # Sort index verification
        sorted_df = df.sort_values("timestamp")

        # Time delta in seconds from previous event in same service
        time_deltas = sorted_df.groupby("service")["timestamp"].diff().dt.total_seconds().fillna(60.0)
        feat_df["time_delta_service"] = time_deltas.reindex(df.index).clip(lower=0.0, upper=3600.0)

        # Rolling 60-second burst activity count properly aligned to df.index
        rolling_counts = sorted_df.groupby("service", group_keys=False).apply(
            lambda g: pd.Series(
                g.set_index("timestamp")["message"].rolling("60s").count().values,
                index=g.index
            ),
            include_groups=False
        )
        feat_df["rolling_burst_60s"] = rolling_counts.reindex(df.index).fillna(1.0).astype(float)
    else:
        feat_df["time_delta_service"] = 60.0
        feat_df["rolling_burst_60s"] = 1.0

    # Ensure no NaN or infinite values exist
    feat_df = feat_df.fillna(0.0)
    feat_df = feat_df.replace([np.inf, -np.inf], 0.0)

    feature_names = list(feat_df.columns)
    feature_matrix = feat_df.to_numpy(dtype=np.float32)

    return feature_matrix, feature_names, feat_df
