"""
Preprocessing module for LogSleuth AI.
Handles schema normalization, duplicate removal, robust timestamp parsing,
missing value imputation, severity categorization, and data-quality reporting.
"""

from typing import Any

import pandas as pd

from src.ingestion import detect_column_mappings

# Severity level canonical map
SEVERITY_MAP = {
    # INFO
    "info": "INFO",
    "information": "INFO",
    "notice": "INFO",
    "informational": "INFO",
    # WARNING
    "warn": "WARNING",
    "warning": "WARNING",
    "degraded": "WARNING",
    # ERROR
    "err": "ERROR",
    "error": "ERROR",
    "exception": "ERROR",
    "failed": "ERROR",
    "failure": "ERROR",
    # CRITICAL
    "crit": "CRITICAL",
    "critical": "CRITICAL",
    "fatal": "CRITICAL",
    "panic": "CRITICAL",
    "alert": "CRITICAL",
    "emergency": "CRITICAL",
    # DEBUG
    "debug": "DEBUG",
    "trace": "DEBUG",
    "verbose": "DEBUG"
}


def normalize_severity(val: Any) -> str:
    """Normalizes arbitrary severity strings or numeric codes into canonical levels."""
    if pd.isna(val) or val is None:
        return "INFO"
    s = str(val).strip().lower()
    return SEVERITY_MAP.get(s, "INFO" if not s else s.upper())


def preprocess_logs(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Cleans and standardizes a raw log DataFrame into canonical representation.

    Returns:
        Tuple of (clean_df, data_quality_report)
    """
    initial_count = len(df_raw)
    
    # 1. Deduplicate exact full-row duplicates
    # Repeated messages across different timestamps or IDs are preserved!
    df = df_raw.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df)

    # 2. Detect and map columns to canonical names
    mapping = detect_column_mappings(list(df.columns))
    df = df.rename(columns=mapping)

    # 3. Ensure canonical columns exist, impute missing columns if absent
    missing_imputations = {}

    # Preserve raw message before altering
    if "raw_message" not in df.columns:
        if "message" in df.columns:
            df["raw_message"] = df["message"].astype(str)
        else:
            # Construct from row representation
            df["raw_message"] = df.astype(str).agg(" | ".join, axis=1)

    # Message column
    if "message" not in df.columns:
        df["message"] = df["raw_message"]
        missing_imputations["message"] = len(df)
    else:
        nan_msgs = df["message"].isna().sum()
        if nan_msgs > 0:
            missing_imputations["message"] = int(nan_msgs)
            df["message"] = df["message"].fillna("").astype(str)
        else:
            df["message"] = df["message"].astype(str)

    # Service column
    if "service" not in df.columns:
        df["service"] = "unknown-service"
        missing_imputations["service"] = len(df)
    else:
        nan_services = df["service"].isna().sum()
        if nan_services > 0:
            missing_imputations["service"] = int(nan_services)
            df["service"] = df["service"].fillna("unknown-service").astype(str)
        else:
            df["service"] = df["service"].astype(str)

    # Normalize service names: lowercase, strip whitespace, replace spaces with hyphens
    df["service"] = df["service"].apply(lambda s: s.strip().lower().replace(" ", "-") if s else "unknown-service")

    # Severity level column
    if "level" not in df.columns:
        # Heuristic detection from message content if level column was absent
        df["level"] = df["message"].apply(
            lambda m: "CRITICAL" if any(w in m.lower() for w in ["fatal", "critical", "panic"])
            else "ERROR" if any(w in m.lower() for w in ["error", "exception", "failed", "failure"])
            else "WARNING" if any(w in m.lower() for w in ["warn", "warning", "timeout", "retry"])
            else "INFO"
        )
        missing_imputations["level"] = len(df)
    else:
        nan_levels = df["level"].isna().sum()
        if nan_levels > 0:
            missing_imputations["level"] = int(nan_levels)
        df["level"] = df["level"].apply(normalize_severity)

    # Timestamp column
    invalid_timestamps_count = 0
    if "timestamp" in df.columns:
        parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_ts_mask = parsed_ts.isna()
        invalid_timestamps_count = int(invalid_ts_mask.sum())

        if invalid_timestamps_count > 0:
            # Impute invalid timestamps using forward/backward fill or synthetic sequence
            now = pd.Timestamp.now()
            # If all are NaT
            if parsed_ts.notna().sum() == 0:
                # Generate synthetic 1-second spaced timestamps
                parsed_ts = pd.date_range(end=now, periods=len(df), freq="1s")
            else:
                parsed_ts = parsed_ts.ffill().bfill()
        df["timestamp"] = parsed_ts
    else:
        # Generate synthetic chronological timestamp sequence
        now = pd.Timestamp.now()
        df["timestamp"] = pd.date_range(end=now, periods=len(df), freq="1s")
        missing_imputations["timestamp"] = len(df)

    # Sort chronologically while maintaining stable order
    df = df.sort_values(by="timestamp", kind="mergesort").reset_index(drop=True)

    # Calculate date range
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    duration_secs = (max_ts - min_ts).total_seconds() if pd.notna(min_ts) and pd.notna(max_ts) else 0

    # Summary quality report
    service_counts = df["service"].value_counts().to_dict()
    severity_counts = df["level"].value_counts().to_dict()

    data_quality_report = {
        "raw_record_count": initial_count,
        "clean_record_count": len(df),
        "duplicates_removed": duplicates_removed,
        "missing_imputations": missing_imputations,
        "invalid_timestamps_count": invalid_timestamps_count,
        "time_range": {
            "start": str(min_ts) if pd.notna(min_ts) else "N/A",
            "end": str(max_ts) if pd.notna(max_ts) else "N/A",
            "duration_seconds": round(duration_secs, 2),
            "duration_human": f"{int(duration_secs // 60)}m {int(duration_secs % 60)}s" if duration_secs < 3600 else f"{duration_secs / 3600:.1f}h"
        },
        "services_count": len(service_counts),
        "service_distribution": service_counts,
        "severity_distribution": severity_counts,
        "data_health_score": round(max(0.0, 100.0 - (duplicates_removed * 2.0 + invalid_timestamps_count * 5.0 + sum(missing_imputations.values()) * 3.0) / max(1, initial_count) * 100.0), 1)
    }

    return df, data_quality_report
