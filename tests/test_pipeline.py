"""
Comprehensive pytest test suite for LogSleuth AI pipeline and API.
Tests data ingestion, preprocessing, feature extraction, anomaly detection,
root-cause heuristic ranking, edge cases, and REST endpoints.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from src.anomaly_detection import detect_anomalies
from src.feature_extraction import extract_features
from src.ingestion import IngestionError, ingest_log_file
from src.preprocessing import preprocess_logs
from src.report_generator import generate_html_report
from src.root_cause import (
    rank_root_cause_candidates,
)


@pytest.fixture
def client():
    return TestClient(app)


# 1. Ingestion Tests
def test_csv_ingestion():
    csv_data = (
        b"Date,Component,Severity,Content\n"
        b"2026-10-27 10:00:00,auth-service,INFO,User authenticated successfully\n"
        b"2026-10-27 10:00:05,order-service,ERROR,Database connection failed\n"
    )
    df, meta = ingest_log_file(csv_data, filename="test.csv")
    assert len(df) == 2
    assert "Component" in df.columns or "service" in df.columns
    assert meta["file_format"] == ".csv"


def test_json_array_ingestion():
    data = [
        {"timestamp": "2026-10-27 10:00:00", "service": "payment-service", "level": "INFO", "message": "Authorized"},
        {"timestamp": "2026-10-27 10:00:01", "service": "payment-service", "level": "ERROR", "message": "Timeout"}
    ]
    json_bytes = json.dumps(data).encode("utf-8")
    df, meta = ingest_log_file(json_bytes, filename="test.json")
    assert len(df) == 2
    assert meta["file_format"] == ".json"


def test_json_lines_ingestion():
    ndjson = (
        b'{"timestamp": "2026-10-27 10:00:00", "service": "auth-service", "level": "INFO", "message": "Token OK"}\n'
        b'{"timestamp": "2026-10-27 10:00:01", "service": "auth-service", "level": "WARNING", "message": "Slow token"}\n'
    )
    df, meta = ingest_log_file(ndjson, filename="test.json")
    assert len(df) == 2


def test_text_log_ingestion():
    text_data = (
        b"[2026-10-27 10:14:02] [ERROR] [payment-service]: Connection timed out after 5000ms\n"
        b"[2026-10-27 10:14:03] [INFO] [order-service]: Order created id=4592\n"
    )
    df, meta = ingest_log_file(text_data, filename="app.log")
    assert len(df) == 2
    assert meta["file_format"] == ".log"
    assert "payment-service" in df["service"].values or "order-service" in df["service"].values


def test_empty_file_ingestion():
    with pytest.raises(IngestionError, match="empty"):
        ingest_log_file(b"", filename="empty.csv")


def test_unsupported_file_format():
    with pytest.raises(IngestionError, match="Unsupported file format"):
        ingest_log_file(b"some content", filename="test.pdf")


# 2. Preprocessing & Column Normalization Tests
def test_preprocessing_column_normalization():
    df_raw = pd.DataFrame({
        "Date": ["2026-10-27 10:00:00", "2026-10-27 10:01:00"],
        "Component": ["auth-service", "order-service"],
        "Severity": ["WARN", "ERR"],
        "Content": ["High latency", "Unhandled exception in thread"]
    })
    clean_df, report = preprocess_logs(df_raw)
    assert "timestamp" in clean_df.columns
    assert "service" in clean_df.columns
    assert "level" in clean_df.columns
    assert "message" in clean_df.columns
    assert clean_df.iloc[0]["level"] == "WARNING"
    assert clean_df.iloc[1]["level"] == "ERROR"
    assert report["clean_record_count"] == 2


def test_preprocessing_deduplication():
    df_raw = pd.DataFrame({
        "timestamp": ["2026-10-27 10:00:00", "2026-10-27 10:00:00", "2026-10-27 10:05:00"],
        "service": ["svc-a", "svc-a", "svc-a"],
        "level": ["INFO", "INFO", "INFO"],
        "message": ["Msg A", "Msg A", "Msg A"]
    })
    # Notice that rows 1 & 2 are exact duplicates, while row 3 has a different timestamp
    clean_df, report = preprocess_logs(df_raw)
    assert report["duplicates_removed"] == 1
    assert len(clean_df) == 2  # repeated message with different timestamp preserved!


def test_preprocessing_missing_and_invalid_values():
    df_raw = pd.DataFrame({
        "timestamp": ["invalid-date-string", None],
        "service": [None, "payment-service"],
        "level": [None, "CRIT"],
        "message": [None, "Fatal database error"]
    })
    clean_df, report = preprocess_logs(df_raw)
    assert clean_df.iloc[0]["service"] == "unknown-service"
    assert clean_df.iloc[0]["level"] == "INFO"
    assert clean_df.iloc[1]["level"] == "CRITICAL"
    assert pd.api.types.is_datetime64_any_dtype(clean_df["timestamp"])
    assert report["invalid_timestamps_count"] >= 1


# 3. Feature Extraction Tests
def test_feature_extraction():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-27 10:00:00", periods=5, freq="10s"),
        "service": ["auth-service", "payment-service", "order-service", "payment-service", "order-service"],
        "level": ["INFO", "ERROR", "WARNING", "CRITICAL", "INFO"],
        "message": [
            "User login success",
            "Connection timed out waiting for database pool",
            "High latency warning during payment check",
            "Fatal panic out of memory error",
            "Order processed normally"
        ]
    })
    X, names, feat_df = extract_features(df)
    assert X.shape[0] == 5
    assert X.shape[1] > 5
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
    assert "has_error_kw" in names
    assert "has_timeout_kw" in names
    assert "msg_length" in names


# 4. Anomaly Detection Tests
def test_anomaly_detection_basic():
    # Construct 50 normal logs and 5 anomalous logs
    normal_msgs = ["User logged in", "Cache hit item", "Health check OK", "Record saved"]
    data = []
    ts = pd.date_range("2026-10-27 10:00:00", periods=60, freq="2s")
    for i in range(55):
        data.append({
            "timestamp": ts[i],
            "service": "order-service",
            "level": "INFO",
            "message": normal_msgs[i % len(normal_msgs)]
        })
    for i in range(55, 60):
        data.append({
            "timestamp": ts[i],
            "service": "payment-service",
            "level": "CRITICAL",
            "message": "CRITICAL: Connection timed out after 30000ms HikariPool exhausted exception"
        })
    df = pd.DataFrame(data)
    scored_df, summary = detect_anomalies(df, contamination=0.08, random_state=42)

    assert "is_anomaly" in scored_df.columns
    assert "anomaly_score" in scored_df.columns
    assert summary["anomaly_count"] > 0
    # Higher scores indicate more anomalous records:
    crit_scores = scored_df[scored_df["level"] == "CRITICAL"]["anomaly_score"].mean()
    info_scores = scored_df[scored_df["level"] == "INFO"]["anomaly_score"].mean()
    assert crit_scores > info_scores


def test_anomaly_detection_empty_dataset():
    empty_df = pd.DataFrame(columns=["timestamp", "service", "level", "message"])
    scored_df, summary = detect_anomalies(empty_df)
    assert len(scored_df) == 0
    assert summary["anomaly_count"] == 0
    assert summary["status"] == "empty_dataset"


def test_anomaly_detection_very_small_dataset():
    small_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-27 10:00:00", periods=3, freq="1s"),
        "service": ["auth-service", "payment-service", "order-service"],
        "level": ["INFO", "ERROR", "INFO"],
        "message": ["Login ok", "Connection timeout error", "Order ok"]
    })
    scored_df, summary = detect_anomalies(small_df)
    assert len(scored_df) == 3
    assert scored_df.iloc[1]["is_anomaly"] == True


# 5. Root Cause Ranking Tests
def test_root_cause_ranking():
    data = []
    ts = pd.date_range("2026-10-27 10:00:00", periods=20, freq="5s")
    for i in range(12):
        data.append({
            "timestamp": ts[i],
            "service": "auth-service",
            "level": "INFO",
            "message": "Regular token check",
            "is_anomaly": False,
            "anomaly_score": 0.1
        })
    for i in range(12, 18):
        data.append({
            "timestamp": ts[i],
            "service": "payment-service",
            "level": "CRITICAL",
            "message": "HikariPool database timeout connection failure",
            "is_anomaly": True,
            "anomaly_score": 0.95
        })
    for i in range(18, 20):
        data.append({
            "timestamp": ts[i],
            "service": "order-service",
            "level": "WARNING",
            "message": "Retrying payment call timeout",
            "is_anomaly": True,
            "anomaly_score": 0.70
        })

    df = pd.DataFrame(data)
    candidates, meta = rank_root_cause_candidates(df)

    assert len(candidates) >= 3
    # Payment-service should be ranked as top root cause candidate due to errors, timeouts, and highest anomaly score
    assert candidates[0]["service"] == "payment-service"
    assert "evidence" in candidates[0]
    assert len(candidates[0]["evidence"]) > 0
    assert "disclaimer" in meta


# 6. Report Generation Tests
def test_report_generation():
    df = pd.DataFrame([{
        "timestamp": "2026-10-27 10:00:00",
        "service": "payment-service",
        "level": "ERROR",
        "message": "Connection reset",
        "is_anomaly": True,
        "anomaly_score": 0.92
    }])
    html = generate_html_report(df, {"clean_record_count": 1}, {"anomaly_count": 1, "anomaly_percentage": 100}, [{"service": "payment-service", "root_cause_score": 85}])
    assert "LogSleuth AI Incident Report" in html
    assert "payment-service" in html


# 7. FastAPI Endpoints Tests
def test_api_root(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "LogSleuth AI"
    assert data["status"] == "operational"


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_api_sample_endpoint(client):
    res = client.get("/sample?contamination=0.08")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "anomaly_detection_summary" in data
    assert "root_cause_ranking" in data
    assert data["root_cause_ranking"]["primary_suspect"] == "payment-service"


def test_api_analyze_upload(client):
    csv_bytes = (
        b"timestamp,service,level,message\n"
        b"2026-10-27 10:00:00,auth-service,INFO,Token valid\n"
        b"2026-10-27 10:00:01,auth-service,INFO,User logged in\n"
        b"2026-10-27 10:00:02,order-service,ERROR,Checkout timeout failed\n"
    )
    files = {"file": ("test_logs.csv", csv_bytes, "text/csv")}
    res = client.post("/analyze?contamination=0.1", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["data_quality_report"]["clean_record_count"] == 3


def test_api_empty_file_upload(client):
    files = {"file": ("empty.csv", b"", "text/csv")}
    res = client.post("/analyze", files=files)
    assert res.status_code == 400


def test_api_sample_report(client):
    res = client.get("/sample/report?contamination=0.08")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "LogSleuth AI - Incident & Root Cause Report" in res.text
    assert "payment-service" in res.text


def test_api_analyze_report(client):
    csv_bytes = (
        b"timestamp,service,level,message\n"
        b"2026-10-27 10:00:00,auth-service,INFO,Token valid\n"
        b"2026-10-27 10:00:01,auth-service,INFO,User logged in\n"
        b"2026-10-27 10:00:02,order-service,ERROR,Checkout timeout failed\n"
    )
    files = {"file": ("test_logs.csv", csv_bytes, "text/csv")}
    res = client.post("/analyze/report?contamination=0.1", files=files)
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "LogSleuth AI - Incident & Root Cause Report" in res.text


def test_api_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "anomaly_count" in data

