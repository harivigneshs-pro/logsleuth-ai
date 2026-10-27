"""
FastAPI REST backend for LogSleuth AI.
Provides standalone REST endpoints for log ingestion, anomaly detection,
service dependency mapping, and root-cause candidate ranking.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.anomaly_detection import detect_anomalies
from src.ingestion import IngestionError, ingest_log_file
from src.preprocessing import preprocess_logs
from src.report_generator import generate_html_report, generate_json_summary
from src.root_cause import (
    build_dependency_graph,
    compute_service_metrics,
    rank_root_cause_candidates,
)

app = FastAPI(
    title="LogSleuth AI API",
    description="Intelligent Log Anomaly Detection and Root Cause Analysis for Cloud Microservices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_FILE_PATH = Path(__file__).resolve().parent / "data" / "sample_logs.csv"


def _execute_pipeline(
    file_bytes: bytes,
    filename: str,
    contamination: float = 0.08
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Internal runner returning both the structured JSON dictionary and intermediate dataframes."""
    # 1. Ingestion
    df_raw, ingestion_meta = ingest_log_file(file_bytes, filename=filename)

    # 2. Preprocessing & Data Quality
    clean_df, data_quality = preprocess_logs(df_raw)

    # 3. Anomaly Detection
    scored_df, anomaly_summary = detect_anomalies(clean_df, contamination=contamination)

    # 4. Dependency Graph & Root Cause Ranking
    services = list(clean_df["service"].unique())
    dep_graph = build_dependency_graph(discovered_services=services)
    service_metrics = compute_service_metrics(scored_df)
    candidates, root_cause_meta = rank_root_cause_candidates(scored_df, dependency_graph=dep_graph)

    # 5. Extract top suspicious events for quick API response
    suspicious_events = []
    if not scored_df.empty:
        anomalies_only = scored_df[scored_df["is_anomaly"] == True].sort_values(
            by="anomaly_score", ascending=False
        ).head(20)
        for _, row in anomalies_only.iterrows():
            suspicious_events.append({
                "timestamp": str(row.get("timestamp", "")),
                "service": str(row.get("service", "")),
                "level": str(row.get("level", "")),
                "anomaly_score": float(row.get("anomaly_score", 0.0)),
                "message": str(row.get("message", ""))
            })

    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "ingestion_metadata": ingestion_meta,
        "data_quality_report": data_quality,
        "anomaly_detection_summary": anomaly_summary,
        "root_cause_ranking": {
            "primary_suspect": candidates[0]["service"] if candidates else None,
            "disclaimer": root_cause_meta.get("disclaimer"),
            "candidates": candidates
        },
        "service_metrics": service_metrics,
        "top_suspicious_events": suspicious_events
    }

    return result, scored_df, data_quality, anomaly_summary, candidates


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    contamination: float = 0.08
) -> dict[str, Any]:
    """Helper executing the complete LogSleuth analysis pipeline on raw bytes and returning JSON result."""
    result, _, _, _, _ = _execute_pipeline(file_bytes, filename=filename, contamination=contamination)
    return result


@app.get("/", tags=["General"])
def root() -> dict[str, Any]:
    """Returns project status, overview, and API capabilities."""
    return {
        "project": "LogSleuth AI",
        "tagline": "Intelligent Log Anomaly Detection and Root Cause Analysis for Cloud Microservices",
        "status": "operational",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "GET /": "Project status and metadata",
            "GET /health": "Liveness and readiness health check",
            "GET /sample": "Run analysis on bundled synthetic microservice sample dataset",
            "GET /sample/report": "Generate standalone HTML incident report for sample dataset",
            "POST /analyze": "Upload CSV/JSON/TXT/LOG file for automated anomaly and root cause analysis",
            "POST /analyze/report": "Upload log file and receive a styled standalone HTML incident report",
            "GET /metrics": "Summary telemetry and service degradation health metrics"
        }
    }


@app.get("/health", tags=["General"])
def health_check() -> dict[str, Any]:
    """Health check endpoint for container probes and monitoring."""
    return {
        "status": "healthy",
        "service": "LogSleuth AI",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", tags=["General"])
def metrics() -> dict[str, Any]:
    """Returns high-level health and operational metrics for monitoring integrations."""
    if SAMPLE_FILE_PATH.exists():
        with open(SAMPLE_FILE_PATH, "rb") as f:
            file_bytes = f.read()
        res = run_pipeline(file_bytes, filename="sample_logs.csv")
        return {
            "status": "healthy",
            "dataset": "sample_logs.csv",
            "total_records": res["data_quality_report"].get("clean_record_count", 0),
            "anomaly_count": res["anomaly_detection_summary"].get("anomaly_count", 0),
            "anomaly_rate_pct": res["anomaly_detection_summary"].get("anomaly_percentage", 0.0),
            "primary_suspect": res["root_cause_ranking"].get("primary_suspect")
        }
    return {"status": "healthy", "service": "LogSleuth AI"}


@app.post("/analyze", tags=["Analysis"])
async def analyze_logs(
    file: UploadFile = File(..., description="Log file in CSV, JSON, TXT, or LOG format"),
    contamination: float = Query(
        0.08, ge=0.01, le=0.45, description="Expected proportion of anomalies (Isolation Forest contamination parameter)"
    )
) -> dict[str, Any]:
    """
    Accepts an uploaded log file and executes full data ingestion, preprocessing,
    Isolation Forest anomaly detection, and heuristic root-cause ranking.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty. Please provide a non-empty log file."
            )

        result = run_pipeline(content, filename=file.filename or "upload.csv", contamination=contamination)
        return result

    except HTTPException:
        raise
    except IngestionError as ie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Log ingestion error: {ie!s}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {e!s}"
        )


@app.post("/analyze/report", response_class=HTMLResponse, tags=["Analysis"])
async def analyze_logs_and_render_report(
    file: UploadFile = File(..., description="Log file in CSV, JSON, TXT, or LOG format"),
    contamination: float = Query(
        0.08, ge=0.01, le=0.45, description="Expected proportion of anomalies"
    )
) -> HTMLResponse:
    """
    Uploads a log file and returns a standalone, publication-ready HTML Incident Report.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty. Please provide a non-empty log file."
            )

        _, scored_df, data_quality, anomaly_summary, candidates = _execute_pipeline(
            content, filename=file.filename or "upload.csv", contamination=contamination
        )
        html_content = generate_html_report(scored_df, data_quality, anomaly_summary, candidates)
        return HTMLResponse(content=html_content, status_code=200)

    except HTTPException:
        raise
    except IngestionError as ie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Log ingestion error: {ie!s}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {e!s}"
        )


@app.get("/sample", tags=["Analysis"])
def analyze_sample_dataset(
    contamination: float = Query(
        0.08, ge=0.01, le=0.45, description="Contamination factor for Isolation Forest"
    )
) -> dict[str, Any]:
    """Runs the analysis pipeline on the bundled sample microservice dataset."""
    if not SAMPLE_FILE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample dataset not found at {SAMPLE_FILE_PATH}"
        )

    try:
        with open(SAMPLE_FILE_PATH, "rb") as f:
            file_bytes = f.read()

        return run_pipeline(file_bytes, filename="sample_logs.csv", contamination=contamination)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process sample dataset: {e!s}"
        )


@app.get("/sample/report", response_class=HTMLResponse, tags=["Analysis"])
def analyze_sample_and_render_report(
    contamination: float = Query(
        0.08, ge=0.01, le=0.45, description="Contamination factor for Isolation Forest"
    )
) -> HTMLResponse:
    """Returns a standalone, styled HTML incident report for the bundled sample logs dataset."""
    if not SAMPLE_FILE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample dataset not found at {SAMPLE_FILE_PATH}"
        )

    try:
        with open(SAMPLE_FILE_PATH, "rb") as f:
            file_bytes = f.read()

        _, scored_df, data_quality, anomaly_summary, candidates = _execute_pipeline(
            file_bytes, filename="sample_logs.csv", contamination=contamination
        )
        html_content = generate_html_report(scored_df, data_quality, anomaly_summary, candidates)
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sample report: {e!s}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
