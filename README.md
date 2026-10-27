# LogSleuth AI: Intelligent Log Anomaly Detection and Root Cause Analysis for Cloud Microservices

LogSleuth AI is an end-to-end cloud-native microservice observability platform designed to automatically detect anomalies, isolate outlier signals, and prioritize incident root causes without requiring paid external APIs, proprietary models, or heavy database infrastructure.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw Log Ingestion (CSV/JSON/TXT/LOG)  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Schema Normalization & Preprocessing  |
                      | (Aliases, Dedup, Timestamps, Health)  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Feature Extraction Engine        |
                      | (Keywords, Severities, Windows, Burst)|
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Isolation Forest Anomaly Detection    |
                      |  (Tree Path Outlier Scoring: [0, 1])  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | NetworkX Microservice Topology Graph  |
                      | & Multi-Factor Heuristic Root Cause   |
                      +---------------------------------------+
                                  /               \
                                 v                 v
                 +----------------------+  +---------------------+
                 | Streamlit Dashboard  |  | FastAPI REST Server |
                 | (Port 8501)          |  | (Port 8000)         |
                 +----------------------+  +---------------------+
```

---

## Key Features

1. **Multi-Format Ingestion**: Ingest CSV, JSON (array or JSON-Lines), TXT, and LOG files with automatic alias recognition (`Date`, `Component`, `Content`, `Severity`, `LogLevel`, `message`).
2. **Data Quality & Hygiene**: Deduplication, robust timestamp parsing with automatic fallback, missing value imputation, and a real-time data health scoring metric.
3. **Isolation Forest Machine Learning**: Unsupervised outlier detection returning continuous normalized anomaly scores where **higher scores indicate more anomalous records**. Edge-case safe on small or empty datasets.
4. **Transparent Root Cause Heuristic**: Combines error/timeout concentration (35%), anomaly severity (30%), temporal onset precedence (20%), and NetworkX dependency graph topology (15%) with transparent evidence narratives.
5. **Interactive Streamlit UI**: 8 dedicated tabs covering Overview, Quality, Anomaly Details, Interactive Topology Graph, Service Health Profiles, Ask LogSleuth Natural Language Q&A, and Incident Report Generation.
6. **FastAPI REST API**: Fully independent OpenAPI-compliant server providing `/health`, `/sample`, and `/analyze` endpoints.
7. **Zero Cloud Dependencies**: Runs entirely on local CPU using standard scientific Python libraries.

---

## Technology Stack

- **Language:** Python 3.10+
- **Machine Learning:** `scikit-learn` (IsolationForest)
- **Graph & Topology:** `networkx`
- **Data Processing:** `pandas`, `numpy`
- **User Interface:** `streamlit`
- **Visualizations:** `plotly`
- **API Framework:** `fastapi`, `uvicorn`
- **Testing:** `pytest`, `httpx`

---

## Directory Structure

```
LogSleuth-AI/
├── .gitignore
├── .streamlit/
│   └── config.toml             # Custom modern navy/blue dark UI theme
├── api.py                      # Standalone FastAPI REST application
├── app.py                      # Interactive Streamlit dashboard
├── data/
│   ├── generate_sample_logs.py # Reproducible synthetic incident log generator
│   └── sample_logs.csv         # 260+ synthetic microservice incident logs
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── src/
│   ├── __init__.py
│   ├── anomaly_detection.py    # Isolation Forest model & score normalization
│   ├── feature_extraction.py   # Keyword, severity, and temporal feature builder
│   ├── ingestion.py            # CSV/JSON/TXT/LOG parsing & schema mapper
│   ├── preprocessing.py        # Normalization, dedup, and data health audits
│   ├── report_generator.py     # HTML, JSON, and CSV report generator
│   └── root_cause.py           # NetworkX dependency graph & heuristic ranking
└── tests/
    └── test_pipeline.py        # Complete 20-test automated test suite
```

---

## Setup & Installation

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Unified Operational Runner (`run.py`)

LogSleuth AI includes a unified CLI runner for all operational tasks:

```powershell
# Launch Streamlit interactive dashboard
python run.py ui

# Launch FastAPI REST backend with auto-reload
python run.py api --reload

# Run complete automated test suite
python run.py test

# Analyze a log file directly from the terminal
python run.py analyze data/sample_logs.csv --contamination 0.08

# Regenerate synthetic incident sample dataset
python run.py generate-data --records 260
```

### Quick Launch Scripts
- **Windows**: Double-click or run `start.bat` for an interactive menu.
- **Linux / macOS**: Run `./start.sh` for an interactive shell menu.

---

## Docker & Container Deployment

LogSleuth AI is fully containerized and can run locally or in production via Docker:

### 1. Run with Docker Compose (Spins up both Dashboard & API)
```bash
docker compose up --build
```
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI REST API**: [http://localhost:8000](http://localhost:8000)
- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Build and Run Standalone Docker Container
```bash
# Build image
docker build -t logsleuth-ai .

# Run Streamlit UI
docker run -p 8501:8501 logsleuth-ai

# Or run FastAPI REST Server
docker run -p 8000:8000 logsleuth-ai python run.py api --host 0.0.0.0 --port 8000
```

---

## REST API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Operational status, project version, and API capability catalog |
| `GET` | `/health` | Container liveness probe & health status |
| `GET` | `/metrics` | High-level system telemetry & operational degradation metrics |
| `GET` | `/sample` | Runs pipeline on bundled synthetic logs and returns structured JSON |
| `GET` | `/sample/report` | Returns a standalone, publication-ready **HTML Incident Report** |
| `POST` | `/analyze` | Accepts log upload (CSV/JSON/LOG/TXT) and returns JSON results |
| `POST` | `/analyze/report` | Accepts log upload and renders a standalone **HTML Incident Report** |

---

## Supported Log Formats and Schema Mapping

LogSleuth AI dynamically parses structured files and unstructured log lines into four standard fields:

| Canonical Field | Recognized Aliases | Description |
| :--- | :--- | :--- |
| `timestamp` | `date`, `datetime`, `time`, `ts`, `logged_at`, `@timestamp` | Chronological record timestamp |
| `service` | `component`, `service_name`, `app`, `application`, `source`, `host` | Name of the microservice |
| `level` | `severity`, `loglevel`, `log_level`, `type`, `priority` | Standardized to INFO, WARNING, ERROR, CRITICAL, DEBUG |
| `message` | `content`, `msg`, `log`, `raw_message`, `payload`, `description` | Full event log payload |

---

## Understanding Anomaly Scores

- **Score Nature**: The anomaly score is a continuous value between `0.0` and `1.0` computed from the average tree path-length in the Isolation Forest.
- **Directionality**: **Higher scores indicate more anomalous records** (1.0 represents extreme deviation from the nominal baseline, while 0.0 is completely normal).
- **Non-Probabilistic**: Anomaly scores are isolation path-length distance metrics, **not probabilities**. They should not be interpreted as likelihood percentages.

---

## Root-Cause Ranking Methodology & Limitations

LogSleuth AI computes a candidate ranking using a 4-factor heuristic:
1. **Failure Intensity (35% weight)**: Concentration of errors, critical failures, and timeouts.
2. **Anomaly Severity (30% weight)**: Peak and average Isolation Forest anomaly scores.
3. **Temporal Precedence (20% weight)**: Timing of earliest anomaly onset (initiating services typically exhibit degradation earlier).
4. **Topology Impact (15% weight)**: In-degree and downstream caller impact within the NetworkX dependency graph.

### Important Limitation & Disclaimer
> **Notice:** This heuristic ranking prioritizes investigation targets based on anomaly density, temporal precedence, and dependency depth. It is **not definitive mathematical causal proof**. In production distributed architectures, exact deterministic causality requires distributed tracing spans (e.g. OpenTelemetry trace IDs and span contexts).

---

## Running the Automated Test Suite

Execute Pytest to validate ingestion, preprocessing, models, heuristics, and API endpoints:
```powershell
python run.py test
# or
python -m pytest -v
```

All 23 tests validate:
- CSV, JSON array, JSON Lines, and unstructured text parsing
- Schema alias normalization and duplicate removal
- Invalid timestamps and missing value imputation
- Feature matrix integrity (no NaN or inf values)
- Isolation Forest anomaly scoring and threshold cut-offs
- Edge cases: empty files and datasets under 5 records
- Root cause ranking and evidence synthesis
- Standalone HTML incident report generation
- FastAPI REST endpoints (`/`, `/health`, `/metrics`, `/sample`, `/sample/report`, `/analyze`, `/analyze/report`)
#   l o g s l e u t h - a i  
 