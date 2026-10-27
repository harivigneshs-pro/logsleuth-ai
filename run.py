"""
LogSleuth AI - Unified CLI & Operational Runner.

Provides a unified interface for launching:
  - Streamlit Dashboard (python run.py ui)
  - FastAPI REST Server (python run.py api)
  - Test Suite (python run.py test)
  - Synthetic Data Generator (python run.py generate-data)
  - Headless CLI Log Analysis (python run.py analyze <filepath>)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_ui(port: int = 8501, host: str = "0.0.0.0"):
    """Launch the Streamlit interactive dashboard."""
    app_path = Path(__file__).resolve().parent / "app.py"
    print(f"[LogSleuth AI] Starting Streamlit dashboard on http://{host}:{port}...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        f"--server.address={host}"
    ]
    subprocess.run(cmd)


def run_api(port: int = 8000, host: str = "0.0.0.0", reload: bool = False):
    """Launch the FastAPI REST API server."""
    print(f"[LogSleuth AI] Starting FastAPI REST server on http://{host}:{port}...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        f"--host={host}",
        f"--port={port}"
    ]
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd)


def run_tests():
    """Run automated Pytest test suite."""
    print("[LogSleuth AI] Running complete test suite...")
    cmd = [sys.executable, "-m", "pytest", "-v"]
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


def run_generate_data(records: int = 260):
    """Generate reproducible sample microservice incident logs."""
    from data.generate_sample_logs import generate_sample_logs
    output_path = Path(__file__).resolve().parent / "data" / "sample_logs.csv"
    print(f"[LogSleuth AI] Generating {records} synthetic microservice log records...")
    df = generate_sample_logs(num_records=records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[LogSleuth AI] Successfully saved sample dataset to {output_path} ({len(df)} rows)")


def run_cli_analysis(filepath: str, contamination: float = 0.08):
    """Run headless command-line analysis on a local log file."""
    p = Path(filepath)
    if not p.exists():
        print(f"[LogSleuth AI] Error: File not found at {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"[LogSleuth AI] Ingesting and analyzing {filepath} (contamination={contamination})...")
    from api import run_pipeline

    with open(p, "rb") as f:
        file_bytes = f.read()

    res = run_pipeline(file_bytes, filename=p.name, contamination=contamination)
    dq = res["data_quality_report"]
    summary = res["anomaly_detection_summary"]
    rc = res["root_cause_ranking"]

    print("\n" + "=" * 65)
    print("           LOGSLEUTH AI - CLI ANALYSIS REPORT            ")
    print("=" * 65)
    print(f"File Ingested     : {p.name}")
    print(f"Clean Records     : {dq.get('clean_record_count', 0)}")
    print(f"Duplicates Dropped: {dq.get('duplicates_removed', 0)}")
    print(f"Data Health Score : {dq.get('data_health_score', 0)}%")
    print(f"Flagged Anomalies : {summary.get('anomaly_count', 0)} ({summary.get('anomaly_percentage', 0)}%)")
    print("-" * 65)
    print(f"PRIMARY SUSPECT   : {rc.get('primary_suspect', 'None')}")
    print("-" * 65)
    print("Top Root Cause Candidates:")
    for c in rc.get("candidates", [])[:5]:
        print(f"  #{c.get('rank')} {c.get('service'):<24} Score: {c.get('root_cause_score'):>3}/100 "
              f"| Anomalies: {c.get('anomaly_count'):>3} | Errors: {c.get('error_count'):>3} | Timeouts: {c.get('timeout_count'):>3}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LogSleuth AI - Intelligent Microservice Log Anomaly Detection & Root Cause Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational mode to run")

    # UI Command
    ui_parser = subparsers.add_parser("ui", help="Launch Streamlit dashboard")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port to bind dashboard (default: 8501)")
    ui_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")

    # API Command
    api_parser = subparsers.add_parser("api", help="Launch FastAPI REST server")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind API (default: 8000)")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    # Test Command
    subparsers.add_parser("test", help="Run automated Pytest test suite")

    # Generate Data Command
    data_parser = subparsers.add_parser("generate-data", help="Generate synthetic microservice incident logs")
    data_parser.add_argument("--records", type=int, default=260, help="Number of records to generate")

    # Analyze CLI Command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze log file directly in terminal")
    analyze_parser.add_argument("file", type=str, help="Path to log file (CSV, JSON, TXT, LOG)")
    analyze_parser.add_argument("--contamination", type=float, default=0.08, help="Isolation Forest contamination factor")

    args = parser.parse_args()

    if args.command == "ui":
        run_ui(port=args.port, host=args.host)
    elif args.command == "api":
        run_api(port=args.port, host=args.host, reload=args.reload)
    elif args.command == "test":
        run_tests()
    elif args.command == "generate-data":
        run_generate_data(records=args.records)
    elif args.command == "analyze":
        run_cli_analysis(filepath=args.file, contamination=args.contamination)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
