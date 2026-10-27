#!/usr/bin/env bash
# LogSleuth AI Unix/Linux/macOS Quick Launcher

set -e

echo "======================================================="
echo "         LogSleuth AI Quick Launcher"
echo "======================================================="
echo "1) Launch Streamlit Interactive Dashboard (Port 8501)"
echo "2) Launch FastAPI REST Server (Port 8000)"
echo "3) Run Automated Pytest Test Suite"
echo "4) Analyze Sample Dataset in Terminal"
echo "5) Regenerate Synthetic Sample Dataset"
echo "6) Exit"
echo "======================================================="
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "Starting Streamlit Dashboard..."
        python run.py ui
        ;;
    2)
        echo "Starting FastAPI REST Server..."
        python run.py api --reload
        ;;
    3)
        echo "Running Tests..."
        python run.py test
        ;;
    4)
        echo "Analyzing sample logs..."
        python run.py analyze data/sample_logs.csv
        ;;
    5)
        echo "Generating sample dataset..."
        python run.py generate-data
        ;;
    *)
        echo "Exiting."
        ;;
esac
