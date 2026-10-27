@echo off
REM LogSleuth AI Windows Quick Launcher

echo =======================================================
echo          LogSleuth AI Launcher
echo =======================================================
echo 1. Launch Streamlit Interactive Dashboard (Port 8501)
echo 2. Launch FastAPI REST Server (Port 8000)
echo 3. Run Automated Pytest Test Suite
echo 4. Analyze Sample Dataset in Terminal
echo 5. Regenerate Synthetic Sample Dataset
echo 6. Exit
echo =======================================================
set /p choice="Enter choice [1-6]: "

if "%choice%"=="1" (
    echo Starting Streamlit Dashboard...
    python run.py ui
) else if "%choice%"=="2" (
    echo Starting FastAPI REST Server...
    python run.py api --reload
) else if "%choice%"=="3" (
    echo Running Tests...
    python run.py test
) else if "%choice%"=="4" (
    echo Analyzing sample logs...
    python run.py analyze data\sample_logs.csv
) else if "%choice%"=="5" (
    echo Generating sample dataset...
    python run.py generate-data
) else (
    echo Exiting.
)
pause
