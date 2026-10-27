# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential needed for some scikit-learn / scientific wheels if not binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Default healthcheck targeting the FastAPI endpoint or Streamlit healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command launches the Streamlit interactive UI
CMD ["python", "run.py", "ui", "--host", "0.0.0.0", "--port", "8501"]
