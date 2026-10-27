"""
Dataset ingestion module for LogSleuth AI.
Supports CSV, JSON, TXT, and LOG file formats with flexible schema mapping and robust error handling.
"""

import io
import json
import re
from pathlib import Path

import pandas as pd


class IngestionError(Exception):
    """Raised when log ingestion fails due to format, size, or content issues."""


# Canonical standard column names
CANONICAL_COLUMNS = ["timestamp", "service", "level", "message"]

# Field alias mapping
FIELD_ALIASES: dict[str, list[str]] = {
    "timestamp": [
        "timestamp", "date", "datetime", "time", "ts", "logged_at",
        "event_time", "@timestamp", "created_at", "record_time"
    ],
    "service": [
        "service", "component", "service_name", "app", "application",
        "source", "host", "microservice", "module", "service_id"
    ],
    "level": [
        "level", "severity", "loglevel", "log_level", "type", "priority",
        "log_severity", "status_level"
    ],
    "message": [
        "message", "content", "msg", "log", "raw_message", "payload",
        "description", "text", "event", "log_message", "details"
    ]
}

# Supported file extensions
SUPPORTED_EXTENSIONS = {".csv", ".json", ".txt", ".log"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit


def detect_column_mappings(columns: list[str]) -> dict[str, str]:
    """
    Detects and maps source column names to canonical columns based on alias definitions.
    Returns a dictionary of {source_col: canonical_col}.
    """
    mapping = {}
    normalized_cols = {col: re.sub(r'[^a-zA-Z0-9_]', '', col.lower().strip()) for col in columns}

    for canonical, aliases in FIELD_ALIASES.items():
        matched_source = None
        for alias in aliases:
            norm_alias = re.sub(r'[^a-zA-Z0-9_]', '', alias.lower().strip())
            for orig_col, norm_col in normalized_cols.items():
                if norm_col == norm_alias and orig_col not in mapping:
                    matched_source = orig_col
                    break
            if matched_source:
                break
        if matched_source:
            mapping[matched_source] = canonical

    return mapping


def parse_unstructured_log_line(line: str) -> dict[str, str]:
    """
    Parses unstructured text log lines using common log line patterns.
    Falls back to populating the message field if parsing does not fully match.
    """
    line = line.strip()
    if not line:
        return {}

    # Common pattern 1: [2026-10-27 10:14:02] [ERROR] [payment-service]: Connection timed out
    m1 = re.match(
        r'^\[(?P<timestamp>[^\]]+)\]\s+\[(?P<level>[^\]]+)\]\s+\[(?P<service>[^\]]+)\]:\s*(?P<message>.*)$',
        line
    )
    if m1:
        return m1.groupdict()

    # Common pattern 2: 2026-10-27 10:14:02 - payment-service - ERROR - Connection timed out
    m2 = re.match(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*[-|]\s*(?P<service>[\w\-]+)\s*[-|]\s*(?P<level>[A-Za-z]+)\s*[-|:]\s*(?P<message>.*)$',
        line
    )
    if m2:
        return m2.groupdict()

    # Common pattern 3: 2026-10-27T10:14:02Z ERROR [order-service] Payment gateway unresponsive
    m3 = re.match(
        r'^(?P<timestamp>\S+)\s+(?P<level>[A-Z]{3,8})\s+(?:\[(?P<service>[\w\-]+)\]|\((?P<service_paren>[\w\-]+)\))\s+(?P<message>.*)$',
        line
    )
    if m3:
        d = m3.groupdict()
        if not d.get("service"):
            d["service"] = d.get("service_paren")
        d.pop("service_paren", None)
        return d

    # Fallback: Treat entire line as raw message
    return {"message": line}


def ingest_log_file(
    file_input: str | Path | io.BytesIO | io.StringIO | bytes,
    filename: str | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Ingests a log file from a path, buffer, or raw bytes into a pandas DataFrame.

    Returns:
        Tuple of (DataFrame, metadata_dict)
    Raises:
        IngestionError on invalid format, empty file, or oversized file.
    """
    # 1. Determine filename and file size
    file_bytes: bytes | None = None
    file_path: Path | None = None

    if isinstance(file_input, (str, Path)):
        file_path = Path(file_input)
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")
        if not filename:
            filename = file_path.name
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise IngestionError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB"
            )
        if file_size == 0:
            raise IngestionError("The uploaded log file is completely empty.")
        with open(file_path, "rb") as f:
            file_bytes = f.read()

    elif isinstance(file_input, bytes):
        file_bytes = file_input
        file_size = len(file_bytes)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise IngestionError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB"
            )
        if file_size == 0:
            raise IngestionError("The uploaded log file is completely empty.")

    elif isinstance(file_input, (io.BytesIO, io.StringIO)):
        content = file_input.getvalue()
        if isinstance(content, str):
            file_bytes = content.encode("utf-8")
        else:
            file_bytes = content
        file_size = len(file_bytes)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise IngestionError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB"
            )
        if file_size == 0:
            raise IngestionError("The uploaded log file is completely empty.")
    else:
        raise IngestionError("Unsupported input type for log ingestion.")

    # Validate file extension
    ext = Path(filename).suffix.lower() if filename else ".csv"
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # 2. Parse based on extension
    df: pd.DataFrame
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")

        elif ext == ".json":
            raw_text = file_bytes.decode("utf-8", errors="replace").strip()
            if not raw_text:
                raise IngestionError("JSON file contains no data.")
            try:
                # Try standard JSON list of objects
                data = json.loads(raw_text)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Could be { "logs": [...] }
                    for key in ["logs", "events", "records", "data"]:
                        if key in data and isinstance(data[key], list):
                            df = pd.DataFrame(data[key])
                            break
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise ValueError("JSON must be a list of objects or a valid log dictionary")
            except Exception:
                # Try JSON Lines (NDJSON)
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                parsed_lines = []
                for line in lines:
                    try:
                        parsed_lines.append(json.loads(line))
                    except Exception:
                        parsed_lines.append({"message": line})
                df = pd.DataFrame(parsed_lines)

        elif ext in {".txt", ".log"}:
            raw_text = file_bytes.decode("utf-8", errors="replace")
            lines = [l for l in raw_text.splitlines() if l.strip()]
            if not lines:
                raise IngestionError("Text log file contains no valid non-empty lines.")
            parsed_records = [parse_unstructured_log_line(l) for l in lines]
            df = pd.DataFrame(parsed_records)

        else:
            raise IngestionError(f"Unsupported format: {ext}")

    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Failed to parse {ext.upper()} file content: {e!s}")

    if df.empty:
        raise IngestionError("No log records could be extracted from the file.")

    metadata = {
        "filename": filename or "uploaded_logs",
        "file_format": ext,
        "file_size_bytes": file_size,
        "raw_record_count": len(df),
        "columns_detected": list(df.columns)
    }

    return df, metadata
