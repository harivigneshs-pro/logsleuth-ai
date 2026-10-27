"""
Incident report generator module for LogSleuth AI.
Generates CSV exports, structured JSON summaries, and polished standalone HTML incident reports.
"""

import json
from datetime import datetime
from typing import Any

import pandas as pd


def generate_json_summary(
    data_quality: dict[str, Any],
    anomaly_summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    service_metrics: dict[str, Any]
) -> str:
    """Generates a complete JSON serializable string summarizing the entire log analysis."""
    report = {
        "report_metadata": {
            "title": "LogSleuth AI Incident Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0"
        },
        "data_quality_summary": data_quality,
        "anomaly_detection_summary": anomaly_summary,
        "root_cause_analysis": {
            "primary_suspect": candidates[0]["service"] if candidates else None,
            "disclaimer": (
                "Heuristic ranking based on anomaly density, temporal sequence, and service topology. "
                "Not definitive causal proof."
            ),
            "ranked_candidates": candidates
        },
        "service_health_metrics": service_metrics
    }
    return json.dumps(report, indent=2, default=str)


def generate_html_report(
    scored_df: pd.DataFrame,
    data_quality: dict[str, Any],
    anomaly_summary: dict[str, Any],
    candidates: list[dict[str, Any]]
) -> str:
    """
    Generates a self-contained, beautifully styled HTML incident report
    ready for viewing or printing.
    """
    primary_suspect = candidates[0]["service"] if candidates else "None Identified"
    primary_score = candidates[0]["root_cause_score"] if candidates else 0

    # Top suspicious logs
    suspicious_df = scored_df[scored_df["is_anomaly"] == True].sort_values(
        by="anomaly_score", ascending=False
    ).head(15) if not scored_df.empty else pd.DataFrame()

    candidate_rows = ""
    for c in candidates:
        evidence_list = "".join(f"<li>{ev}</li>" for ev in c.get("evidence", []))
        candidate_rows += f"""
        <tr>
            <td style="font-weight: 600; text-align: center;">#{c.get('rank', '-')}</td>
            <td style="font-weight: 600; color: #1E40AF;">{c.get('service', 'Unknown')}</td>
            <td style="text-align: center;"><span class="badge {'badge-danger' if c.get('root_cause_score', 0) > 60 else 'badge-warning' if c.get('root_cause_score', 0) > 30 else 'badge-info'}">{c.get('root_cause_score', 0)} / 100</span></td>
            <td style="text-align: center;">{c.get('anomaly_count', 0)} ({c.get('anomaly_rate_pct', 0)}%)</td>
            <td style="text-align: center;">{c.get('error_count', 0)}</td>
            <td style="text-align: center;">{c.get('timeout_count', 0)}</td>
            <td style="font-size: 13px;">{c.get('earliest_anomaly_ts', 'N/A')}</td>
            <td><ul style="margin: 0; padding-left: 18px; font-size: 13px;">{evidence_list}</ul></td>
        </tr>
        """

    log_rows = ""
    for _, row in suspicious_df.iterrows():
        level = str(row.get("level", "INFO"))
        level_class = "badge-danger" if level in ["ERROR", "CRITICAL"] else "badge-warning" if level == "WARNING" else "badge-info"
        log_rows += f"""
        <tr>
            <td style="white-space: nowrap; font-size: 13px;">{row.get('timestamp', '')}</td>
            <td style="font-weight: 600;">{row.get('service', '')}</td>
            <td><span class="badge {level_class}">{level}</span></td>
            <td style="text-align: center; font-weight: 600; color: #DC2626;">{row.get('anomaly_score', 0.0):.4f}</td>
            <td style="font-family: monospace; font-size: 12px; color: #1E293B;">{row.get('message', '')!s}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LogSleuth AI - Incident & Root Cause Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #F8FAFC;
            color: #0F172A;
            margin: 0;
            padding: 30px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            border: 1px solid #E2E8F0;
        }}
        .header {{
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 24px;
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            color: #1E3A8A;
        }}
        .header p {{
            margin: 4px 0 0 0;
            color: #64748B;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #F1F5F9;
            padding: 18px;
            border-radius: 8px;
            border-left: 4px solid #3B82F6;
        }}
        .metric-card.alert {{
            border-left-color: #EF4444;
            background: #FEF2F2;
        }}
        .metric-card.success {{
            border-left-color: #10B981;
            background: #ECFDF5;
        }}
        .metric-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
            color: #0F172A;
        }}
        .section {{
            margin-bottom: 36px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: #1E3A8A;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
        }}
        th {{
            background: #F8FAFC;
            color: #475569;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-danger {{ background: #FEE2E2; color: #991B1B; }}
        .badge-warning {{ background: #FEF3C7; color: #92400E; }}
        .badge-info {{ background: #DBEAFE; color: #1E40AF; }}
        .disclaimer-box {{
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            padding: 16px;
            border-radius: 8px;
            font-size: 13px;
            color: #1E3A8A;
            margin-top: 24px;
        }}
        .rec-box {{
            background: #F8FAFC;
            border-left: 4px solid #2563EB;
            padding: 16px 20px;
            border-radius: 4px;
            margin-top: 12px;
        }}
        .rec-box h4 {{
            margin: 0 0 6px 0;
            color: #1E3A8A;
        }}
        .rec-box p {{
            margin: 0;
            color: #334155;
            font-size: 14px;
        }}
        @media print {{
            body {{ padding: 0; background: #fff; }}
            .container {{ box-shadow: none; border: none; padding: 0; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>LogSleuth AI Incident Report</h1>
            <p>Automated Anomaly Detection & Microservice Root Cause Analysis</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 13px; color: #64748B;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</div>
            <div style="font-size: 13px; font-weight: 600; color: #2563EB;">Model: Isolation Forest</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Total Log Records</div>
            <div class="metric-value">{data_quality.get('clean_record_count', len(scored_df))}</div>
        </div>
        <div class="metric-card alert">
            <div class="metric-title">Anomalies Flagged</div>
            <div class="metric-value">{anomaly_summary.get('anomaly_count', 0)} <span style="font-size: 16px; font-weight: normal;">({anomaly_summary.get('anomaly_percentage', 0)}%)</span></div>
        </div>
        <div class="metric-card alert">
            <div class="metric-title">Primary Suspect</div>
            <div class="metric-value" style="font-size: 20px;">{primary_suspect}</div>
        </div>
        <div class="metric-card success">
            <div class="metric-title">Data Quality Health</div>
            <div class="metric-value">{data_quality.get('data_health_score', 100)}%</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Root Cause Analysis Ranking</div>
        <p style="color: #64748B; font-size: 14px; margin-top: -6px;">
            Services ranked by multi-factor composite heuristic (anomaly intensity, error/timeout count, temporal precedence, and dependency topology).
        </p>
        <table>
            <thead>
                <tr>
                    <th style="text-align: center;">Rank</th>
                    <th>Service Name</th>
                    <th style="text-align: center;">Suspect Score</th>
                    <th style="text-align: center;">Anomalies</th>
                    <th style="text-align: center;">Errors</th>
                    <th style="text-align: center;">Timeouts</th>
                    <th>Earliest Anomaly</th>
                    <th>Supporting Evidence</th>
                </tr>
            </thead>
            <tbody>
                {candidate_rows if candidate_rows else "<tr><td colspan='8'>No candidates available.</td></tr>"}
            </tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-title">Top Suspicious & Anomalous Events</div>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Service</th>
                    <th>Level</th>
                    <th style="text-align: center;">Score</th>
                    <th>Log Message</th>
                </tr>
            </thead>
            <tbody>
                {log_rows if log_rows else "<tr><td colspan='5'>No anomalous logs identified.</td></tr>"}
            </tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-title">Recommended Engineering Actions</div>
        <div class="rec-box">
            <h4>1. Inspect Primary Suspect: {primary_suspect}</h4>
            <p>Verify internal connection pool metrics, database connection saturation, and thread executor queues for <code>{primary_suspect}</code> around the initial anomaly timestamp.</p>
        </div>
        <div class="rec-box">
            <h4>2. Check Upstream Cascade & Circuit Breakers</h4>
            <p>Ensure upstream services (e.g. <code>order-service</code>) have configured circuit breakers and graceful retry backoffs to prevent cascading timeout loops.</p>
        </div>
        <div class="rec-box">
            <h4>3. Verify Distributed Trace Corroboration</h4>
            <p>Corroborate findings with OpenTelemetry / Jaeger distributed trace spans corresponding to the flagged error timestamps.</p>
        </div>
    </div>

    <div class="disclaimer-box">
        <strong>Technical & Methodological Disclaimer:</strong><br>
        This report was generated using unsupervised Isolation Forest anomaly detection. Anomaly scores reflect tree path-length isolation distances and are non-probabilistic metrics. Root-cause candidates are prioritized via heuristic multi-factor scoring (temporal onset, failure concentration, and dependency depth) and do not constitute deterministic mathematical causal proof.
    </div>
</div>
</body>
</html>
"""
    return html
