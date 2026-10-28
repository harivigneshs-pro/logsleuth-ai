"""
LogSleuth AI: Intelligent Log Anomaly Detection and Root Cause Analysis for Cloud Microservices.
Interactive Streamlit Dashboard.
"""

import io
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import detect_anomalies
from src.ingestion import IngestionError, ingest_log_file
from src.preprocessing import preprocess_logs
from src.report_generator import generate_html_report, generate_json_summary
from src.root_cause import (
    build_dependency_graph,
    compute_service_metrics,
    rank_root_cause_candidates,
)

# Set page config
st.set_page_config(
    page_title="LogSleuth AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Sleek Modern Navy/Blue Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Header Banner */
    .banner-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .banner-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60A5FA, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .banner-subtitle {
        font-size: 15px;
        color: #94A3B8;
        margin-top: 6px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #38BDF8;
        margin: 4px 0;
    }
    .metric-value.alert {
        color: #F87171;
    }
    .metric-value.success {
        color: #34D399;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
    }
    
    /* Disclaimer Alert Box */
    .disclaimer-banner {
        background: rgba(30, 58, 138, 0.3);
        border: 1px solid #1E40AF;
        border-radius: 8px;
        padding: 14px 18px;
        color: #BFDBFE;
        font-size: 13px;
        margin-top: 14px;
        margin-bottom: 20px;
    }
    
    /* Candidate Card */
    .candidate-card {
        background: #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 5px solid #3B82F6;
    }
    .candidate-card.primary {
        border-left-color: #EF4444;
        background: #231B2C;
    }
    
    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 6px;
        color: #94A3B8;
        padding: 8px 18px;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "pipeline_results" not in st.session_state:
    st.session_state["pipeline_results"] = None
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "current_file_source" not in st.session_state:
    st.session_state["current_file_source"] = None


def execute_analysis_pipeline(file_bytes: bytes, filename: str, contamination: float):
    """Executes the end-to-end processing pipeline and saves outputs in session state."""
    with st.spinner("Analyzing microservice logs with Isolation Forest & NetworkX..."):
        # 1. Ingestion
        df_raw, ingestion_meta = ingest_log_file(file_bytes, filename=filename)

        # 2. Preprocessing & Quality Check
        clean_df, data_quality = preprocess_logs(df_raw)

        # 3. Anomaly Detection
        scored_df, anomaly_summary = detect_anomalies(clean_df, contamination=contamination)

        # 4. Dependency Graph & Root Cause
        services = list(clean_df["service"].unique())
        dep_graph = build_dependency_graph(discovered_services=services)
        service_metrics = compute_service_metrics(scored_df)
        candidates, root_cause_meta = rank_root_cause_candidates(scored_df, dependency_graph=dep_graph)

        # Store in session state
        st.session_state["raw_df"] = df_raw
        st.session_state["current_file_source"] = filename
        st.session_state["pipeline_results"] = {
            "scored_df": scored_df,
            "data_quality": data_quality,
            "anomaly_summary": anomaly_summary,
            "candidates": candidates,
            "root_cause_meta": root_cause_meta,
            "service_metrics": service_metrics,
            "dep_graph": dep_graph,
            "ingestion_meta": ingestion_meta
        }


def render_network_graph(G: nx.DiGraph, candidates: list[dict[str, Any]]):
    """Renders a clean, interactive NetworkX dependency graph using Plotly."""
    if G.number_of_nodes() == 0:
        st.info("No service nodes to display in graph.")
        return

    # Use spring layout for node positioning
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Prepare node attributes
    candidate_lookup = {c["service"]: c for c in candidates}
    top_suspect = candidates[0]["service"] if candidates else None

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        c_info = candidate_lookup.get(node, {})
        anom_cnt = c_info.get("anomaly_count", 0)
        err_cnt = c_info.get("error_count", 0)
        score = c_info.get("root_cause_score", 0)

        # Determine node color & size
        if node == top_suspect:
            node_color.append("#EF4444")  # Red for primary root cause suspect
            node_size.append(36)
        elif anom_cnt > 0:
            node_color.append("#F59E0B")  # Amber for affected services
            node_size.append(28)
        else:
            node_color.append("#10B981")  # Green for healthy services
            node_size.append(22)

        hover_info = (
            f"<b>Service: {node}</b><br>"
            f"Suspect Score: {score}/100<br>"
            f"Anomalies: {anom_cnt}<br>"
            f"Errors: {err_cnt}<br>"
            f"Role: {'Primary Suspect' if node == top_suspect else 'Affected' if anom_cnt > 0 else 'Healthy'}"
        )
        node_text.append(hover_info)

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.8, color="#64748B"),
        hoverinfo="none",
        mode="lines"
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[f"  <b>{n}</b>" for n in G.nodes()],
        textposition="top center",
        textfont=dict(color="#F8FAFC", size=11),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color="#FFFFFF")
        )
    )
    node_trace.hovertext = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="<b>Microservice Dependency Topology & Failure State</b>",
                font=dict(color="#F8FAFC", size=16)
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=450
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def answer_natural_language_query(query: str, res: dict[str, Any]) -> str:
    """Answers user queries directly from computed pipeline results."""
    q = query.lower().strip()
    candidates = res["candidates"]
    data_quality = res["data_quality"]
    anomaly_summary = res["anomaly_summary"]
    service_metrics = res["service_metrics"]
    scored_df = res["scored_df"]

    if not candidates:
        return "No log records are currently analyzed. Please upload or load logs first."

    # Top root cause / primary suspect
    if any(k in q for k in ["root cause", "primary suspect", "culprit", "who caused", "main issue"]):
        top = candidates[0]
        evidence_str = "; ".join(top.get("evidence", []))
        return (
            f"**Primary Suspect:** `{top['service']}` (Heuristic Score: {top['root_cause_score']}/100, Rank #1).\n\n"
            f"**Key Supporting Evidence:**\n- {evidence_str}\n\n"
            f"*Note: This is an empirical heuristic ranking based on anomaly concentration, temporal precedence, and dependency topology.*"
        )

    # Most anomalies
    if any(k in q for k in ["most anomalies", "highest anomaly", "anomaly count"]):
        sorted_by_anom = sorted(candidates, key=lambda c: c["anomaly_count"], reverse=True)
        top_anom = sorted_by_anom[0]
        return (
            f"The service with the most flagged anomalies is **`{top_anom['service']}`** with **{top_anom['anomaly_count']}** anomalies "
            f"({top_anom['anomaly_rate_pct']}% of its events flagged). Max anomaly score: {top_anom['max_anomaly_score']}."
        )

    # First / earliest anomaly timestamp
    if any(k in q for k in ["earliest", "first anomaly", "timeline", "when did it start"]):
        earliest_cand = min(
            [c for c in candidates if c["earliest_anomaly_ts"] != "None"],
            key=lambda x: x["earliest_anomaly_ts"],
            default=None
        )
        if earliest_cand:
            return f"The earliest anomaly onset was detected in **`{earliest_cand['service']}`** at timestamp **`{earliest_cand['earliest_anomaly_ts']}`**."
        return "No specific anomalous timestamps were recorded."

    # Total logs or record count
    if any(k in q for k in ["total logs", "how many logs", "how many events", "total events"]):
        return (
            f"A total of **{data_quality.get('raw_record_count', len(scored_df))}** raw logs were ingested. "
            f"After removing {data_quality.get('duplicates_removed', 0)} exact duplicates, **{data_quality.get('clean_record_count', len(scored_df))}** "
            f"clean events were analyzed across **{data_quality.get('services_count', len(service_metrics))}** microservices."
        )

    # Specific service query
    for s_name in service_metrics.keys():
        if s_name in q:
            m = service_metrics[s_name]
            return (
                f"**Service Summary for `{s_name}`:**\n"
                f"- Total Events: {m['total_events']}\n"
                f"- Flagged Anomalies: {m['anomaly_count']} ({round(m['anomaly_rate'] * 100, 1)}%)\n"
                f"- Error Events: {m['error_count']}\n"
                f"- Timeout Events: {m['timeout_count']}\n"
                f"- Retry Events: {m['retry_count']}\n"
                f"- Earliest Anomaly: {m['earliest_anomaly_ts'] or 'None'}\n"
                f"- Peak Anomaly Score: {m['max_anomaly_score']}"
            )

    # Common errors / timeout count
    if any(k in q for k in ["common error", "timeouts", "timeout count", "errors"]):
        total_errors = sum(m["error_count"] for m in service_metrics.values())
        total_timeouts = sum(m["timeout_count"] for m in service_metrics.values())
        return (
            f"System-wide error breakdown:\n"
            f"- Total Error/Critical Events: **{total_errors}**\n"
            f"- Total Timeout Events: **{total_timeouts}**\n"
            f"- Service with most errors: **`{candidates[0]['service']}`** ({candidates[0]['error_count']} errors)"
        )

    # Fallback general summary
    return (
        f"LogSleuth AI analyzed {data_quality.get('clean_record_count', 0)} records. "
        f"Flagged {anomaly_summary.get('anomaly_count', 0)} anomalies ({anomaly_summary.get('anomaly_percentage', 0)}%). "
        f"The primary root cause suspect is **`{candidates[0]['service']}`** (score {candidates[0]['root_cause_score']}/100)."
    )


# --- MAIN UI ---

# Banner
st.markdown("""
<div class="banner-container">
    <div class="banner-title">🔍 LogSleuth AI</div>
    <div class="banner-subtitle">
        Intelligent Microservice Log Anomaly Detection & Heuristic Root Cause Analysis
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.header("⚙️ Analysis Settings")
contamination = st.sidebar.slider(
    "Contamination Factor",
    min_value=0.01,
    max_value=0.30,
    value=0.08,
    step=0.01,
    help="Expected proportion of anomalies in the dataset for Isolation Forest."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Quick Actions")

# Load Bundled Sample Logs button
sample_path = ROOT_DIR / "data" / "sample_logs.csv"
if st.sidebar.button("🚀 Load Bundled Sample Logs", use_container_width=True):
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            sample_bytes = f.read()
        execute_analysis_pipeline(sample_bytes, filename="sample_logs.csv", contamination=contamination)
        st.sidebar.success("Loaded and analyzed sample logs!")
    else:
        st.sidebar.error("Sample dataset not found on disk.")

st.sidebar.markdown("---")
st.sidebar.caption("LogSleuth AI • Unsupervised Machine Learning • NetworkX Graph Topology")


# Main Tabs Navigation
tab_upload, tab_overview, tab_quality, tab_anomaly, tab_rootcause, tab_health, tab_ask, tab_report = st.tabs([
    "📤 Upload Logs",
    "📊 Overview",
    "🛡️ Data Quality",
    "⚡ Anomaly Detection",
    "🎯 Root Cause",
    "📈 Service Health",
    "💬 Ask LogSleuth",
    "📑 Incident Report"
])


# 1. TAB: UPLOAD LOGS
with tab_upload:
    st.subheader("Upload Microservice Logs")
    st.markdown(
        "Upload raw or structured logs in **CSV**, **JSON**, **TXT**, or **LOG** format. "
        "LogSleuth AI automatically detects column aliases (e.g. `Component`, `Severity`, `Content`)."
    )

    uploaded_file = st.file_uploader(
        "Choose a log file",
        type=["csv", "json", "txt", "log"],
        help="Upload up to 25MB log file."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_btn = st.button("Run Full Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            try:
                execute_analysis_pipeline(file_bytes, filename=uploaded_file.name, contamination=contamination)
                st.success(f"Successfully processed `{uploaded_file.name}`!")
            except IngestionError as ie:
                st.error(f"Ingestion Error: {ie!s}")
            except Exception as e:
                st.error(f"Analysis failed: {e!s}")
        else:
            st.warning("Please upload a file or click 'Load Bundled Sample Logs' in the sidebar.")

    # Show preview if data is loaded
    if st.session_state["raw_df"] is not None:
        st.markdown("---")
        st.markdown(f"**Loaded Dataset Preview:** `{st.session_state['current_file_source']}` ({len(st.session_state['raw_df'])} rows)")
        st.dataframe(st.session_state["raw_df"].head(10), use_container_width=True)


# Check if pipeline results are available for the analytical tabs
res = st.session_state["pipeline_results"]

if res is None:
    for tab in [tab_overview, tab_quality, tab_anomaly, tab_rootcause, tab_health, tab_ask, tab_report]:
        with tab:
            st.info("👈 Please upload a log file in the **Upload Logs** tab or click **'Load Bundled Sample Logs'** in the sidebar to begin analysis.")
else:
    scored_df = res["scored_df"]
    data_quality = res["data_quality"]
    anomaly_summary = res["anomaly_summary"]
    candidates = res["candidates"]
    root_cause_meta = res["root_cause_meta"]
    service_metrics = res["service_metrics"]
    dep_graph = res["dep_graph"]

    # 2. TAB: OVERVIEW
    with tab_overview:
        st.subheader("System Anomaly & Incident Overview")

        # Metric Cards Row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Events</div>
                <div class="metric-value">{data_quality.get('clean_record_count', len(scored_df))}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Microservices</div>
                <div class="metric-value">{data_quality.get('services_count', len(service_metrics))}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Flagged Anomalies</div>
                <div class="metric-value alert">{anomaly_summary.get('anomaly_count', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Anomaly Rate</div>
                <div class="metric-value alert">{anomaly_summary.get('anomaly_percentage', 0.0)}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Primary Suspect</div>
                <div class="metric-value" style="font-size: 20px;">{candidates[0]['service'] if candidates else 'None'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            # Severity distribution chart
            sev_dist = data_quality.get("severity_distribution", {})
            if sev_dist:
                color_map = {
                    "INFO": "#3B82F6",
                    "WARNING": "#F59E0B",
                    "ERROR": "#EF4444",
                    "CRITICAL": "#DC2626",
                    "DEBUG": "#64748B"
                }
                fig_sev = px.pie(
                    names=list(sev_dist.keys()),
                    values=list(sev_dist.values()),
                    title="<b>Log Severity Distribution</b>",
                    hole=0.45,
                    color=list(sev_dist.keys()),
                    color_discrete_map=color_map
                )
                fig_sev.update_layout(
                    plot_bgcolor="#0F172A",
                    paper_bgcolor="#0F172A",
                    font_color="#F8FAFC",
                    legend=dict(orientation="h", y=-0.1)
                )
                st.plotly_chart(fig_sev, use_container_width=True)

        with col_chart2:
            # Anomaly vs Normal distribution
            anom_data = {
                "Classification": ["Normal Events", "Anomalous Events"],
                "Count": [anomaly_summary.get("normal_count", 0), anomaly_summary.get("anomaly_count", 0)]
            }
            fig_anom = px.pie(
                anom_data,
                names="Classification",
                values="Count",
                title="<b>Isolation Forest Anomaly Classification</b>",
                hole=0.45,
                color="Classification",
                color_discrete_map={"Normal Events": "#10B981", "Anomalous Events": "#EF4444"}
            )
            fig_anom.update_layout(
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                font_color="#F8FAFC",
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig_anom, use_container_width=True)

        # Timeline plot
        if "timestamp" in scored_df.columns and pd.api.types.is_datetime64_any_dtype(scored_df["timestamp"]):
            st.markdown("#### ⏱️ Log Event & Anomaly Timeline")
            # Group by 1-minute intervals
            timeline_df = scored_df.set_index("timestamp").resample("1min").agg({
                "message": "count",
                "is_anomaly": "sum"
            }).rename(columns={"message": "Total Events", "is_anomaly": "Anomalies"}).reset_index()

            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=timeline_df["timestamp"],
                y=timeline_df["Total Events"],
                name="Total Logs",
                marker_color="#3B82F6",
                opacity=0.6
            ))
            fig_time.add_trace(go.Scatter(
                x=timeline_df["timestamp"],
                y=timeline_df["Anomalies"],
                name="Flagged Anomalies",
                mode="lines+markers",
                marker=dict(size=8, color="#EF4444"),
                line=dict(width=2, color="#EF4444")
            ))
            fig_time.update_layout(
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                font_color="#F8FAFC",
                xaxis=dict(title="Timestamp (UTC)", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="Event Count", showgrid=True, gridcolor="#334155"),
                legend=dict(orientation="h", y=1.1),
                height=350,
                margin=dict(t=30, b=30, l=40, r=20)
            )
            st.plotly_chart(fig_time, use_container_width=True)

    # 3. TAB: DATA QUALITY
    with tab_quality:
        st.subheader("Data Quality & Preprocessing Hygiene")

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Health Score</div>
                <div class="metric-value success">{data_quality.get('data_health_score', 100)}%</div>
            </div>
            """, unsafe_allow_html=True)
        with q2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Duplicates Removed</div>
                <div class="metric-value">{data_quality.get('duplicates_removed', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with q3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Invalid Timestamps</div>
                <div class="metric-value">{data_quality.get('invalid_timestamps_count', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with q4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Time Span Analyzed</div>
                <div class="metric-value" style="font-size: 20px;">{data_quality.get('time_range', {}).get('duration_human', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_q_left, col_q_right = st.columns(2)
        with col_q_left:
            st.markdown("#### 📋 Preprocessing Audit Log")
            st.markdown(f"- **Raw Record Count:** `{data_quality.get('raw_record_count', 0)}`")
            st.markdown(f"- **Clean Record Count:** `{data_quality.get('clean_record_count', 0)}`")
            st.markdown(f"- **Exact Full-Row Duplicates Dropped:** `{data_quality.get('duplicates_removed', 0)}`")
            st.markdown("- **Repeated Message Policy:** Repeated messages across distinct events are *retained* to preserve burst patterns.")
            imputations = data_quality.get("missing_imputations", {})
            if imputations:
                st.markdown("**Imputed Missing Values:**")
                for col, count in imputations.items():
                    st.markdown(f"  - `{col}`: `{count}` missing entries handled safely")
            else:
                st.markdown("- **Missing Values:** No null values detected in required fields.")

        with col_q_right:
            st.markdown("#### 🕒 Timestamp Coverage")
            tr = data_quality.get("time_range", {})
            st.markdown(f"- **Start Timestamp:** `{tr.get('start', 'N/A')}`")
            st.markdown(f"- **End Timestamp:** `{tr.get('end', 'N/A')}`")
            st.markdown(f"- **Duration:** `{tr.get('duration_seconds', 0)}` seconds")

    # 4. TAB: ANOMALY DETECTION
    with tab_anomaly:
        st.subheader("Isolation Forest Anomaly Detection")

        st.markdown("""
        <div class="disclaimer-banner">
            ℹ️ <b>Technical Interpretation:</b> Anomaly scores are continuous non-probabilistic metrics derived from tree path-length isolation distances. 
            Scores are normalized to a consistent [0.0, 1.0] scale where <b>higher scores indicate more anomalous records</b>. Scores are NOT probabilities.
        </div>
        """, unsafe_allow_html=True)

        # Anomaly score distribution chart
        fig_hist = px.histogram(
            scored_df,
            x="anomaly_score",
            color="is_anomaly",
            nbins=35,
            title="<b>Anomaly Score Distribution with Outlier Boundary</b>",
            color_discrete_map={True: "#EF4444", False: "#3B82F6"},
            labels={"anomaly_score": "Continuous Anomaly Score", "is_anomaly": "Is Flagged Anomaly"}
        )
        thresh = anomaly_summary.get("suspicious_threshold", 0.7)
        fig_hist.add_vline(x=thresh, line_dash="dash", line_color="#F59E0B", annotation_text=f"Cut-off (~{thresh})", annotation_position="top right")
        fig_hist.update_layout(
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            font_color="#F8FAFC",
            height=320
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🚨 Suspicious & Anomalous Events")
        
        # Interactive Filtering Controls
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        with c_filter1:
            all_services = ["All Services"] + sorted(list(scored_df["service"].unique()))
            selected_svc = st.selectbox("Filter by Service", all_services)
        with c_filter2:
            min_score = st.slider("Minimum Anomaly Score", 0.0, 1.0, float(thresh), 0.05)
        with c_filter3:
            only_anomalies = st.checkbox("Show only flagged anomalies", value=True)

        filtered_df = scored_df.copy()
        if selected_svc != "All Services":
            filtered_df = filtered_df[filtered_df["service"] == selected_svc]
        filtered_df = filtered_df[filtered_df["anomaly_score"] >= min_score]
        if only_anomalies:
            filtered_df = filtered_df[filtered_df["is_anomaly"] == True]

        st.markdown(f"Displaying **{len(filtered_df)}** records matching criteria:")

        # Display table with clear formatting
        display_cols = ["timestamp", "service", "level", "anomaly_score", "message"]
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[available_cols].sort_values(by="anomaly_score", ascending=False),
            use_container_width=True,
            height=350
        )

    # 5. TAB: ROOT CAUSE ANALYSIS
    with tab_rootcause:
        st.subheader("Microservice Root Cause Candidate Ranking")

        st.markdown(f"""
        <div class="disclaimer-banner">
            ⚠️ <b>Transparency & Methodology Disclaimer:</b> {root_cause_meta.get('disclaimer')}
        </div>
        """, unsafe_allow_html=True)

        col_top, col_graph = st.columns([1, 1])

        with col_top:
            st.markdown("#### 🏆 Ranked Suspect Services")
            for c in candidates:
                is_primary = c.get("rank") == 1
                score = c.get("root_cause_score", 0)
                card_class = "candidate-card primary" if is_primary else "candidate-card"

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 18px; font-weight: 700; color: {'#F87171' if is_primary else '#60A5FA'};">
                            #{c.get('rank')} {c.get('service')}
                        </span>
                        <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">
                            {score} / 100
                        </span>
                    </div>
                    <div style="font-size: 13px; color: #94A3B8; margin-top: 6px;">
                        Anomalies: <b>{c.get('anomaly_count')}</b> ({c.get('anomaly_rate_pct')}%) &nbsp;|&nbsp; 
                        Errors: <b>{c.get('error_count')}</b> &nbsp;|&nbsp; 
                        Timeouts: <b>{c.get('timeout_count')}</b> &nbsp;|&nbsp; 
                        Retries: <b>{c.get('retry_count')}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Supporting Evidence & Score Breakdown for {c.get('service')}"):
                    st.markdown("**Evidence:**")
                    for ev in c.get("evidence", []):
                        st.markdown(f"- {ev}")
                    st.markdown("**Score Component Contribution:**")
                    sb = c.get("score_breakdown", {})
                    st.markdown(f"- Failure Intensity (Errors & Timeouts): `{sb.get('failure_intensity', 0)} pts`")
                    st.markdown(f"- Anomaly Severity: `{sb.get('anomaly_severity', 0)} pts`")
                    st.markdown(f"- Temporal Precedence (Onset Timing): `{sb.get('temporal_precedence', 0)} pts`")
                    st.markdown(f"- Graph Topology & Impact: `{sb.get('graph_topology', 0)} pts`")

        with col_graph:
            render_network_graph(dep_graph, candidates)

    # 6. TAB: SERVICE HEALTH
    with tab_health:
        st.subheader("Microservice Activity & Degradation Profiles")

        # Service metrics comparison table
        health_rows = []
        for s, m in service_metrics.items():
            health_rows.append({
                "Service": s,
                "Total Events": m["total_events"],
                "Anomalies": m["anomaly_count"],
                "Anomaly Rate (%)": round(m["anomaly_rate"] * 100, 1),
                "Errors": m["error_count"],
                "Timeouts": m["timeout_count"],
                "Retries": m["retry_count"],
                "Max Anomaly Score": m["max_anomaly_score"],
                "Earliest Anomaly": m["earliest_anomaly_ts"] or "None"
            })

        health_df = pd.DataFrame(health_rows)
        st.dataframe(health_df.sort_values(by="Anomalies", ascending=False), use_container_width=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_h_bar = px.bar(
                health_df,
                x="Service",
                y=["Anomalies", "Errors", "Timeouts"],
                barmode="group",
                title="<b>Degradation Signals by Microservice</b>",
                color_discrete_map={"Anomalies": "#EF4444", "Errors": "#F59E0B", "Timeouts": "#38BDF8"}
            )
            fig_h_bar.update_layout(
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                font_color="#F8FAFC",
                xaxis=dict(title="", tickangle=-20)
            )
            st.plotly_chart(fig_h_bar, use_container_width=True)

        with col_h2:
            fig_h_scores = px.bar(
                health_df,
                x="Service",
                y="Max Anomaly Score",
                color="Max Anomaly Score",
                color_continuous_scale="Reds",
                title="<b>Peak Anomaly Score by Microservice</b>"
            )
            fig_h_scores.update_layout(
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                font_color="#F8FAFC",
                xaxis=dict(title="", tickangle=-20)
            )
            st.plotly_chart(fig_h_scores, use_container_width=True)

    # 7. TAB: ASK LOGSLEUTH (Rule-based Natural Language Q&A)
    with tab_ask:
        st.subheader("💬 Ask LogSleuth AI")
        st.markdown(
            "Query the incident data using natural language questions. "
            "LogSleuth AI computes answers directly from the currently active dataset."
        )

        # Quick prompt buttons
        st.markdown("**Quick Questions:**")
        q_cols = st.columns(4)
        selected_prompt = None
        if q_cols[0].button("Who is the primary suspect?", use_container_width=True):
            selected_prompt = "Who is the primary suspect?"
        if q_cols[1].button("Which service has most anomalies?", use_container_width=True):
            selected_prompt = "Which service has the most anomalies?"
        if q_cols[2].button("When did the first anomaly start?", use_container_width=True):
            selected_prompt = "When did the first anomaly start?"
        if q_cols[3].button("Show timeout counts across services", use_container_width=True):
            selected_prompt = "Show timeout counts across services"

        user_query = st.text_input(
            "Enter your question:",
            value=selected_prompt or "",
            placeholder="e.g. Which service has the most anomalies? or payment-service status"
        )

        if user_query:
            answer = answer_natural_language_query(user_query, res)
            st.markdown(f"""
            <div style="background: #1E293B; border-left: 4px solid #3B82F6; padding: 18px 24px; border-radius: 8px; margin-top: 16px;">
                <div style="font-weight: 600; color: #60A5FA; margin-bottom: 8px;">LogSleuth Response:</div>
                <div style="color: #F8FAFC; line-height: 1.6;">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

    # 8. TAB: INCIDENT REPORT & EXPORTS
    with tab_report:
        st.subheader("📥 Incident Report & Artifact Downloads")
        st.markdown("Download computed analytical assets or generate a publication-ready HTML incident report.")

        # Prepare artifacts
        # 1. Scored CSV
        csv_buffer = io.StringIO()
        scored_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode("utf-8")

        # 2. JSON Summary
        json_data = generate_json_summary(data_quality, anomaly_summary, candidates, service_metrics).encode("utf-8")

        # 3. HTML Report
        html_report = generate_html_report(scored_df, data_quality, anomaly_summary, candidates)
        html_bytes = html_report.encode("utf-8")

        # Download Buttons
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                label="📄 Download Scored Logs (CSV)",
                data=csv_data,
                file_name="logsleuth_scored_logs.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            st.download_button(
                label="📊 Download Summary (JSON)",
                data=json_data,
                file_name="logsleuth_summary_report.json",
                mime="application/json",
                use_container_width=True
            )
        with d3:
            st.download_button(
                label="📑 Download Incident Report (HTML)",
                data=html_bytes,
                file_name="logsleuth_incident_report.html",
                mime="text/html",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("#### 👁️ Incident Report Live Preview")
        st.components.v1.html(html_report, height=750, scrolling=True)
