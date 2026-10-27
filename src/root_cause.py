"""
Root-cause analysis module for LogSleuth AI.
Uses NetworkX directed dependency graphs and a multi-factor transparent heuristic
to identify and rank root-cause candidate services.
"""

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

# Default microservice dependency edges (Caller -> Callee)
DEFAULT_DEPENDENCY_EDGES = [
    ("authentication-service", "order-service"),
    ("product-service", "order-service"),
    ("inventory-service", "order-service"),
    ("order-service", "payment-service"),
]


def build_dependency_graph(
    discovered_services: list[str] | None = None,
    custom_edges: list[tuple[str, str]] | None = None
) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph representing microservice call dependencies.
    """
    G = nx.DiGraph()

    # Add edges
    edges = custom_edges if custom_edges is not None else DEFAULT_DEPENDENCY_EDGES
    for src, dst in edges:
        G.add_edge(src, dst)

    # Add any remaining discovered services as standalone nodes if not present
    if discovered_services:
        for s in discovered_services:
            if not G.has_node(s):
                G.add_node(s)

    return G


def compute_service_metrics(scored_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Computes per-service anomaly counts, score statistics, error counts, timeout counts,
    and earliest anomalous timestamps.
    """
    if scored_df.empty or "service" not in scored_df.columns:
        return {}

    services = scored_df["service"].unique()
    metrics = {}

    for s in services:
        sdf = scored_df[scored_df["service"] == s]
        total_events = len(sdf)
        anomalies = sdf[sdf["is_anomaly"] == True]
        anomaly_count = len(anomalies)

        # Anomaly scores
        scores = sdf["anomaly_score"] if "anomaly_score" in sdf.columns else pd.Series([0.0])
        max_score = float(scores.max()) if not scores.empty else 0.0
        avg_score = float(scores.mean()) if not scores.empty else 0.0

        # Errors and timeouts
        messages = sdf["message"].astype(str).str.lower()
        err_mask = sdf["level"].isin(["ERROR", "CRITICAL"]) | messages.str.contains(r'error|exception|fail', regex=True)
        timeout_mask = messages.str.contains(r'timeout|timed out|deadline exceeded', regex=True)
        retry_mask = messages.str.contains(r'retry|retrying|attempt', regex=True)

        error_count = int(err_mask.sum())
        timeout_count = int(timeout_mask.sum())
        retry_count = int(retry_mask.sum())

        # Earliest anomalous / failure timestamp
        earliest_anomaly_ts = None
        if not anomalies.empty and "timestamp" in anomalies.columns:
            ts_series = anomalies["timestamp"].dropna()
            if not ts_series.empty:
                earliest_anomaly_ts = str(ts_series.min())
        elif err_mask.any() and "timestamp" in sdf.columns:
            err_ts = sdf[err_mask]["timestamp"].dropna()
            if not err_ts.empty:
                earliest_anomaly_ts = str(err_ts.min())

        metrics[s] = {
            "service": s,
            "total_events": total_events,
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_count / max(1, total_events), 4),
            "max_anomaly_score": round(max_score, 4),
            "avg_anomaly_score": round(avg_score, 4),
            "error_count": error_count,
            "timeout_count": timeout_count,
            "retry_count": retry_count,
            "earliest_anomaly_ts": earliest_anomaly_ts
        }

    return metrics


def rank_root_cause_candidates(
    scored_df: pd.DataFrame,
    dependency_graph: nx.DiGraph | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Ranks microservices as potential incident root causes using a multi-factor transparent heuristic.

    Factors considered:
    1. Anomaly severity (max & mean anomaly score): 30% weight
    2. Failure intensity (error & timeout count proportion): 35% weight
    3. Temporal precedence (earliest onset of anomalous behavior): 20% weight
    4. Graph dependency impact (downstream propagation and caller blast radius): 15% weight

    Returns:
        Tuple of (ranked_candidates_list, analysis_meta)
    """
    if scored_df.empty:
        return [], {
            "status": "empty",
            "disclaimer": "This is a heuristic ranking, not definitive causal proof."
        }

    services = list(scored_df["service"].unique())
    G = dependency_graph if dependency_graph is not None else build_dependency_graph(discovered_services=services)

    service_metrics = compute_service_metrics(scored_df)
    if not service_metrics:
        return [], {"status": "no_metrics"}

    # Aggregate global totals for normalization
    total_anomalies = max(1, sum(m["anomaly_count"] for m in service_metrics.values()))
    total_errors = max(1, sum(m["error_count"] for m in service_metrics.values()))
    total_timeouts = max(1, sum(m["timeout_count"] for m in service_metrics.values()))

    # Find global earliest anomaly timestamp across all services
    all_earliest_timestamps = []
    for m in service_metrics.values():
        if m["earliest_anomaly_ts"]:
            try:
                all_earliest_timestamps.append(pd.to_datetime(m["earliest_anomaly_ts"]))
            except Exception:
                pass

    global_earliest_ts = min(all_earliest_timestamps) if all_earliest_timestamps else None

    # Calculate heuristic score for each service
    candidates = []

    for s, m in service_metrics.items():
        # 1. Anomaly Severity Component (0.0 to 1.0)
        # Weighted blend of max score and average score
        severity_comp = (m["max_anomaly_score"] * 0.6) + (m["avg_anomaly_score"] * 0.4)

        # 2. Failure Intensity Component (0.0 to 1.0)
        err_ratio = m["error_count"] / total_errors if total_errors > 0 else 0.0
        to_ratio = m["timeout_count"] / total_timeouts if total_timeouts > 0 else 0.0
        failure_comp = min(1.0, (err_ratio * 0.6) + (to_ratio * 0.4))

        # 3. Temporal Precedence Component (0.0 to 1.0)
        # Services that failed first receive higher precedence scores
        temporal_comp = 0.0
        if m["earliest_anomaly_ts"] and global_earliest_ts:
            try:
                svc_ts = pd.to_datetime(m["earliest_anomaly_ts"])
                delta_seconds = max(0.0, (svc_ts - global_earliest_ts).total_seconds())
                # Decay score with time delta (if it failed first, delta=0 -> temporal_comp=1.0)
                temporal_comp = float(np.exp(-delta_seconds / 180.0))  # 3-minute half-decay
            except Exception:
                temporal_comp = 0.5
        elif m["anomaly_count"] > 0:
            temporal_comp = 0.5

        # 4. Graph Topology Component (0.0 to 1.0)
        # In microservice dependencies:
        # A service called by others (in-degree > 0 or downstream endpoint) whose failure causes caller retries
        # Or upstream services with high betweenness
        topo_score = 0.5
        if G.has_node(s):
            in_degree = G.in_degree(s)
            out_degree = G.out_degree(s)
            # High in-degree (many depend on this service, e.g. payment or auth) amplifies impact
            topo_score = min(1.0, 0.4 + (in_degree * 0.2) + (out_degree * 0.1))

        # Combined Heuristic Root Cause Score (0 to 100 scale)
        composite_score = (
            (failure_comp * 0.35) +
            (severity_comp * 0.30) +
            (temporal_comp * 0.20) +
            (topo_score * 0.15)
        ) * 100.0

        # Attenuate score if the service experienced zero errors, timeouts, or retries
        if m["error_count"] == 0 and m["timeout_count"] == 0 and m["retry_count"] == 0:
            composite_score = composite_score * 0.25

        # Construct specific evidence narrative
        evidence = []
        if m["earliest_anomaly_ts"]:
            if temporal_comp > 0.95:
                evidence.append(f"Earliest anomaly initiator: first observed incident event at {m['earliest_anomaly_ts']}.")
            else:
                evidence.append(f"First anomaly recorded at {m['earliest_anomaly_ts']}.")

        if m["error_count"] > 0:
            evidence.append(f"{m['error_count']} error/exception events ({int(err_ratio * 100)}% of system-wide errors).")

        if m["timeout_count"] > 0:
            evidence.append(f"{m['timeout_count']} timeout/connection-reset events.")

        if m["retry_count"] > 0:
            evidence.append(f"{m['retry_count']} client retry attempts triggered.")

        if m["anomaly_count"] > 0:
            evidence.append(f"{m['anomaly_count']} total flagged anomalies (Max score: {m['max_anomaly_score']:.2f}).")
        else:
            evidence.append("No anomalous events detected in this service.")

        if G.has_node(s):
            predecessors = list(G.predecessors(s))
            successors = list(G.successors(s))
            if predecessors:
                evidence.append(f"Direct callers dependent on {s}: {', '.join(predecessors)}.")
            if successors:
                evidence.append(f"Calls downstream dependencies: {', '.join(successors)}.")

        candidates.append({
            "service": s,
            "root_cause_score": round(composite_score, 1),
            "anomaly_count": m["anomaly_count"],
            "anomaly_rate_pct": round(m["anomaly_rate"] * 100, 1),
            "max_anomaly_score": m["max_anomaly_score"],
            "avg_anomaly_score": m["avg_anomaly_score"],
            "error_count": m["error_count"],
            "timeout_count": m["timeout_count"],
            "retry_count": m["retry_count"],
            "earliest_anomaly_ts": m["earliest_anomaly_ts"] or "None",
            "score_breakdown": {
                "failure_intensity": round(failure_comp * 35, 1),
                "anomaly_severity": round(severity_comp * 30, 1),
                "temporal_precedence": round(temporal_comp * 20, 1),
                "graph_topology": round(topo_score * 15, 1)
            },
            "evidence": evidence
        })

    # Sort descending by composite score
    candidates.sort(key=lambda c: c["root_cause_score"], reverse=True)

    # Assign rank
    for idx, c in enumerate(candidates, 1):
        c["rank"] = idx

    analysis_meta = {
        "ranked_candidates_count": len(candidates),
        "primary_suspect": candidates[0]["service"] if candidates else None,
        "disclaimer": (
            "NOTICE: This is a transparent heuristic ranking based on anomaly density, failure severity, "
            "temporal precedence, and dependency topology. It is NOT definitive causal proof. "
            "Trace instrumentation (e.g. distributed span IDs) is required for deterministic causality."
        ),
        "graph_node_count": G.number_of_nodes(),
        "graph_edge_count": G.number_of_edges()
    }

    return candidates, analysis_meta
