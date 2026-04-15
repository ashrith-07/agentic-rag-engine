import json
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_cards import render_metric_row

st.set_page_config(page_title="Retrieval Metrics", page_icon="📊", layout="wide")
st.title("📊 Retrieval Metrics")
st.caption("Evaluation results from your test dataset")

API_URL = st.session_state.get("api_url", "http://0.0.0.0:8000")
REPORT_PATH = Path("data/evaluation/benchmark_report.json")

col_load, col_gen, col_run = st.columns([1, 1, 1])

with col_load:
    if st.button("📂 Load last report", use_container_width=True):
        if REPORT_PATH.exists():
            with open(REPORT_PATH) as f:
                data = json.load(f)
            # If hit_rate is perfectly 0, it means it's a corrupted/empty offline artifact
            if data.get("retrieval_metrics", {}).get("hit_rate", 0) == 0.0:
                st.warning("No valid queries ran till now. Click 'Generate Test Dataset' and then 'Run benchmark now' to run your first evaluation.")
                st.session_state.pop("benchmark_report", None)
            else:
                st.session_state["benchmark_report"] = data
                st.success("Report loaded")
        else:
            st.warning("No queries ran till now. Click 'Generate Test Dataset' and then 'Run benchmark now' to run your first evaluation.")

with col_gen:
    if st.button("📝 Generate Test Dataset", use_container_width=True):
        with st.spinner("Generating 15 QA pairs from ingested documents (takes ~20s)..."):
            try:
                resp = httpx.post(f"{API_URL}/eval/generate?target_size=15", timeout=120)
                if resp.status_code == 200:
                    st.success(f"Generated {resp.json().get('num_pairs')} evaluation pairs!")
                else:
                    st.error(f"Generation failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

with col_run:
    if st.button("▶️ Run benchmark now", use_container_width=True, type="primary"):
        with st.spinner("Running benchmark (may take 60s)..."):
            try:
                resp = httpx.get(f"{API_URL}/eval/run", timeout=300)
                if resp.status_code == 200:
                    st.session_state["benchmark_report"] = resp.json()
                    st.success("Benchmark complete")
                else:
                    st.error(f"Benchmark failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

report = st.session_state.get("benchmark_report")
if not report:
    st.info("👋 Welcome! Click **Load last report** to view past metrics, or generate a test dataset and run a new benchmark.")
    st.stop()

cfg = report.get("benchmark_config", {})
metrics = report.get("retrieval_metrics", {})
evaluated_queries = cfg.get("evaluated_queries", 0)

# Also block rendering if hit rate perfectly 0 (a corrupted or unauthenticated offline artifact)
if evaluated_queries == 0 or metrics.get("hit_rate", 0) == 0.0:
    st.warning("No valid queries ran till now. Please click 'Generate Test Dataset' over ingested documents and then click 'Run benchmark now'.")
    st.stop()

st.divider()
render_metric_row(
    [
        {"label": "Queries evaluated", "value": str(cfg.get("evaluated_queries", 0))},
        {"label": "Avg latency", "value": f"{cfg.get('avg_retrieval_latency_ms', 0):.0f}ms"},
        {"label": "Top-K", "value": str(cfg.get("top_k", 0))},
        {"label": "MRR", "value": f"{metrics.get('mrr', 0):.3f}"},
        {"label": "Hit Rate", "value": f"{metrics.get('hit_rate', 0):.3f}"},
    ]
)

st.divider()
st.subheader("Metrics at K")

k_metrics = {
    "Precision@K": metrics.get("precision_at_k", {}),
    "Recall@K": metrics.get("recall_at_k", {}),
    "F1@K": metrics.get("f1_at_k", {}),
    "NDCG@K": metrics.get("ndcg_at_k", {}),
}

fig = go.Figure()
colours = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]
for (name, values), colour in zip(k_metrics.items(), colours):
    if not values:
        continue
    k_vals = sorted(int(k) for k in values.keys())
    scores = [values[str(k)] for k in k_vals]
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=scores,
            mode="lines+markers",
            name=name,
            line=dict(color=colour, width=2),
            marker=dict(size=8),
        )
    )

fig.update_layout(
    xaxis_title="K",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1.05]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#334155",
        borderwidth=1,
    ),
    height=380,
)
fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
st.plotly_chart(fig, use_container_width=True)

per_query = metrics.get("per_query", [])
if per_query:
    st.divider()
    st.subheader("Per-Query Results")
    rows = []
    for q in per_query:
        ndcg = q.get("ndcg_at_k", {})
        p_at = q.get("precision_at_k", {})
        rows.append(
            {
                "Query ID": q.get("query_id", "")[:12] + "...",
                "MRR": q.get("mrr", 0),
                "Hit Rate": q.get("hit_rate", 0),
                "NDCG@5": ndcg.get("5", ndcg.get(5, 0)),
                "P@5": p_at.get("5", p_at.get(5, 0)),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
