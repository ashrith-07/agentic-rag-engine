import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_cards import render_metric_row

st.set_page_config(page_title="Hallucination Report", page_icon="🛡️", layout="wide")

from dashboard.css import apply_minimal_theme
apply_minimal_theme()
st.title("🛡️ Hallucination Report")
st.caption("Per-query confidence scores and flagged claims")

LOG_PATH = Path("data/logs/query_history.json")


def load_history() -> list[dict]:
    """Load query history from session state or disk."""
    if "query_history" in st.session_state:
        return st.session_state["query_history"]
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            return json.load(f)
    return []


def save_to_history(query_response: dict) -> None:
    """Append a query response to history."""
    history = load_history()
    history.append(
        {
            "query": query_response.get("query"),
            "query_type": query_response.get("query_type"),
            "confidence_score": query_response.get("hallucination", {}).get(
                "confidence_score", 0
            ),
            "is_reliable": query_response.get("hallucination", {}).get(
                "is_reliable", False
            ),
            "hallucinated_claims": query_response.get("hallucination", {}).get(
                "hallucinated_claims", []
            ),
            "supported_claims": query_response.get("hallucination", {}).get(
                "supported_claims", []
            ),
            "total_ms": query_response.get("trace", {}).get("total_ms", 0),
            "cost_usd": query_response.get("token_usage", {}).get("total_cost_usd", 0),
        }
    )
    st.session_state["query_history"] = history
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)


st.info(
    "Run queries from the **Query Explorer** page — results are automatically "
    "tracked here. Or paste a raw API response below."
)

with st.expander("Paste API response to add to history"):
    raw = st.text_area("Paste JSON response", height=150)
    if st.button("Add to history"):
        try:
            data = json.loads(raw)
            save_to_history(data)
            st.success("Added to history")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

history = load_history()
if not history:
    st.info("👋 Welcome! You haven't run any queries yet. Go to the **Query Explorer** to ask a question, and its results will be tracked here.", icon="ℹ️")
    st.stop()

st.divider()
reliable_count = sum(1 for h in history if h.get("is_reliable", False))
avg_confidence = sum(h.get("confidence_score", 0) for h in history) / len(history)
total_hallucinated = sum(len(h.get("hallucinated_claims", [])) for h in history)

render_metric_row(
    [
        {"label": "Queries tracked", "value": str(len(history))},
        {"label": "Reliable answers", "value": f"{reliable_count}/{len(history)}"},
        {"label": "Avg confidence", "value": f"{avg_confidence:.2f}"},
        {"label": "Total hallucinations", "value": str(total_hallucinated)},
    ]
)

st.divider()
st.subheader("Confidence Score Over Queries")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(1, len(history) + 1)),
        y=[h.get("confidence_score", 0) for h in history],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(
            size=8,
            color=[
                "#22c55e"
                if h.get("confidence_score", 0) >= 0.8
                else "#f59e0b"
                if h.get("confidence_score", 0) >= 0.5
                else "#ef4444"
                for h in history
            ],
        ),
        hovertemplate="Query %{x}: score=%{y:.2f}<extra></extra>",
    )
)

fig.add_hline(
    y=0.8,
    line_dash="dash",
    line_color="#22c55e",
    annotation_text="Reliable (0.8)",
    annotation_position="right",
)
fig.add_hline(
    y=0.5,
    line_dash="dash",
    line_color="#f59e0b",
    annotation_text="Caution (0.5)",
    annotation_position="right",
)

fig.update_layout(
    xaxis_title="Query #",
    yaxis_title="Confidence Score",
    yaxis=dict(range=[0, 1.05]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#475569"),
    height=320,
)
fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Per-Query Detail")

df = pd.DataFrame(
    [
        {
            "Query": h.get("query", "")[:60] + "...",
            "Type": h.get("query_type", ""),
            "Confidence": h.get("confidence_score", 0),
            "Reliable": "✅" if h.get("is_reliable") else "⚠️",
            "Hallucinations": len(h.get("hallucinated_claims", [])),
            "Latency (ms)": round(h.get("total_ms", 0)),
            "Cost ($)": f"{h.get('cost_usd', 0):.6f}",
        }
        for h in history
    ]
)
st.dataframe(df, use_container_width=True, hide_index=True)

flagged = [h for h in history if h.get("hallucinated_claims")]
if flagged:
    st.divider()
    st.subheader(f"⚠️ Flagged Hallucinations ({len(flagged)} queries)")
    for h in flagged:
        with st.expander(f"Query: {h.get('query', '')[:80]}..."):
            st.markdown(f"**Confidence:** {h.get('confidence_score', 0):.2f}")
            st.markdown("**Hallucinated claims:**")
            for claim in h.get("hallucinated_claims", []):
                st.markdown(f"- ❌ {claim}")
else:
    st.success("✅ No hallucinated claims detected across all queries.")
