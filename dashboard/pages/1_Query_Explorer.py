import httpx
import streamlit as st

from dashboard.components.latency_tracer import render_cost_breakdown, render_waterfall
from dashboard.components.metric_cards import (
    render_citation_card,
    render_metric_row,
    render_query_type_badge,
    render_score_badge,
)

st.set_page_config(page_title="Query Explorer", page_icon="💬", layout="wide")
st.title("💬 Query Explorer")
st.caption("Ask questions against your ingested documents")

API_URL = st.session_state.get("api_url", "http://127.0.0.1:8000")

with st.form("query_form"):
    question = st.text_area(
        "Your question",
        placeholder="What chunking strategies are described in this document?",
        height=100,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        run_ab = st.checkbox("Run A/B comparison (adds ~300ms)", value=False)
    with col2:
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question.strip():
    with st.spinner("Running pipeline..."):
        try:
            resp = httpx.post(
                f"{API_URL}/query",
                json={"question": question, "run_ab_comparison": run_ab},
                timeout=120,
            )
            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()
            data = resp.json()
            st.session_state["last_query_data"] = data
            
            # Auto-save to history for Hallucination Report
            import json
            from pathlib import Path
            log_path = Path("data/logs/query_history.json")
            history = st.session_state.get("query_history", [])
            if not history and log_path.exists():
                with open(log_path) as f:
                    history = json.load(f)
            history.append({
                "query": data.get("query"),
                "query_type": data.get("query_type"),
                "confidence_score": data.get("hallucination", {}).get("confidence_score", 0),
                "is_reliable": data.get("hallucination", {}).get("is_reliable", False),
                "hallucinated_claims": data.get("hallucination", {}).get("hallucinated_claims", []),
                "supported_claims": data.get("hallucination", {}).get("supported_claims", []),
                "total_ms": data.get("trace", {}).get("total_ms", 0),
                "cost_usd": data.get("token_usage", {}).get("total_cost_usd", 0),
            })
            st.session_state["query_history"] = history
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)

        except httpx.ConnectError:
            st.error("Cannot connect to API. Is the server running?")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

# Render from session state if available, so it persists across page navigations
data = st.session_state.get("last_query_data")
if data:
    st.divider()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Answer")
        render_query_type_badge(data.get("query_type", "UNKNOWN"))
        st.write("")
        
        if data.get("query_type") == "OUT_OF_SCOPE":
            st.warning(
                "🛡️ **Query Blocked by Guardrails**\n\n"
                "Your question appears out-of-scope or attempts prompt injection. "
                "Our agentic system utilizes strict zero-trust routing and only provides answers explicitly grounded in the uploaded documents."
            )
        else:
            st.markdown(data.get("answer", "No answer returned."))

        citations = data.get("citations", [])
        if citations:
            st.subheader(f"📎 Sources ({len(citations)})")
            for c in citations:
                render_citation_card(c)
        else:
            st.caption("No citations extracted.")

    with col_right:
        h = data.get("hallucination", {})
        score = h.get("confidence_score", 0.0)
        st.subheader("Reliability")
        render_score_badge(score, "Confidence")
        st.write("")

        render_metric_row(
            [
                {
                    "label": "Chunks retrieved",
                    "value": str(data.get("chunks_retrieved", 0)),
                    "help": "Number of chunks after re-ranking",
                },
                {
                    "label": "Total latency",
                    "value": f"{data.get('trace', {}).get('total_ms', 0):.0f}ms",
                },
            ]
        )

        if h.get("hallucinated_claims"):
            st.warning("⚠️ Hallucinated claims detected:")
            for claim in h["hallucinated_claims"]:
                st.markdown(f"- {claim}")

        if h.get("unsupported_inferences"):
            with st.expander("Unsupported inferences"):
                for inf in h["unsupported_inferences"]:
                    st.markdown(f"- {inf}")

    st.divider()
    st.subheader("⏱ Pipeline Trace")
    trace = data.get("trace", {})
    stages = trace.get("stages", {})
    render_waterfall(stages, title="Stage Latency")

    with st.expander("💰 Token Cost Breakdown"):
        render_cost_breakdown(data.get("token_usage", {}))

    ab = data.get("ab_comparison")
    if ab:
        st.divider()
        st.subheader("🔬 A/B Comparison — Baseline vs Re-ranked")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Pipeline A — Hybrid only**")
            a = ab.get("pipeline_a_baseline", {})
            st.metric("Latency", f"{a.get('latency_ms', 0):.0f}ms")
            for r in a.get("results", []):
                st.caption(f"`{r['chunk_id'][:8]}` · {r['text_preview'][:80]}...")
        with col_b:
            st.markdown("**Pipeline B — + Cross-encoder + MMR**")
            b = ab.get("pipeline_b_reranked", {})
            st.metric(
                "Latency",
                f"{b.get('latency_ms', 0):.0f}ms",
                delta=f"+{ab.get('delta', {}).get('latency_overhead_ms', 0):.0f}ms",
            )
            for r in b.get("results", []):
                st.caption(f"`{r['chunk_id'][:8]}` · {r['text_preview'][:80]}...")

    with st.expander("🔧 Raw API response"):
        import json
        st.code(json.dumps(data, indent=2), language="json")

    # Clear state if user wants to start over
    if st.button("Clear Results"):
        st.session_state.pop("last_query_data", None)
        st.rerun()

elif submitted:
    st.warning("Please enter a question.")
