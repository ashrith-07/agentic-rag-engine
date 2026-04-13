import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Chunking Comparison", page_icon="✂️", layout="wide")
st.title("✂️ Chunking Strategy Comparison")
st.caption("Compare all 4 chunking strategies on your documents")

pdf_files = list(Path("data/raw").glob("*.pdf")) if Path("data/raw").exists() else []
if not pdf_files:
    st.warning("No PDFs found in data/raw/. Upload a document first.")
    st.stop()

selected_pdf = st.selectbox(
    "Select document",
    options=pdf_files,
    format_func=lambda p: p.name,
)

col_size, col_overlap, col_run = st.columns([2, 2, 1])
with col_size:
    chunk_size = st.slider("Chunk size (tokens)", 128, 1024, 512, step=64)
with col_overlap:
    overlap = st.slider("Overlap (tokens)", 0, 256, 102, step=16)
with col_run:
    st.write("")
    run = st.button("▶️ Compare", type="primary", use_container_width=True)

if run:
    from src.ingestion.chunker import chunk_document
    from src.ingestion.parser import parse_pdf
    from src.utils.correlation_id import set_correlation_id

    set_correlation_id()
    strategies = ["fixed", "semantic", "hierarchical", "structure"]
    results: dict = {}

    with st.spinner("Running all 4 strategies..."):
        doc = parse_pdf(str(selected_pdf))
        for strategy in strategies:
            t0 = time.perf_counter()
            chunks = chunk_document(
                doc,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            token_counts = [c.metadata.token_count for c in chunks]
            results[strategy] = {
                "chunks": chunks,
                "count": len(chunks),
                "token_counts": token_counts,
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "avg_tokens": sum(token_counts) // max(len(token_counts), 1),
                "elapsed_ms": round(elapsed_ms, 1),
                "has_tables": sum(1 for c in chunks if c.metadata.has_table),
                "has_code": sum(1 for c in chunks if c.metadata.has_code),
            }

    st.session_state["chunking_results"] = results
    st.session_state["chunking_doc_name"] = doc.source_file

results = st.session_state.get("chunking_results")
if not results:
    st.info("Select a document and click Compare to see results.")
    st.stop()

doc_name = st.session_state.get("chunking_doc_name", "")
st.divider()
st.subheader(f"Results for: {doc_name}")

rows = []
for strategy, r in results.items():
    rows.append(
        {
            "Strategy": strategy,
            "Chunks": r["count"],
            "Min tokens": r["min_tokens"],
            "Max tokens": r["max_tokens"],
            "Avg tokens": r["avg_tokens"],
            "Time (ms)": r["elapsed_ms"],
            "Has tables": r["has_tables"],
            "Has code": r["has_code"],
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Token Distribution per Strategy")

colours = {
    "fixed": "#3b82f6",
    "semantic": "#22c55e",
    "hierarchical": "#f59e0b",
    "structure": "#8b5cf6",
}

fig = go.Figure()
for strategy, r in results.items():
    token_counts = r["token_counts"]
    if token_counts:
        fig.add_trace(
            go.Box(
                y=token_counts,
                name=strategy,
                marker_color=colours[strategy],
                boxmean=True,
            )
        )

fig.update_layout(
    yaxis_title="Tokens per chunk",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8"),
    height=380,
    showlegend=False,
)
fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Chunk Preview")
selected_strategy = st.selectbox(
    "Strategy to preview",
    options=list(results.keys()),
)
chunks = results[selected_strategy]["chunks"]
for i, chunk in enumerate(chunks[:5]):
    with st.expander(
        f"Chunk {i + 1} — {chunk.metadata.token_count} tokens · "
        f"page {chunk.metadata.page_number}"
    ):
        st.markdown(f"**Strategy:** `{chunk.metadata.strategy_used}`")
        if chunk.metadata.section_title:
            st.markdown(f"**Section:** {chunk.metadata.section_title}")
        st.text(chunk.text[:600] + ("..." if len(chunk.text) > 600 else ""))
