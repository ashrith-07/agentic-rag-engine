import streamlit as st

st.set_page_config(
    page_title="Agentic RAG Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Agentic RAG Engine")
    st.caption("Production RAG Pipeline · v0.1.0")
    st.divider()

    api_url = st.text_input(
        "API URL",
        value="http://127.0.0.1:8000",
        help="FastAPI backend URL",
    )
    st.session_state["api_url"] = api_url

    if st.button("Check Health", use_container_width=True):
        import httpx

        try:
            resp = httpx.get(f"{api_url}/health", timeout=5)
            health = resp.json()
            status = health.get("status", "unknown")
            if status == "healthy":
                st.success("✅ All systems healthy")
            else:
                st.warning(f"⚠️ Status: {status}")
            st.json(health)
        except Exception as e:
            st.error(f"❌ Cannot reach API: {e}")

    st.divider()
    st.subheader("📄 Ingest Document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    strategy = st.selectbox(
        "Chunking strategy",
        ["auto", "fixed", "semantic", "hierarchical", "structure"],
        index=0,
    )

    if uploaded and st.button("Ingest", use_container_width=True, type="primary"):
        import httpx

        with st.spinner("Ingesting..."):
            try:
                resp = httpx.post(
                    f"{api_url}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    data={"strategy": strategy},
                    timeout=120,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"✅ Ingested {result['source_file']}\n"
                        f"{result['total_chunks']} chunks · {result['strategy_used']}"
                    )
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.caption("Pages: Query Explorer · Metrics · Chunking · Hallucination")

# ── Home page ─────────────────────────────────────────────────────────────────

st.title("🔍 Agentic RAG Engine")
st.markdown(
    """
**Production-grade RAG pipeline** with hybrid retrieval, cross-encoder re-ranking,
agentic query routing, and hallucination detection.

---

### Navigate using the sidebar pages:

| Page | What it shows |
|------|--------------|
| **1 · Query Explorer** | Ask questions, see answers with citations and stage traces |
| **2 · Retrieval Metrics** | NDCG, MRR, Precision over your evaluation dataset |
| **3 · Chunking Comparison** | Side-by-side comparison of all 4 chunking strategies |
| **4 · Hallucination Report** | Per-query confidence scores and flagged claims |

### Quick start
1. Upload a PDF using the sidebar
2. Go to **Query Explorer** and ask a question
3. See the full pipeline trace — routing, retrieval, re-ranking, LLM, audit

---
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Stack**\nGroq · Qdrant · Redis\nFastAPI · Docker")
with col2:
    st.info("**Retrieval**\nHybrid RRF\nCross-encoder · MMR")
with col3:
    st.info("**Evaluation**\nRAGAS · 6 custom metrics\nHallucination audit")
