import streamlit as st


def render_metric_row(metrics: list[dict]) -> None:
    """
    Render a horizontal row of metric cards.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
            )


def render_score_badge(score: float, label: str = "Confidence") -> None:
    """
    Render a colour-coded score badge.
    Green >= 0.8, Yellow 0.5-0.8, Red < 0.5
    """
    if score >= 0.8:
        colour = "#22c55e"
        icon = "✅"
    elif score >= 0.5:
        colour = "#f59e0b"
        icon = "⚠️"
    else:
        colour = "#ef4444"
        icon = "❌"

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background: {colour}22;
            border: 1.5px solid {colour};
            border-radius: 8px;
            padding: 4px 14px;
            font-weight: 600;
            color: {colour};
            font-size: 1rem;
        ">
            {icon} {label}: {score:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_type_badge(query_type: str) -> None:
    """Render a colour-coded query type badge."""
    colours = {
        "SIMPLE": "#3b82f6",
        "ANALYTICAL": "#8b5cf6",
        "COMPARATIVE": "#f59e0b",
        "MULTI_HOP": "#ec4899",
        "OUT_OF_SCOPE": "#6b7280",
    }
    colour = colours.get(query_type, "#6b7280")
    st.markdown(
        f"""
        <span style="
            background: {colour}22;
            border: 1.5px solid {colour};
            border-radius: 6px;
            padding: 2px 10px;
            font-weight: 600;
            color: {colour};
            font-size: 0.85rem;
        ">{query_type}</span>
        """,
        unsafe_allow_html=True,
    )


def render_citation_card(citation: dict) -> None:
    """Render a single citation as a styled card."""
    st.markdown(
        f"""
        <div style="
            background: #f8fafc; border: 1px solid #e2e8f0;
            border-left: 3px solid #3b82f6;
            border-radius: 6px;
            padding: 8px 14px;
            margin: 4px 0;
            font-size: 0.85rem;
        ">
            <code style="color: #60a5fa;">[{citation.get('chunk_id_short', '?')}]</code>
            &nbsp;
            <span style="color: #94a3b8;">{citation.get('source_file', 'unknown')}</span>
            &nbsp;·&nbsp;
            <span style="color: #64748b;">
                page {citation.get('page_number', '?')}
            </span>
            {f"&nbsp;·&nbsp;<span style='color: #475569;'>{citation.get('section_title', '')}</span>" if citation.get('section_title') else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
