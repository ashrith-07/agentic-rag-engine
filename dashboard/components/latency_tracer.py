import plotly.graph_objects as go
import streamlit as st


def render_waterfall(stages: dict[str, float], title: str = "Stage Latency") -> None:
    """
    Render a horizontal bar chart showing per-stage latency.
    """
    if not stages:
        st.info("No trace data available.")
        return

    filtered = {k: v for k, v in stages.items() if v > 0.1}
    if not filtered:
        st.info("All stages completed in < 0.1ms.")
        return

    labels = list(filtered.keys())
    values = list(filtered.values())
    total = sum(values)

    colour_map = {
        "routing_ms": "#3b82f6",
        "retrieval_ms": "#22c55e",
        "reranking_ms": "#f59e0b",
        "llm_ms": "#8b5cf6",
        "hallucination_ms": "#ec4899",
        "ab_comparison_ms": "#06b6d4",
    }
    colours = [colour_map.get(l, "#64748b") for l in labels]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:.0f}ms ({v / total * 100:.0f}%)" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}ms<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title} — total {total:.0f}ms",
        xaxis_title="Milliseconds",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        height=max(200, len(labels) * 50 + 80),
        margin=dict(l=10, r=80, t=40, b=30),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cost_breakdown(token_usage: dict) -> None:
    """Render token cost breakdown as a small table."""
    calls = token_usage.get("calls", [])
    if not calls:
        st.info("No token usage data.")
        return

    import pandas as pd

    df = pd.DataFrame(calls)
    df["cost_usd"] = df["cost_usd"].apply(lambda x: f"${x:.6f}")
    st.dataframe(
        df[["stage", "input_tokens", "output_tokens", "cost_usd"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"**Total cost:** `${token_usage.get('total_cost_usd', 0):.6f}`  "
        f"| **Total tokens:** `{token_usage.get('total_input_tokens', 0) + token_usage.get('total_output_tokens', 0):,}`"
    )
