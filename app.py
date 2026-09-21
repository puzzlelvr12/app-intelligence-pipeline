"""
app.py
======
Executive Product Intelligence Dashboard for App Store & Play Store Feedback.
Provides real-time NLP classification, customer sentiment monitoring,
release regression tracking, and an interactive inference playground.

Run:
    streamlit run app.py
"""

from pathlib import Path
import re
import sqlite3
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import hardened classification engine and database utilities from pipeline
from pipeline import (
    CATEGORY_KEYWORDS,
    DB_PATH,
    _PATTERNS,
    classify_category,
    get_db,
    vader_sentiment,
)

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="App Store Product Intelligence",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global Styling (Apple Human Interface / Dark Executive Aesthetic) ───────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif;
}

/* Base App Background */
.stApp {
    background: radial-gradient(circle at 50% 0%, #17153B 0%, #0C0A1D 55%, #05040A 100%);
    min-height: 100vh;
    color: #F8FAFC;
}

/* Executive Header Hero */
.hero-header {
    text-align: center;
    padding: 2.2rem 1rem 1.2rem;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    background: rgba(167, 139, 250, 0.12);
    border: 1px solid rgba(167, 139, 250, 0.35);
    color: #C4B5FD;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #FFFFFF 20%, #C4B5FD 60%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.35rem;
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Executive KPI Cards Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.6rem 0 1.2rem;
}
.kpi-card {
    background: rgba(255, 255, 255, 0.035);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(167, 139, 250, 0.35);
    background: rgba(255, 255, 255, 0.05);
}
.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.kpi-label {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.kpi-icon {
    font-size: 1.25rem;
    opacity: 0.9;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.35rem;
}

/* Section Title Headers */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #F1F5F9;
    margin: 2rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-line {
    height: 1px;
    background: linear-gradient(90deg, rgba(167, 139, 250, 0.5) 0%, rgba(255, 255, 255, 0.05) 100%);
    margin-bottom: 1.2rem;
}

/* Interactive Playground Card */
.test-panel {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(167, 139, 250, 0.25);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-top: 0.8rem;
    backdrop-filter: blur(12px);
}
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
}
.badge-bug      { background: rgba(239, 68, 68, 0.15);  color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.badge-feature  { background: rgba(96, 165, 250, 0.15); color: #60A5FA; border: 1px solid rgba(96, 165, 250, 0.4); }
.badge-friction { background: rgba(251, 191, 36, 0.15); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.4); }
.badge-praise   { background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.4); }

.token-chip {
    display: inline-block;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    margin: 0.2rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: #E2E8F0;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.82rem;
    padding: 2.5rem 0 1.2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 3.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Constant Design Tokens ──────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Bug Report":      "#F87171",
    "Feature Request": "#60A5FA",
    "UI Friction":     "#FBBF24",
    "Praise":          "#34D399",
}

SENTIMENT_COLORS = {
    "Positive": "#34D399",
    "Neutral":  "#FBBF24",
    "Negative": "#F87171",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0"),
    margin=dict(l=20, r=20, t=35, b=20),
)

PLOTLY_LEGEND = dict(
    bgcolor="rgba(255,255,255,0.04)",
    bordercolor="rgba(255,255,255,0.08)",
    borderwidth=1,
)


# ─── Data Access Layer ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """Loads validated reviews with connection lifecycle safety."""
    if not Path(db_path).exists():
        return pd.DataFrame()

    with get_db(db_path) as conn:
        df = pd.read_sql("SELECT * FROM reviews ORDER BY review_date DESC", conn)

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["app_version"] = df["app_version"].fillna("Unknown").astype(str).str.strip()
    return df


# ─── Semantic Version Parser ─────────────────────────────────────────────────

def parse_semantic_version(v_str: str) -> Tuple[int, ...]:
    """Parses dotted version strings into sortable integer tuples."""
    parts = re.split(r"[.\-_]", str(v_str).strip())
    nums = []
    for p in parts:
        digits = re.sub(r"\D", "", p)
        if digits:
            try:
                nums.append(int(digits))
            except ValueError:
                pass
    return tuple(nums) if nums else (0,)


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-pill">⚡ Enterprise Intelligence Platform</div>
  <div class="hero-title">App Store Product Intelligence</div>
  <div class="hero-subtitle">
    Automated semantic telemetry, customer sentiment triage, and version regression monitoring engineered for product engineering teams.
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Dataset ────────────────────────────────────────────────────────────
df = load_data()

if df.empty:
    st.warning(
        "**Database is empty or missing.** Run the ingestion and ETL pipeline first:\n\n"
        "```powershell\npython scraper.py\npython pipeline.py\n```",
        icon="⚠️",
    )
    st.stop()

# ─── Sidebar Filtering ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Data Filters")

    apps = ["All"] + sorted(df["app_name"].dropna().unique().tolist())
    sel_app = st.selectbox("Application", apps)

    all_categories = sorted(list(CATEGORY_COLORS.keys()))
    sel_cats = st.multiselect("Category Triage", all_categories, default=all_categories)

    sentiments = ["All", "Positive", "Neutral", "Negative"]
    sel_sent = st.selectbox("Sentiment Status", sentiments)

    min_rating, max_rating = st.slider("Star Rating Range", 1, 5, (1, 5))

    st.markdown("---")
    st.markdown(
        f"<div style='color:#64748B;font-size:0.78rem;line-height:1.6;'>"
        f"<b>Storage:</b> SQLite WAL (<code>{DB_PATH}</code>)<br>"
        f"<b>Total Ingested:</b> {len(df):,} records<br>"
        f"<b>Schema Version:</b> 2.0 (Relational + Indexes)</div>",
        unsafe_allow_html=True,
    )

# Filter Dataframe
fdf = df.copy()
if sel_app != "All":
    fdf = fdf[fdf["app_name"] == sel_app]
if sel_cats:
    fdf = fdf[fdf["category"].isin(sel_cats)]
if sel_sent != "All":
    fdf = fdf[fdf["sentiment_label"] == sel_sent]
fdf = fdf[fdf["rating"].between(min_rating, max_rating)]

# ─── Executive KPI Metrics Grid ──────────────────────────────────────────────
total_count   = len(fdf)
bug_count     = (fdf["category"] == "Bug Report").sum()
bug_rate      = (bug_count / total_count * 100) if total_count else 0.0
praise_count  = (fdf["category"] == "Praise").sum()
praise_rate   = (praise_count / total_count * 100) if total_count else 0.0
avg_sentiment = fdf["sentiment_score"].mean() if total_count else 0.0
critical_1_star = (fdf["rating"] == 1).sum()
critical_rate = (critical_1_star / total_count * 100) if total_count else 0.0

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-top">
      <span class="kpi-label">Analyzed Volume</span>
      <span class="kpi-icon">📊</span>
    </div>
    <div class="kpi-value">{total_count:,}</div>
    <div class="kpi-sub">Filtered customer reviews</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-top">
      <span class="kpi-label">Bug Report Rate</span>
      <span class="kpi-icon">🐛</span>
    </div>
    <div class="kpi-value" style="color:#F87171;">{bug_rate:.1f}%</div>
    <div class="kpi-sub">{bug_count:,} defects reported</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-top">
      <span class="kpi-label">Net Sentiment</span>
      <span class="kpi-icon">💬</span>
    </div>
    <div class="kpi-value" style="color:{'#34D399' if avg_sentiment >= 0 else '#F87171'};">
      {avg_sentiment:+.3f}
    </div>
    <div class="kpi-sub">Scale: −1.000 to +1.000</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-top">
      <span class="kpi-label">Praise & CSAT</span>
      <span class="kpi-icon">⭐</span>
    </div>
    <div class="kpi-value" style="color:#34D399;">{praise_rate:.1f}%</div>
    <div class="kpi-sub">{praise_count:,} positive endorsements</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Visualizations: Row 1 (Breakdown & Sentiment) ───────────────────────────
c1, c2 = st.columns([1.1, 0.9], gap="large")

with c1:
    st.markdown('<div class="section-title">📋 Category Distribution & Volume</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

    cat_summary = (
        fdf.groupby("category")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    cat_summary["share"] = (cat_summary["count"] / max(total_count, 1) * 100).round(1)
    cat_summary["color"] = cat_summary["category"].map(CATEGORY_COLORS).fillna("#94A3B8")

    fig_cat = go.Figure(
        go.Bar(
            x=cat_summary["count"],
            y=cat_summary["category"],
            orientation="h",
            marker=dict(color=cat_summary["color"], line=dict(width=0)),
            text=[f"{c:,} ({s}%)" for c, s in zip(cat_summary["count"], cat_summary["share"])],
            textposition="outside",
            textfont=dict(color="#E2E8F0", size=12),
            hovertemplate="<b>%{y}</b><br>Volume: %{x:,}<br>Share: %{text}<extra></extra>",
        )
    )
    fig_cat.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, color="#94A3B8"),
        yaxis=dict(showgrid=False, color="#F8FAFC", tickfont=dict(size=12, weight=600)),
    )
    st.plotly_chart(fig_cat, width="stretch")

with c2:
    st.markdown('<div class="section-title">🎭 Customer Sentiment Polarization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

    sent_counts = fdf["sentiment_label"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]

    fig_sent = go.Figure(
        go.Pie(
            labels=sent_counts["Sentiment"],
            values=sent_counts["Count"],
            hole=0.62,
            marker=dict(
                colors=[SENTIMENT_COLORS.get(s, "#94A3B8") for s in sent_counts["Sentiment"]],
                line=dict(color="#0C0A1D", width=2.5),
            ),
            textinfo="percent+label",
            textfont=dict(size=12, color="#F8FAFC"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Proportion: %{percent}<extra></extra>",
        )
    )
    fig_sent.update_layout(
        **PLOTLY_LAYOUT,
        legend=PLOTLY_LEGEND,
        height=320,
        annotations=[
            dict(
                text=f"<b>{total_count:,}</b><br><span style='font-size:11px;color:#94A3B8'>REVIEWS</span>",
                x=0.5, y=0.5, font_size=18, font_color="#FFFFFF", showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig_sent, width="stretch")


# ─── Visualizations: Row 2 (Version Regression Analysis) ──────────────────────
st.markdown('<div class="section-title">📈 Release Regression Telemetry (Bug Rate % by App Version)</div>', unsafe_allow_html=True)
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

# Exclude 'Unknown' versions for cleaner release tracking
ver_df = fdf[fdf["app_version"] != "Unknown"].copy()

if not ver_df.empty and ver_df["app_version"].nunique() > 1:
    ver_stats = (
        ver_df.groupby("app_version")
        .agg(
            total_reviews=("id", "count"),
            bug_count=("category", lambda s: (s == "Bug Report").sum()),
        )
        .reset_index()
    )
    ver_stats["bug_rate_pct"] = (ver_stats["bug_count"] / ver_stats["total_reviews"] * 100).round(1)

    # Sort semantically by version numbers
    ver_stats["_semver"] = ver_stats["app_version"].apply(parse_semantic_version)
    ver_stats = ver_stats.sort_values("_semver").drop(columns="_semver").tail(15)

    # Detect regression spike
    mean_rate = ver_stats["bug_rate_pct"].mean()
    high_regressions = ver_stats[ver_stats["bug_rate_pct"] > (mean_rate * 1.5)]
    if not high_regressions.empty:
        spike_ver = high_regressions.iloc[-1]["app_version"]
        spike_val = high_regressions.iloc[-1]["bug_rate_pct"]
        st.info(
            f"🚨 **Release Regression Alert:** Version `{spike_ver}` exhibited a **{spike_val}% bug incidence rate** "
            f"(benchmark avg: {mean_rate:.1f}%). Recommend targeted audio/core diagnostics.",
            icon="⚠️",
        )

    # Dual line/bar chart for Bug Volume & Bug Rate %
    fig_reg = go.Figure()

    # Bar: Total review volume per version
    fig_reg.add_trace(go.Bar(
        x=ver_stats["app_version"],
        y=ver_stats["total_reviews"],
        name="Total Version Reviews",
        marker=dict(color="rgba(148, 163, 184, 0.2)", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)),
        yaxis="y",
        hovertemplate="Version %{x}<br>Total Reviews: %{y:,}<extra></extra>",
    ))

    # Line: Bug Rate %
    fig_reg.add_trace(go.Scatter(
        x=ver_stats["app_version"],
        y=ver_stats["bug_rate_pct"],
        mode="lines+markers",
        name="Bug Rate (%)",
        line=dict(color="#F87171", width=3.5),
        marker=dict(size=8, color="#F87171", line=dict(color="#FFFFFF", width=1.5)),
        yaxis="y2",
        hovertemplate="Version %{x}<br>Bug Rate: <b>%{y}%</b><extra></extra>",
    ))

    fig_reg.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(title="App Release Version", showgrid=False, color="#94A3B8", tickangle=-30),
        yaxis=dict(title="Total Reviews", showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#94A3B8"),
        yaxis2=dict(
            title="Bug Rate (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            color="#F87171",
            ticksuffix="%",
        ),
        legend=dict(**PLOTLY_LEGEND, x=0.01, y=0.98),
    )
    st.plotly_chart(fig_reg, width="stretch")
else:
    st.info("Insufficient multi-version release telemetry in current filter selection.", icon="ℹ️")


# ─── Rating Distribution ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">⭐ Customer Star Rating Spectrum</div>', unsafe_allow_html=True)
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

rating_dist = fdf["rating"].value_counts().sort_index().reset_index()
rating_dist.columns = ["Rating", "Count"]
r_palette = ["#F87171", "#FB923C", "#FBBF24", "#A3E635", "#34D399"]

fig_r = go.Figure(
    go.Bar(
        x=[f"{r} Star{'s' if r > 1 else ''}" for r in rating_dist["Rating"]],
        y=rating_dist["Count"],
        marker=dict(
            color=[r_palette[int(r) - 1] for r in rating_dist["Rating"]],
            line=dict(width=0),
        ),
        text=[f"{c:,}" for c in rating_dist["Count"]],
        textposition="outside",
        textfont=dict(color="#E2E8F0"),
        hovertemplate="%{x}<br>Count: <b>%{y:,}</b><extra></extra>",
    )
)
fig_r.update_layout(
    **PLOTLY_LAYOUT,
    height=260,
    xaxis=dict(color="#94A3B8", showgrid=False),
    yaxis=dict(title="Volume", color="#94A3B8", showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
)
st.plotly_chart(fig_r, width="stretch")


# ─── Tabular Data Explorer ───────────────────────────────────────────────────
with st.expander("📄 Interactive Feedback Ledger & Drill-down", expanded=False):
    view_cols = [
        "review_date", "app_name", "app_version", "rating",
        "category", "sentiment_label", "sentiment_score", "review_text"
    ]
    present_cols = [c for c in view_cols if c in fdf.columns]
    st.dataframe(
        fdf[present_cols].head(300),
        width="stretch",
        hide_index=True,
        column_config={
            "sentiment_score": st.column_config.ProgressColumn(
                "Sentiment Polarity", min_value=-1.0, max_value=1.0, format="%.3f"
            ),
            "rating": st.column_config.NumberColumn("Rating", format="%d ⭐"),
            "review_date": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
        },
    )


# ─── Interactive Inference Playground ────────────────────────────────────────
st.markdown('<div class="section-title">🧪 Real-time NLP Inference & Triage Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

st.markdown('<div class="test-panel">', unsafe_allow_html=True)

user_input = st.text_area(
    "Enter raw customer feedback to test real-time classification and sentiment scoring:",
    value="The latest update completely broke audio crossfade! The app keeps crashing when switching playlists.",
    height=90,
    placeholder="Type any review text, e.g., 'Please add lossless audio streaming to the car mode!'",
)

if user_input.strip():
    pred_cat = classify_category(user_input)
    pred_score, pred_label = vader_sentiment(user_input)

    cat_badge_cls = {
        "Bug Report":      "badge-bug",
        "Feature Request": "badge-feature",
        "UI Friction":     "badge-friction",
        "Praise":          "badge-praise",
    }.get(pred_cat, "badge-feature")

    sent_badge_cls = {
        "Positive": "badge-praise",
        "Negative": "badge-bug",
        "Neutral":  "badge-friction",
    }.get(pred_label, "badge-friction")

    col_res1, col_res2, col_res3 = st.columns([1.2, 1.2, 2.0])

    with col_res1:
        st.markdown("**Assigned Category**")
        st.markdown(f'<span class="result-badge {cat_badge_cls}">{pred_cat}</span>', unsafe_allow_html=True)

    with col_res2:
        st.markdown("**Sentiment Classification**")
        st.markdown(f'<span class="result-badge {sent_badge_cls}">{pred_label}</span>', unsafe_allow_html=True)

    with col_res3:
        st.markdown(f"**VADER Polarity Score: `{pred_score:+.4f}`**")
        norm_pos = (pred_score + 1.0) / 2.0
        bar_color = "#34D399" if pred_score > 0 else ("#F87171" if pred_score < 0 else "#FBBF24")
        st.markdown(
            f"""<div style="background:rgba(255,255,255,0.08);border-radius:6px;overflow:hidden;height:10px;margin-top:6px;">
              <div style="width:{norm_pos * 100:.1f}%;background:{bar_color};height:10px;transition:width 0.3s ease;"></div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Scale: −1.0 (Maximum Negative) → 0.0 (Neutral) → +1.0 (Maximum Positive)")

    # Highlight matched keywords
    matched_tokens: list[str] = []
    for cat_name, pattern in _PATTERNS.items():
        found = pattern.findall(user_input)
        if found:
            for token in found:
                matched_tokens.append(f"{token.lower()} ({cat_name})")

    if matched_tokens:
        unique_tokens = list(dict.fromkeys(matched_tokens))
        st.markdown(
            "**Triggered Semantic Entities:** "
            + "".join(f'<span class="token-chip">{t}</span>' for t in unique_tokens),
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# ─── Executive Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <b>App Store Product Intelligence Pipeline</b> · Enterprise Data Engineering Reference Architecture<br>
  Engineered with Streamlit, Plotly, VADER NLP & SQLite (WAL Mode)
</div>
""", unsafe_allow_html=True)
