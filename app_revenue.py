"""
Revenue Analytics Agent — World-Class Edition
==============================================
AI-powered marketing & CRM analytics for B2SMB payments companies.
Built to demonstrate full-stack data analyst capabilities.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Revenue Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark theme tweaks */
  .main { background: #0d0f1a; }
  section[data-testid="stSidebar"] { background: #151825 !important; }

  /* KPI card */
  .kpi-card {
    background: #1c2033;
    border: 1px solid #252a3d;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .kpi-label { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .6px; }
  .kpi-value { font-size: 28px; font-weight: 800; color: #e2e8f0; margin: 6px 0 4px; }
  .kpi-delta-up   { color: #10b981; font-size: 12px; font-weight: 600; }
  .kpi-delta-down { color: #ef4444; font-size: 12px; font-weight: 600; }

  /* Insight card */
  .insight-positive { background: rgba(16,185,129,.08); border: 1px solid rgba(16,185,129,.25); border-radius: 10px; padding: 14px; margin-bottom: 10px; }
  .insight-warning  { background: rgba(245,158,11,.08);  border: 1px solid rgba(245,158,11,.25);  border-radius: 10px; padding: 14px; margin-bottom: 10px; }
  .insight-opportunity { background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.25); border-radius: 10px; padding: 14px; margin-bottom: 10px; }
  .insight-title { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
  .insight-text  { font-size: 12px; color: #94a3b8; line-height: 1.5; }

  /* Chat bubbles */
  .user-msg { background: #1e293b; border-radius: 12px 12px 0 12px; padding: 12px 16px; margin: 8px 0; max-width: 85%; float: right; clear: both; }
  .ai-msg   { background: #1c2033; border: 1px solid #252a3d; border-radius: 12px 12px 12px 0; padding: 12px 16px; margin: 8px 0; max-width: 90%; clear: both; }

  /* Suggested question chip */
  div[data-testid="stButton"] button {
    background: #1c2033 !important;
    border: 1px solid #252a3d !important;
    color: #94a3b8 !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 4px 12px !important;
    transition: all .2s !important;
  }
  div[data-testid="stButton"] button:hover {
    border-color: #6366f1 !important;
    color: #6366f1 !important;
  }

  /* Header */
  .header-bar {
    background: linear-gradient(135deg, #1c2033, #151825);
    border: 1px solid #252a3d;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .header-title { font-size: 22px; font-weight: 800; color: #e2e8f0; }
  .header-sub   { font-size: 13px; color: #64748b; margin-top: 4px; }

  /* Tab styling */
  button[data-baseweb="tab"] { font-size: 13px !important; }

  .stTabs [data-baseweb="tab-list"] { background: #151825; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from demo_data import DATASETS, SUGGESTED_QUESTIONS
from insights import generate_insights

# ── Plotly theme ──────────────────────────────────────────────────────────────
COLORS = ["#6366f1", "#10b981", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#f97316", "#8b5cf6"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1c2033",
    plot_bgcolor="#1c2033",
    font=dict(color="#e2e8f0", size=12),
    xaxis=dict(gridcolor="#252a3d", linecolor="#252a3d"),
    yaxis=dict(gridcolor="#252a3d", linecolor="#252a3d"),
    legend=dict(bgcolor="#1c2033", bordercolor="#252a3d"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages = []
if "active_df"     not in st.session_state: st.session_state.active_df = None
if "active_name"   not in st.session_state: st.session_state.active_name = None
if "chat_input_val"not in st.session_state: st.session_state.chat_input_val = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Revenue Analytics Agent")
    st.markdown("*AI-powered marketing & CRM analytics for B2SMB payments*")
    st.divider()

    mode = st.radio("Data Source", ["🎯 SumUp Demo Data", "📂 Upload Your CSV"], index=0)
    st.divider()

    if mode == "🎯 SumUp Demo Data":
        selected_dataset = st.selectbox("Choose Dataset", list(DATASETS.keys()))
        df = DATASETS[selected_dataset].copy()
        st.session_state.active_df   = df
        st.session_state.active_name = selected_dataset

        # Filters
        st.markdown("**🔍 Filters**")
        if "channel" in df.columns:
            channels = ["All"] + sorted(df["channel"].unique().tolist())
            sel_ch = st.selectbox("Channel", channels)
            if sel_ch != "All":
                df = df[df["channel"] == sel_ch]

        if "country" in df.columns:
            countries = ["All"] + sorted(df["country"].unique().tolist())
            sel_co = st.selectbox("Country", countries)
            if sel_co != "All":
                df = df[df["country"] == sel_co]

        if "month" in df.columns:
            months = sorted(df["month"].unique().tolist())
            sel_range = st.select_slider("Month Range", options=months, value=(months[0], months[-1]))
            df = df[(df["month"] >= sel_range[0]) & (df["month"] <= sel_range[1])]

        st.session_state.filtered_df = df

    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.active_df   = df
            st.session_state.active_name = uploaded.name
            st.session_state.filtered_df = df
            st.success(f"✅ {uploaded.name} loaded — {len(df):,} rows")
        else:
            st.info("Upload a CSV to analyse your own data")
            if st.session_state.active_df is None:
                st.stop()

    st.divider()

    # Ollama is used locally - no API key needed
    st.info("This app uses Ollama with qwen2.5:7b locally (no API key required)")

    st.divider()
    st.markdown("**Built by Ganesh**")
    st.markdown("[GitHub](https://github.com/ganesh123381) · [LinkedIn](https://linkedin.com/in/v-ganesh2024)")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
  <div style="width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
              display:flex;align-items:center;justify-content:center;font-size:22px;">📊</div>
  <div>
    <div class="header-title">Revenue Analytics Agent</div>
    <div class="header-sub">Marketing & Engagement Intelligence · {st.session_state.get('active_name','')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Get filtered df ───────────────────────────────────────────────────────────
df = st.session_state.get("filtered_df", st.session_state.active_df)
if df is None:
    st.info("Select a dataset from the sidebar to get started.")
    st.stop()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🎯 Funnel Analysis", "📡 Channel Attribution", "🤖 AI Chat Analyst", "🔍 Auto Insights"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    agg_name = st.session_state.active_name or ""

    if "Marketing Funnel" in agg_name:
        # KPIs
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        total_spend  = df["spend_eur"].sum()
        total_gmv    = df["gmv_eur"].sum()
        total_rev    = df["revenue_eur"].sum()
        total_leads  = df["leads"].sum()
        total_act    = df["activated"].sum()
        avg_roas     = round(total_rev / total_spend, 2) if total_spend > 0 else 0

        for col, label, val, delta in [
            (c1, "Total GMV",       f"€{total_gmv/1e6:.1f}M", "▲ 18.4% vs LY"),
            (c2, "Revenue (Fees)",  f"€{total_rev/1e3:.0f}K",  "▲ 21.2% vs LY"),
            (c3, "Total Spend",     f"€{total_spend/1e3:.0f}K","▲ 12.1% vs LY"),
            (c4, "Total Leads",     f"{total_leads:,}",        "▲ 9.3% vs LY"),
            (c5, "Activations",     f"{total_act:,}",          "▲ 12.7% vs LY"),
            (c6, "Avg ROAS",        f"{avg_roas}×",            "▲ 0.4× vs LY"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{val}</div>
                  <div class="kpi-delta-up">{delta}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # GMV trend
        col1, col2 = st.columns(2)
        with col1:
            monthly = df.groupby("month").agg(gmv=("gmv_eur","sum"), rev=("revenue_eur","sum")).reset_index()
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=monthly["month"], y=monthly["gmv"], name="GMV (€)",
                                  marker_color="#6366f1", opacity=0.7), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["rev"], name="Revenue (€)",
                                      line=dict(color="#10b981",width=2), mode="lines+markers"), secondary_y=True)
            fig.update_layout(title="Monthly GMV & Revenue", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            ch_agg = df.groupby("channel").agg(leads=("leads","sum")).reset_index()
            fig = px.pie(ch_agg, names="channel", values="leads", hole=0.6,
                         title="Lead Share by Channel", color_discrete_sequence=COLORS)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Spend vs ROAS
        col3, col4 = st.columns(2)
        with col3:
            ch = df.groupby("channel").agg(spend=("spend_eur","sum"), rev=("revenue_eur","sum")).reset_index()
            ch["roas"] = (ch["rev"] / ch["spend"].replace(0, np.nan)).round(2)
            ch = ch.dropna().sort_values("roas", ascending=True)
            fig = px.bar(ch, x="roas", y="channel", orientation="h", title="ROAS by Channel",
                         color="roas", color_continuous_scale=["#ef4444","#f59e0b","#10b981"],
                         text="roas")
            fig.update_traces(texttemplate="%{text}×", textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            country_gmv = df.groupby("country").agg(gmv=("gmv_eur","sum"), spend=("spend_eur","sum")).reset_index()
            fig = px.bar(country_gmv, x="country", y="gmv", title="GMV by Country",
                         color="country", color_discrete_sequence=COLORS, text_auto=".2s")
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    elif "CRM" in agg_name:
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, label, val in [
            (c1, "Emails Sent",    f"{df['emails_sent'].sum():,}"),
            (c2, "Avg Open Rate",  f"{df['email_open_rate'].mean():.1f}%"),
            (c3, "Avg Click Rate", f"{df['email_ctr'].mean():.1f}%"),
            (c4, "Push Open Rate", f"{df['push_open_rate'].mean():.1f}%"),
            (c5, "SMS CTR",        f"{df['sms_ctr'].mean():.1f}%"),
        ]:
            with col:
                st.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(df.groupby("month").agg(open=("email_open_rate","mean"), ctr=("email_ctr","mean")).reset_index(),
                          x="month", y=["open","ctr"], title="Email Engagement Trend",
                          color_discrete_sequence=["#6366f1","#10b981"])
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            seg = df.groupby("segment").agg(open=("email_open_rate","mean")).sort_values("open").reset_index()
            fig = px.bar(seg, x="open", y="segment", orientation="h", title="Open Rate by Segment",
                         color="open", color_continuous_scale=["#ef4444","#10b981"], text_auto=".1f")
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.dataframe(df.describe(), use_container_width=True)
        fig = px.histogram(df.select_dtypes(include=np.number).iloc[:, 0:1], title="Distribution",
                           color_discrete_sequence=["#6366f1"])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FUNNEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if "Marketing Funnel" not in (st.session_state.active_name or ""):
        st.info("Funnel analysis is available for the Marketing Funnel dataset. Select it from the sidebar.")
    else:
        st.markdown("### 🎯 Full Conversion Funnel")

        col1, col2 = st.columns([1, 1])
        with col1:
            # Funnel chart
            funnel_steps = {
                "Impressions": int(df["impressions"].sum()),
                "Clicks":      int(df["clicks"].sum()),
                "Leads":       int(df["leads"].sum()),
                "Activated":   int(df["activated"].sum()),
                "Transacting": int(df["transacting"].sum()),
            }
            fig = go.Figure(go.Funnel(
                y=list(funnel_steps.keys()),
                x=list(funnel_steps.values()),
                textinfo="value+percent initial",
                marker=dict(color=["#6366f1","#8b5cf6","#ec4899","#10b981","#14b8a6"]),
                connector=dict(line=dict(color="#252a3d", width=2)),
            ))
            fig.update_layout(title="Conversion Funnel", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Activation rate by channel
            ch_conv = df.groupby("channel").agg(
                leads=("leads","sum"), activated=("activated","sum")
            ).reset_index()
            ch_conv["act_rate"] = (ch_conv["activated"] / ch_conv["leads"] * 100).round(1)
            ch_conv = ch_conv.sort_values("act_rate")
            fig = px.bar(ch_conv, x="act_rate", y="channel", orientation="h",
                         title="Lead → Activation Rate by Channel",
                         color="act_rate", color_continuous_scale=["#ef4444","#f59e0b","#10b981"],
                         text_auto=".1f")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Cohort trend
        st.markdown("### 📅 Monthly Cohort Performance")
        monthly_cohort = df.groupby("month").agg(
            leads=("leads","sum"), activated=("activated","sum"),
            transacting=("transacting","sum"), gmv=("gmv_eur","sum")
        ).reset_index()
        monthly_cohort["act_rate"] = (monthly_cohort["activated"] / monthly_cohort["leads"] * 100).round(1)

        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=monthly_cohort["month"], y=monthly_cohort["leads"],
                              name="Leads", marker_color="#6366f1", opacity=0.6), secondary_y=False)
        fig.add_trace(go.Bar(x=monthly_cohort["month"], y=monthly_cohort["activated"],
                              name="Activated", marker_color="#10b981", opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly_cohort["month"], y=monthly_cohort["act_rate"],
                                  name="Act. Rate %", line=dict(color="#f59e0b",width=2.5),
                                  mode="lines+markers"), secondary_y=True)
        fig.update_layout(barmode="group", title="Monthly Leads vs Activations", **PLOTLY_LAYOUT)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Activation Rate %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHANNEL ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if "Marketing Funnel" not in (st.session_state.active_name or ""):
        st.info("Channel attribution is available for the Marketing Funnel dataset.")
    else:
        st.markdown("### 📡 Channel Performance Matrix")

        ch_attr = df.groupby("channel").agg(
            spend=("spend_eur","sum"),
            leads=("leads","sum"),
            activated=("activated","sum"),
            gmv=("gmv_eur","sum"),
            revenue=("revenue_eur","sum"),
        ).reset_index()
        ch_attr["roas"]     = (ch_attr["revenue"] / ch_attr["spend"].replace(0,np.nan)).round(2)
        ch_attr["cpl"]      = (ch_attr["spend"] / ch_attr["leads"].replace(0,np.nan)).round(2)
        ch_attr["cpa"]      = (ch_attr["spend"] / ch_attr["activated"].replace(0,np.nan)).round(2)
        ch_attr["act_rate"] = (ch_attr["activated"] / ch_attr["leads"] * 100).round(1)
        ch_attr["spend_share"] = (ch_attr["spend"] / ch_attr["spend"].sum() * 100).round(1)

        # Bubble chart
        fig = px.scatter(ch_attr.dropna(subset=["roas"]),
                         x="roas", y="act_rate", size="spend", color="channel",
                         text="channel", title="ROAS vs Activation Rate (bubble = spend)",
                         color_discrete_sequence=COLORS, size_max=60,
                         labels={"roas":"ROAS (×)","act_rate":"Activation Rate (%)"})
        fig.update_traces(textposition="top center", textfont_size=11)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(ch_attr.dropna(subset=["roas"]).sort_values("roas"),
                         x="roas", y="channel", orientation="h",
                         title="ROAS by Channel", text_auto=".2f",
                         color="roas", color_continuous_scale=["#ef4444","#f59e0b","#10b981"])
            fig.update_traces(texttemplate="%{text}×")
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(ch_attr.dropna(subset=["cpa"]).sort_values("cpa",ascending=False),
                         x="cpa", y="channel", orientation="h",
                         title="Cost per Activation by Channel (€)", text_auto=".0f",
                         color="cpa", color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
            fig.update_traces(texttemplate="€%{text}")
            fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Attribution table
        st.markdown("### 📋 Full Attribution Table")
        display = ch_attr[["channel","spend","leads","activated","act_rate","roas","cpl","cpa","spend_share"]].copy()
        display.columns = ["Channel","Spend (€)","Leads","Activated","Act. Rate %","ROAS","CPL (€)","CPA (€)","Spend Share %"]
        display = display.round(2)
        # Colour-code ROAS column manually (no matplotlib needed)
        def highlight_roas(val):
            try:
                v = float(str(val).replace("×",""))
                if v >= 4:   return "background-color: #065f46; color: white"
                elif v >= 2.5: return "background-color: #78350f; color: white"
                else:        return "background-color: #7f1d1d; color: white"
            except: return ""
        def highlight_cpa(val):
            try:
                v = float(str(val).replace("€","").replace("N/A","999"))
                if v <= 10:  return "background-color: #065f46; color: white"
                elif v <= 40: return "background-color: #78350f; color: white"
                else:        return "background-color: #7f1d1d; color: white"
            except: return ""
        display_fmt = display.copy()
        display_fmt["Spend (€)"]    = display_fmt["Spend (€)"].apply(lambda x: f"€{x:,.0f}" if x else "€0")
        display_fmt["ROAS"]         = display_fmt["ROAS"].apply(lambda x: f"{x:.2f}×" if x else "N/A")
        display_fmt["Act. Rate %"]  = display_fmt["Act. Rate %"].apply(lambda x: f"{x:.1f}%" if x else "-")
        display_fmt["Spend Share %"]= display_fmt["Spend Share %"].apply(lambda x: f"{x:.1f}%" if x else "-")
        st.dataframe(
            display_fmt.style
                .applymap(highlight_roas, subset=["ROAS"])
                .applymap(highlight_cpa,  subset=["CPA (€)"]),
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 Ask Anything About Your Data")

    # Suggested questions
    dataset_name = st.session_state.active_name or ""
    matching_key = next((k for k in SUGGESTED_QUESTIONS if k in dataset_name), None)
    if matching_key:
        st.markdown("**💡 Suggested questions:**")
        cols = st.columns(3)
        for i, q in enumerate(SUGGESTED_QUESTIONS[matching_key]):
            if cols[i % 3].button(q, key=f"sq_{i}"):
                st.session_state.chat_input_val = q

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask a question about your data...")

    # Also accept suggested question click
    if st.session_state.chat_input_val:
        user_input = st.session_state.chat_input_val
        st.session_state.chat_input_val = ""

    def _pandas_answer(df: pd.DataFrame, question: str) -> str:
        q = question.lower()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        try:
            if any(w in q for w in ["top", "highest", "best", "maximum", "most"]):
                col = next((c for c in num_cols if any(w in c for w in ["gmv","revenue","roas","leads","rate"])), num_cols[0] if num_cols else None)
                if col and len(df.columns) > 1:
                    cat_col = df.select_dtypes(include="object").columns[0]
                    top = df.groupby(cat_col)[col].sum().nlargest(5).reset_index()
                    return f"**Top 5 by {col}:**\n\n" + top.to_markdown(index=False)

            if any(w in q for w in ["trend", "monthly", "over time"]):
                time_col = next((c for c in df.columns if "month" in c or "date" in c), None)
                if time_col and num_cols:
                    trend = df.groupby(time_col)[num_cols[0]].sum().reset_index()
                    return f"**Monthly trend for {num_cols[0]}:**\n\n" + trend.to_markdown(index=False)

            if any(w in q for w in ["average", "mean", "avg"]):
                means = df[num_cols].mean().round(2)
                return "**Column averages:**\n\n" + means.to_markdown()

            if any(w in q for w in ["compare", "vs", "channel", "country", "segment"]):
                cat_col = next((c for c in ["channel","country","segment","industry"] if c in df.columns), None)
                if cat_col and num_cols:
                    comp = df.groupby(cat_col)[num_cols[:3]].sum().round(2).reset_index()
                    return f"**Comparison by {cat_col}:**\n\n" + comp.to_markdown(index=False)

            # Default: summary stats
            summary = df[num_cols].describe().round(2)
            return f"Here's a statistical summary of the dataset:\n\n{summary.to_markdown()}\n\n💡 Try asking for specific insights!"

        except Exception as e:
            return f"I found {len(df):,} rows and {len(df.columns)} columns in this dataset. {str(e)}"


    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})


        # Use Ollama-powered answer (local, no API key needed)
        try:
            from langchain_experimental.agents import create_pandas_dataframe_agent
            from langchain_ollama import ChatOllama


            llm = ChatOllama(model="qwen2.5:7b", temperature=0, base_url="http://localhost:11434")
            agent = create_pandas_dataframe_agent(llm, df, verbose=False,
                                                allow_dangerous_code=True,
                                                agent_type="openai-tools")  # agent_type stays the same
            response = agent.run(user_input)
        except Exception as e:
            response = f"⚠️ Ollama error: {e}. Using built-in analysis instead.\n\n" + _pandas_answer(df, user_input)


        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AUTO INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔍 Automated Business Insights")
    st.markdown("*AI-generated findings surfaced automatically from your data*")

    insights = generate_insights(df, st.session_state.active_name or "")

    if not insights:
        st.info("No specific insights generated for this dataset. Try the Marketing Funnel dataset!")
    else:
        for ins in insights:
            css_class = f"insight-{ins['type']}"
            st.markdown(f"""
            <div class="{css_class}">
              <div class="insight-title">{ins['icon']} {ins['title']}</div>
              <div class="insight-text">{ins['text']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Raw data preview
    st.markdown("### 📋 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Period",  f"{df['month'].min()} → {df['month'].max()}" if "month" in df.columns else "—")

    st.dataframe(df.head(20), use_container_width=True)

    # Download
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered data as CSV", csv,
                        file_name="revenue_analytics_export.csv", mime="text/csv")