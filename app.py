import streamlit as st
import pandas as pd
import plotly.express as px
from agent import run_agent, clean_data, generate_report
from charts import auto_chart, suggest_charts
from fpdf import FPDF
import io

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #4a6fa5;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    div[data-testid="stChatMessage"] {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🤖 Data Analyst Agent")
st.markdown("**Chat with your data in plain English — powered by GPT & LangChain**")
st.markdown("---")

# --- Feature Cards ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("💬 **Chat with Data**\nAsk questions in plain English and get instant AI answers")
with col2:
    st.info("📊 **Smart Charts**\nAuto-suggested charts + 5 chart types with color grouping")
with col3:
    st.info("📋 **Column Profiler**\nInstant data quality overview per column")
with col4:
    st.info("📥 **PDF Export**\nDownload your full chat + insights as a report")

st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR — Data Loading
# ─────────────────────────────────────────
st.sidebar.title("📂 Load Your Data")

st.sidebar.markdown("### 🗂 Try a Sample Dataset")
sample_choice = st.sidebar.selectbox(
    "Pick a built-in dataset:",
    ["None", "🛒 Superstore Sales", "🚢 Titanic", "🏠 Airbnb NYC"]
)

SAMPLE_URLS = {
    "🛒 Superstore Sales": "C:\Data Analyst agent project\superstore.csv",
    "🚢 Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "🏠 Airbnb NYC":"https://raw.githubusercontent.com/YueminLi/Airbnb_NYC_2019/refs/heads/master/AB_NYC_2019.csv"
}

st.sidebar.markdown("### ⬆️ Upload CSV File(s)")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV files",
    type="csv",
    accept_multiple_files=True
)

st.sidebar.markdown("### 🔗 Or Paste a URL")
url_input = st.sidebar.text_input("Public CSV or Google Sheets URL:")
st.sidebar.caption("For Google Sheets: File → Share → Publish to web → CSV")

# ─────────────────────────────────────────
# DATA LOADING LOGIC
# ─────────────────────────────────────────
df = None
data_source = None

def load_csv_with_encoding(file_or_path):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    return None

if uploaded_files:
    dfs = [load_csv_with_encoding(f) for f in uploaded_files]
    dfs = [d for d in dfs if d is not None]
    if dfs:
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        data_source = f"{len(uploaded_files)} uploaded file(s)"

elif url_input.strip():
    try:
        if "docs.google.com" in url_input:
            url_input = url_input.replace("/edit#gid=", "/export?format=csv&gid=").replace("/edit", "/export?format=csv")
        df = pd.read_csv(url_input)
        data_source = "URL"
    except Exception as e:
        st.sidebar.error(f"Could not load URL: {str(e)}")

elif sample_choice != "None":
    source = SAMPLE_URLS[sample_choice]
    try:
        df = pd.read_csv(source)
        data_source = f"sample: {sample_choice}"
    except Exception as e:
        st.sidebar.error(f"Could not load sample: {str(e)}")

# ─────────────────────────────────────────
# MAIN APP — After data is loaded
# ─────────────────────────────────────────
if df is not None:
    df = clean_data(df)

    st.success(f"✅ Loaded from {data_source} — **{df.shape[0]:,} rows**, **{df.shape[1]} columns**")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    missing_pct = round(df.isnull().mean().mean() * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{df.shape[0]:,}")
    k2.metric("Columns", df.shape[1])
    k3.metric("Numeric Cols", len(numeric_cols))
    k4.metric("Missing Data", f"{missing_pct}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Visualizations", "📋 Auto Report", "🔍 Column Profiler"])

    # ── Tab 1: Chat ──────────────────────────────────────────────────────────
    with tab1:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.chat_input("Ask a question about your data...")

        if question:
            with st.spinner("Thinking..."):
                try:
                    answer = run_agent(df, question, st.session_state.chat_history)
                except Exception as e:
                    answer = f"Sorry, I couldn't process that. Try rephrasing. Error: {str(e)}"
            st.session_state.chat_history.append((question, answer))

            # Auto chart after each answer
            fig = auto_chart(df, question)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        if st.session_state.chat_history:
            for q, a in reversed(st.session_state.chat_history):
                st.chat_message("user").write(q)
                st.chat_message("assistant").write(a)

            col_clear, col_export = st.columns([1, 1])
            with col_clear:
                if st.button("🗑 Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col_export:
                if st.button("📥 Export Chat as PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 16)
                    pdf.cell(0, 10, "Data Analyst Agent — Chat Report", ln=True)
                    pdf.set_font("Helvetica", size=10)
                    pdf.ln(4)
                    for q, a in st.session_state.chat_history:
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.multi_cell(0, 7, f"Q: {q}")
                        pdf.set_font("Helvetica", size=10)
                        safe_a = a.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 7, f"A: {safe_a}")
                        pdf.ln(3)
                    pdf_bytes = bytes(pdf.output())
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name="chat_report.pdf",
                        mime="application/pdf"
                    )
        else:
            st.markdown("**💡 Try asking:**")
            examples = [
                "What are the top 5 products by sales?",
                "Which region has the highest profit?",
                "What is the average discount by category?",
                "How many duplicate rows are there?",
                "Show a summary of numeric columns"
            ]
            for ex in examples:
                st.code(ex)

    # ── Tab 2: Visualizations ─────────────────────────────────────────────────
    with tab2:
        st.subheader("📊 Explore Your Data")
        all_cols = df.columns.tolist()

        suggestions = suggest_charts(df)
        if suggestions:
            st.markdown("**✨ Suggested Charts:**")
            sug_cols = st.columns(len(suggestions))
            chosen = None
            for i, sug in enumerate(suggestions):
                if sug_cols[i].button(sug["label"]):
                    chosen = sug

        if not numeric_cols:
            st.warning("No numeric columns found for visualization.")
        else:
            chart_type_default = chosen["type"] if chosen else "Bar"
            x_default = chosen["x"] if chosen else (categorical_cols[0] if categorical_cols else all_cols[0])
            y_default = chosen["y"] if chosen else numeric_cols[0]

            chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"],
                                      index=["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"].index(chart_type_default)
                                      if chart_type_default in ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"] else 0)
            c1, c2 = st.columns(2)
            with c1:
                x_axis = st.selectbox("X axis", all_cols, index=all_cols.index(x_default) if x_default in all_cols else 0)
            with c2:
                y_axis = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index(y_default) if y_default in numeric_cols else 0)

            color = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            color_col = None if color == "None" else color

            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, template="plotly_dark")
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_col, template="plotly_dark")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, template="plotly_dark")
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, color=color_col, template="plotly_dark")
                elif chart_type == "Box":
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_col, template="plotly_dark")
                elif chart_type == "Pie":
                    grouped = df.groupby(x_axis)[y_axis].sum().reset_index().sort_values(y_axis, ascending=False).head(8)
                    fig = px.pie(grouped, names=x_axis, values=y_axis, hole=0.35, template="plotly_dark")

                fig.update_layout(plot_bgcolor="#1a1a2e", paper_bgcolor="#0f1117", font_color="#f0f0f0")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render chart: {str(e)}")

    # ── Tab 3: Auto Report ────────────────────────────────────────────────────
    with tab3:
        st.subheader("📋 Auto-Generated Data Insights")
        st.markdown(generate_report(df))
        st.markdown("**Preview (first 20 rows):**")
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("📥 Export Insights Report as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Data Analyst Agent — Insights Report", ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.ln(4)
            report_text = generate_report(df)
            for line in report_text.split("\n"):
                safe = line.replace("**", "").encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 7, safe)
            pdf_bytes = bytes(pdf.output())
            st.download_button("⬇️ Download PDF", pdf_bytes, "insights_report.pdf", "application/pdf")

    # ── Tab 4: Column Profiler ────────────────────────────────────────────────
    with tab4:
        st.subheader("🔍 Column-by-Column Data Profile")

        profile_data = []
        for col in df.columns:
            profile_data.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing (%)": f"{round(df[col].isnull().mean() * 100, 1)}%",
                "Unique Values": df[col].nunique(),
                "Sample Values": ", ".join([str(v) for v in df[col].dropna().unique()[:3]])
            })

        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, height=400)

        st.markdown("**Numeric Column Statistics:**")
        st.dataframe(df.describe().T.round(2), use_container_width=True)

else:
    st.markdown("### 👈 Load data from the sidebar to get started")
    st.markdown("You can upload a CSV file, paste a public URL, or try one of the built-in sample datasets.")
    st.markdown("**Example questions you can ask once loaded:**")
    c1, c2 = st.columns(2)
    with c1:
        st.code("What are the top 5 products by sales?")
        st.code("Show me the average profit by category")
        st.code("Which region has the highest revenue?")
    with c2:
        st.code("How many orders were placed in 2023?")
        st.code("What is the total discount given?")
        st.code("Which customer has the most orders?")