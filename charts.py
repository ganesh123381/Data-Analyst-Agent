import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# ── Shared dark theme layout ───────────────────────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#0f1117",
    font_color="#f0f0f0",
    title_font_size=16,
    margin=dict(t=50, l=40, r=20, b=40)
)

EXCLUDE_COLS = {'row id', 'row_id', 'postal code', 'zip', 'id', 'index'}

X_KEYWORD_MAP = {
    ('category', 'categories'): 'Category',
    ('region',): 'Region',
    ('segment',): 'Segment',
    ('state', 'states'): 'State',
    ('city', 'cities'): 'City',
    ('country',): 'Country',
    ('product', 'products'): 'Product Name',
    ('customer', 'customers'): 'Customer Name',
    ('ship mode', 'shipping'): 'Ship Mode',
    ('sub-category', 'subcategory'): 'Sub-Category',
    ('year',): 'year',
    ('month',): 'month',
}

Y_KEYWORD_MAP = {
    ('sales', 'revenue'): 'Sales',
    ('profit', 'profitability'): 'Profit',
    ('discount',): 'Discount',
    ('quantity',): 'Quantity',
    ('orders', 'order count'): 'Order ID',
}


def _best_column(question_lower, keyword_map, df_columns, fallback_cols):
    for keywords, preferred in keyword_map.items():
        if any(k in question_lower for k in keywords):
            for col in df_columns:
                if col.lower() == preferred.lower():
                    return col
    return fallback_cols[0] if fallback_cols else None


def _detect_date_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ['date', 'time', 'year', 'month', 'day']):
            return col
    for col in df.select_dtypes(include='object').columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            if parsed.notna().mean() > 0.7:
                return col
        except Exception:
            pass
    return None


def _detect_chart_type(question_lower, has_date):
    if any(k in question_lower for k in ['trend', 'over time', 'monthly', 'yearly', 'timeline', 'growth']):
        return 'line'
    if any(k in question_lower for k in ['distribution', 'spread', 'histogram', 'frequency']):
        return 'histogram'
    if any(k in question_lower for k in ['correlation', 'vs', 'versus', 'compare two', 'scatter']):
        return 'scatter'
    if any(k in question_lower for k in ['proportion', 'share', 'percentage', 'pie', 'breakdown']):
        return 'pie'
    if any(k in question_lower for k in ['box', 'range', 'outlier', 'variance']):
        return 'box'
    if has_date and any(k in question_lower for k in ['monthly', 'daily', 'weekly', 'over', 'time']):
        return 'line'
    return 'bar'


def auto_chart(df, question):
    try:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = [c for c in df.select_dtypes(include='object').columns
                    if c.lower() not in EXCLUDE_COLS]
        all_cols = df.columns.tolist()

        if not numeric_cols:
            return None

        question_lower = question.lower()
        date_col = _detect_date_col(df)
        chart_type = _detect_chart_type(question_lower, date_col is not None)

        x_col = _best_column(question_lower, X_KEYWORD_MAP, all_cols, cat_cols)
        y_col = _best_column(question_lower, Y_KEYWORD_MAP, all_cols, numeric_cols)

        if not x_col:
            x_col = date_col if date_col else (cat_cols[0] if cat_cols else numeric_cols[0])
        if not y_col:
            y_col = numeric_cols[0]

        title = f"{y_col} by {x_col}"

        if chart_type == 'line':
            col = date_col if date_col else x_col
            temp = df.copy()
            try:
                temp[col] = pd.to_datetime(temp[col], infer_datetime_format=True, errors='coerce')
                temp = temp.dropna(subset=[col])
                temp['_period'] = temp[col].dt.to_period('M').astype(str)
                grouped = temp.groupby('_period')[y_col].sum().reset_index()
                grouped.columns = [col, y_col]
            except Exception:
                grouped = temp.groupby(col)[y_col].sum().reset_index()
            fig = px.line(grouped, x=col, y=y_col, title=f"{y_col} Over Time", markers=True)

        elif chart_type == 'histogram':
            fig = px.histogram(df, x=y_col, nbins=30, title=f"Distribution of {y_col}",
                               color_discrete_sequence=["#4a6fa5"])

        elif chart_type == 'scatter':
            y2 = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            color_col = cat_cols[0] if cat_cols else None
            fig = px.scatter(df, x=y_col, y=y2, color=color_col,
                             title=f"{y_col} vs {y2}", opacity=0.7)

        elif chart_type == 'pie':
            grouped = df.groupby(x_col)[y_col].sum().reset_index()
            grouped = grouped.sort_values(y_col, ascending=False).head(8)
            fig = px.pie(grouped, names=x_col, values=y_col,
                         title=f"{y_col} Share by {x_col}", hole=0.35)

        elif chart_type == 'box':
            color_col = cat_cols[0] if cat_cols else None
            fig = px.box(df, x=x_col, y=y_col, color=color_col,
                         title=f"Distribution of {y_col} by {x_col}")

        else:
            grouped = df.groupby(x_col)[y_col].sum().reset_index()
            grouped = grouped.sort_values(y_col, ascending=False).head(12)
            fig = px.bar(grouped, x=x_col, y=y_col, title=title,
                         color=y_col, color_continuous_scale="Blues")

        fig.update_layout(**DARK_LAYOUT)
        return fig

    except Exception as e:
        print(f"Chart error: {e}")
        return None


def suggest_charts(df):
    numeric = df.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in df.select_dtypes(include='object').columns
                   if c.lower() not in EXCLUDE_COLS]
    date_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ['date', 'time', 'year', 'month'])]

    suggestions = []
    if date_cols and numeric:
        suggestions.append({"label": f"📈 {numeric[0]} over time", "type": "Line",
                             "x": date_cols[0], "y": numeric[0]})
    if categorical and numeric:
        suggestions.append({"label": f"📊 {categorical[0]} by {numeric[0]}", "type": "Bar",
                             "x": categorical[0], "y": numeric[0]})
    if len(numeric) >= 2:
        suggestions.append({"label": f"🔵 {numeric[0]} vs {numeric[1]}", "type": "Scatter",
                             "x": numeric[0], "y": numeric[1]})
    if categorical and numeric:
        suggestions.append({"label": f"🥧 {numeric[0]} share by {categorical[0]}", "type": "Pie",
                             "x": categorical[0], "y": numeric[0]})
    if numeric:
        suggestions.append({"label": f"📉 Distribution of {numeric[0]}", "type": "Histogram",
                             "x": numeric[0], "y": numeric[0]})
    return suggestions