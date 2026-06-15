"""
insights.py — Automated insight generation engine
Scans the active dataframe and surfaces key business findings.
"""

import pandas as pd
import numpy as np


def generate_insights(df: pd.DataFrame, dataset_name: str) -> list[dict]:
    """Return a list of insight dicts: {icon, title, text, type}"""
    insights = []

    try:
        if "Marketing Funnel" in dataset_name:
            insights += _funnel_insights(df)
        elif "CRM" in dataset_name:
            insights += _crm_insights(df)
        elif "Brand" in dataset_name:
            insights += _brand_insights(df)
        elif "Transaction" in dataset_name:
            insights += _transaction_insights(df)
    except Exception:
        pass

    return insights[:6]  # max 6 insights


def _funnel_insights(df):
    insights = []
    agg = df.groupby("channel").agg(
        spend=("spend_eur", "sum"),
        revenue=("revenue_eur", "sum"),
        leads=("leads", "sum"),
        activated=("activated", "sum"),
    ).reset_index()
    agg["roas"] = (agg["revenue"] / agg["spend"].replace(0, np.nan)).round(2)
    agg["cpl"]  = (agg["spend"] / agg["leads"].replace(0, np.nan)).round(2)
    agg["act_rate"] = (agg["activated"] / agg["leads"].replace(0, np.nan) * 100).round(1)

    total_spend = agg["spend"].sum()
    agg["spend_share"] = (agg["spend"] / total_spend * 100).round(1)

    # Best ROAS channel
    best = agg.dropna(subset=["roas"]).nlargest(1, "roas").iloc[0]
    worst = agg.dropna(subset=["roas"]).nsmallest(1, "roas").iloc[0]
    insights.append({
        "icon": "🏆", "type": "positive",
        "title": f"{best['channel']} leads with {best['roas']}× ROAS",
        "text": f"Highest return on ad spend. Generates €{best['roas']} in revenue per €1 spent. "
                f"Only {best['spend_share']}% of total budget allocated here — strong case to scale up."
    })

    # Underinvested high-performer
    high_roas_low_spend = agg.dropna(subset=["roas"]).query("roas > 4 and spend_share < 15")
    if not high_roas_low_spend.empty:
        row = high_roas_low_spend.iloc[0]
        insights.append({
            "icon": "💡", "type": "opportunity",
            "title": f"Budget reallocation opportunity: {row['channel']}",
            "text": f"{row['channel']} has {row['roas']}× ROAS but only {row['spend_share']}% of total spend. "
                    f"Reallocating 10% of {worst['channel']} budget here could unlock significant incremental revenue."
        })

    # Worst performer
    insights.append({
        "icon": "⚠️", "type": "warning",
        "title": f"{worst['channel']} needs attention — {worst['roas']}× ROAS",
        "text": f"Lowest ROAS in the portfolio. Lead-to-activation rate is "
                f"{agg.loc[agg.channel==worst['channel'], 'act_rate'].values[0]}%. "
                f"Consider pausing low-performing ad sets or testing new creatives."
    })

    # Best activation rate
    best_act = agg.nlargest(1, "act_rate").iloc[0]
    insights.append({
        "icon": "✅", "type": "positive",
        "title": f"{best_act['channel']} drives fastest merchant activation",
        "text": f"{best_act['act_rate']}% lead-to-activation rate — "
                f"{round(best_act['act_rate'] / agg['act_rate'].mean(), 1)}× the portfolio average. "
                f"Cost per activation: €{best_act['cpl'] * (100/best_act['act_rate']):.0f}."
    })

    # Monthly trend
    monthly = df.groupby("month").agg(gmv=("gmv_eur","sum"), spend=("spend_eur","sum")).reset_index()
    if len(monthly) >= 3:
        first3  = monthly.head(3)["gmv"].mean()
        last3   = monthly.tail(3)["gmv"].mean()
        growth  = round((last3 - first3) / first3 * 100, 1)
        emoji   = "📈" if growth > 0 else "📉"
        insights.append({
            "icon": emoji, "type": "positive" if growth > 0 else "warning",
            "title": f"GMV trend: {'+' if growth>0 else ''}{growth}% (H2 vs H1)",
            "text": f"Average monthly GMV grew from €{first3:,.0f} in H1 to €{last3:,.0f} in H2. "
                    f"{'Strong growth momentum heading into year-end.' if growth > 10 else 'Moderate growth — consider acceleration strategies.'}"
        })

    return insights


def _crm_insights(df):
    insights = []
    seg_agg = df.groupby("segment").agg(
        open_rate=("email_open_rate", "mean"),
        ctr=("email_ctr", "mean"),
        push=("push_open_rate", "mean"),
    ).round(1).reset_index()

    best_seg  = seg_agg.nlargest(1, "open_rate").iloc[0]
    worst_seg = seg_agg.nsmallest(1, "open_rate").iloc[0]

    insights.append({
        "icon": "🏆", "type": "positive",
        "title": f"{best_seg['segment'].replace('_',' ').title()} segment leads engagement",
        "text": f"{best_seg['open_rate']}% email open rate — "
                f"{round(best_seg['open_rate']/seg_agg['open_rate'].mean(),1)}× the average. "
                f"Click rate: {best_seg['ctr']}%. These merchants are highly engaged and ripe for upsell."
    })
    insights.append({
        "icon": "🔄", "type": "opportunity",
        "title": f"Re-engagement opportunity: {worst_seg['segment'].replace('_',' ').title()}",
        "text": f"Only {worst_seg['open_rate']}% open rate. "
                f"A targeted win-back campaign with personalised offers could reactivate these merchants."
    })

    # Monthly trend
    monthly = df.groupby("month").agg(open=("email_open_rate","mean")).reset_index()
    if len(monthly) >= 3:
        delta = round(monthly["open"].iloc[-1] - monthly["open"].iloc[0], 1)
        insights.append({
            "icon": "📈" if delta > 0 else "📉", "type": "positive" if delta > 0 else "warning",
            "title": f"Email open rate {'improved' if delta>0 else 'declined'} {abs(delta)}pp YTD",
            "text": f"From {monthly['open'].iloc[0]}% in Jan to {monthly['open'].iloc[-1]}% in Dec. "
                    f"{'Consistent improvement suggests better targeting and content.' if delta>0 else 'Review subject line strategy and send-time optimisation.'}"
        })

    return insights


def _brand_insights(df):
    insights = []
    latest = df[df["month"] == df["month"].max()]
    best_aware = latest.nlargest(1, "aided_awareness").iloc[0]
    insights.append({
        "icon": "🌟", "type": "positive",
        "title": f"{best_aware['country']} leads on brand awareness",
        "text": f"{best_aware['aided_awareness']}% aided awareness, {best_aware['unaided_awareness']}% unaided. "
                f"NPS: {best_aware['nps_score']}. Strong brand equity in this market."
    })

    # SOV growth
    sov_by_month = df.groupby("month").agg(sov=("share_of_voice","mean")).reset_index()
    if len(sov_by_month) >= 2:
        growth = round(sov_by_month["sov"].iloc[-1] - sov_by_month["sov"].iloc[0], 1)
        insights.append({
            "icon": "📊", "type": "positive" if growth > 0 else "warning",
            "title": f"Share of Voice {'grew' if growth>0 else 'declined'} {abs(growth)}pp YTD",
            "text": f"Average SOV moved from {sov_by_month['sov'].iloc[0]}% to {sov_by_month['sov'].iloc[-1]}% over the year. "
                    f"{'Brand investment is paying off in competitive visibility.' if growth>0 else 'Competitors gaining ground — review media investment strategy.'}"
        })

    return insights


def _transaction_insights(df):
    insights = []
    ind_agg = df.groupby("industry").agg(gmv=("gmv_eur","mean"), txn=("transactions","sum")).round(1).reset_index()
    best_ind = ind_agg.nlargest(1,"gmv").iloc[0]
    insights.append({
        "icon": "💰", "type": "positive",
        "title": f"{best_ind['industry']} merchants drive highest basket size",
        "text": f"Average GMV per merchant: €{best_ind['gmv']}. "
                f"Focus acquisition efforts here for maximum revenue per merchant."
    })

    ch_agg = df.groupby("channel").agg(gmv=("gmv_eur","sum")).reset_index()
    best_ch = ch_agg.nlargest(1,"gmv").iloc[0]
    insights.append({
        "icon": "📡", "type": "positive",
        "title": f"{best_ch['channel']} drives highest merchant GMV",
        "text": f"€{best_ch['gmv']:,.0f} total GMV from merchants acquired via {best_ch['channel']}."
    })

    return insights