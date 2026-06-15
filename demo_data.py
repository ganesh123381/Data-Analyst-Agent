"""
demo_data.py — Pre-loaded SumUp-style Revenue Analytics demo datasets
Generates realistic B2SMB marketing funnel data for the demo mode.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

CHANNELS  = ["Paid Search", "Paid Social", "Email CRM", "Organic SEO", "Display", "Referral"]
COUNTRIES = ["DE", "GB", "FR", "IT", "ES"]
INDUSTRIES= ["Retail", "Food & Beverage", "Beauty", "Services", "Health", "Hospitality"]
SEGMENTS  = ["new_lead", "onboarding", "inactive_90d", "high_value", "churned"]

START = datetime(2024, 1, 1)
MONTHS = pd.date_range("2024-01-01", periods=12, freq="MS")


def make_funnel_df():
    """Monthly full-funnel data by channel and country."""
    rows = []
    channel_params = {
        "Paid Search":  {"spend": 12000, "leads": 180, "act_rate": 0.38, "roas_base": 3.4},
        "Paid Social":  {"spend": 7000,  "leads": 130, "act_rate": 0.29, "roas_base": 2.2},
        "Email CRM":    {"spend": 3500,  "leads": 90,  "act_rate": 0.52, "roas_base": 5.2},
        "Organic SEO":  {"spend": 0,     "leads": 80,  "act_rate": 0.41, "roas_base": 0},
        "Display":      {"spend": 2500,  "leads": 55,  "act_rate": 0.21, "roas_base": 1.4},
        "Referral":     {"spend": 1500,  "leads": 52,  "act_rate": 0.60, "roas_base": 6.1},
    }
    for month in MONTHS:
        trend = (MONTHS.tolist().index(month)) / 11
        for ch, p in channel_params.items():
            for country in COUNTRIES:
                country_factor = {"DE": 1.0, "GB": 0.85, "FR": 0.65, "IT": 0.55, "ES": 0.50}[country]
                noise = np.random.uniform(0.88, 1.12)
                spend   = round(p["spend"] * country_factor * noise * (1 + trend * 0.15), 0)
                leads   = max(1, int(p["leads"] * country_factor * noise * (1 + trend * 0.10)))
                act_r   = min(0.75, p["act_rate"] * np.random.uniform(0.92, 1.08))
                activated = max(0, int(leads * act_r))
                transacting = max(0, int(activated * np.random.uniform(0.78, 0.88)))
                gmv     = round(transacting * np.random.uniform(280, 420) * (1 + trend * 0.08), 0)
                revenue = round(gmv * np.random.uniform(0.016, 0.022), 2)
                impressions = int(spend * np.random.uniform(80, 120)) if spend > 0 else int(leads * np.random.uniform(8000, 15000))
                clicks  = int(spend * np.random.uniform(0.6, 1.4) / max(0.5, np.random.uniform(0.8, 3.0))) if spend > 0 else leads * 8

                rows.append({
                    "month":              month.strftime("%Y-%m"),
                    "channel":            ch,
                    "country":            country,
                    "spend_eur":          spend,
                    "impressions":        impressions,
                    "clicks":             clicks,
                    "leads":              leads,
                    "activated":          activated,
                    "transacting":        transacting,
                    "gmv_eur":            gmv,
                    "revenue_eur":        revenue,
                    "cost_per_lead":      round(spend / leads, 2) if leads > 0 and spend > 0 else 0,
                    "cost_per_activation":round(spend / activated, 2) if activated > 0 and spend > 0 else 0,
                    "lead_to_act_rate":   round(act_r * 100, 1),
                    "roas":               round(revenue / spend, 2) if spend > 0 else None,
                })
    return pd.DataFrame(rows)


def make_crm_df():
    """Monthly CRM engagement by segment and country."""
    rows = []
    seg_params = {
        "high_value":   {"open": 0.41, "click": 0.12, "push_open": 0.28},
        "onboarding":   {"open": 0.35, "click": 0.09, "push_open": 0.22},
        "new_lead":     {"open": 0.28, "click": 0.07, "push_open": 0.18},
        "inactive_90d": {"open": 0.18, "click": 0.04, "push_open": 0.11},
        "churned":      {"open": 0.12, "click": 0.02, "push_open": 0.07},
    }
    for month in MONTHS:
        trend = MONTHS.tolist().index(month) / 11
        for seg, p in seg_params.items():
            for country in COUNTRIES:
                noise = np.random.uniform(0.92, 1.08)
                sent   = random.randint(800, 2500)
                opened = int(sent * p["open"] * noise * (1 + trend * 0.05))
                clicked= int(opened * p["click"] / p["open"] * noise)
                push_s = random.randint(400, 1200)
                push_o = int(push_s * p["push_open"] * noise)
                sms_s  = random.randint(200, 600)
                sms_c  = int(sms_s * np.random.uniform(0.10, 0.16))
                rows.append({
                    "month":          month.strftime("%Y-%m"),
                    "segment":        seg,
                    "country":        country,
                    "emails_sent":    sent,
                    "emails_opened":  opened,
                    "emails_clicked": clicked,
                    "email_open_rate":round(opened / sent * 100, 1),
                    "email_ctr":      round(clicked / sent * 100, 1),
                    "email_ctor":     round(clicked / max(1, opened) * 100, 1),
                    "push_sent":      push_s,
                    "push_opened":    push_o,
                    "push_open_rate": round(push_o / push_s * 100, 1),
                    "sms_sent":       sms_s,
                    "sms_clicked":    sms_c,
                    "sms_ctr":        round(sms_c / sms_s * 100, 1),
                })
    return pd.DataFrame(rows)


def make_brand_df():
    """Monthly brand health metrics by country."""
    rows = []
    country_base = {
        "DE": {"aided": 0.58, "unaided": 0.24, "consider": 0.31, "nps": 48, "sov": 0.19},
        "GB": {"aided": 0.52, "unaided": 0.20, "consider": 0.27, "nps": 44, "sov": 0.16},
        "FR": {"aided": 0.46, "unaided": 0.17, "consider": 0.23, "nps": 40, "sov": 0.13},
        "IT": {"aided": 0.43, "unaided": 0.15, "consider": 0.21, "nps": 38, "sov": 0.11},
        "ES": {"aided": 0.41, "unaided": 0.14, "consider": 0.19, "nps": 36, "sov": 0.10},
    }
    for month in MONTHS:
        trend = MONTHS.tolist().index(month) / 11
        for country, b in country_base.items():
            noise = np.random.uniform(0.97, 1.03)
            rows.append({
                "month":             month.strftime("%Y-%m"),
                "country":           country,
                "aided_awareness":   round(min(0.85, b["aided"] * noise * (1 + trend * 0.08)) * 100, 1),
                "unaided_awareness": round(min(0.50, b["unaided"] * noise * (1 + trend * 0.10)) * 100, 1),
                "consideration":     round(min(0.60, b["consider"] * noise * (1 + trend * 0.09)) * 100, 1),
                "nps_score":         round(b["nps"] * noise * (1 + trend * 0.10), 1),
                "share_of_voice":    round(min(0.40, b["sov"] * noise * (1 + trend * 0.15)) * 100, 1),
            })
    return pd.DataFrame(rows)


def make_transactions_df():
    """Sample merchant-level transaction data."""
    rows = []
    for i in range(500):
        month = random.choice(MONTHS)
        ch    = random.choice(CHANNELS)
        ind   = random.choice(INDUSTRIES)
        country = random.choices(COUNTRIES, weights=[30,22,18,16,14])[0]
        gmv   = round(abs(np.random.normal(85, 55)), 2)
        rows.append({
            "merchant_id":   f"M{i+1:05d}",
            "month":         month.strftime("%Y-%m"),
            "channel":       ch,
            "country":       country,
            "industry":      ind,
            "gmv_eur":       gmv,
            "fee_eur":       round(gmv * np.random.uniform(0.015, 0.022), 4),
            "transactions":  random.randint(1, 80),
            "avg_basket_eur":round(gmv / random.randint(1, 30), 2),
        })
    return pd.DataFrame(rows)


# Pre-load all datasets once
FUNNEL_DF       = make_funnel_df()
CRM_DF          = make_crm_df()
BRAND_DF        = make_brand_df()
TRANSACTIONS_DF = make_transactions_df()

DATASETS = {
    "📊 Marketing Funnel (by channel/country/month)": FUNNEL_DF,
    "💌 CRM Engagement (email/push/SMS by segment)":  CRM_DF,
    "🌟 Brand Health (awareness/NPS/SOV)":             BRAND_DF,
    "💳 Merchant Transactions":                        TRANSACTIONS_DF,
}

SUGGESTED_QUESTIONS = {
    "📊 Marketing Funnel (by channel/country/month)": [
        "Which channel has the highest ROAS?",
        "What is the lead to activation rate by channel?",
        "Show me monthly GMV trend for Germany",
        "Which country has the lowest cost per activation?",
        "Compare spend vs revenue across all channels",
        "What is the top performing channel in Q4?",
    ],
    "💌 CRM Engagement (email/push/SMS by segment)": [
        "Which segment has the highest email open rate?",
        "Compare email vs push vs SMS engagement rates",
        "Show me click-to-open rate trend over the year",
        "Which country has the best CRM engagement?",
        "How does the high_value segment perform vs churned?",
    ],
    "🌟 Brand Health (awareness/NPS/SOV)": [
        "Which country has the highest brand awareness?",
        "Show me NPS trend over the year",
        "How has share of voice grown in Germany?",
        "Compare aided vs unaided awareness by country",
    ],
    "💳 Merchant Transactions": [
        "Which industry has the highest average GMV?",
        "Show me transaction volume by channel",
        "What is the average basket size by country?",
        "Which channel drives the most merchant revenue?",
    ],
}