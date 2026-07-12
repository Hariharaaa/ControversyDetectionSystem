"""
Streamlit Dashboard for the ESG Controversy Detection System.

Premium dark-themed dashboard with glassmorphism design, animated score gauges,
interactive company navigation, and polished chart styling.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.scorer import ControversyScorer

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="ESG Controversy Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Premium CSS — Glassmorphism Dark Theme
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        background: #f8fafc;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* ── Hide default chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stStatusWidget"] { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
        border-right: 1px solid rgba(0,0,0,0.04);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label {
        color: #64748b !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* ── Header ── */
    .dash-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.06) 50%, rgba(14,165,233,0.04) 100%);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        backdrop-filter: blur(20px);
    }
    .dash-header-icon {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 16px rgba(99,102,241,0.2);
        flex-shrink: 0;
        color: white;
    }
    .dash-header h1 {
        color: #0f172a;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .dash-header p {
        color: #475569;
        font-size: 13px;
        margin: 2px 0 0 0;
        font-weight: 400;
    }

    /* ── Score Gauge ── */
    .score-gauge {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    }
    .score-gauge::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at center, var(--glow-color, rgba(99,102,241,0.06)) 0%, transparent 70%);
        pointer-events: none;
    }
    .score-gauge .score-num {
        font-size: 72px;
        font-weight: 900;
        letter-spacing: -3px;
        line-height: 1;
        margin: 6px 0 4px;
        position: relative;
    }
    .score-gauge .score-of {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 2px;
        position: relative;
    }
    .score-gauge .score-label {
        color: #64748b;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        position: relative;
    }

    /* ── Risk Badges ── */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        position: relative;
    }
    .risk-badge::before {
        content: '';
        width: 6px; height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .risk-low {
        background: rgba(16,185,129,0.12);
        color: #059669;
        border: 1px solid rgba(16,185,129,0.2);
    }
    .risk-low::before { background: #10b981; box-shadow: 0 0 6px #10b981; }
    .risk-medium {
        background: rgba(245,158,11,0.12);
        color: #d97706;
        border: 1px solid rgba(245,158,11,0.2);
    }
    .risk-medium::before { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
    .risk-high {
        background: rgba(249,115,22,0.12);
        color: #ea580c;
        border: 1px solid rgba(249,115,22,0.2);
    }
    .risk-high::before { background: #f97316; box-shadow: 0 0 6px #f97316; }
    .risk-critical {
        background: rgba(239,68,68,0.12);
        color: #dc2626;
        border: 1px solid rgba(239,68,68,0.2);
    }
    .risk-critical::before { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .kpi-card:hover {
        border-color: rgba(99,102,241,0.3);
        box-shadow: 0 4px 14px rgba(99,102,241,0.1);
    }
    .kpi-label {
        color: #64748b;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 26px;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-value.positive { color: #dc2626; }
    .kpi-value.negative { color: #059669; }
    .kpi-value.neutral { color: #64748b; }
    .kpi-value.small { font-size: 15px; font-weight: 600; color: #4f46e5; }

    /* ── Section Headers ── */
    .section-title {
        color: #0f172a;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin: 4px 0 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-title .icon {
        width: 28px; height: 28px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-size: 14px;
    }

    /* ── Spike Alert ── */
    .spike-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.05) 100%);
        border: 1px solid rgba(239,68,68,0.25);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .spike-banner .pulse {
        width: 10px; height: 10px;
        background: #ef4444;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(239,68,68,0.5);
        animation: pulse-ring 1.5s infinite;
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
        70% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    .spike-banner .spike-text {
        color: #991b1b;
        font-size: 13px;
        font-weight: 500;
        line-height: 1.5;
    }
    .spike-banner .spike-text strong { color: #7f1d1d; font-weight: 700; }

    /* ── Article Cards ── */
    .article-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .article-card:hover {
        border-color: rgba(99,102,241,0.3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.08);
    }
    .article-card .accent-bar {
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }
    .article-card .a-title {
        color: #1e293b;
        font-size: 13.5px;
        font-weight: 600;
        line-height: 1.45;
        margin-bottom: 10px;
        padding-left: 12px;
    }
    .article-card .a-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        padding-left: 12px;
    }
    .cat-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .meta-tag {
        color: #64748b;
        font-size: 11.5px;
        font-weight: 400;
    }
    .conf-tag {
        font-size: 11.5px;
        font-weight: 600;
        margin-left: auto;
    }
    .conf-bar-track {
        margin-top: 8px;
        margin-left: 12px;
        height: 4px;
        background: rgba(0,0,0,0.06);
        border-radius: 2px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.4s ease;
    }

    /* ── Sidebar Company Buttons ── */
    .company-btn {
        width: 100%;
        padding: 10px 14px;
        margin-bottom: 4px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.04);
        background: rgba(255,255,255,0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: default;
        transition: all 0.2s ease;
    }
    .company-btn:hover {
        background: rgba(99,102,241,0.08);
        border-color: rgba(99,102,241,0.2);
    }
    .company-btn .cb-name {
        color: #0f172a;
        font-size: 13px;
        font-weight: 600;
    }
    .company-btn .cb-sector {
        color: #64748b;
        font-size: 10px;
        font-weight: 500;
        display: block;
    }

    /* ── Sector Badge ── */
    .sector-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 8px;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        color: #4f46e5;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ── Chart container ── */
    .chart-box {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
    }

    /* ── Divider ── */
    .soft-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,0,0,0.06), transparent);
        margin: 20px 0;
    }

    /* ── Streamlit element tweaks ── */
    .stSelectbox > div > div { border-color: rgba(0,0,0,0.1) !important; background-color: #ffffff; color: #0f172a; }
    .stDateInput > div > div > input { border-color: rgba(0,0,0,0.1) !important; background-color: #ffffff; color: #0f172a; }
    div[data-testid="stMarkdownContainer"] > p { margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Plotly shared config
# ============================================================================
PLOTLY_CONFIG = {"displayModeBar": False, "staticPlot": False}
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#475569"),
)

RISK_COLORS = {
    "Low": "#34d399",
    "Medium": "#fbbf24",
    "High": "#fb923c",
    "Critical": "#f87171",
}

CATEGORY_COLORS = {
    "environmental_violation": "#22c55e",
    "carbon_fraud": "#f97316",
    "labour_dispute": "#eab308",
    "supply_chain_abuse": "#06b6d4",
    "data_breach": "#3b82f6",
    "privacy_violation": "#8b5cf6",
    "bribery_corruption": "#ec4899",
    "board_misconduct": "#f43f5e",
    "community_impact": "#14b8a6",
}


# ============================================================================
# Seed Data Generator (cached)
# ============================================================================
@st.cache_data
def generate_seed_data() -> dict:
    """Generate comprehensive seed data for 5 companies."""
    random.seed(42)
    np.random.seed(42)
    scorer = ControversyScorer()
    companies_data = {}
    categories = config.CONTROVERSY_CATEGORIES

    case_studies = {
        "Apple": {
            "sector": "Technology", "primary_categories": ["supply_chain_abuse", "privacy_violation", "labour_dispute"],
            "num_articles": 18, "recent_spike": False,
        },
        "Shell": {
            "sector": "Energy", "primary_categories": ["environmental_violation", "carbon_fraud", "community_impact"],
            "num_articles": 25, "recent_spike": True,
        },
        "Meta": {
            "sector": "Technology", "primary_categories": ["data_breach", "privacy_violation", "board_misconduct"],
            "num_articles": 22, "recent_spike": False,
        },
        "Nestlé": {
            "sector": "Consumer Staples",
            "primary_categories": ["supply_chain_abuse", "environmental_violation", "community_impact"],
            "num_articles": 15, "recent_spike": False,
        },
        "JPMorgan": {
            "sector": "Financials", "primary_categories": ["bribery_corruption", "board_misconduct", "data_breach"],
            "num_articles": 20, "recent_spike": True,
        },
    }

    headline_templates = {
        "environmental_violation": [
            "{co} faces EPA fine for emissions violations at {loc} plant",
            "Environmental groups sue {co} over water contamination",
            "{co} ordered to clean up toxic waste site in {loc}",
        ],
        "carbon_fraud": [
            "{co} under investigation for inflated carbon offset claims",
            "Regulators probe {co}'s emissions reporting methodology",
            "{co} accused of greenwashing in sustainability report",
        ],
        "labour_dispute": [
            "Workers at {co} facility stage walkout over safety concerns",
            "{co} faces class-action lawsuit over wage theft allegations",
            "Union demands better conditions at {co} manufacturing plant",
        ],
        "supply_chain_abuse": [
            "Investigation reveals forced labour in {co} supply chain",
            "{co} supplier accused of child labour violations",
            "Audit finds safety violations at {co} supplier factories",
        ],
        "data_breach": [
            "{co} discloses breach affecting millions of customers",
            "Hackers access {co} employee database in cyber attack",
            "{co} customer data found on dark web forums",
        ],
        "privacy_violation": [
            "{co} fined for tracking users without consent",
            "Regulators investigate {co}'s data collection practices",
            "{co} faces GDPR penalty over privacy violations",
        ],
        "bribery_corruption": [
            "{co} executives indicted on foreign bribery charges",
            "Anti-corruption probe targets {co} overseas operations",
            "{co} settles bribery allegations for record amount",
        ],
        "board_misconduct": [
            "{co} board member resigns amid insider trading probe",
            "Shareholders revolt over {co} executive compensation",
            "{co} CEO terminated for undisclosed conflicts of interest",
        ],
        "community_impact": [
            "Residents protest {co} factory expansion plan",
            "{co} project threatens indigenous community lands",
            "Health study links {co} operations to local disease spike",
        ],
    }

    locations = ["Houston", "London", "Mumbai", "São Paulo", "Berlin", "Singapore", "Lagos"]
    sources = ["Reuters", "Bloomberg", "BBC", "Financial Times", "The Guardian", "CNBC", "Wall Street Journal"]

    for company_name, study in case_studies.items():
        articles = []
        now = datetime.now()
        for i in range(study["num_articles"]):
            category = random.choice(study["primary_categories"]) if random.random() < 0.7 else random.choice(categories)
            days_ago = random.randint(0, 3) if study["recent_spike"] and i < 5 else random.randint(0, 60)
            pub_date = now - timedelta(days=days_ago)
            location = random.choice(locations)
            title = random.choice(headline_templates[category]).format(co=company_name, loc=location)
            articles.append({
                "title": title, "category": category, "confidence": round(random.uniform(0.65, 0.98), 3),
                "source": random.choice(sources),
                "published_at": pub_date.strftime("%Y-%m-%dT%H:%M:%SZ"), "company": company_name,
            })

        company_score = scorer.compute_company_score(articles)
        rolling_scores = scorer.compute_rolling_score(articles, end_date=now, num_days=30)
        weekly_delta = scorer.compute_weekly_delta(rolling_scores)
        risk_level = scorer.get_risk_level(company_score)
        cat_counts = {}
        for art in articles:
            cat_counts[art["category"]] = cat_counts.get(art["category"], 0) + 1

        companies_data[company_name] = {
            "sector": study["sector"], "score": company_score, "risk_level": risk_level,
            "articles": sorted(articles, key=lambda x: x["published_at"], reverse=True),
            "rolling_scores": rolling_scores, "weekly_delta": weekly_delta, "category_breakdown": cat_counts,
        }

    sectors = {}
    for name, data in companies_data.items():
        sector = data["sector"]
        sectors.setdefault(sector, []).append({"name": name, "score": data["score"]})
    for name, data in companies_data.items():
        sector_scores = [c["score"] for c in sectors[data["sector"]]]
        data["sector_benchmark"] = scorer.benchmark_sector(data["score"], sector_scores)
        data["sector_peers"] = sectors[data["sector"]]

    return companies_data


# ============================================================================
# Helper: render a KPI card
# ============================================================================
def kpi_html(label: str, value: str, css_class: str = "") -> str:
    return f"""<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value {css_class}">{value}</div></div>"""


# ============================================================================
# Main Dashboard
# ============================================================================
def main():
    data = generate_seed_data()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 16px; display:flex; align-items:center; gap:10px;">
            <div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                        display:flex;align-items:center;justify-content:center;font-size:16px;
                        box-shadow:0 2px 12px rgba(99,102,241,0.3);color:white;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg></div>
            <div><span style="color:#0f172a;font-size:15px;font-weight:700;">ESG Monitor</span>
                 <span style="color:#64748b;font-size:10px;display:block;margin-top:-2px;">Controversy Detection</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="kpi-label" style="margin-bottom:4px;">COMPANY</div>', unsafe_allow_html=True)
        company = st.selectbox("Company", list(data.keys()), index=0, label_visibility="collapsed")

        sector = data[company]["sector"]
        st.markdown(f'<div class="kpi-label" style="margin:12px 0 4px;">SECTOR</div>'
                    f'<div class="sector-badge">{sector}</div>', unsafe_allow_html=True)

        st.markdown('<div class="kpi-label" style="margin:16px 0 4px;">DATE RANGE</div>', unsafe_allow_html=True)
        st.date_input("Date", value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
                       label_visibility="collapsed")

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label" style="margin-bottom:8px;">COMPANIES</div>', unsafe_allow_html=True)

        for name, d in data.items():
            rc = d["risk_level"].lower()
            active_style = "background:rgba(99,102,241,0.1);border-color:rgba(99,102,241,0.2);" if name == company else ""
            st.markdown(f"""
            <div class="company-btn" style="{active_style}">
                <div><span class="cb-name">{name}</span><span class="cb-sector">{d['sector']}</span></div>
                <span class="risk-badge risk-{rc}" style="font-size:9px;padding:3px 8px;">{d['risk_level']}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:24px;padding:12px 14px;border-radius:10px;background:rgba(0,0,0,0.02);
                    border:1px solid rgba(0,0,0,0.04);">
            <div style="color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:4px;">
                Powered by
            </div>
            <div style="color:#475569;font-size:11px;line-height:1.6;">
                DistilBERT · FinBERT<br>
                Synthetic demo data
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Main Content ─────────────────────────────────────────────────────────
    cd = data[company]
    score = cd["score"]
    risk = cd["risk_level"]
    rc = risk.lower()
    color = RISK_COLORS.get(risk, "#94a3b8")
    delta_info = cd["weekly_delta"]
    benchmark = cd["sector_benchmark"]

    # Header
    st.markdown(f"""
    <div class="dash-header">
        <div class="dash-header-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg></div>
        <div>
            <h1>{company} — ESG Controversy Report</h1>
            <p>{sector} Sector · Last 30 days · {len(cd['articles'])} flagged articles</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # Spike alert
    if delta_info["is_spike"]:
        st.markdown(f"""
        <div class="spike-banner">
            <div class="pulse"></div>
            <div class="spike-text">
                <strong>Spike detected</strong> — {company}'s controversy score surged
                <strong>{delta_info['delta']:+.1f} pts</strong> over the past 7 days
                (threshold: ±{delta_info['spike_threshold']:.0f}). Immediate review recommended.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Row 1 — Score + KPIs ──
    c1, c2 = st.columns([0.32, 0.68])

    with c1:
        st.markdown(f"""
        <div class="score-gauge" style="--glow-color: {color}22;">
            <div class="score-label">Controversy Score</div>
            <div class="score-num" style="color:{color}">{score:.0f}</div>
            <div class="score-of">/ 100</div>
            <div style="margin-top:12px;">
                <span class="risk-badge risk-{rc}">{risk} Risk</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        k1, k2, k3, k4 = st.columns(4)
        delta_class = "positive" if delta_info["delta"] > 0 else ("negative" if delta_info["delta"] < 0 else "neutral")
        dev = benchmark["deviation_from_mean"]
        dev_class = "positive" if dev > 0 else ("negative" if dev < 0 else "neutral")
        top_cat = max(cd["category_breakdown"], key=cd["category_breakdown"].get)

        with k1:
            st.markdown(kpi_html("7-Day Delta", f"{delta_info['delta']:+.1f}", delta_class), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_html("Percentile", f"{benchmark['percentile_rank']:.0f}th"), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_html("Sector Avg", f"{benchmark['sector_mean']:.1f}"), unsafe_allow_html=True)
        with k4:
            st.markdown(kpi_html("Articles", f"{len(cd['articles'])}"), unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        k5, k6, k7, k8 = st.columns(4)
        with k5:
            st.markdown(kpi_html("Deviation", f"{dev:+.1f}", dev_class), unsafe_allow_html=True)
        with k6:
            st.markdown(kpi_html("Sector Median", f"{benchmark['sector_median']:.1f}"), unsafe_allow_html=True)
        with k7:
            st.markdown(kpi_html("Top Category", top_cat.replace("_", " ").title(), "small"), unsafe_allow_html=True)
        with k8:
            cat_count = len(cd["category_breakdown"])
            st.markdown(kpi_html("Categories Hit", f"{cat_count} / 9"), unsafe_allow_html=True)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── Row 2 — Charts ──
    ch1, ch2 = st.columns([1.6, 1], gap="medium")

    with ch1:
        with st.container(border=True):
            st.markdown('<div class="section-title"><span class="icon" style="background:rgba(99,102,241,0.1);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg></span>30-Day Score Trend</div>', unsafe_allow_html=True)
            rolling = cd["rolling_scores"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling["date"], y=rolling["score"], mode="lines",
                line=dict(color="#6366f1", width=2.5, shape="spline"),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.06)", name="Score",
                hovertemplate="<b>%{x|%b %d}</b><br>Score: %{y:.1f}<extra></extra>",
            ))
            for threshold, lbl, clr_rgba in [
                (25, "Low", "rgba(52,211,153,0.18)"), (50, "Med", "rgba(251,191,36,0.18)"),
                (75, "High", "rgba(249,115,22,0.18)"),
            ]:
                fig.add_hline(y=threshold, line_dash="dot", line_color=clr_rgba, line_width=1,
                              annotation_text=lbl, annotation_font_color=clr_rgba, annotation_font_size=10)
            fig.update_layout(**PLOTLY_LAYOUT, height=280, margin=dict(l=35, r=12, t=8, b=30),
                              xaxis=dict(gridcolor="rgba(0,0,0,0.05)", showgrid=False),
                              yaxis=dict(gridcolor="rgba(0,0,0,0.05)", range=[0, 100], dtick=25))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    with ch2:
        with st.container(border=True):
            st.markdown('<div class="section-title"><span class="icon" style="background:rgba(139,92,246,0.1);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg></span>Category Breakdown</div>', unsafe_allow_html=True)
            bd = cd["category_breakdown"]
            labels = [k.replace("_", " ").title() for k in bd]
            colors = [CATEGORY_COLORS.get(k, "#6366f1") for k in bd]
            fig2 = go.Figure(data=[go.Pie(
                labels=labels, values=list(bd.values()), hole=0.6,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="percent", textfont=dict(size=10, color="#475569"),
                hovertemplate="<b>%{label}</b><br>%{value} articles (%{percent})<extra></extra>",
            )])
            fig2.update_layout(**PLOTLY_LAYOUT, height=280, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    # ── Row 3 — Peer Comparison ──
    peers = cd["sector_peers"]
    with st.container(border=True):
        st.markdown(f'<div class="section-title"><span class="icon" style="background:rgba(14,165,233,0.1);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" stroke-width="2"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg></span>Peer Comparison — {sector}</div>', unsafe_allow_html=True)
    
        if len(peers) < 2:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;padding:24px;">
                <div class="kpi-label">No sector peers available</div>
                <div style="color:#64748b;font-size:13px;margin-top:6px;">
                    {company} is the only seeded company in the {sector} sector.
                    Add more companies to enable peer comparison.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            peer_names = [p["name"] for p in peers]
            peer_scores = [p["score"] for p in peers]
            bar_colors = [color if p["name"] == company else "#e2e8f0" for p in peers]
            border_colors = [color if p["name"] == company else "rgba(0,0,0,0.06)" for p in peers]
    
            fig3 = go.Figure(data=[go.Bar(
                x=peer_scores, y=peer_names, orientation="h",
                marker=dict(color=bar_colors, line=dict(color=border_colors, width=1), cornerradius=6),
                text=[f"  {s:.0f}" for s in peer_scores], textposition="outside",
                textfont=dict(color="#334155", size=12, family="Outfit"), width=0.5,
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
            )])
            sector_mean = benchmark["sector_mean"]
            fig3.add_vline(x=sector_mean, line_dash="dash", line_color="#fbbf24", line_width=1,
                           annotation_text=f"Avg {sector_mean:.0f}", annotation_font_color="#fbbf24",
                           annotation_font_size=11)
            fig3.update_layout(**PLOTLY_LAYOUT, height=max(120, len(peers) * 52 + 40),
                               margin=dict(l=80, r=50, t=10, b=10),
                               xaxis=dict(gridcolor="rgba(0,0,0,0.05)", range=[0, 105], showgrid=False),
                               yaxis=dict(gridcolor="rgba(0,0,0,0.05)"))
            st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── Row 4 — Flagged Articles ──
    with st.container(border=True):
        st.markdown(f'<div class="section-title"><span class="icon" style="background:rgba(249,115,22,0.1);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg></span>Flagged Articles</div>', unsafe_allow_html=True)
    
        articles = cd["articles"]
        show_count = st.session_state.get(f"show_{company}", 8)
    
        for article in articles[:show_count]:
            cat = article["category"]
            cat_display = cat.replace("_", " ").title()
            cat_color = CATEGORY_COLORS.get(cat, "#6366f1")
            conf = article["confidence"]
            conf_pct = int(conf * 100)
            conf_color = "#ef4444" if conf > 0.85 else "#f97316" if conf > 0.7 else "#eab308"
            pub_date = article.get("published_at", "")[:10]
    
            st.markdown(f"""
            <div class="article-card">
                <div class="accent-bar" style="background:{cat_color};"></div>
                <div class="a-title">{article['title']}</div>
                <div class="a-meta">
                    <span class="cat-pill" style="background:{cat_color}14;color:{cat_color};border:1px solid {cat_color}25;">
                        {cat_display}
                    </span>
                    <span class="meta-tag">{article['source']}</span>
                    <span class="meta-tag">{pub_date}</span>
                    <span class="conf-tag" style="color:{conf_color};">{conf_pct}%</span>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill" style="width:{conf_pct}%;background:{conf_color};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
    
        remaining = len(articles) - show_count
        if remaining > 0:
            if st.button(f"Show {min(remaining, 8)} more articles", key=f"more_{company}", use_container_width=True):
                st.session_state[f"show_{company}"] = show_count + 8
                st.rerun()
        elif show_count > 8 and len(articles) > 8:
            if st.button("Show less", key=f"less_{company}", use_container_width=True):
                st.session_state[f"show_{company}"] = 8
                st.rerun()


if __name__ == "__main__":
    main()
