"""
Visual theme for the PS52 Defence Acoustic Intelligence Lab dashboard.
Graphite / deep navy background, restrained cyan-teal instrumentation
accents, green/amber/red status colors, monospace telemetry accents.
"""

import streamlit as st

# ---- Color tokens -----------------------------------------------------
BG_PRIMARY = "#0a0e14"
BG_PANEL = "#111722"
BG_PANEL_ALT = "#151c29"
BORDER = "#243044"
TEXT_PRIMARY = "#e6ebf2"
TEXT_MUTED = "#8b98ac"
ACCENT_CYAN = "#3dd6d0"
ACCENT_TEAL = "#2b9e9a"
STATUS_GOOD = "#4fd07a"
STATUS_WARN = "#e8b84b"
STATUS_BAD = "#e8593f"

FONT_SANS = "'IBM Plex Sans', 'Inter', sans-serif"
FONT_MONO = "'JetBrains Mono', 'IBM Plex Mono', monospace"

PLOTLY_TEMPLATE = "plotly_dark"

METHOD_COLORS = {
    "raw": "#5b6478",
    "wiener": "#5aa9e6",
    "neural": "#b98ae0",
    "fxlms": "#e8b84b",
    "hybrid": "#3dd6d0",
}

REGIME_COLORS = {
    "stationary": "#3dd6d0",
    "non_stationary": "#e8b84b",
    "impulse": "#e8593f",
    "unknown": "#5b6478",
}


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_SANS};
        }}

        .stApp {{
            background-color: {BG_PRIMARY};
            background-image:
                linear-gradient(180deg, {BG_PRIMARY} 0%, #0c111a 100%);
        }}

        /* Responsive wide container for Streamlit Cloud */
        .main .block-container,
        div[data-testid="stAppViewBlockContainer"] {{
            max-width: 95% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {BG_PANEL};
            border-right: 1px solid {BORDER};
        }}

        /* Headline */
        .lab-header {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 18px 22px;
            background: linear-gradient(135deg, {BG_PANEL} 0%, {BG_PANEL_ALT} 100%);
            border: 1px solid {BORDER};
            border-radius: 6px;
            margin-bottom: 14px;
        }}
        .lab-title {{
            font-family: {FONT_MONO};
            letter-spacing: 0.08em;
            font-size: 0.95rem;
            color: {ACCENT_CYAN};
            text-transform: uppercase;
        }}
        .lab-subtitle {{
            font-size: 1.05rem;
            color: {TEXT_PRIMARY};
            font-weight: 500;
        }}
        .lab-pipeline {{
            font-family: {FONT_MONO};
            font-size: 0.82rem;
            color: {TEXT_MUTED};
            letter-spacing: 0.03em;
            margin-top: 2px;
        }}
        .badge-row {{
            display: flex;
            gap: 10px;
            margin-top: 8px;
            flex-wrap: wrap;
        }}
        .badge {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            padding: 3px 9px;
            border-radius: 3px;
            border: 1px solid {BORDER};
            background: {BG_PRIMARY};
            color: {TEXT_MUTED};
            letter-spacing: 0.03em;
        }}
        .badge.good {{ color: {STATUS_GOOD}; border-color: {STATUS_GOOD}44; }}
        .badge.cyan {{ color: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}44; }}
        .badge.warn {{ color: {STATUS_WARN}; border-color: {STATUS_WARN}44; }}

        /* KPI cards */
        .kpi-card {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 12px 14px;
            height: 100%;
        }}
        .kpi-label {{
            font-family: {FONT_MONO};
            font-size: 0.68rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 600;
            color: {TEXT_PRIMARY};
            margin-top: 2px;
            font-family: {FONT_MONO};
        }}
        .kpi-sub {{
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            margin-top: 2px;
        }}

        /* Section headers */
        .section-tag {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            color: {ACCENT_CYAN};
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 1px solid {BORDER};
            padding-bottom: 6px;
            margin-bottom: 10px;
            margin-top: 4px;
        }}

        /* Status pill */
        .pill {{
            display: inline-block;
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: 600;
        }}
        .pill.met {{ background: {STATUS_GOOD}22; color: {STATUS_GOOD}; }}
        .pill.partial {{ background: {STATUS_WARN}22; color: {STATUS_WARN}; }}
        .pill.notmet {{ background: {STATUS_BAD}22; color: {STATUS_BAD}; }}

        .claim-card {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-left: 3px solid {ACCENT_TEAL};
            border-radius: 4px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}
        .claim-key {{
            font-family: {FONT_MONO};
            font-size: 0.68rem;
            color: {ACCENT_CYAN};
            letter-spacing: 0.06em;
        }}
        .claim-val {{
            font-size: 0.86rem;
            color: {TEXT_PRIMARY};
            margin-top: 2px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {BG_PANEL};
            border: 1px solid {BORDER};
            border-radius: 4px 4px 0 0;
            font-family: {FONT_MONO};
            font-size: 0.8rem;
        }}

        div[data-testid="stMetricValue"] {{
            font-family: {FONT_MONO};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_div = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_div}'
        f'</div>'
    )


def status_pill(status: str, text: str) -> str:
    cls = {"met": "met", "partial": "partial", "notmet": "notmet"}.get(status, "partial")
    return f'<span class="pill {cls}">{text}</span>'


def claim_card(key: str, value: str) -> str:
    return (
        f'<div class="claim-card">'
        f'<div class="claim-key">{key}</div>'
        f'<div class="claim-val">{value}</div>'
        f'</div>'
    )
