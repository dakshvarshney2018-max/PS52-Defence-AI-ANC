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

FONT_SANS = (
    "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)
FONT_MONO = (
    "'JetBrains Mono', 'IBM Plex Mono', 'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', monospace"
)

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
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Global typography hierarchy & resets */
        html, body, .stApp,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stText"],
        div[data-testid="stWidgetLabel"] label,
        .stMarkdown, p, span, label {{
            font-family: {FONT_SANS};
        }}

        .stApp {{
            background-color: {BG_PRIMARY} !important;
            background-image: linear-gradient(180deg, {BG_PRIMARY} 0%, #0c111a 100%) !important;
            color: {TEXT_PRIMARY} !important;
        }}

        /* Responsive wide container for desktop view */
        .main .block-container,
        div[data-testid="stAppViewBlockContainer"] {{
            max-width: 95% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {BG_PANEL} !important;
            border-right: 1px solid {BORDER} !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 2rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }}

        /* Headline / Header Card */
        .lab-header {{
            display: flex !important;
            flex-direction: column !important;
            gap: 6px !important;
            padding: 18px 22px !important;
            background: linear-gradient(135deg, {BG_PANEL} 0%, {BG_PANEL_ALT} 100%) !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
        }}
        .lab-title {{
            font-family: {FONT_MONO} !important;
            letter-spacing: 0.08em !important;
            font-size: 0.95rem !important;
            color: {ACCENT_CYAN} !important;
            text-transform: uppercase !important;
            font-weight: 600 !important;
        }}
        .lab-subtitle {{
            font-family: {FONT_SANS} !important;
            font-size: 1.05rem !important;
            color: {TEXT_PRIMARY} !important;
            font-weight: 500 !important;
            margin: 1px 0 !important;
        }}
        .lab-pipeline {{
            font-family: {FONT_MONO} !important;
            font-size: 0.82rem !important;
            color: {TEXT_MUTED} !important;
            letter-spacing: 0.04em !important;
            margin-top: 2px !important;
        }}
        .badge-row {{
            display: flex !important;
            gap: 10px !important;
            margin-top: 10px !important;
            flex-wrap: wrap !important;
            align-items: center !important;
        }}
        .badge {{
            font-family: {FONT_MONO} !important;
            font-size: 0.72rem !important;
            padding: 4px 10px !important;
            border-radius: 4px !important;
            border: 1px solid {BORDER} !important;
            background: {BG_PRIMARY} !important;
            color: {TEXT_MUTED} !important;
            letter-spacing: 0.03em !important;
            display: inline-flex !important;
            align-items: center !important;
            line-height: 1.2 !important;
            font-weight: 500 !important;
        }}
        .badge.good {{
            color: {STATUS_GOOD} !important;
            border-color: rgba(79, 208, 122, 0.45) !important;
            background: rgba(79, 208, 122, 0.12) !important;
        }}
        .badge.cyan {{
            color: {ACCENT_CYAN} !important;
            border-color: rgba(61, 214, 208, 0.45) !important;
            background: rgba(61, 214, 208, 0.12) !important;
        }}
        .badge.warn {{
            color: {STATUS_WARN} !important;
            border-color: rgba(232, 184, 75, 0.45) !important;
            background: rgba(232, 184, 75, 0.12) !important;
        }}

        /* KPI cards */
        .kpi-card {{
            background-color: {BG_PANEL} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 6px !important;
            padding: 12px 14px !important;
            height: 100% !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        }}
        .kpi-label {{
            font-family: {FONT_MONO} !important;
            font-size: 0.68rem !important;
            color: {TEXT_MUTED} !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            font-weight: 500 !important;
            line-height: 1.2 !important;
            margin-bottom: 2px !important;
        }}
        .kpi-value {{
            font-size: 1.55rem !important;
            font-weight: 600 !important;
            color: {TEXT_PRIMARY} !important;
            margin-top: 2px !important;
            margin-bottom: 2px !important;
            font-family: {FONT_MONO} !important;
            letter-spacing: -0.01em !important;
            line-height: 1.2 !important;
        }}
        .kpi-sub {{
            font-size: 0.72rem !important;
            color: {TEXT_MUTED} !important;
            margin-top: 2px !important;
            font-family: {FONT_SANS} !important;
            line-height: 1.3 !important;
        }}

        /* Section headers */
        .section-tag {{
            font-family: {FONT_MONO} !important;
            font-size: 0.82rem !important;
            color: {ACCENT_CYAN} !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            border-bottom: 1px solid {BORDER} !important;
            padding-bottom: 6px !important;
            margin-bottom: 12px !important;
            margin-top: 6px !important;
            font-weight: 600 !important;
        }}

        /* Status pill */
        .pill {{
            display: inline-block !important;
            font-family: {FONT_MONO} !important;
            font-size: 0.72rem !important;
            padding: 2px 8px !important;
            border-radius: 3px !important;
            font-weight: 600 !important;
        }}
        .pill.met {{ background: rgba(79, 208, 122, 0.18) !important; color: {STATUS_GOOD} !important; border: 1px solid rgba(79, 208, 122, 0.35) !important; }}
        .pill.partial {{ background: rgba(232, 184, 75, 0.18) !important; color: {STATUS_WARN} !important; border: 1px solid rgba(232, 184, 75, 0.35) !important; }}
        .pill.notmet {{ background: rgba(232, 89, 63, 0.18) !important; color: {STATUS_BAD} !important; border: 1px solid rgba(232, 89, 63, 0.35) !important; }}

        /* Claim Cards */
        .claim-card {{
            background: {BG_PANEL} !important;
            border: 1px solid {BORDER} !important;
            border-left: 3px solid {ACCENT_TEAL} !important;
            border-radius: 4px !important;
            padding: 10px 12px !important;
            margin-bottom: 8px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }}
        .claim-key {{
            font-family: {FONT_MONO} !important;
            font-size: 0.68rem !important;
            color: {ACCENT_CYAN} !important;
            letter-spacing: 0.06em !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
        }}
        .claim-val {{
            font-size: 0.86rem !important;
            color: {TEXT_PRIMARY} !important;
            margin-top: 3px !important;
            font-family: {FONT_SANS} !important;
            line-height: 1.4 !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {BG_PANEL} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 4px 4px 0 0 !important;
            font-family: {FONT_MONO} !important;
            font-size: 0.8rem !important;
            color: {TEXT_MUTED} !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT_CYAN} !important;
            border-bottom: 2px solid {ACCENT_CYAN} !important;
        }}

        div[data-testid="stMetricValue"] {{
            font-family: {FONT_MONO} !important;
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
