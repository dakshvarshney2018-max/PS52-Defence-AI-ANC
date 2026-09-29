"""
PS52 Defence AI + Adaptive Noise Cancellation
Streamlit Cloud Entry Point
"""
import sys
from pathlib import Path
import streamlit as st

# Configure page layout as the very first Streamlit call for Streamlit Community Cloud
try:
    st.set_page_config(
        page_title="PS52 - Defence Acoustic Intelligence Lab",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# Add dashboard and repo root to Python path
ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from dashboard.theme import inject_css
from dashboard.app import main

if __name__ == "__main__":
    inject_css()
    main()

