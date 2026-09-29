"""
PS52 Defence AI + Adaptive Noise Cancellation
Streamlit Cloud Entry Point
"""
import sys
from pathlib import Path

# Add dashboard and repo root to Python path
ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from dashboard.app import main

if __name__ == "__main__":
    main()
