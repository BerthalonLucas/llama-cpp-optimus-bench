"""03 – Sweeps page (fallback for classic Streamlit multi-page)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streamlit_dashboard.ui.page_sweeps import render
render()
