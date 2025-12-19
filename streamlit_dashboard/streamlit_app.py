from __future__ import annotations

import sys
from pathlib import Path

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from streamlit_dashboard.ui import page_bench, page_history, page_models, page_hyperoptimus
# from streamlit_dashboard.ui import page_visualizer # Optional, keeping it if useful


st.set_page_config(
    page_title="GGUF Dashboard (local)",
    layout="wide",
)


def _has_new_nav_api() -> bool:
    return hasattr(st, "navigation") and hasattr(st, "Page")


if _has_new_nav_api():
    pages = [
        st.Page(page_models.render, title="Modèles", icon="📦", url_path="models"),
        st.Page(page_hyperoptimus.render, title="HyperOptimus", icon="🚀", url_path="hyperoptimus"),
        st.Page(page_bench.render, title="Bench", icon="🧪", url_path="bench"),
        st.Page(page_history.render, title="Historique", icon="📈", url_path="history"),
        # st.Page(page_visualizer.render, title="Visualiser", icon="📊", url_path="visualizer"),
    ]
    st.navigation(pages).run()
else:
    # Fallback: Streamlit's classic multipage uses the ./pages directory.
    st.title("GGUF Dashboard")
    st.warning(
        "Votre version Streamlit ne supporte pas `st.Page` / `st.navigation`. "
        "Le mode multipage classique (dossier pages/) est actif."
    )
    page_models.render()
