from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..core.gguf_meta import get_cached_meta
from ..core.paths import RepoPaths, get_model_source_label
from ..core.settings import (
    SESSION_SETTINGS_KEY,
    SESSION_SELECTED_MODEL_KEY,
    AppSettings,
)
from ..core.storage import list_local_models
from ..core.docker import (
    check_container_running,
    service_up,
    service_stop,
    service_restart,
)


def st_dataframe(data, **kwargs):
    """Compatibility wrapper around `st.dataframe`.

    Streamlit is deprecating `use_container_width` in favor of `width=...`.
    This wrapper keeps the codebase compatible with older Streamlit versions
    while removing the deprecation warnings on newer ones.
    """
    use_container_width = kwargs.pop("use_container_width", None)
    if use_container_width is None:
        return st.dataframe(data, **kwargs)

    width = "stretch" if bool(use_container_width) else "content"
    try:
        return st.dataframe(data, width=width, **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=use_container_width, **kwargs)


def get_paths() -> RepoPaths:
    paths = RepoPaths.detect()
    paths.ensure_dirs()
    return paths


def get_settings() -> AppSettings:
    if SESSION_SETTINGS_KEY not in st.session_state:
        st.session_state[SESSION_SETTINGS_KEY] = AppSettings()
    return st.session_state[SESSION_SETTINGS_KEY]


def get_selected_model_path(paths: RepoPaths) -> Path | None:
    p = st.session_state.get(SESSION_SELECTED_MODEL_KEY)
    if not p:
        return None
    try:
        pp = Path(p)
    except Exception:
        return None
    if pp.is_file():
        return pp
    return None


def set_selected_model_path(model_path: Path | None) -> None:
    st.session_state[SESSION_SELECTED_MODEL_KEY] = str(model_path) if model_path else None


def sidebar(paths: RepoPaths, settings: AppSettings) -> None:
    st.sidebar.header("Llama Perf Dashboard")
    
    # Container controls
    with st.sidebar.expander("Container llama-cpp", expanded=False):
        running = check_container_running(settings.docker_service)
        status_label = "🟢 running" if running else "🔴 stopped"
        st.sidebar.caption(f"Status: {status_label}")
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("Start", key="container_start"):
            ok, msg = service_up(paths, settings)
            st.sidebar.success(msg) if ok else st.sidebar.error(msg)
        if col_b.button("Stop", key="container_stop"):
            ok, msg = service_stop(paths, settings)
            st.sidebar.success(msg) if ok else st.sidebar.error(msg)
        if col_c.button("Restart", key="container_restart"):
            ok, msg = service_restart(paths, settings)
            st.sidebar.success(msg) if ok else st.sidebar.error(msg)

    # Get models from local + external directories
    models = list_local_models(paths, settings.external_model_dirs)

    current = get_selected_model_path(paths)
    current_path_str = str(current) if current else None
    
    # Build display names with source indicator
    model_display = []
    for m in models:
        source = get_model_source_label(m, paths)
        if source == "local":
            model_display.append(m.name)
        else:
            model_display.append(f"{source} {m.name}")
    
    # Find default index
    default_index = 0
    if current_path_str:
        for i, m in enumerate(models):
            if str(m) == current_path_str:
                default_index = i
                break

    if models:
        chosen_idx = st.sidebar.selectbox(
            "Modèle GGUF",
            options=range(len(models)),
            index=default_index,
            format_func=lambda i: model_display[i],
        )
        chosen_path = models[chosen_idx]
        set_selected_model_path(chosen_path)

        meta = get_cached_meta(paths, chosen_path)
        moe_badge = "MoE" if meta.is_moe else "Dense"
        source_label = get_model_source_label(chosen_path, paths)
        st.sidebar.caption(
            f"**{moe_badge}** · experts: {meta.expert_count if meta.expert_count else '—'}"
        )
        if source_label != "local":
            st.sidebar.caption(f"📂 Source: {source_label}")
    else:
        st.sidebar.warning("Aucun modèle GGUF trouvé")

    st.sidebar.divider()
    st.sidebar.subheader("Settings")

    # MAX_COMBOS moved to Advanced section (legacy sweeps only)
    with st.sidebar.expander("Avancé", expanded=False):
        settings.max_combos = int(
            st.number_input(
                "MAX_COMBOS (sweeps legacy)",
                min_value=1,
                max_value=100_000,
                value=int(settings.max_combos),
                step=100,
                help="Pour l'ancien mode sweeps uniquement. HyperOptimus utilise le slider 'Nombre d'essais'.",
            )
        )
        settings.llama_bin_dir = st.text_input(
            "LLAMA_BIN_DIR (dans le container)",
            value=settings.llama_bin_dir,
            help="Par défaut: /app (image ghcr.io/ggml-org/llama.cpp)",
        )
        settings.docker_service = st.text_input(
            "Docker compose service", value=settings.docker_service
        )
        settings.docker_compose_file = st.text_input(
            "Compose file", value=settings.docker_compose_file
        )


def require_selected_model(paths: RepoPaths) -> Path:
    m = get_selected_model_path(paths)
    if not m:
        st.error("Sélectionne d'abord un modèle dans la sidebar (page **Modèles**).")
        st.stop()
    return m
