from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from ..core.gguf_meta import get_cached_meta
from ..core.hf_download import (
    download_gguf,
    get_default_hf_token,
    list_gguf_files,
    match_gguf_by_quant,
    parse_hf_spec,
)
from ..core.storage import list_local_models, list_models_by_source
from ..core.paths import get_model_source_label
from ..core.validation import ValidationError
from .common import get_paths, get_settings, set_selected_model_path, sidebar, st_dataframe
from .components import status as status_box


def _human_bytes(n: int) -> str:
    step = 1024.0
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < step:
            return f"{x:.1f} {u}"
        x /= step
    return f"{x:.1f} PiB"


def render() -> None:
    paths = get_paths()
    settings = get_settings()
    sidebar(paths, settings)

    st.title("Modèles GGUF")
    
    # Get models from all sources
    models = list_local_models(paths, settings.external_model_dirs)
    
    if not models:
        st.info("Aucun fichier *.gguf trouvé. Utilise le download HF ci-dessous.")
    else:
        # Show model count by source
        models_by_source = list_models_by_source(paths, settings.external_model_dirs)
        source_counts = {k: len(v) for k, v in models_by_source.items() if v}
        
        if len(source_counts) > 1:
            cols = st.columns(len(source_counts))
            for i, (src, count) in enumerate(source_counts.items()):
                label = "📁 Local" if src == "local" else ("🔷 LM Studio" if "lmstudio" in src.lower() else f"📂 {Path(src).name}")
                cols[i].metric(label, count)
        
        # Build rows with source info
        rows = []
        for p in models:
            try:
                stt = p.stat()
                meta = get_cached_meta(paths, p)
                source = get_model_source_label(p, paths)
                rows.append(
                    {
                        "source": source,
                        "file": p.name,
                        "size": _human_bytes(stt.st_size),
                        "type": "MoE" if meta.is_moe else "Dense",
                        "experts": str(meta.expert_count) if meta.expert_count else "—",
                    }
                )
            except Exception:
                # File might be inaccessible
                rows.append({
                    "source": "?",
                    "file": p.name,
                    "size": "?",
                    "type": "?",
                    "experts": "?",
                })
        
        st_dataframe(rows, use_container_width=True, hide_index=True)
        
        # Show configured external directories
        if settings.external_model_dirs:
            with st.expander("📂 Répertoires externes configurés"):
                for d in settings.external_model_dirs:
                    if d.exists():
                        st.markdown(f"✅ `{d}`")
                    else:
                        st.markdown(f"❌ `{d}` (non trouvé)")

    st.divider()
    st.header("Téléchargement Hugging Face")
    st.caption(
        "Formats acceptés : `hf.co/<org>/<repo>:<quant>` · `<org>/<repo>:<quant>` · `<org>/<repo>` + sélection du fichier."
    )

    if "hf_input_last" not in st.session_state:
        st.session_state["hf_input_last"] = ""
    if "hf_files" not in st.session_state:
        st.session_state["hf_files"] = None
    if "hf_logs" not in st.session_state:
        st.session_state["hf_logs"] = []

    hf_raw = st.text_input(
        "HF input",
        value=st.session_state.get("hf_input", ""),
        placeholder="hf.co/unsloth/Ministral-...-GGUF:Q4_K_XL",
        key="hf_input",
    )

    # Reset cached file list when input changes
    if hf_raw != st.session_state.get("hf_input_last"):
        st.session_state["hf_input_last"] = hf_raw
        st.session_state["hf_files"] = None

    spec = None
    parse_error = None
    if hf_raw.strip():
        try:
            spec = parse_hf_spec(hf_raw)
            st.success(f"Repo: `{spec.repo_id}` · Quant: `{spec.quant or '—'}`")
        except ValidationError as e:
            parse_error = str(e)
            st.error(parse_error)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        list_btn = st.button("Lister les fichiers GGUF", disabled=spec is None)
    with col_b:
        st.write("")

    if list_btn and spec is not None:
        try:
            token = get_default_hf_token()
            files = list_gguf_files(spec.repo_id, token=token)
            if not files:
                st.warning("Aucun fichier .gguf trouvé sur ce repo.")
                st.session_state["hf_files"] = []
            else:
                st.session_state["hf_files"] = files
        except Exception as e:
            st.session_state["hf_files"] = None
            st.error(f"Erreur listing HF: {e}")

    files = st.session_state.get("hf_files")
    selected_file = None
    candidates: list[str] = []

    if spec is not None and isinstance(files, list) and files:
        if spec.quant:
            best, candidates = match_gguf_by_quant(files, spec.quant)
            if best:
                st.info(f"Fichier auto-sélectionné (match quant): `{best}`")
                selected_file = best
            else:
                st.warning("Aucun match automatique pour ce quant. Choisis un fichier.")

        if selected_file is None:
            # Manual selection
            selected_file = st.selectbox(
                "Fichier GGUF à télécharger",
                options=files,
                index=0,
            )

        if candidates and len(candidates) > 1:
            with st.expander("Autres fichiers correspondant au quant"):
                st.write(candidates)

    overwrite = st.checkbox("Overwrite si le fichier existe déjà dans ./models", value=False)

    download_disabled = spec is None or not (isinstance(files, list) and files) or not selected_file
    dl_btn = st.button("Download", disabled=download_disabled)

    log_box = st.empty()

    def log(msg: str) -> None:
        st.session_state["hf_logs"].append(msg)
        # Keep last N
        st.session_state["hf_logs"] = st.session_state["hf_logs"][-400:]

    if dl_btn and spec is not None and selected_file:
        st.session_state["hf_logs"] = []
        with status_box("Téléchargement en cours...", expanded=True) as status:
            try:
                dest = download_gguf(
                    paths,
                    repo_id=spec.repo_id,
                    filename=selected_file,
                    overwrite=overwrite,
                    token=get_default_hf_token(),
                    log_fn=log,
                )
                status.update(label="Téléchargement terminé ✅", state="complete")
                st.success(f"Téléchargé : `{dest}`")
                # Auto-select downloaded model
                set_selected_model_path(dest)
                st.rerun()
            except Exception as e:
                status.update(label="Téléchargement échoué ❌", state="error")
                st.error(str(e))

    # Logs
    logs = st.session_state.get("hf_logs") or []
    if logs:
        log_box.text_area("Logs", value="\n".join(logs), height=220)
