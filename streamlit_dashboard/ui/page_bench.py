from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ..core.bench import BenchConfig, CACHE_TYPES
from ..core.gguf_meta import get_cached_meta
from ..core.profiles import ProfileManager, create_bench_profile
from ..core.tasks import BenchTask, SweepTask
from ..core.settings import SESSION_ACTIVE_TASK_KEY
from .common import get_paths, get_settings, require_selected_model, sidebar, st_dataframe


# Session keys for relaunch
SESSION_RELAUNCH_CONFIG_KEY = "_bench_relaunch_config"


def _ncmoe_bounds(layer_count: int | None) -> tuple[int, int, int]:
    """Return (min, max, default) for --n-cpu-moe.

    IMPORTANT: ncmoe = number of LAYERS whose MoE experts stay on CPU.
    NOT the number of experts! This is about layers (blocks).
    
    - Higher ncmoe = more layers' experts on CPU = less VRAM usage but slower
    - Lower ncmoe = more layers' experts on GPU = faster but more VRAM
    
    Default: ALL layers on CPU (safest, then user decrements to find optimal).
    For benchmarking MoE, we typically sweep from max (all on CPU) down to 0 (all on GPU).
    """
    if layer_count is None or layer_count <= 0:
        # unknown -> allow a reasonable range, default to safe value
        return 0, 80, 80  # Most models have < 80 layers
    # Default = all layers' experts on CPU (safest starting point)
    return 0, layer_count, layer_count


def _compute_ncmoe_step(layer_count: int) -> int:
    """Compute a sensible step size for ncmoe based on layer count."""
    if layer_count <= 16:
        return 1
    if layer_count <= 32:
        return 2
    if layer_count <= 48:
        return 4
    if layer_count <= 80:
        return 8
    return max(1, layer_count // 10)  # ~10 steps total


def _load_results_df(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _render_task(task: BenchTask) -> None:
    st.subheader("Exécution")

    cols = st.columns([1, 1, 2])
    cols[0].markdown(f"**Status:** `{task.status}`")
    cols[1].markdown(f"**Run ID:** `{task.run_id}`")

    stop_disabled = task.status != "running"
    if cols[2].button("Stop", disabled=stop_disabled, use_container_width=True):
        task.stop()
        st.warning("Stop demandé (SIGINT).")

    proc = task.proc()

    if proc is None:
        st.info("Process non démarré.")
        return

    def _logs_text() -> str:
        return proc.lines_text(last_n=800)

    # Check if task just finished - trigger rerun to show results
    if task.status != "running" and task.result is not None:
        # Task finished, show final logs and results
        st.text_area("Logs (stdout/stderr)", value=_logs_text(), height=320)
        st.success("Run terminé.")
    elif hasattr(st, "fragment"):
        @st.fragment(run_every=0.5)
        def _live() -> None:
            st.text_area("Logs (stdout/stderr)", value=_logs_text(), height=320)
            # Trigger full page rerun when task completes
            if task.result is not None and task.status != "running":
                st.rerun()
        _live()
    else:
        st.text_area("Logs (stdout/stderr)", value=_logs_text(), height=320)
        st.caption("(Astuce: Streamlit ancien -> rafraîchir la page pour mettre à jour les logs)")
        # Auto-refresh for older Streamlit
        if task.status == "running":
            import time
            time.sleep(1)
            st.rerun()

    if task.result is not None and task.status != "running":
        st.divider()
        st.subheader("Résultats")

        df = _load_results_df(task.run_directory())
        if df is not None:
            # Select only the most relevant columns for display
            display_cols = [
                "model_type", "n_batch", "n_ubatch", "n_threads", "n_gpu_layers", 
                "n_cpu_moe", "flash_attn", "type_k", "type_v",
                "n_prompt", "n_gen", "avg_ts"
            ]
            available_cols = [c for c in display_cols if c in df.columns]
            if available_cols:
                st_dataframe(df[available_cols], use_container_width=True, hide_index=True)
            else:
                st_dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("Impossible de lire results.csv (voir stdout.log)")

        # Key metrics - find tg and pp keys dynamically
        metrics = task.result.get("metrics") or {}
        tg_key = next((k for k in metrics.keys() if k.startswith("tg")), None)
        pp_key = next((k for k in metrics.keys() if k.startswith("pp")), None)
        
        c1, c2, c3 = st.columns(3)
        if tg_key and isinstance(metrics.get(tg_key), (int, float)):
            c1.metric(f"{tg_key} (tok/s)", f"{metrics[tg_key]:.2f}")
        else:
            c1.metric("tg (tok/s)", "—")
        if pp_key and isinstance(metrics.get(pp_key), (int, float)):
            c2.metric(f"{pp_key} (tok/s)", f"{metrics[pp_key]:.2f}")
        else:
            c2.metric("pp (tok/s)", "—")
        c3.metric("wall_time (s)", f"{task.result.get('wall_time_sec', 0):.2f}" if isinstance(task.result.get('wall_time_sec'), (int, float)) else "—")

        # Suggested llama-server command
        if task.result.get("llama_server_cmd"):
            st.markdown("**Commande llama-server recommandée (rejouer les paramètres du bench)**")
            st.code(task.result["llama_server_cmd"], language="bash")
            ctx_tokens = task.result.get("ctx_size_tokens")
            if ctx_tokens:
                st.caption(f"ctx-size ≈ {int(ctx_tokens)} tokens (prompt + génération + profondeur).")

        # Exports
        st.subheader("Export")
        run_json_path = task.run_directory() / "run.json"
        results_csv_path = task.run_directory() / "results.csv"
        results_rows_path = task.run_directory() / "results_rows.json"

        col_a, col_b, col_c = st.columns(3)
        if run_json_path.exists():
            col_a.download_button(
                "run.json",
                data=run_json_path.read_bytes(),
                file_name=f"{task.run_id}_run.json",
                mime="application/json",
            )
        if results_csv_path.exists():
            col_b.download_button(
                "results.csv",
                data=results_csv_path.read_bytes(),
                file_name=f"{task.run_id}_results.csv",
                mime="text/csv",
            )
        if results_rows_path.exists():
            col_c.download_button(
                "results_rows.json",
                data=results_rows_path.read_bytes(),
                file_name=f"{task.run_id}_results_rows.json",
                mime="application/json",
            )

        with st.expander("run.json (aperçu)"):
            try:
                st.json(json.loads(run_json_path.read_text(encoding="utf-8")))
            except Exception:
                st.code(run_json_path.read_text(encoding="utf-8"))

        # Action buttons: Relaunch and Clear
        st.divider()
        btn_cols = st.columns([1, 1, 2])
        
        if btn_cols[0].button("🔄 Relancer", type="primary", use_container_width=True, 
                              help="Relancer avec les mêmes paramètres"):
            # Store config for relaunch and clear current task
            st.session_state[SESSION_RELAUNCH_CONFIG_KEY] = task.cfg
            st.session_state[SESSION_ACTIVE_TASK_KEY] = None
            st.rerun()
        
        if btn_cols[1].button("🗑️ Effacer", type="secondary", use_container_width=True):
            st.session_state[SESSION_ACTIVE_TASK_KEY] = None
            st.rerun()


def render() -> None:
    paths = get_paths()
    settings = get_settings()
    sidebar(paths, settings)

    st.title("Bench (llama-bench)")

    model_path = require_selected_model(paths)
    meta = get_cached_meta(paths, model_path)

    st.caption(f"Modèle: `{model_path.name}` · {'MoE' if meta.is_moe else 'Dense'}")

    active = st.session_state.get(SESSION_ACTIVE_TASK_KEY)
    if isinstance(active, SweepTask) and active.status == "running":
        st.warning("Un sweep est en cours dans cette session. Termine-le ou stoppe-le avant de lancer un bench.")
        return

    # Profile manager
    profile_mgr = ProfileManager(paths.runs_dir)
    
    # Check for relaunch config
    relaunch_cfg = st.session_state.pop(SESSION_RELAUNCH_CONFIG_KEY, None)
    
    # Defaults: from relaunch, profile, or settings
    def _get_default(key: str, default, relaunch: BenchConfig | None = None):
        """Get default value: relaunch > session_state > settings default."""
        if relaunch is not None:
            cfg_dict = asdict(relaunch)
            # Map session key to config field
            key_map = {
                "bench_t": "threads", "bench_b": "batch", "bench_ub": "ubatch",
                "bench_ngl": "gpu_layers", "bench_fa": "flash_attn", 
                "bench_nkvo": "no_kv_offload", "bench_ctk": "ctk", "bench_ctv": "ctv",
                "bench_ncmoe": "n_cpu_moe", "bench_p": "n_prompt", "bench_n": "n_gen",
                "bench_r": "repeats", "bench_depth": "n_depth",
            }
            if key in key_map and key_map[key] in cfg_dict:
                val = cfg_dict[key_map[key]]
                if val is not None:
                    return val
        if key in st.session_state:
            return st.session_state[key]
        return default

    # ─── Profiles Section ─────────────────────────────────────────────────
    with st.expander("📁 Profils", expanded=False):
        profiles = profile_mgr.list_bench_profiles()
        profile_names = ["(Aucun)"] + [p.name for p in profiles]
        
        col_load, col_del = st.columns([3, 1])
        selected_profile = col_load.selectbox(
            "Charger un profil", 
            options=profile_names, 
            key="bench_profile_select",
            label_visibility="collapsed"
        )
        
        # Load profile button
        if selected_profile != "(Aucun)":
            profile = profile_mgr.get_bench_profile(selected_profile)
            if profile:
                if col_load.button("📥 Charger", use_container_width=True):
                    # Apply profile to session state
                    cfg = profile.config
                    st.session_state["bench_t"] = cfg.threads
                    st.session_state["bench_b"] = cfg.batch
                    st.session_state["bench_ub"] = cfg.ubatch
                    st.session_state["bench_ngl"] = cfg.gpu_layers
                    st.session_state["bench_fa"] = cfg.flash_attn
                    st.session_state["bench_nkvo"] = cfg.no_kv_offload
                    st.session_state["bench_ctk"] = cfg.ctk
                    st.session_state["bench_ctv"] = cfg.ctv
                    st.session_state["bench_ncmoe"] = cfg.n_cpu_moe or 0
                    st.session_state["bench_p"] = cfg.n_prompt
                    st.session_state["bench_n"] = cfg.n_gen
                    st.session_state["bench_r"] = cfg.repeats
                    st.session_state["bench_depth"] = cfg.n_depth
                    st.success(f"Profil '{selected_profile}' chargé!")
                    st.rerun()
                
                if col_del.button("🗑️", use_container_width=True, help="Supprimer ce profil"):
                    profile_mgr.delete_bench_profile(selected_profile)
                    st.success(f"Profil '{selected_profile}' supprimé!")
                    st.rerun()
        
        st.divider()
        st.markdown("**Sauvegarder configuration actuelle:**")
        st.caption("💡 Les valeurs du formulaire ci-dessous seront sauvegardées.")
        save_cols = st.columns([3, 1])
        new_profile_name = save_cols[0].text_input(
            "Nom du profil", 
            placeholder="Ex: MoE-optimal-100K",
            key="bench_new_profile_name",
            label_visibility="collapsed"
        )
        # Set flag to save after form values are captured
        if save_cols[1].button("💾 Sauver", use_container_width=True):
            if new_profile_name.strip():
                st.session_state["_bench_pending_save_profile"] = new_profile_name.strip()
            else:
                st.error("Nom de profil requis")

    # Show relaunch banner if applicable
    if relaunch_cfg is not None:
        st.info("🔄 **Relance** - Paramètres du run précédent chargés. Modifiez si besoin et lancez.")

    with st.form("bench_form"):
        c1, c2, c3, c4 = st.columns(4)
        t = c1.number_input("Threads (-t)", min_value=1, max_value=512, value=int(_get_default("bench_t", settings.default_threads, relaunch_cfg)), step=1)
        b = c2.number_input("Batch (-b)", min_value=1, max_value=131072, value=int(_get_default("bench_b", settings.default_batch, relaunch_cfg)), step=128)
        ub = c3.number_input("UBatch (-ub)", min_value=1, max_value=131072, value=int(_get_default("bench_ub", settings.default_ubatch, relaunch_cfg)), step=64)
        ngl = c4.number_input("GPU layers (-ngl)", min_value=0, max_value=999, value=int(_get_default("bench_ngl", settings.default_gpu_layers, relaunch_cfg)), step=1)

        c5, c6, c7, c8 = st.columns(4)
        fa = c5.selectbox("Flash Attention (-fa)", options=[0, 1], index=int(_get_default("bench_fa", settings.default_flash_attn, relaunch_cfg)))
        nkvo = c6.selectbox("No KV offload (-nkvo)", options=[0, 1], index=int(_get_default("bench_nkvo", settings.default_no_kv_offload, relaunch_cfg)))
        ctk = c7.selectbox("Cache type K (-ctk)", options=list(CACHE_TYPES), index=list(CACHE_TYPES).index(_get_default("bench_ctk", settings.default_ctk, relaunch_cfg)))
        ctv = c8.selectbox("Cache type V (-ctv)", options=list(CACHE_TYPES), index=list(CACHE_TYPES).index(_get_default("bench_ctv", settings.default_ctv, relaunch_cfg)))

        if meta.is_moe:
            mn, mx, dflt = _ncmoe_bounds(meta.layer_count)
            # Clamp stored value to valid range for current model
            stored_ncmoe = int(_get_default("bench_ncmoe", dflt, relaunch_cfg))
            clamped_ncmoe = max(mn, min(mx, stored_ncmoe))
            
            expert_info = f"{meta.expert_count} experts" if meta.expert_count else "experts"
            layer_detected = meta.layer_count is not None and meta.layer_count > 0
            layer_info = f"{meta.layer_count} couches" if layer_detected else f"couches inconnues (garde-fou: {mx})"
            st.markdown(f"**🔀 Modèle MoE** - {expert_info}, {layer_info}")
            if not layer_detected:
                st.warning(f"⚠️ Le nombre de couches n'a pas pu être détecté. Valeur max ncmoe = {mx} (garde-fou).")
            ncmoe = st.number_input(
                "Couches MoE sur CPU (-ncmoe)",
                min_value=int(mn),
                max_value=int(mx),
                value=int(clamped_ncmoe),
                step=_compute_ncmoe_step(meta.layer_count or 40),
                help=f"Nombre de COUCHES dont les experts MoE restent sur CPU (0 à {mx}). "
                     f"**Max ({mx})** = toutes les couches MoE sur CPU (safe, lent). "
                     f"**0** = tous sur GPU (rapide mais risque OOM). "
                     f"Décrémentez progressivement pour trouver l'optimal.",
            )
        else:
            ncmoe = None
            st.caption("ncmoe désactivé (modèle non-MoE)")

        c9, c10, c11, c12 = st.columns(4)
        p = c9.number_input("Prompt tokens (-p)", min_value=0, max_value=16384, value=int(_get_default("bench_p", settings.default_prompt, relaunch_cfg)), step=64)
        n = c10.number_input("Gen tokens (-n)", min_value=0, max_value=8192, value=int(_get_default("bench_n", settings.default_gen, relaunch_cfg)), step=32)
        r = c11.number_input("Repeat (-r)", min_value=1, max_value=50, value=int(_get_default("bench_r", settings.default_repeats, relaunch_cfg)), step=1)
        depth = c12.number_input(
            "Context depth (-d)", 
            min_value=0, 
            max_value=200000, 
            value=int(_get_default("bench_depth", 0, relaunch_cfg)), 
            step=8192,
            help="Contexte pré-rempli (total = p + n + depth). "
                 "Presets: **8K**=7360, **32K**=31360, **64K**=63360, **100K**=99360, **128K**=127360"
        )

        submitted = st.form_submit_button("Run bench")

    # Persist UI values
    st.session_state["bench_t"] = int(t)
    st.session_state["bench_b"] = int(b)
    st.session_state["bench_ub"] = int(ub)
    st.session_state["bench_ngl"] = int(ngl)
    st.session_state["bench_fa"] = int(fa)
    st.session_state["bench_nkvo"] = int(nkvo)
    st.session_state["bench_ctk"] = str(ctk)
    st.session_state["bench_ctv"] = str(ctv)
    if meta.is_moe:
        st.session_state["bench_ncmoe"] = int(ncmoe) if ncmoe is not None else 0
    st.session_state["bench_p"] = int(p)
    st.session_state["bench_n"] = int(n)
    st.session_state["bench_r"] = int(r)
    st.session_state["bench_depth"] = int(depth)

    # ─── Handle pending profile save (after form values are captured) ─────
    pending_save_name = st.session_state.pop("_bench_pending_save_profile", None)
    if pending_save_name:
        current_cfg = BenchConfig(
            threads=int(t),
            batch=int(b),
            ubatch=int(ub),
            gpu_layers=int(ngl),
            no_kv_offload=int(nkvo),
            flash_attn=int(fa),
            n_cpu_moe=int(ncmoe) if (meta.is_moe and ncmoe is not None) else None,
            ctk=str(ctk),
            ctv=str(ctv),
            n_prompt=int(p),
            n_gen=int(n),
            repeats=int(r),
            n_depth=int(depth),
        )
        profile = create_bench_profile(
            name=pending_save_name,
            config=current_cfg,
            model_pattern=model_path.stem,
        )
        profile_mgr.save_bench_profile(profile)
        st.success(f"✅ Profil '{pending_save_name}' sauvegardé!")
        st.rerun()

    if submitted:
        if isinstance(active, BenchTask) and active.status == "running":
            st.error("Un bench est déjà en cours dans cette session.")
        else:
            cfg = BenchConfig(
                threads=int(t),
                batch=int(b),
                ubatch=int(ub),
                gpu_layers=int(ngl),
                no_kv_offload=int(nkvo),
                flash_attn=int(fa),
                n_cpu_moe=int(ncmoe) if (meta.is_moe and ncmoe is not None) else None,
                ctk=str(ctk),
                ctv=str(ctv),
                n_prompt=int(p),
                n_gen=int(n),
                repeats=int(r),
                n_depth=int(depth),
            )
            task = BenchTask(paths=paths, settings=settings, model_path_host=model_path, cfg=cfg)
            st.session_state[SESSION_ACTIVE_TASK_KEY] = task
            task.start()
            st.rerun()

    active = st.session_state.get(SESSION_ACTIVE_TASK_KEY)
    if isinstance(active, BenchTask):
        _render_task(active)
