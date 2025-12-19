"""History page – past bench runs, sweeps & optimus runs."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from ..core.storage import (
    list_runs,
    load_run_json,
    delete_run,
    list_sweeps,
    load_sweep_results,
    delete_sweep,
    list_optimus_runs,
    load_all_optimus_runs,
    delete_optimus_run,
    list_optimizer_runs,
    load_optimizer_summary,
    delete_optimizer_run,
    delete_all_failed_runs,
    load_json,
)
from .common import get_paths, get_settings, sidebar, st_dataframe


def _timestamp_fmt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def _render_runs(paths) -> None:
    runs = list_runs(paths)
    if not runs:
        st.info("Aucun run enregistré.")
        return

    st.caption(f"{len(runs)} run(s) trouvé(s)")

    # Build a summary DataFrame for quick filtering
    rows = []
    for r in runs:
        data = load_run_json(r)
        if data is None:
            continue
        metrics = data.get("metrics", {})
        # Find tg and pp keys dynamically
        tg_val = next((v for k, v in metrics.items() if k.startswith("tg")), None)
        pp_val = next((v for k, v in metrics.items() if k.startswith("pp")), None)
        
        # Status with error info
        status = data.get("status", "?")
        error_type = data.get("error_type")
        is_oom = data.get("is_oom", False)
        
        if is_oom:
            status_display = "⚠️ OOM"
        elif status == "completed":
            status_display = "✅ OK"
        elif status == "failed":
            if error_type and error_type != "none":
                status_display = f"❌ {error_type}"
            else:
                status_display = "❌ failed"
        else:
            status_display = status
        
        rows.append({
            "run_id": r.name,
            "model": data.get("model_file", data.get("model", "?")),
            "status": status_display,
            "error": data.get("error_message", "—") if status != "completed" else "—",
            "started": _timestamp_fmt(data.get("started_ts", 0)) if data.get("started_ts") else data.get("created_at", "?"),
            "wall_time": f"{data.get('wall_time_sec', 0):.1f}s" if data.get("wall_time_sec") else "—",
            "tg (t/s)": f"{tg_val:.1f}" if isinstance(tg_val, (int, float)) else "—",
            "pp (t/s)": f"{pp_val:.1f}" if isinstance(pp_val, (int, float)) else "—",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("Aucun run lisible.")
        return

    st_dataframe(df, use_container_width=True, hide_index=True)

    # Detailed expanders
    with st.expander("Détails par run"):
        selected_run = st.selectbox("Choisir un run", options=[r["run_id"] for r in rows])
        run_dir = paths.runs_dir / "bench" / selected_run
        data = load_run_json(run_dir)
        if data:
            # Show key info in a cleaner format
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Configuration:**")
                cfg = data.get("bench_config", {})
                st.markdown(f"""
- **Threads:** {cfg.get('threads', '?')}
- **Batch:** {cfg.get('batch', '?')}
- **UBatch:** {cfg.get('ubatch', '?')}
- **GPU Layers:** {cfg.get('gpu_layers', '?')}
- **Flash Attn:** {cfg.get('flash_attn', '?')}
- **ncmoe:** {cfg.get('n_cpu_moe', 'N/A')}
""")
            with col2:
                st.markdown("**Résultats:**")
                metrics = data.get("metrics", {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        st.markdown(f"- **{k}:** {v:.2f} t/s")
                st.markdown(f"- **Wall time:** {data.get('wall_time_sec', 0):.2f}s")
                st.markdown(f"- **Status:** {data.get('status', '?')}")

            if data.get("llama_server_cmd"):
                st.markdown("**Commande llama-server (recommandée):**")
                st.code(data["llama_server_cmd"], language="bash")
                if data.get("ctx_size_tokens"):
                    st.caption(f"ctx-size ≈ {int(data['ctx_size_tokens'])} tokens.")
            
            with st.expander("JSON complet"):
                st.json(data)
            
            csv_path = run_dir / "results.csv"
            if csv_path.exists():
                st.download_button("Télécharger results.csv", csv_path.read_bytes(), file_name=f"{selected_run}_results.csv")

        if st.button("Supprimer ce run", type="secondary", key=f"del_run_{selected_run}"):
            delete_run(run_dir)
            st.success("Supprimé.")
            st.rerun()


def _render_optimus(paths) -> None:
    """Render Optimus optimization runs."""
    runs = list_optimus_runs(paths)
    if not runs:
        st.info("Aucune optimisation Optimus enregistrée.")
        return

    st.caption(f"{len(runs)} optimisation(s) trouvée(s)")

    # Build summary DataFrame
    rows = []
    for r in runs:
        data = load_run_json(r)
        if data is None:
            continue
        final_cfg = data.get("final_config", {})
        rows.append({
            "run_id": r.name,
            "model": data.get("model_file", "?"),
            "status": data.get("status", "?"),
            "date": data.get("created_at", "?")[:16] if data.get("created_at") else "?",
            "metric": data.get("config", {}).get("metric", "?"),
            "trials": data.get("config", {}).get("trials", "?"),
            "tg (t/s)": f"{data.get('optimized_results', {}).get('tg', 0):.1f}" if data.get('optimized_results', {}).get('tg') else "—",
            # Cast to str to avoid mixed-type Arrow issues
            "ngl": str(final_cfg.get("ngl", "—")),
            "batch": str(final_cfg.get("batch", "—")),
            "threads": str(final_cfg.get("threads", "—")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("Aucune optimisation lisible.")
        return

    st_dataframe(df, use_container_width=True, hide_index=True)

    # Detailed view
    with st.expander("Détails par optimisation"):
        selected = st.selectbox("Choisir une optimisation", options=[r["run_id"] for r in rows])
        run_dir = paths.runs_dir / "optimus" / selected
        data = load_run_json(run_dir)
        
        if data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Configuration:**")
                cfg = data.get("config", {})
                st.markdown(f"""
- **Trials:** {cfg.get('trials', '?')}
- **Repeat:** {cfg.get('repeat', '?')}
- **Metric:** {cfg.get('metric', '?')}
- **N Tokens:** {cfg.get('n_tokens', '?')}
- **Override Mode:** {cfg.get('override_mode', '?')}
""")
                if data.get("is_moe"):
                    st.markdown(f"- **MoE:** Oui ({data.get('expert_count', '?')} experts)")
            
            with col2:
                st.markdown("**Résultat Final:**")
                final = data.get("final_config", {})
                for k, v in final.items():
                    st.markdown(f"- **{k}:** {v}")
                opt_res = data.get("optimized_results", {})
                if opt_res:
                    st.markdown("**Scores:**")
                    for k, v in opt_res.items():
                        st.markdown(f"- **{k}:** {v:.2f} t/s")
            
            # Show stages
            stages = data.get("stages", [])
            if stages:
                st.markdown("**Stages:**")
                for s in stages:
                    st.markdown(f"- **Stage {s['stage']}** ({s['stage_name']}): {s['best_value']:.2f} t/s")
            
            # Commands
            if data.get("llama_server_cmd"):
                st.markdown("**llama-server command:**")
                st.code(data["llama_server_cmd"], language="bash")
                if data.get("ctx_size_tokens"):
                    st.caption(f"ctx-size ≈ {int(data['ctx_size_tokens'])} tokens.")
            if data.get("llama_bench_cmd"):
                st.markdown("**llama-bench command:**")
                st.code(data["llama_bench_cmd"], language="bash")
            
            with st.expander("JSON complet"):
                st.json({k: v for k, v in data.items() if k != "raw_output"})
            
            with st.expander("Raw output"):
                st.code(data.get("raw_output", ""), language="text")
        
        if st.button("Supprimer cette optimisation", type="secondary", key=f"del_optimus_{selected}"):
            delete_optimus_run(run_dir)
            st.success("Supprimé.")
            st.rerun()


def _parse_optimizer_created_at(run_id: str) -> str:
    # Format: optimizer_YYYYMMDD_HHMMSS_xxxxxx
    try:
        parts = run_id.split("_")
        if len(parts) >= 3:
            ymd = parts[1]
            hms = parts[2]
            return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]} {hms[:2]}:{hms[2:4]}:{hms[4:6]}"
    except Exception:
        pass
    return "—"


def _render_optimizer(paths) -> None:
    runs = list_optimizer_runs(paths)
    if not runs:
        st.info("Aucun run Optimizer enregistré.")
        return

    rows = []
    summaries: dict[str, dict[str, Any]] = {}
    for r in runs:
        s = load_optimizer_summary(r)
        if not s:
            continue
        summaries[r.name] = s
        cfg = s.get("config", {}) or {}
        model_path = cfg.get("model_path_host")
        model_name = Path(model_path).name if isinstance(model_path, str) else "?"
        results = s.get("results", []) or []
        best = s.get("best") or {}
        best_is_valid = isinstance(best, dict) and best.get("status") == "completed"
        best_score = best.get("score") if best_is_valid else None
        best_ctx = best.get("ctx_tokens") if best_is_valid else None
        budget = cfg.get("budget_runs", "?")
        rows.append({
            "run_id": r.name,
            "model": model_name,
            "status": s.get("status", "?"),
            "created": _parse_optimizer_created_at(r.name),
            "done/total": f"{len(results)}/{budget}",
            "best_score": f"{best_score:.3f}" if isinstance(best_score, (int, float)) else "—",
            "best_ctx": str(int(best_ctx)) if isinstance(best_ctx, (int, float)) else "—",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        st_dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Aucun run Optimizer lisible.")
        return

    with st.expander("Détails par run Optimizer"):
        selected = st.selectbox("Choisir un run", options=[r["run_id"] for r in rows])
        run_dir = paths.runs_dir / "optimizer" / selected
        summary = summaries.get(selected) or load_optimizer_summary(run_dir) or {}
        if summary:
            cfg = summary.get("config", {}) or {}
            st.markdown("**Config:**")
            st.json(cfg)

            best = summary.get("best")
            if best and isinstance(best, dict):
                st.markdown("**Best:**")
                if best.get("status") == "completed" and best.get("llama_server_cmd"):
                    st.code(best["llama_server_cmd"], language="bash")
                elif best.get("status") != "completed":
                    st.caption("Aucun best run valide (aucun run 'completed').")
                st.json({k: v for k, v in best.items() if k != "raw_output"})

            logs_path = run_dir / "logs.txt"
            if logs_path.exists():
                with st.expander("Logs (tail)"):
                    txt = logs_path.read_text(encoding="utf-8", errors="replace")
                    st.code("\n".join(txt.splitlines()[-200:]), language="text")

            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                st.download_button(
                    "Télécharger summary.json",
                    summary_path.read_bytes(),
                    file_name=f"{selected}_summary.json",
                )

        if st.button("Supprimer ce run Optimizer", type="secondary", key=f"del_optimizer_{selected}"):
            delete_optimizer_run(run_dir)
            st.success("Supprimé.")
            st.rerun()


def _render_sweeps(paths) -> None:
    sweeps = list_sweeps(paths)
    if not sweeps:
        st.info("Aucun sweep enregistré.")
        return

    st.caption(f"{len(sweeps)} sweep(s) trouvé(s)")

    rows = []
    for s in sweeps:
        # Try sweep.json first (new format), then sweep_info.json (legacy)
        info_path = s / "sweep.json"
        if not info_path.exists():
            info_path = s / "sweep_info.json"
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        rows.append({
            "sweep_id": s.name,
            "model": info.get("model_file", info.get("model", "?")),
            "status": info.get("status", "?"),
            "done/total": f"{info.get('done', '?')}/{info.get('total', '?')}",
            "created": info.get("created_at", "?")[:16] if info.get("created_at") else "?",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("Aucun sweep lisible.")
        return

    st_dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Détails par sweep"):
        selected = st.selectbox("Choisir un sweep", options=[r["sweep_id"] for r in rows])
        sweep_d = paths.runs_dir / "sweeps" / selected
        
        # Load sweep info
        info_path = sweep_d / "sweep.json"
        if not info_path.exists():
            info_path = sweep_d / "sweep_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            
            # Show summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Configuration:**")
                base_cfg = info.get("base_config", {})
                st.markdown(f"""
- **Threads:** {base_cfg.get('threads', '?')}
- **Batch:** {base_cfg.get('batch', '?')}
- **UBatch:** {base_cfg.get('ubatch', '?')}
- **GPU Layers:** {base_cfg.get('gpu_layers', '?')}
""")
            with col2:
                st.markdown("**Sweep Parameters:**")
                sweep_vals = info.get("sweep_values", {})
                for k, v in sweep_vals.items():
                    st.markdown(f"- **{k}:** {v}")
                weights = info.get("weights", {})
                st.markdown(f"- **Weights:** tg={weights.get('w_tg', '?')}, pp={weights.get('w_pp', '?')}")
            
            # Best run
            if info.get("best_run_id"):
                st.success(f"🏆 Best run: `{info['best_run_id']}`")
            if info.get("best_llama_server_cmd"):
                st.markdown("**Commande llama-server (best run):**")
                st.code(info["best_llama_server_cmd"], language="bash")
                if info.get("best_ctx_size_tokens"):
                    st.caption(f"ctx-size ≈ {int(info['best_ctx_size_tokens'])} tokens.")
            
            with st.expander("JSON complet"):
                st.json({k: v for k, v in info.items() if k != "results"})

        csv_path = sweep_d / "sweep_results.csv"
        if csv_path.exists():
            df_s = pd.read_csv(csv_path).convert_dtypes()
            for col in ("ngl", "gpu_layers", "threads", "batch", "ubatch", "n_cpu_moe"):
                if col in df_s.columns:
                    df_s[col] = pd.to_numeric(df_s[col], errors="coerce")
            st_dataframe(df_s, use_container_width=True, hide_index=True)
            st.download_button("Télécharger sweep_results.csv", csv_path.read_bytes(), file_name=f"{selected}_sweep_results.csv")

            # Quick chart - find tg column dynamically
            tg_col = next((c for c in df_s.columns if c.startswith("tg")), None)
            if tg_col:
                num_cols = [c for c in ["ngl", "threads", "batch", "ubatch", "n_cpu_moe", "gpu_layers"] if c in df_s.columns]
                if num_cols:
                    x_col = st.selectbox("X axis", options=num_cols, key=f"xcol_{selected}")
                    st.line_chart(df_s, x=x_col, y=tg_col)
        else:
            st.warning("Pas de fichier sweep_results.csv")

        if st.button("Supprimer ce sweep", type="secondary", key=f"del_sweep_{selected}"):
            delete_sweep(sweep_d)
            st.success("Supprimé.")
            st.rerun()
            st.rerun()


def render() -> None:
    paths = get_paths()
    settings = get_settings()
    sidebar(paths, settings)

    st.title("Historique")
    
    # Bulk delete section at the top
    with st.expander("🗑️ Suppression en masse"):
        st.warning("Attention: cette action est irréversible!")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Supprimer tous les runs failed", type="secondary"):
                count = delete_all_failed_runs(paths, "bench")
                st.success(f"{count} run(s) supprimé(s)")
                st.rerun()
        
        with col2:
            if st.button("Supprimer tous les sweeps failed", type="secondary"):
                count = delete_all_failed_runs(paths, "sweeps")
                st.success(f"{count} sweep(s) supprimé(s)")
                st.rerun()
        
        with col3:
            if st.button("Supprimer tous les optimus failed", type="secondary"):
                count = delete_all_failed_runs(paths, "optimus")
                st.success(f"{count} optimus run(s) supprimé(s)")
                st.rerun()
        
        with col4:
            if st.button("🧹 Supprimer TOUS les failed", type="primary"):
                count = delete_all_failed_runs(paths, "all")
                st.success(f"{count} entrée(s) supprimée(s)")
                st.rerun()

    tab_runs, tab_sweeps, tab_optimus, tab_optimizer = st.tabs(["Bench Runs", "Sweeps", "Optimus", "Optimizer"])

    with tab_runs:
        _render_runs(paths)

    with tab_sweeps:
        _render_sweeps(paths)
    
    with tab_optimus:
        _render_optimus(paths)

    with tab_optimizer:
        _render_optimizer(paths)
