from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Any

import streamlit as st

from ..core.gguf_meta import get_cached_meta
from ..core.hyperoptimus import (
    HyperOptConfig,
    HyperOptimusRunner,
    TrialResult,
)
from ..core.bench import format_llama_server_command
from ..core.paths import RepoPaths, host_path_to_container_path
from ..core.settings import AppSettings, SESSION_ACTIVE_TASK_KEY
from .common import sidebar, get_paths, get_settings, require_selected_model, st_dataframe


def _is_moe_model(meta) -> bool:
    if getattr(meta, "is_moe", False):
        return True
    name = (meta.path.name if hasattr(meta, "path") else "").lower()
    indicators = ["moe", "mixtral", "gpt-oss", "deepseek", "dbrx"]
    return any(x in name for x in indicators)


def render() -> None:
    paths = get_paths()
    settings = get_settings()
    sidebar(paths, settings)

    st.title("HyperOptimus")
    st.caption("🧠 Optimisation intelligente des paramètres llama.cpp (Bayésien)")
    
    with st.expander("🔍 Comment ça fonctionne ?"):
        st.markdown("""
        **HyperOptimus** utilise une stratégie d'optimisation **agressive** inspirée de llama-optimus.
        
        - 🚀 **Exploration maximale** : Teste toutes les combinaisons possibles (batch jusqu'à 16K, ubatch jusqu'à 8K)
        - 🧠 **Apprentissage intelligent** : Optuna (algorithme Bayésien) apprend quelles configurations fonctionnent
        - 🛡️ **Gestion des échecs** : Les combinaisons qui dépassent la VRAM sont automatiquement ignorées (score=0)
        - 🎯 **Objectif** : Trouver LA configuration qui pousse votre GPU au maximum pour pp, tg et ctx
        
        ⚠️ **C'est normal d'avoir des échecs** - c'est comme ça qu'on trouve les limites réelles de votre matériel !
        """)

    # --- State Initialization ---
    if "runner" not in st.session_state:
        st.session_state.runner = HyperOptimusRunner()
    
    runner: HyperOptimusRunner = st.session_state.runner

    # --- 1. CONFIGURATION ---
    st.header("⚙️ Configuration")

    model_path = require_selected_model(paths)
    meta = get_cached_meta(paths, model_path)
    is_moe = _is_moe_model(meta)
    
    # Get model limits from GGUF metadata
    model_max_ctx = meta.context_length if meta.context_length else 32768
    model_expert_count = meta.expert_count if meta.expert_count else 0

    if is_moe:
        st.info(f"🔀 Modèle MoE détecté ({model_expert_count} experts) - Optimisation ncmoe activée")
    
    # Display model info
    st.caption(f"📊 **{model_path.name}** | Max ctx: {model_max_ctx:,} tokens | {'MoE' if is_moe else 'Dense'}")

    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        metric_choice = st.radio(
            "🎯 Objectif",
            ["Mixte (Recommandé)", "Génération (tg)", "Prompt (pp)"],
            help="""**Mixte** (recommandé) : Score = 50% TG + 30% PP + 20% CTX\n
- TG (text generation) : vitesse de génération de tokens\n- PP (prompt processing) : vitesse de traitement du contexte\n- CTX : bonus pour contexte élevé\n\n**Génération** : Optimise uniquement la vitesse TG (100%)\n**Prompt** : Optimise uniquement la vitesse PP (100%)"""
        )
    
    with col_conf2:
        trials = st.slider("🔢 Nombre d'essais", 10, 5000, 100, step=10, help="Plus d'essais = meilleur résultat. 100-500 recommandé.")
        n_threads = st.number_input("🧵 Threads (CPU Cores)", 1, 128, 8, help="Fixer le nombre de threads (ex: cœurs physiques)")

    with col_conf3:
        # Simplified advanced options
        with st.expander("⚙️ Options avancées"):
            timeout = st.number_input("Timeout (s)", 30, 600, 120, help="Court = crash rapide détecté vite")
            repeats = st.number_input("Répétitions", 1, 5, 2)
            n_tokens = st.number_input("Tokens générés", 16, 512, 128)
            cache_type = st.selectbox("Cache Type (K/V)", ["f16", "q8_0", "q4_0"], index=0, help="f16 est plus sûr, q8_0 économise de la VRAM")
            
            st.divider()
            st.markdown("🔥 **Stratégie Top-Down** : Part du MAX et descend")
            st.caption("💡 Les crashes sont rapides (score=0), Optuna apprend les vraies limites du hardware")
            
            # Context range - defaults to model's MAX (top-down approach)
            ctx_max_ui = max(512000, model_max_ctx)
            col_ctx1, col_ctx2 = st.columns(2)
            with col_ctx1:
                ctx_min = st.number_input("Contexte min", 512, 65536, 2048, step=512)
            with col_ctx2:
                # DEFAULT TO MODEL MAX - top-down approach!
                ctx_max = st.number_input("Contexte max", 2048, ctx_max_ui, model_max_ctx, step=1024, 
                                          help=f"🎯 Max du modèle: {model_max_ctx:,}")
            
            col1, col2 = st.columns(2)
            with col1:
                batch_min = st.number_input("Batch min", 8, 8192, 512, step=256)
                batch_max = st.number_input("Batch max", 512, 32768, 16384, step=512)
            with col2:
                ubatch_min = st.number_input("Ubatch min", 4, 4096, 256, step=128)
                ubatch_max = st.number_input("Ubatch max", 128, 16384, 8192, step=256)
            
            # ncmoe range for MoE models
            if is_moe and model_expert_count > 0:
                st.divider()
                st.caption(f"🧠 **MoE** : {model_expert_count} experts détectés")
                # Generate ncmoe values from 0 to expert_count
                max_ncmoe = model_expert_count
                ncmoe_options = [0] + [2**i for i in range(1, 8) if 2**i <= max_ncmoe]
                if max_ncmoe not in ncmoe_options:
                    ncmoe_options.append(max_ncmoe)
                ncmoe_options = sorted(set(ncmoe_options))
            else:
                ncmoe_options = [0]
            
            st.caption("⚠️ Crashes = score 0 → Optuna évite ces zones automatiquement")

    # Map metric choice to weights - default is balanced 33/33/33
    w_tg, w_pp, w_ctx = 0.333, 0.333, 0.334  # Équilibré
    if "Génération" in metric_choice:
        w_tg, w_pp, w_ctx = 1.0, 0.0, 0.0
    elif "Prompt" in metric_choice:
        w_tg, w_pp, w_ctx = 0.0, 1.0, 0.0

    # --- 2. CONTROLS ---
    st.header("🎮 Contrôles")
    
    c1, c2, c3 = st.columns(3)
    
    is_running = runner.is_running
    
    with c1:
        if st.button("🚀 LANCER L'OPTIMISATION", type="primary", disabled=is_running, use_container_width=True):
            cfg = HyperOptConfig(
                model_path_host=model_path,
                is_moe=is_moe,
                # Model limits from GGUF
                model_max_ctx=model_max_ctx,
                model_expert_count=model_expert_count,
                # Optimization ranges - TOP-DOWN approach
                ctx_range=(ctx_min, ctx_max, 1024),
                weight_tg=w_tg,
                weight_pp=w_pp,
                weight_ctx=w_ctx,
                trials=trials,
                repeats=repeats,
                ctk=cache_type,
                ctv=cache_type,
                n_tokens=n_tokens,
                timeout_sec=timeout,  # Short timeout for fast crash detection
                # AGGRESSIVE ranges - start from MAX, let crashes guide Optuna down
                batch_range=(batch_min, batch_max, 512),
                ubatch_range=(ubatch_min, ubatch_max, 256),
                threads_range=(n_threads, n_threads, 1),
                ncmoe_values=ncmoe_options,  # Dynamic based on model's expert count
            )
            runner.start(cfg, paths, settings)
            st.session_state.logs = []
            st.session_state.best_score = 0.0
            st.session_state.best_cfg = None
            st.session_state.run_dir = None
            st.session_state.progress = 0.0
            st.session_state.current_trial = 0
            st.rerun()

    with c2:
        if st.button("⏹️ STOP", disabled=not is_running, use_container_width=True):
            runner.stop()
            st.warning("Arrêt demandé...")

    with c3:
        if is_running:
            st.warning("🔄 En cours...")
        elif st.session_state.get("best_cfg"):
            st.success("✅ Terminé")
            if st.session_state.get("best_score", 0) > 0:
                st.metric("🏆 Score Final", f"{st.session_state['best_score']:.1f}")
        else:
            st.info("⏸️ Prêt")

    # --- 3. PROGRESSION ---
    if is_running or st.session_state.get("best_cfg"):
        st.divider()
        st.subheader("📊 Progression")
        
        prog_val = st.session_state.get("progress", 0.0)
        cur_trial = st.session_state.get("current_trial", 0)
        best_sc = st.session_state.get("best_score", 0.0)
        
        st.progress(prog_val, text=f"Trial {cur_trial}/{trials}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Trial", f"{cur_trial}/{trials}")
        m2.metric("Meilleur Score", f"{best_sc:.3f}")
        
        # Show breakdown of best config if available
        if st.session_state.get("best_metrics") and st.session_state.get("best_cfg"):
            metrics = st.session_state["best_metrics"]
            best_cfg_progress = st.session_state["best_cfg"]
            tg = metrics.get("tg", 0)
            pp = metrics.get("pp", 0)
            ctx = best_cfg_progress.get("ctx", 0)
            
            # Always show key metrics prominently
            col1, col2, col3 = st.columns(3)
            col1.metric("🚀 TG", f"{tg:.1f} t/s")
            col2.metric("⚡ PP", f"{pp:.1f} t/s")
            col3.metric("📐 CTX", f"{ctx:,}")
        
        # --- 4. LOGS ---
        st.subheader("📜 Logs")
        
        # Count failed trials
        zero_count = sum(1 for log in st.session_state.get("logs", []) if "score=0.0" in log or "parse_error" in log or "OOM" in log)
        if zero_count > 0:
            st.warning(f"⚠️ {zero_count} essais ont échoué (score=0). Cela peut indiquer des paramètres batch/ubatch trop élevés causant un manque de mémoire (OOM).")
        
        log_container = st.empty()
        logs = st.session_state.get("logs", [])
        log_container.code("\n".join(logs[-20:]), language="text") # Show last 20 lines
        
        with st.expander("Voir tous les logs"):
            st.code("\n".join(logs), language="text")

    # --- 5. RÉSULTATS ---
    best_cfg = st.session_state.get("best_cfg")
    if best_cfg:
        st.divider()
        st.header("🏆 Meilleure Configuration")
        
        # Display config card - now includes optimized context!
        with st.container():
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            cc1.metric("📐 Contexte", f"{best_cfg.get('ctx', 'N/A'):,}")
            cc2.metric("Batch", best_cfg.get("batch"))
            cc3.metric("Ubatch", best_cfg.get("ubatch"))
            cc4.metric("Threads", best_cfg.get("threads"))
            cc5.metric("NGL", best_cfg.get("ngl"))
            
            if is_moe:
                st.metric("ncmoe (Experts CPU)", best_cfg.get("ncmoe"))

        st.subheader("📋 Commande llama-server")
        
        # Generate command using the best config dict directly (use optimized ctx)
        ctx_optimized = best_cfg.get("ctx", 8192)
        model_container = host_path_to_container_path(model_path, paths)
        server_cmd = format_llama_server_command(model_container, best_cfg, ctx_size=ctx_optimized)
        
        st.code(server_cmd, language="bash")
        st.caption("Copiez cette commande pour lancer votre serveur avec les paramètres optimaux.")
        
        # Show run directory
        if st.session_state.get("run_dir"):
            st.info(f"📂 Tous les logs et données sont sauvegardés dans:\n`{st.session_state['run_dir']}`")
            st.caption("Consultez `summary.csv` pour toutes les métriques de chaque essai.")

    # --- POLLING LOOP ---
    if is_running:
        # Process logs
        while not runner.log_queue.empty():
            msg = runner.log_queue.get()
            if "logs" not in st.session_state:
                st.session_state.logs = []
            st.session_state.logs.append(msg)
        
        # Process results
        while not runner.result_queue.empty():
            data = runner.result_queue.get()
            dtype = data.get("type")
            
            if dtype == "progress":
                st.session_state.current_trial = data["trial"]
                st.session_state.progress = data["trial"] / data["total"]
                st.session_state.best_score = data["best_score"]
            
            elif dtype == "new_best":
                st.session_state.best_score = data["score"]
                st.session_state.best_cfg = data["cfg"]
                st.session_state.best_metrics = data.get("metrics", {})
                st.toast(f"🎉 Nouveau record: {data['score']:.2f}")
            
            elif dtype == "complete":
                st.success("Optimisation terminée !")
                st.session_state.best_cfg = data["best"].cfg if data.get("best") else None
                if data.get("run_dir"):
                    st.session_state.run_dir = str(data["run_dir"])
                st.rerun()
            
            elif dtype == "error":
                st.error(f"Erreur: {data['message']}")

        time.sleep(1)
        st.rerun()
