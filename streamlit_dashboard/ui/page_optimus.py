"""
llama-optimus page - Intelligent Bayesian optimization for llama.cpp parameters.

Uses Optuna to find optimal batch, ubatch, threads, ngl, flash_attn, and tensor overrides.
For MoE models, optimizes ncmoe (experts on CPU) instead of ngl.
"""

import streamlit as st
import subprocess
import threading
import queue
import time
import re
import shlex
from typing import Any
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.optimus import (
    OptimusConfig,
    OptimusResult,
    MoeSweepResult,
    run_optimus_blocking,
    get_container_model_path,
    estimate_optimus_duration,
    parse_optimus_output,
    run_moe_ncmoe_sweep,
)
from core.bench import format_llama_server_command
from core.paths import RepoPaths, get_model_source_label
from core.docker import check_container_running, start_container, kill_in_container
from core.gguf_meta import get_cached_meta
from core.storage import new_id, save_optimus_run, list_models_by_source
from core.settings import AppSettings


def list_available_models(paths: RepoPaths, settings: AppSettings) -> list[tuple[str, Path]]:
    """List GGUF models across local + external sources (LM Studio, etc.).
    
    Returns list of (label, absolute_path).
    """
    grouped = list_models_by_source(paths, settings.external_model_dirs)
    models: list[tuple[str, Path]] = []
    for source, files in grouped.items():
        for f in files:
            label = str(f.relative_to(paths.models_dir)) if source == "local" else f"{f.name} ({get_model_source_label(f, paths)})"
            models.append((label, f))
    models.sort(key=lambda x: x[0].lower())
    return models


def run_moe_sweep_with_queue(
    model_path: str,
    max_layers: int,
    metric: str,
    n_tokens: int,
    step: int,
    output_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event | None = None,
):
    """Run MoE ncmoe sweep in a thread, putting output lines in a queue."""
    try:
        gen = run_moe_ncmoe_sweep(
            model_path=model_path,
            max_layers=max_layers,
            metric=metric,
            n_tokens=n_tokens,
            step=step,
            container_name="llama-cpp",
            stop_event=stop_event,
        )
        # Consume generator while capturing its return value (StopIteration.value)
        while True:
            try:
                line = next(gen)
            except StopIteration as e:
                result_queue.put(e.value)
                output_queue.put(None)  # Signal completion
                return
            output_queue.put(line)
        
    except Exception as e:
        output_queue.put(f"[MoE] Error: {str(e)}\n")
        output_queue.put(None)
        result_queue.put(MoeSweepResult(ok=False, stopped=False, best_ncmoe=None, best_score=None, results=[]))


def run_optimus_with_queue(
    config: OptimusConfig,
    output_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event | None = None,
    proc_holder: dict[str, Any] | None = None,
):
    """Run llama-optimus in a thread, putting output lines in a queue, stoppable via stop_event."""
    output_lines = []
    workspace_root = Path(__file__).parent.parent.parent

    moe_wrapper_dir = "/tmp/llama-bin-moe"

    # For MoE: llama-optimus doesn't know about -ncmoe, so we create a small
    # wrapper `llama-bench` that injects `-ncmoe $LLAMA_BENCH_NCMOE`.
    if config.is_moe and config.moe_ncmoe is not None:
        config.llama_bin_dir = moe_wrapper_dir
        setup_script = f"""set -e
mkdir -p {moe_wrapper_dir}
cat > {moe_wrapper_dir}/llama-bench <<'EOF'
#!/bin/sh
set -eu
REAL="/app/llama-bench"
if [ -n "${{LLAMA_BENCH_NCMOE:-}}" ]; then
  case " $* " in
    *" -ncmoe "*) exec "$REAL" "$@" ;;
    *) exec "$REAL" -ncmoe "$LLAMA_BENCH_NCMOE" "$@" ;;
  esac
else
  exec "$REAL" "$@"
fi
EOF
chmod +x {moe_wrapper_dir}/llama-bench
for f in /app/llama-*; do
  name="$(basename "$f")"
  [ "$name" = "llama-bench" ] && continue
  ln -sf "$f" "{moe_wrapper_dir}/$name" 2>/dev/null || true
done
"""
        try:
            setup = subprocess.run(
                ["docker", "compose", "exec", "-T", "llama-cpp", "sh", "-c", setup_script],
                capture_output=True,
                text=True,
                cwd=str(workspace_root),
                timeout=30,
            )
            if setup.returncode != 0:
                output_queue.put(
                    "[MoE] ⚠️ Failed to create llama-bench wrapper; continuing anyway.\n"
                    + (setup.stderr or setup.stdout or "")
                    + "\n"
                )
        except Exception as e:
            output_queue.put(f"[MoE] ⚠️ Failed to create llama-bench wrapper: {e}\n")

        inner = shlex.join(["llama-optimus"] + config.to_cli_args())
        inner = f"LLAMA_BENCH_NCMOE={int(config.moe_ncmoe)} {inner}"
        cmd = ["docker", "compose", "exec", "-T", "llama-cpp", "sh", "-lc", inner]
    else:
        cmd = [
            "docker", "compose", "exec", "-T", "llama-cpp",
            "llama-optimus"
        ] + config.to_cli_args()
    
    # Log the command being run
    cmd_str = ' '.join(cmd)
    output_lines.append(f"[CMD] {cmd_str}\n")
    output_queue.put(f"[CMD] {cmd_str}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(workspace_root)
        )
        if proc_holder is not None:
            proc_holder["proc"] = process
        
        for line in iter(process.stdout.readline, ''):
            if stop_event and stop_event.is_set():
                process.terminate()
                output_lines.append("[Optimus] Stopped by user\n")
                break
            output_lines.append(line)
            output_queue.put(line)
        
        process.wait()
        full_output = ''.join(output_lines)
        
        # If stopped, return early with a failure result
        if stop_event and stop_event.is_set():
            try:
                kill_in_container(RepoPaths.detect(), AppSettings(), patterns=["llama-optimus", "llama-bench"])
            except Exception:
                pass
            result_queue.put(OptimusResult(
                success=False,
                error_message="Stopped by user",
                raw_output=full_output
            ))
            output_queue.put(None)
            return
        
        # Always parse output to detect errors in content
        result = parse_optimus_output(full_output)
        
        # If returncode is non-zero but parse didn't catch an error, set it
        if process.returncode != 0 and result.success:
            result.success = False
            result.error_message = f"Process exited with code {process.returncode}"
        
        result_queue.put(result)
        output_queue.put(None)  # Signal completion
        
    except Exception as e:
        result_queue.put(OptimusResult(
            success=False,
            error_message=str(e),
            raw_output=''.join(output_lines)
        ))
        output_queue.put(None)


def render_stage_results(result: OptimusResult):
    """Render the stage results as nice cards."""
    if not result.stages:
        return
    
    st.subheader("📊 Optimization Stages")
    
    cols = st.columns(len(result.stages))
    for i, stage in enumerate(result.stages):
        with cols[i]:
            st.markdown(f"### Stage {stage.stage}")
            st.caption(stage.stage_name)
            st.metric("Best Score", f"{stage.best_value:.2f} t/s")
            
            with st.expander("Parameters"):
                for key, value in stage.best_params.items():
                    st.text(f"{key}: {value}")


def render_final_config(result: OptimusResult):
    """Render the final optimized configuration."""
    if not result.final_config:
        return
    
    st.subheader("🎯 Final Optimized Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numeric Parameters**")
        for key in ["batch", "ubatch", "threads", "ngl", "ncmoe"]:
            if key in result.final_config:
                st.metric(key, result.final_config[key])
    
    with col2:
        st.markdown("**Categorical Parameters**")
        for key in ["flash_attn", "override_tensor"]:
            if key in result.final_config:
                st.metric(key, result.final_config[key])


def render_commands(result: OptimusResult):
    """Render the generated commands."""
    st.subheader("📋 Generated Commands")
    
    if result.llama_server_cmd:
        st.markdown("**llama-server command:**")
        st.code(result.llama_server_cmd, language="bash")
        if getattr(result, "ctx_size_tokens", None):
            st.caption(f"ctx-size recommandé ≈ {int(result.ctx_size_tokens)} tokens.")
    
    if result.llama_bench_cmd:
        st.markdown("**llama-bench command:**")
        st.code(result.llama_bench_cmd, language="bash")


def render():
    """Main render function for the Optimus page."""
    st.title("🧠 llama-optimus")
    st.markdown("""
    **Intelligent Bayesian Optimization** for llama.cpp parameters using Optuna.
    
    This tool automatically finds optimal values for:
    - `batch` / `ubatch` - Batch sizes
    - `threads` - CPU thread count  
    - `ngl` - GPU layers offload (or `ncmoe` for MoE models)
    - `flash_attn` - Flash attention toggle
    - `override_tensor` - Tensor placement strategy
    """)
    
    paths = RepoPaths.detect()
    settings = AppSettings()
    
    # Check container status
    container_ok = check_container_running("llama-cpp")
    if not container_ok:
        st.error("⚠️ Container `llama-cpp` is not running.")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Start Container", type="primary"):
                with st.spinner("Starting container..."):
                    success, message = start_container("llama-cpp", cwd=paths.repo_root)
                    if success:
                        st.success(message)
                        time.sleep(2)  # Wait for container to be ready
                        st.rerun()
                    else:
                        st.error(message)
        with col2:
            st.code("docker compose up -d", language="bash")
        return
    
    st.success("✅ Container `llama-cpp` is running")
    
    # Model selection
    models = list_available_models(paths, settings)
    if not models:
        st.warning("No GGUF models found in models/ directory. Download one first!")
        return
    
    st.divider()
    
    # Configuration form
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model selection with detection of MoE
        model_options = [m[0] for m in models]
        selected_model_idx = st.selectbox(
            "Model",
            options=range(len(model_options)),
            format_func=lambda i: model_options[i],
            help="Select the GGUF model to optimize"
        )
        
        selected_model_label, selected_model_path = models[selected_model_idx]
        
        # Auto-detect MoE from GGUF metadata
        meta = get_cached_meta(paths, selected_model_path)
        is_moe = meta.is_moe
        expert_count = meta.expert_count or 0
        layer_count = meta.layer_count or 0
        
        # Show model info
        if is_moe:
            st.info(f"🔀 **MoE Model Detected** - {expert_count} experts, {layer_count} couches")
            st.caption("""
            For MoE models, `ngl` is set to max (99) to keep non-expert layers on GPU.
            We optimize `ncmoe` (layers with experts on CPU) instead.
            """)
        else:
            st.caption(f"📦 Dense model · Architecture: `{meta.architecture or 'unknown'}`")
        
        trials = st.slider(
            "Optimization Trials",
            min_value=3,
            max_value=100,
            value=20,
            step=1,
            help="More trials = better results but longer runtime"
        )
        
        metric = st.selectbox(
            "Optimization Metric",
            options=["tg", "pp", "mean"],
            index=0,
            help="tg=Text Generation, pp=Prompt Processing, mean=Both"
        )
        
        # MoE ncmoe step configuration - based on LAYER_COUNT, not expert_count!
        if is_moe and layer_count > 0:
            default_step = max(1, layer_count // 8)
            ncmoe_step = st.number_input(
                "ncmoe Sweep Step",
                min_value=1,
                max_value=max(1, layer_count // 2),
                value=default_step,
                step=1,
                help=f"Step size for ncmoe sweep (0 to {layer_count} layers). Smaller = more precise but slower."
            )
        else:
            ncmoe_step = 1
    
    with col2:
        repeat = st.slider(
            "Repeats per Config",
            min_value=1,
            max_value=5,
            value=2,
            help="More repeats = more accurate measurements"
        )
        
        n_tokens = st.number_input(
            "Tokens per Benchmark",
            min_value=16,
            max_value=512,
            value=128,
            step=16,
            help="Number of tokens for each benchmark run"
        )
        
        override_mode = st.selectbox(
            "Override Tensor Mode",
            options=["scan", "none", "custom"],
            index=0,
            help="scan=Try all strategies, none=Skip, custom=Specific override"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        no_warmup = st.checkbox(
            "Skip Warmup",
            value=False,
            help="Skip warmup phase (faster but less accurate)"
        )
        
        custom_override = None
        if override_mode == "custom":
            custom_override = st.text_input(
                "Custom Override Key",
                placeholder="e.g., ffn_cpu_even",
                help="Specific tensor override to use"
            )
        
        extra_args_str = st.text_input(
            "Extra Arguments",
            placeholder="e.g., --n-warmup-tokens 64 --warmup-runs 10",
            help="Additional arguments passed to llama-optimus/llama-bench (optional)"
        )
    
    # Duration estimate
    estimated_time = estimate_optimus_duration(trials, repeat, is_moe)
    st.info(f"⏱️ Estimated duration: **{estimated_time}**" + (" (includes ncmoe sweep)" if is_moe else ""))
    
    st.divider()
    
    # Session state for results
    if "optimus_running" not in st.session_state:
        st.session_state.optimus_running = False
    if "optimus_result" not in st.session_state:
        st.session_state.optimus_result = None
    if "optimus_output" not in st.session_state:
        st.session_state.optimus_output = ""
    if "moe_best_ncmoe" not in st.session_state:
        st.session_state.moe_best_ncmoe = None
    if "optimus_stop_event" not in st.session_state:
        st.session_state.optimus_stop_event = None
    if "optimus_proc_holder" not in st.session_state:
        st.session_state.optimus_proc_holder = {}
    
    # Run button
    run_button = st.button(
        "🚀 Start Optimization" + (" (MoE: ncmoe sweep + optimus)" if is_moe else ""),
        type="primary",
        disabled=st.session_state.optimus_running,
        use_container_width=True
    )
    if st.session_state.optimus_running:
        if st.button("🛑 Stop run", type="secondary", use_container_width=True):
            if st.session_state.optimus_stop_event:
                st.session_state.optimus_stop_event.set()
            proc = (st.session_state.optimus_proc_holder or {}).get("proc")
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                kill_in_container(paths, settings, patterns=["llama-optimus", "llama-bench"])
            except Exception:
                pass
            st.warning("Arrêt demandé (process en cours).")
    
    if run_button and not st.session_state.optimus_running:
        # Build config
        extra_args = extra_args_str.split() if extra_args_str else []
        stop_event = threading.Event()
        proc_holder: dict[str, Any] = {}
        st.session_state.optimus_stop_event = stop_event
        st.session_state.optimus_proc_holder = proc_holder
        
        # Note: For MoE models, we do NOT add -ngl/-ncmoe to extra_args
        # because llama-optimus doesn't support them. Instead:
        # - MoE models: We run our own ncmoe sweep, then llama-optimus with --ngl-max 99
        # - llama-optimus will optimize batch, ubatch, threads, flash_attn, override
        
        config = OptimusConfig(
            model_path=get_container_model_path(selected_model_path, paths),
            trials=trials,
            repeat=repeat,
            metric=metric,
            n_tokens=n_tokens,
            no_warmup=no_warmup,
            override_mode=override_mode,
            custom_override=custom_override if override_mode == "custom" else None,
            is_moe=is_moe,
            moe_experts=expert_count if is_moe else 0,
            extra_args=extra_args
        )
        
        st.session_state.optimus_running = True
        st.session_state.optimus_result = None
        st.session_state.optimus_output = ""
        st.session_state.moe_best_ncmoe = None
        
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        full_output = ""
        abort_error_message: str | None = None
        
        # Phase 1: MoE ncmoe sweep (if MoE model)
        # Use layer_count for ncmoe bounds, NOT expert_count!
        if is_moe and layer_count > 0:
            status_placeholder.info(f"🔀 **Phase 1/2**: Finding optimal ncmoe value (0-{layer_count} layers, step={ncmoe_step})...")
            
            # Run ncmoe sweep in a thread
            model_container_path = get_container_model_path(selected_model_path, paths)
            best_ncmoe = 0
            
            moe_output_queue = queue.Queue()
            moe_result_queue = queue.Queue()
            
            moe_thread = threading.Thread(
                target=run_moe_sweep_with_queue,
                args=(model_container_path, layer_count, metric, n_tokens, ncmoe_step, moe_output_queue, moe_result_queue, stop_event)
            )
            moe_thread.start()
            
            # Stream MoE sweep output
            while True:
                try:
                    line = moe_output_queue.get(timeout=0.3)
                    if line is None:
                        break
                    full_output += line
                    # Show last 50 lines
                    lines = full_output.split('\n')
                    display_lines = '\n'.join(lines[-50:])
                    output_placeholder.code(display_lines, language="text")
                except queue.Empty:
                    # Check if thread is still alive
                    if stop_event.is_set() or not moe_thread.is_alive():
                        # Drain any remaining messages
                        while True:
                            try:
                                line = moe_output_queue.get_nowait()
                                if line is None:
                                    break
                                full_output += line
                            except queue.Empty:
                                break
                        break
            
            moe_thread.join()
            
            # Check result and use structured best ncmoe
            try:
                moe_res = moe_result_queue.get_nowait()
                if isinstance(moe_res, MoeSweepResult) and moe_res.ok and moe_res.best_ncmoe is not None:
                    best_ncmoe = int(moe_res.best_ncmoe)
                    best_score = float(moe_res.best_score or 0.0)
                    st.session_state.moe_best_ncmoe = best_ncmoe
                    config.moe_ncmoe = best_ncmoe
                    config.llama_bin_dir = "/tmp/llama-bin-moe"
                    full_output += f"\n[MoE] ✅ Best ncmoe={best_ncmoe} ({best_score:.2f} t/s)\n"
                    full_output += "[MoE] ℹ️ Note: llama-optimus will now optimize batch/ubatch/threads\n"
                    full_output += f"[MoE] ℹ️ For final config, combine llama-optimus results with ncmoe={best_ncmoe}\n\n"
                elif isinstance(moe_res, MoeSweepResult) and moe_res.stopped:
                    full_output += "\n[MoE] ⏹️ ncmoe sweep stopped by user\n\n"
                    abort_error_message = "MoE ncmoe sweep stopped by user"
                    stop_event.set()
                else:
                    full_output += "\n[MoE] ❌ ncmoe sweep failed (no valid results)\n\n"
                    abort_error_message = "MoE ncmoe sweep failed (no valid results)"
                    stop_event.set()
            except queue.Empty:
                full_output += "\n[MoE] ❌ ncmoe sweep did not complete\n\n"
                abort_error_message = "MoE ncmoe sweep did not complete"
                stop_event.set()
            
            status_placeholder.info("🧠 **Phase 2/2**: Running llama-optimus with optimal ncmoe...")
        else:
            status_placeholder.info("🧠 Running llama-optimus optimization...")
        
        if stop_event.is_set():
            status_placeholder.info("⏹️ Optimisation stoppée.")
            st.session_state.optimus_output = full_output
            if st.session_state.optimus_result is None:
                st.session_state.optimus_result = OptimusResult(
                    success=False,
                    error_message=abort_error_message or "Stopped by user",
                    raw_output=full_output,
                )
            st.session_state.optimus_running = False
            st.session_state.optimus_stop_event = None
            st.session_state.optimus_proc_holder = {}
            st.rerun()
        
        # Phase 2 (or only phase for non-MoE): Run llama-optimus
        # Run in subprocess and capture output
        output_queue = queue.Queue()
        result_queue = queue.Queue()
        
        thread = threading.Thread(
            target=run_optimus_with_queue,
            args=(config, output_queue, result_queue, stop_event, proc_holder)
        )
        thread.start()
        
        # Stream output
        while True:
            try:
                line = output_queue.get(timeout=0.3)
                if line is None:
                    break
                full_output += line
                # Show last 50 lines
                lines = full_output.split('\n')
                display_lines = '\n'.join(lines[-50:])
                output_placeholder.code(display_lines, language="text")
            except queue.Empty:
                # Check if thread is still alive
                if stop_event.is_set() or not thread.is_alive():
                    # Drain any remaining messages
                    while True:
                        try:
                            line = output_queue.get_nowait()
                            if line is None:
                                break
                            full_output += line
                        except queue.Empty:
                            break
                    break
        
        thread.join()
        
        # Get result
        try:
            result = result_queue.get_nowait()
            # Add MoE info to result if applicable
            if is_moe and st.session_state.moe_best_ncmoe is not None:
                result.final_config["ncmoe"] = st.session_state.moe_best_ncmoe
                result.final_config["ngl"] = 99  # Always max for MoE
            # Build a usable llama-server command (inject ncmoe/ctx-size)
            ctx_tokens = max(int(n_tokens), 128)
            try:
                cmd_params = dict(result.final_config)
                if cmd_params:
                    result.llama_server_cmd = format_llama_server_command(
                        model_path_container=get_container_model_path(selected_model_path, paths),
                        params=cmd_params,
                        ctx_size=ctx_tokens,
                    )
                    result.ctx_size_tokens = ctx_tokens
            except Exception:
                pass
            st.session_state.optimus_result = result
            st.session_state.optimus_output = full_output
            
            # Save to history
            run_id = new_id("optimus")
            run_data = {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "model_file": selected_model_label,
                "is_moe": is_moe,
                "expert_count": expert_count if is_moe else None,
                "config": {
                    "trials": trials,
                    "repeat": repeat,
                    "metric": metric,
                    "n_tokens": n_tokens,
                    "override_mode": override_mode,
                    "no_warmup": no_warmup,
                },
                "status": "completed" if result.success else "failed",
                "error_message": result.error_message if not result.success else None,
                "gpu_info": result.gpu_info,
                "stages": [
                    {
                        "stage": s.stage,
                        "stage_name": s.stage_name,
                        "best_params": s.best_params,
                        "best_value": s.best_value,
                    }
                    for s in result.stages
                ],
                "final_config": result.final_config,
                "llama_server_cmd": result.llama_server_cmd,
                "llama_bench_cmd": result.llama_bench_cmd,
                "optimized_results": result.optimized_results,
                "ctx_size_tokens": getattr(result, "ctx_size_tokens", None),
                "raw_output": full_output,
            }
            save_optimus_run(paths, run_id, run_data)
            st.session_state.optimus_last_run_id = run_id
            
        except queue.Empty:
            st.error("Failed to get optimization result")
        
        status_placeholder.empty()
        st.session_state.optimus_stop_event = None
        st.session_state.optimus_proc_holder = {}
        st.session_state.optimus_running = False
        st.rerun()
    
    # Display results
    if st.session_state.optimus_result:
        result = st.session_state.optimus_result
        
        if result.success:
            st.success("✅ Optimization completed successfully!")
            
            if result.gpu_info:
                st.info(f"🎮 GPU: {result.gpu_info}")
            
            render_stage_results(result)
            st.divider()
            render_final_config(result)
            st.divider()
            render_commands(result)
            
            # Raw output
            with st.expander("📜 Raw Output", expanded=False):
                st.code(st.session_state.optimus_output, language="text")
            
            # Action buttons
            btn_cols = st.columns([1, 1, 2])
            if btn_cols[0].button("🔄 Relancer", type="primary", use_container_width=True,
                                  help="Effacer les résultats et recommencer"):
                st.session_state.optimus_result = None
                st.session_state.optimus_output = ""
                st.rerun()
            if btn_cols[1].button("🗑️ Effacer", type="secondary", use_container_width=True):
                st.session_state.optimus_result = None
                st.session_state.optimus_output = ""
                st.rerun()
        else:
            st.error(f"❌ Optimization failed")
            
            # Show error message prominently
            st.markdown("### Error Details")
            st.code(result.error_message, language="text")
            
            # Show GPU info if available
            if result.gpu_info:
                st.info(f"🎮 GPU detected: {result.gpu_info}")
            
            # Common error hints
            if "failed to load model" in result.error_message.lower():
                st.warning("""
                **Possible causes:**
                - Model file is corrupted or incomplete
                - Model is too large for available RAM/VRAM
                - Model path is incorrect
                
                **Try:**
                - Use a smaller quantization (Q4_K_M instead of Q6_K)
                - Use a smaller model
                - Check if model file exists and is complete
                """)
            elif "out of memory" in result.error_message.lower():
                st.warning("""
                **CUDA Out of Memory**
                
                **Try:**
                - Use a smaller model
                - Use a more aggressive quantization
                - Reduce batch size in extra arguments: `--batch 512`
                """)
            elif "ngl" in result.error_message.lower() or "max ngl = 0" in st.session_state.optimus_output.lower():
                st.warning("""
                **Model cannot fit in VRAM**
                
                The model is too large to offload any layers to GPU.
                
                **Try:**
                - Use a smaller model
                - Use a more aggressive quantization (Q4_K_M, Q4_0, etc.)
                """)
            
            # Always show full output for errors - expanded by default
            st.markdown("### 📜 Full Output Log")
            st.code(st.session_state.optimus_output if st.session_state.optimus_output else result.raw_output, language="text")
            
            # Action buttons
            btn_cols = st.columns([1, 1, 2])
            if btn_cols[0].button("🔄 Réessayer", type="primary", use_container_width=True,
                                  help="Effacer les résultats et réessayer"):
                st.session_state.optimus_result = None
                st.session_state.optimus_output = ""
                st.rerun()
            if btn_cols[1].button("🗑️ Effacer", type="secondary", use_container_width=True, key="optimus_clear_err"):
                st.session_state.optimus_result = None
                st.session_state.optimus_output = ""
                st.rerun()


if __name__ == "__main__":
    render()
