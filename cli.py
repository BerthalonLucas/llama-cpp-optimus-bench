#!/usr/bin/env python3
"""
CLI for llama-cpp-optimus-bench

Usage:
    ./cli.py optimize <model> [options]
    ./cli.py dashboard [options]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add streamlit_dashboard to path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_dashboard.core.paths import RepoPaths
from streamlit_dashboard.core.settings import AppSettings
from streamlit_dashboard.core.gguf_meta import read_gguf_meta_fast


# ─────────────────────────────────────────────────────────────────────────────
# Rich console output (optional, fallback to basic if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class SimpleConsole:
    """Fallback console when rich is not available"""
    def print(self, *args, **kwargs):
        # Strip rich markup
        text = " ".join(str(a) for a in args)
        for tag in ["[bold", "[green", "[red", "[yellow", "[blue", "[cyan", "[dim", "[/", "]"]:
            text = text.replace(tag, "")
        print(text)
    
    def rule(self, title=""):
        print(f"\n{'─' * 60}")
        if title:
            print(f"  {title}")
        print(f"{'─' * 60}\n")


console = Console() if RICH_AVAILABLE else SimpleConsole()


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_hardware() -> dict:
    """Detect available hardware (GPU/CPU)"""
    import subprocess
    import multiprocessing
    
    hw = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_mb": 0,
        "cpu_cores": multiprocessing.cpu_count(),
    }
    
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            hw["gpu_available"] = True
            hw["gpu_name"] = parts[0].strip()
            hw["gpu_vram_mb"] = int(parts[1].strip()) if len(parts) > 1 else 0
    except Exception:
        pass
    
    # Try rocm-smi for AMD
    if not hw["gpu_available"]:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                hw["gpu_available"] = True
                hw["gpu_name"] = "AMD GPU (ROCm)"
        except Exception:
            pass
    
    return hw


def print_hardware_info(hw: dict):
    """Display hardware information"""
    console.print("\n[bold cyan]🖥️  Hardware Detected[/bold cyan]")
    console.print(f"  CPU: {hw['cpu_cores']} cores")
    if hw["gpu_available"]:
        vram_gb = hw["gpu_vram_mb"] / 1024 if hw["gpu_vram_mb"] else 0
        console.print(f"  GPU: [green]{hw['gpu_name']}[/green] ({vram_gb:.1f} GB VRAM)")
    else:
        console.print("  GPU: [yellow]Not detected (CPU-only mode)[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Optimize Command
# ─────────────────────────────────────────────────────────────────────────────
def cmd_optimize(args):
    """Run HyperOptimus optimization"""
    from streamlit_dashboard.core.hyperoptimus import HyperOptConfig, run_optuna
    
    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / "models" / args.model
    
    if not model_path.exists():
        console.print(f"[red]❌ Model not found: {model_path}[/red]")
        sys.exit(1)
    
    # Detect hardware
    hw = detect_hardware()
    print_hardware_info(hw)
    
    # Read model metadata
    console.print(f"\n[bold cyan]📦 Model[/bold cyan]: {model_path.name}")
    try:
        meta = read_gguf_meta_fast(model_path)
        model_ctx = meta.context_length or 32768
        expert_count = meta.expert_count or 0
        console.print(f"  Context: {model_ctx:,} tokens")
        if expert_count:
            console.print(f"  Experts: {expert_count} (MoE model)")
    except Exception:
        model_ctx = 32768
        expert_count = 0
    
    # Determine optimization mode
    if not hw["gpu_available"]:
        console.print("\n[yellow]⚠️  No GPU detected - using CPU-only optimization[/yellow]")
        ngl_default = 0
    else:
        ngl_default = 999
    
    # Setup paths
    paths = RepoPaths()
    settings = AppSettings()
    
    # Configure optimization
    preset = args.preset or "fast"
    presets = {
        "fast": {"trials": 20, "repeats": 1, "n_tokens": 128, "timeout": 60},
        "mid": {"trials": 50, "repeats": 2, "n_tokens": 256, "timeout": 90},
        "high": {"trials": 100, "repeats": 3, "n_tokens": 512, "timeout": 120},
    }
    p = presets.get(preset, presets["fast"])
    
    # Context range based on VRAM
    if hw["gpu_vram_mb"] >= 24000:
        ctx_max = min(model_ctx, 65536)
    elif hw["gpu_vram_mb"] >= 16000:
        ctx_max = min(model_ctx, 32768)
    elif hw["gpu_vram_mb"] >= 8000:
        ctx_max = min(model_ctx, 16384)
    else:
        ctx_max = min(model_ctx, 8192)
    
    ctx_min = args.ctx_min or 2048
    ctx_max = args.ctx_max or ctx_max
    
    cfg = HyperOptConfig(
        model_path_host=model_path,
        is_moe=expert_count > 0,
        model_max_ctx=model_ctx,
        model_expert_count=expert_count,
        ctx_range=(ctx_min, ctx_max, args.ctx_step or 2048),
        batch_range=(args.batch_min or 512, args.batch_max or 8192, 512),
        ubatch_range=(args.ubatch_min or 256, args.ubatch_max or 4096, 256),
        threads_range=(args.threads_min or 4, args.threads_max or min(16, hw["cpu_cores"]), 2),
        trials=args.trials or p["trials"],
        repeats=args.repeats or p["repeats"],
        n_tokens=args.n_tokens or p["n_tokens"],
        timeout_sec=args.timeout or p["timeout"],
        weight_tg=args.weight_tg or 0.5,
        weight_pp=args.weight_pp or 0.3,
        weight_ctx=args.weight_ctx or 0.2,
    )
    
    console.print(f"\n[bold cyan]⚙️  Optimization Settings ({preset})[/bold cyan]")
    console.print(f"  Trials: {cfg.trials}")
    console.print(f"  Context: {cfg.ctx_range[0]:,} - {cfg.ctx_range[1]:,} (step {cfg.ctx_range[2]})")
    console.print(f"  Batch: {cfg.batch_range[0]} - {cfg.batch_range[1]}")
    console.print(f"  Weights: TG={cfg.weight_tg} PP={cfg.weight_pp} CTX={cfg.weight_ctx}")
    
    console.rule("[bold green]Starting Optimization[/bold green]")
    
    # Custom logging function with nice formatting
    best_score = 0.0
    best_cfg = {}
    trial_count = 0
    
    def log_fn(msg: str):
        nonlocal best_score, best_cfg, trial_count
        
        # Parse and reformat the log message
        if "Trial" in msg and ("✅" in msg or "❌" in msg):
            trial_count += 1
            # Extract key info for cleaner display
            if "✅" in msg:
                # Success trial
                parts = msg.split("|")
                score_part = [p for p in parts if "score=" in p]
                ctx_part = [p for p in parts if "ctx=" in p]
                tg_part = [p for p in parts if "TG=" in p]
                pp_part = [p for p in parts if "PP=" in p]
                
                score = float(score_part[0].split("=")[1]) if score_part else 0
                ctx = ctx_part[0].split("=")[1].replace(",", "").strip() if ctx_part else "?"
                tg = tg_part[0].split("=")[1].replace("t/s", "").strip() if tg_part else "?"
                pp = pp_part[0].split("=")[1].replace("t/s", "").strip() if pp_part else "?"
                
                console.print(f"  [green]✓[/green] Trial {trial_count:3d}/{cfg.trials} │ score={score:6.1f} │ ctx={ctx:>6} │ TG={tg:>6} │ PP={pp:>6}")
            else:
                # Failed trial
                error = msg.split("-")[-1].strip() if "-" in msg else "error"
                console.print(f"  [red]✗[/red] Trial {trial_count:3d}/{cfg.trials} │ {error[:50]}")
        
        elif "NEW BEST" in msg:
            console.print(f"  [bold yellow]★ {msg.split(']')[-1].strip()}[/bold yellow]")
        
        elif "🚀" in msg or "📊" in msg or "🎯" in msg or "⚖️" in msg:
            # Config info - skip, we already showed it
            pass
        elif "──" in msg:
            # Separator - skip
            pass
        else:
            # Other messages
            if msg.strip():
                console.print(f"  [dim]{msg.split(']')[-1].strip() if ']' in msg else msg}[/dim]")
    
    start_time = time.time()
    
    try:
        results, best, run_dir = run_optuna(
            cfg,
            paths,
            settings,
            log_fn=log_fn,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Optimization interrupted by user[/yellow]")
        sys.exit(130)
    
    elapsed = time.time() - start_time
    
    # Display results
    console.rule("[bold green]Optimization Complete[/bold green]")
    
    console.print(f"\n⏱️  Duration: {elapsed/60:.1f} minutes")
    console.print(f"📁 Results saved to: {run_dir}")
    
    if best:
        console.print("\n[bold green]🏆 Best Configuration Found:[/bold green]")
        console.print(f"  Score: [bold]{best.score:.2f}[/bold]")
        console.print(f"  Context: {best.ctx_tokens:,} tokens")
        
        tg = best.metrics.get("tg", 0)
        pp = best.metrics.get("pp", 0)
        console.print(f"  TG: {tg:.1f} t/s | PP: {pp:.1f} t/s")
        
        console.print("\n[bold cyan]📋 Recommended llama-server command:[/bold cyan]")
        cmd_parts = [
            "./llama.sh server",
            f"--model {model_path.name}",
            f"-ngl {best.cfg.get('ngl', 99)}",
            f"-c {best.ctx_tokens}",
            f"--batch-size {best.cfg.get('batch', 2048)}",
            f"--ubatch-size {best.cfg.get('ubatch', 512)}",
            f"-t {best.cfg.get('threads', 8)}",
        ]
        if best.cfg.get("flash_attn"):
            cmd_parts.append("--flash-attn")
        if best.cfg.get("no_kv_offload"):
            cmd_parts.append("--no-kv-offload")
        
        console.print(f"\n  [green]{' '.join(cmd_parts)}[/green]")
    else:
        console.print("\n[red]❌ No successful trials found[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Command
# ─────────────────────────────────────────────────────────────────────────────
def cmd_dashboard(args):
    """Launch Streamlit dashboard"""
    import subprocess
    
    port = args.port or 8510
    host = args.host or "0.0.0.0"
    
    console.print(f"\n[bold cyan]🚀 Launching Dashboard[/bold cyan]")
    console.print(f"  URL: http://localhost:{port}")
    console.print(f"  Host: {host}")
    console.print("\n  Press Ctrl+C to stop\n")
    
    app_path = Path(__file__).parent / "streamlit_dashboard" / "streamlit_app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="llama-cpp-optimus-bench CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick optimization
  ./cli.py optimize models/my-model.gguf --preset fast
  
  # Full optimization with custom settings
  ./cli.py optimize models/my-model.gguf --trials 100 --ctx-max 32768
  
  # Launch dashboard
  ./cli.py dashboard --port 8510
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run HyperOptimus optimization")
    opt_parser.add_argument("model", help="Path to GGUF model (relative to models/ or absolute)")
    opt_parser.add_argument("--preset", choices=["fast", "mid", "high"], default="fast",
                           help="Optimization preset (default: fast)")
    opt_parser.add_argument("--trials", type=int, help="Number of trials")
    opt_parser.add_argument("--repeats", type=int, help="Benchmark repeats per trial")
    opt_parser.add_argument("--n-tokens", type=int, help="Tokens to generate")
    opt_parser.add_argument("--timeout", type=int, help="Timeout per trial (seconds)")
    
    # Context range
    opt_parser.add_argument("--ctx-min", type=int, help="Minimum context size")
    opt_parser.add_argument("--ctx-max", type=int, help="Maximum context size")
    opt_parser.add_argument("--ctx-step", type=int, help="Context step size")
    
    # Batch range
    opt_parser.add_argument("--batch-min", type=int, help="Minimum batch size")
    opt_parser.add_argument("--batch-max", type=int, help="Maximum batch size")
    opt_parser.add_argument("--ubatch-min", type=int, help="Minimum ubatch size")
    opt_parser.add_argument("--ubatch-max", type=int, help="Maximum ubatch size")
    
    # Thread range
    opt_parser.add_argument("--threads-min", type=int, help="Minimum threads")
    opt_parser.add_argument("--threads-max", type=int, help="Maximum threads")
    
    # Weights
    opt_parser.add_argument("--weight-tg", type=float, help="TG weight (default: 0.5)")
    opt_parser.add_argument("--weight-pp", type=float, help="PP weight (default: 0.3)")
    opt_parser.add_argument("--weight-ctx", type=float, help="CTX weight (default: 0.2)")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8510, help="Port (default: 8510)")
    dash_parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
