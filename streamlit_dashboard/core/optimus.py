"""
llama-optimus integration for intelligent parameter optimization.

Uses Bayesian optimization (Optuna) to find optimal parameters for llama.cpp.
"""

import subprocess
import re
from dataclasses import dataclass, field
from typing import Generator, Optional, Any
from pathlib import Path

from .paths import RepoPaths, host_path_to_container_path
from .bench import extract_metrics, parse_llama_bench_csv


@dataclass
class OptimusConfig:
    """Configuration for llama-optimus run."""
    model_path: str
    llama_bin_dir: str = "/app"
    trials: int = 20
    repeat: int = 2
    metric: str = "tg"  # tg, pp, mean
    n_tokens: int = 128
    no_warmup: bool = False
    override_mode: str = "scan"  # none, scan, custom
    custom_override: Optional[str] = None
    # MoE specific options
    is_moe: bool = False
    moe_experts: int = 0  # Total number of experts in the model
    moe_ncmoe: int | None = None  # MoE: number of layers whose experts stay on CPU
    extra_args: list[str] = field(default_factory=list)
    
    def to_cli_args(self) -> list[str]:
        """Convert config to llama-optimus CLI arguments."""
        args = [
            "--llama-bin", self.llama_bin_dir,
            "--model", self.model_path,
            "--trials", str(self.trials),
            "-r", str(self.repeat),
            "--metric", self.metric,
            "--n-tokens", str(self.n_tokens),
        ]
        
        # For MoE models, set ngl-max to 99 (all non-expert layers on GPU)
        if self.is_moe:
            args.extend(["--ngl-max", "99"])
        
        if self.no_warmup:
            args.append("--no-warmup")
        
        if self.override_mode == "none":
            args.append("--no-override-scan")
        elif self.override_mode == "custom" and self.custom_override:
            args.extend(["--override-key", self.custom_override])
        # "scan" is the default
        
        args.extend(self.extra_args)
        return args


@dataclass
class TrialResult:
    """Result of a single optimization trial."""
    trial_number: int
    stage: int
    params: dict[str, Any]
    value: float
    best_so_far: bool = False


@dataclass  
class StageResult:
    """Result of an optimization stage."""
    stage: int
    stage_name: str
    best_params: dict[str, Any]
    best_value: float
    total_trials: int


@dataclass
class OptimusResult:
    """Complete result of llama-optimus run."""
    success: bool
    stages: list[StageResult] = field(default_factory=list)
    final_config: dict[str, Any] = field(default_factory=dict)
    llama_server_cmd: str = ""
    llama_bench_cmd: str = ""
    optimized_results: dict[str, float] = field(default_factory=dict)
    baseline_results: dict[str, float] = field(default_factory=dict)
    speedup: dict[str, float] = field(default_factory=dict)
    ctx_size_tokens: int | None = None
    gpu_info: str = ""
    error_message: str = ""
    raw_output: str = ""
    ncmoe: int | None = None  # Added to track ncmoe used during optimization


@dataclass
class MoeSweepResult:
    """Structured result for the MoE ncmoe sweep phase."""

    ok: bool
    stopped: bool
    best_ncmoe: int | None
    best_score: float | None
    results: list[tuple[int, float]] = field(default_factory=list)


def _parse_dict_str(dict_content: str) -> dict[str, str]:
    """Parse a Python dict-like string: 'batch': 4465, 'u_batch': 7758, ..."""
    result = {}
    # Match patterns like 'key': value or 'key': 'value'
    for match in re.finditer(r"'(\w+)':\s*(?:'([^']+)'|(\d+))", dict_content):
        key = match.group(1)
        value = match.group(2) if match.group(2) else match.group(3)
        result[key] = value
    return result


def parse_optimus_output(output: str) -> OptimusResult:
    """Parse llama-optimus output into structured result."""
    result = OptimusResult(success=True, raw_output=output)
    metrics_map: dict[str, float] = {}
    accumulated_params = {}
    
    # Check for Python errors (Traceback)
    if "Traceback (most recent call last):" in output:
        result.success = False
        # Extract the error message
        error_match = re.search(r"(RuntimeError|Error|Exception):\s*(.+?)(?:\n\n|$)", output, re.DOTALL)
        if error_match:
            result.error_message = f"{error_match.group(1)}: {error_match.group(2).strip()}"
        else:
            # Try to get the last line of the traceback
            lines = output.strip().split('\n')
            for line in reversed(lines):
                if line.strip() and not line.startswith(' '):
                    result.error_message = line.strip()
                    break
        # Still try to extract GPU info even on error
        cuda_gpu_match = re.search(r"Device \d+:\s*([^,]+)", output)
        if cuda_gpu_match:
            result.gpu_info = cuda_gpu_match.group(1).strip()
        return result
    
    # Check for llama-bench errors
    if "error: failed to load model" in output.lower():
        result.success = False
        result.error_message = "Failed to load model - model may be too large for available memory"
        # Extract GPU info
        cuda_gpu_match = re.search(r"Device \d+:\s*([^,]+)", output)
        if cuda_gpu_match:
            result.gpu_info = cuda_gpu_match.group(1).strip()
        return result
    
    if "CUDA out of memory" in output or "out of memory" in output.lower():
        result.success = False
        result.error_message = "CUDA out of memory - try a smaller model or reduce batch size"
        return result
    
    # Check for ngl = 0 (model too large for GPU)
    if "Estimated max ngl = 0" in output or "max ngl = 0" in output.lower():
        # This might still succeed with CPU-only, but warn
        pass
    
    # Extract GPU info
    gpu_match = re.search(r"Detected GPU:\s*(.+)", output)
    if gpu_match:
        result.gpu_info = gpu_match.group(1).strip()
    
    # Also try to extract from CUDA init output
    if not result.gpu_info:
        cuda_gpu_match = re.search(r"Device \d+:\s*([^,]+)", output)
        if cuda_gpu_match:
            result.gpu_info = cuda_gpu_match.group(1).strip()
    
    # Extract ncmoe if present in the command log
    ncmoe_match = re.search(r"LLAMA_BENCH_NCMOE=(\d+)", output)
    if ncmoe_match:
        result.ncmoe = int(ncmoe_match.group(1))
        accumulated_params['ncmoe'] = result.ncmoe

    # Parse stage results using actual llama-optimus output format
    # Format: Best config Stage_1: {'batch': 4465, 'u_batch': 7758, 'threads': 6, 'gpu_layers': 66}
    # Format: Best Stage_1 tg tokens/sec: 159.227554
    
    # Stage 1: Initial exploration
    stage1_config_match = re.search(
        r"Best config Stage_1:\s*\{([^}]+)\}",
        output
    )
    stage1_value_match = re.search(
        r"Best Stage_1 (?:tg|pp|mean) tokens/sec:\s*([\d.]+)",
        output
    )
    if stage1_config_match:
        params = _parse_dict_str(stage1_config_match.group(1))
        # Normalize param names
        normalized = {}
        if 'batch' in params: normalized['batch'] = int(params['batch'])
        if 'u_batch' in params: normalized['ubatch'] = int(params['u_batch'])
        if 'threads' in params: normalized['threads'] = int(params['threads'])
        if 'gpu_layers' in params: normalized['ngl'] = int(params['gpu_layers'])
        
        accumulated_params.update(normalized)
        result.stages.append(StageResult(
            stage=1,
            stage_name="Initial Exploration",
            best_params=normalized,
            best_value=float(stage1_value_match.group(1)) if stage1_value_match else 0.0,
            total_trials=0
        ))
    
    # Stage 2: Grid search categorical
    stage2_config_match = re.search(
        r"Best config Stage_2:\s*\{([^}]+)\}",
        output
    )
    stage2_value_match = re.search(
        r"Best Stage_2 (?:tg|pp|mean) tokens/sec:\s*([\d.]+)",
        output
    )
    if stage2_config_match:
        params = _parse_dict_str(stage2_config_match.group(1))
        normalized = {}
        if 'flash_attn' in params: normalized['flash_attn'] = int(params['flash_attn'])
        if 'override_tensor' in params: normalized['override_tensor'] = params['override_tensor'].strip("'\"")
        
        accumulated_params.update(normalized)
        result.stages.append(StageResult(
            stage=2,
            stage_name="Grid Search (Categorical)",
            best_params=normalized,
            best_value=float(stage2_value_match.group(1)) if stage2_value_match else 0.0,
            total_trials=0
        ))
    
    # Stage 3: Finetune
    stage3_config_match = re.search(
        r"Best config Stage_3:\s*\{([^}]+)\}",
        output
    )
    stage3_value_match = re.search(
        r"Best Stage_3 (?:tg|pp|mean) tokens/sec:\s*([\d.]+)",
        output
    )
    if stage3_config_match:
        params = _parse_dict_str(stage3_config_match.group(1))
        normalized = {}
        if 'batch' in params: normalized['batch'] = int(params['batch'])
        if 'u_batch' in params: normalized['ubatch'] = int(params['u_batch'])
        if 'threads' in params: normalized['threads'] = int(params['threads'])
        if 'gpu_layers' in params: normalized['ngl'] = int(params['gpu_layers'])
        if 'flash_attn' in params: normalized['flash_attn'] = int(params['flash_attn'])
        if 'override_tensor' in params: normalized['override_tensor'] = params['override_tensor'].strip("'\"")
        
        accumulated_params.update(normalized)
        result.stages.append(StageResult(
            stage=3,
            stage_name="Fine-tuning",
            best_params=normalized,
            best_value=float(stage3_value_match.group(1)) if stage3_value_match else 0.0,
            total_trials=0
        ))
    
    # Extract final commands - look for the actual command line printed by llama-optimus
    # Format:  $LLAMA_BIN/llama-server --model $MODEL -t 4 --batch-size 3584 --ubatch-size 896 -ngl 43 --flash-attn
    server_cmd_match = re.search(
        r"\s+\$LLAMA_BIN/llama-server\s+--model\s+\$MODEL\s+([^\n]+)",
        output
    )
    if server_cmd_match:
        # Replace $LLAMA_BIN and $MODEL with placeholders for the UI
        args = server_cmd_match.group(1).strip()
        result.llama_server_cmd = f"llama-server --model <MODEL> {args}"
        # If we have ncmoe, inject it if not present
        if result.ncmoe is not None and "-ncmoe" not in result.llama_server_cmd:
            result.llama_server_cmd += f" -ncmoe {result.ncmoe}"
    
    # Fallback: Build the actual command from accumulated params if extraction failed
    if not result.llama_server_cmd and accumulated_params:
        final = accumulated_params
        cmd_parts = ["llama-server --model <MODEL>"]
        if 'threads' in final: cmd_parts.append(f"-t {final['threads']}")
        if 'batch' in final: cmd_parts.append(f"--batch-size {final['batch']}")
        if 'ubatch' in final: cmd_parts.append(f"--ubatch-size {final['ubatch']}")
        if 'ngl' in final: cmd_parts.append(f"-ngl {final['ngl']}")
        if final.get('flash_attn'): cmd_parts.append("--flash-attn")
        if final.get('override_tensor') and final['override_tensor'] != "none":
            # We don't have the pattern mapping here, so we use the key as a placeholder
            cmd_parts.append(f"--override-tensor \"<{final['override_tensor']}>\"")
        if final.get('ncmoe') is not None:
            cmd_parts.append(f"-ncmoe {final['ncmoe']}")
        result.llama_server_cmd = ' '.join(cmd_parts)
    
    # Extract llama-bench command
    bench_match = re.search(
        r"(/app/llama-bench --model [^\n]+--progress)",
        output
    )
    if bench_match:
        result.llama_bench_cmd = bench_match.group(1).strip()
        # Inject ncmoe if present
        if result.ncmoe is not None and "-ncmoe" not in result.llama_bench_cmd:
            result.llama_bench_cmd = result.llama_bench_cmd.replace("--model", f"--model <MODEL> -ncmoe {result.ncmoe} --model")
            # Clean up double model if it happened
            result.llama_bench_cmd = result.llama_bench_cmd.replace("--model <MODEL> --model", "--model <MODEL>")
    
    # Also try simpler pattern
    if not result.llama_bench_cmd:
        bench_match2 = re.search(
            r"(/app/llama-bench --model /models/[^\n]+)",
            output
        )
        if bench_match2:
            result.llama_bench_cmd = bench_match2.group(1).strip()
    
    # Extract benchmark results from final benchmark output
    # Format in table: | mistral3 3B Q8_0 | ... | pp256 | 11017.77 ± 3638.27 |
    # Or: | mistral3 3B Q8_0 | ... | tg128 | 172.85 ± 1.03 |
    # Collect all tg/pp rows to avoid mixing metrics (supports multiple ctx)
    for match in re.finditer(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)\s*±", output):
        label = match.group(1)
        val = float(match.group(2))
        metrics_map[label] = val
    # Keep best/final seen values for pp/tg
    for k, v in metrics_map.items():
        if k.startswith("pp"):
            result.optimized_results["pp"] = v
        elif k.startswith("tg"):
            result.optimized_results["tg"] = v
    # If both present, provide a "mean" helper (average of tg/pp)
    if "pp" in result.optimized_results and "tg" in result.optimized_results:
        try:
            result.optimized_results["mean"] = (
                float(result.optimized_results["pp"]) + float(result.optimized_results["tg"])
            ) / 2.0
        except Exception:
            pass
    
    # Final config from accumulated params
    result.final_config = accumulated_params
    
    return result


def run_optimus_streaming(
    config: OptimusConfig,
    container_name: str = "llama-cpp"
) -> Generator[str, None, OptimusResult]:
    """
    Run llama-optimus and yield output lines as they arrive.
    Returns the final parsed result.
    """
    cmd = [
        "docker", "compose", "exec", "-T", container_name,
        "llama-optimus"
    ] + config.to_cli_args()
    
    output_lines = []
    workspace_root = Path(__file__).parent.parent.parent
    
    # Log the command being run
    cmd_str = ' '.join(cmd)
    output_lines.append(f"[CMD] {cmd_str}\n")
    yield f"[CMD] {cmd_str}\n"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(workspace_root)
    )
    
    try:
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            yield line
        
        process.wait()
        full_output = ''.join(output_lines)
        
        # Always parse to detect errors
        result = parse_optimus_output(full_output)
        
        if process.returncode != 0 and result.success:
            result.success = False
            result.error_message = f"Process exited with code {process.returncode}"
        
        return result
        
    except Exception as e:
        return OptimusResult(
            success=False,
            error_message=str(e),
            raw_output=''.join(output_lines)
        )


def run_optimus_blocking(
    config: OptimusConfig,
    container_name: str = "llama-cpp"
) -> OptimusResult:
    """Run llama-optimus and wait for completion."""
    cmd = [
        "docker", "compose", "exec", "-T", container_name,
        "llama-optimus"
    ] + config.to_cli_args()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        # Combine stdout and stderr, with stderr clearly marked
        full_output = result.stdout
        if result.stderr:
            full_output += "\n[STDERR]\n" + result.stderr
        
        # Parse the output (will detect errors automatically)
        parsed = parse_optimus_output(full_output)
        
        # If returncode is non-zero but parse didn't catch an error, set it
        if result.returncode != 0 and parsed.success:
            parsed.success = False
            parsed.error_message = f"Process exited with code {result.returncode}"
        
        return parsed
        
    except Exception as e:
        return OptimusResult(
            success=False,
            error_message=str(e)
        )


def get_container_model_path(model_path: str | Path, paths: RepoPaths | None = None) -> str:
    """Get the model path inside the container, handling external dirs."""
    p = Path(model_path)
    if paths is None:
        paths = RepoPaths.detect()
    if not p.is_absolute():
        p = paths.models_dir / p
    return host_path_to_container_path(p, paths)


def estimate_optimus_duration(trials: int, repeat: int, is_moe: bool = False) -> str:
    """Estimate duration of optimus run."""
    # Rough estimate: ~10-30 seconds per trial depending on model size
    # MoE ncmoe scan adds extra time
    base_trials = trials
    if is_moe:
        base_trials = trials * 1.5  # Extra time for ncmoe sweep
    
    min_seconds = int(base_trials * repeat * 10)
    max_seconds = int(base_trials * repeat * 30)
    
    def format_time(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    
    return f"{format_time(min_seconds)} - {format_time(max_seconds)}"


def run_moe_ncmoe_sweep(
    model_path: str,
    max_layers: int,
    metric: str = "tg",
    n_tokens: int = 128,
    container_name: str = "llama-cpp",
    step: int | None = None,
    stop_event: Any | None = None,
) -> Generator[str, None, MoeSweepResult]:
    """
    Sweep ncmoe values to find optimal setting for MoE models.
    ngl is always set to max (99) because non-expert layers must stay on GPU.
    
    IMPORTANT: ncmoe = number of LAYERS whose MoE experts stay on CPU (NOT experts!)
    - ncmoe=max_layers -> all MoE layers on CPU (slow but safe)
    - ncmoe=0 -> all MoE layers on GPU (fast but may OOM)
    
    Args:
        model_path: Path to model inside container
        max_layers: Total number of layers (blocks) in the model
        metric: Optimization metric (tg, pp, mean)
        n_tokens: Tokens per benchmark
        container_name: Docker container name
        step: Step size for ncmoe sweep. If None, defaults to max_layers // 8
    
    Yields output lines, returns (best_ncmoe, best_score).
    """
    workspace_root = Path(__file__).parent.parent.parent
    
    # Test values: 0, then increments based on step
    if step is None:
        step = max(1, max_layers // 8)
    step = max(1, step)  # Ensure at least 1
    
    test_values = list(range(0, max_layers + 1, step))
    if max_layers not in test_values:
        test_values.append(max_layers)
    
    best_ncmoe: int | None = None
    best_score: float | None = None
    results: list[tuple[int, float]] = []
    
    yield f"[MoE] Sweeping ncmoe (layers) from 0 to {max_layers} (step={step})\n"
    yield f"[MoE] ngl=99 (all non-expert layers on GPU)\n"
    yield f"[MoE] Testing values: {test_values}\n"
    yield f"[MoE] Total tests: {len(test_values)}\n\n"
    
    for i, ncmoe in enumerate(test_values):
        if stop_event and stop_event.is_set():
            yield "[MoE] Sweep stopped by user\n"
            return MoeSweepResult(
                ok=bool(results),
                stopped=True,
                best_ncmoe=best_ncmoe,
                best_score=best_score,
                results=results,
            )
        yield f"[MoE] [{i+1}/{len(test_values)}] Testing ncmoe={ncmoe}...\n"
        
        cmd = [
            "docker", "compose", "exec", "-T", container_name,
            "/app/llama-bench",
            "-m", model_path,
            "-ngl", "99",
            "-ncmoe", str(ncmoe),
            "-n", str(n_tokens),
            "-p", "512",
            "-r", "1",  # Only 1 repeat for faster sweep
            "-o", "csv"
        ]
        
        try:
            # Use Popen for streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(workspace_root)
            )
            if stop_event and stop_event.is_set():
                process.terminate()
                yield "[MoE]   stopped before run\n"
                return MoeSweepResult(
                    ok=bool(results),
                    stopped=True,
                    best_ncmoe=best_ncmoe,
                    best_score=best_score,
                    results=results,
                )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if stop_event and stop_event.is_set():
                    process.terminate()
                    yield "[MoE]   stopped during run\n"
                    return MoeSweepResult(
                        ok=bool(results),
                        stopped=True,
                        best_ncmoe=best_ncmoe,
                        best_score=best_score,
                        results=results,
                    )
                output_lines.append(line)
                # Yield progress for long-running benchmarks
                if "model size" in line.lower() or "ggml" in line.lower():
                    yield f"[MoE]   {line.strip()}\n"
            
            process.wait(timeout=300)
            output = ''.join(output_lines)
            
            # Check for OOM or load failure
            if "error: failed to load model" in output.lower() or "out of memory" in output.lower():
                yield f"[MoE]   ❌ ncmoe={ncmoe}: OOM or load failed, skipping\n"
                continue
            
            # Parse CSV output for metric (robust to quoted fields with commas)
            # We always run with `-p 512 -n <n_tokens>` so we expect both `pp512` and `tg<n_tokens>` rows.
            score = 0.0
            try:
                rows = parse_llama_bench_csv(output)
                metrics_map = extract_metrics(rows)
                prompt_key = "pp512"
                tg_key = f"tg{int(n_tokens)}"

                def _best_by_prefix(prefix: str) -> float | None:
                    vals = [v for k, v in metrics_map.items() if str(k).startswith(prefix)]
                    return max(vals) if vals else None

                if metric == "tg":
                    score = float(metrics_map.get(tg_key) or (_best_by_prefix("tg") or 0.0))
                elif metric == "pp":
                    score = float(metrics_map.get(prompt_key) or (_best_by_prefix("pp") or 0.0))
                elif metric in ("mean", "both"):
                    tg_v = metrics_map.get(tg_key) or _best_by_prefix("tg")
                    pp_v = metrics_map.get(prompt_key) or _best_by_prefix("pp")
                    vals = [v for v in (tg_v, pp_v) if isinstance(v, (int, float)) and float(v) > 0.0]
                    score = float(sum(vals) / len(vals)) if vals else 0.0
                else:
                    # Unknown metric -> fall back to best available
                    score = float(max(metrics_map.values()) if metrics_map else 0.0)
            except Exception as e:
                yield f"[MoE]   ⚠️ ncmoe={ncmoe}: CSV parse failed ({type(e).__name__}: {e})\n"
                if output.strip():
                    yield f"[MoE]      Last line: {output.strip().split(chr(10))[-1][:120]}\n"
                continue
            
            if score > 0:
                results.append((ncmoe, score))
                is_new_best = best_score is None or float(score) > float(best_score)
                marker = "🏆" if is_new_best else "  "
                yield f"[MoE]   {marker} ncmoe={ncmoe}: {score:.2f} t/s\n"
                
                if is_new_best:
                    best_score = float(score)
                    best_ncmoe = int(ncmoe)
            else:
                yield f"[MoE]   ⚠️ ncmoe={ncmoe}: could not parse score from output\n"
                # Show some output for debugging
                if output.strip():
                    yield f"[MoE]      Last line: {output.strip().split(chr(10))[-1][:80]}\n"
                
        except subprocess.TimeoutExpired:
            yield f"[MoE]   ncmoe={ncmoe}: timeout\n"
        except Exception as e:
            yield f"[MoE]   ncmoe={ncmoe}: error - {str(e)}\n"
    
    if not results:
        yield "\n[MoE] ❌ No valid results (all tests failed)\n\n"
        return MoeSweepResult(
            ok=False,
            stopped=False,
            best_ncmoe=None,
            best_score=None,
            results=[],
        )

    # By construction, results is non-empty -> best must exist.
    yield f"\n[MoE] ✅ Best ncmoe={best_ncmoe} with {float(best_score or 0.0):.2f} t/s\n"
    yield f"[MoE] Results: {results}\n\n"

    return MoeSweepResult(
        ok=True,
        stopped=False,
        best_ncmoe=best_ncmoe,
        best_score=best_score,
        results=results,
    )
