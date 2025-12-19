from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import threading
import queue
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception as _e:
    optuna = None
    TPESampler = None
    _OPTUNA_IMPORT_ERROR = _e
else:
    _OPTUNA_IMPORT_ERROR = None

from .bench import parse_llama_bench_csv, extract_metrics, format_llama_server_command
from .paths import RepoPaths, host_path_to_container_path
from .settings import AppSettings
from .docker import compose_base
from .error_detection import detect_error, ErrorType

IGNORE_PATTERNS = [
    "ggml_",
    "llama_model_loader",
    "load_backend:",
    "compiling shaders",
    "warning:",
    "info:",
]

REQUIRED_COLUMNS = [
    "avg_ts",
    "stddev_ts",
    "n_prompt",
    "n_gen",
    "n_gpu_layers",
    "n_threads",
    "n_batch",
    "n_ubatch",
]


@dataclass
class TrialResult:
    cfg: dict[str, Any]
    metrics: dict[str, float]
    score: float
    status: str
    error: str | None = None
    raw_output: str = ""
    ctx_tokens: int | None = None


@dataclass
class HyperOptConfig:
    model_path_host: Path
    is_moe: bool
    # Model limits from GGUF metadata (for dynamic bounding)
    model_max_ctx: int = 32768  # From GGUF context_length
    model_expert_count: int = 0  # From GGUF expert_count (0 = not MoE)
    # Optimization ranges - defaults to model max, Optuna explores down from top
    ctx_range: tuple[int, int, int] = (2048, 32768, 1024)  # (min, max, step)
    # Balanced weights: 33% TG, 33% PP, 33% CTX for equilibrium
    weight_tg: float = 0.333
    weight_pp: float = 0.333
    weight_ctx: float = 0.334
    trials: int = 100
    repeats: int = 2
    n_tokens: int = 128
    batch_range: tuple[int, int, int] = (512, 16384, 512)  # Aggressive default
    ubatch_range: tuple[int, int, int] = (256, 8192, 256)  # Aggressive default
    threads_range: tuple[int, int, int] = (8, 8, 1)
    ncmoe_values: list[int] = field(default_factory=lambda: [0])
    flash_values: list[int] = field(default_factory=lambda: [1])
    nkvo_values: list[int] = field(default_factory=lambda: [1])
    ctk: str = "f16"
    ctv: str = "f16"
    timeout_sec: int = 120  # Shorter timeout - crashes are fast, no need to wait long


def _ensure_run_dir(paths: RepoPaths) -> Path:
    run_id = time.strftime("hyperoptimus_%Y%m%d_%H%M%S")
    d = paths.runs_dir / "hyperoptimus" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "trials").mkdir(exist_ok=True)
    return d


def _clean_lines(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        low = ln.lower().strip()
        if any(low.startswith(pat) for pat in IGNORE_PATTERNS):
            continue
        lines.append(ln)
    return "\n".join(lines)


def _parse_llama_bench_stdout(stdout_text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Robust parsing: try standard parser, fallback to manual CSV.
    Returns (metrics_map, rows)
    """
    stdout_text = stdout_text or ""
    # Try the robust parser from bench.py first
    rows: list[dict[str, Any]] = []
    metrics_map: dict[str, float] = {}
    try:
        rows = parse_llama_bench_csv(stdout_text)
        metrics_map = extract_metrics(rows)
    except Exception:
        rows = []
        metrics_map = {}

    if not metrics_map:
        # Fallback manual CSV parse
        try:
            cleaned = _clean_lines(stdout_text)
            reader = csv.DictReader(io.StringIO(cleaned))
            for r in reader:
                if not r:
                    continue
                rows.append(dict(r))
        except Exception:
            rows = []

        # Validate columns
        valid_rows: list[dict[str, Any]] = []
        for r in rows:
            ok = all(k in r for k in REQUIRED_COLUMNS)
            if not ok:
                continue
            try:
                avg_ts = float(r.get("avg_ts") or 0.0)
                std_ts = float(r.get("stddev_ts") or 0.0)
                if math.isinf(avg_ts) or math.isnan(avg_ts):
                    continue
                if math.isinf(std_ts) or math.isnan(std_ts):
                    r["stddev_ts"] = 0.0
                valid_rows.append(r)
            except Exception:
                continue
        rows = valid_rows
        metrics_map = extract_metrics(valid_rows) if valid_rows else {}

    # Normalise metrics so we always expose 'pp' and 'tg' keys for scoring
    if rows:
        pp_candidates: list[float] = []
        tg_candidates: list[float] = []
        generic: list[float] = []
        for r in rows:
            try:
                avg_ts = float(r.get("avg_ts") or 0.0)
            except Exception:
                continue
            try:
                n_prompt = int(float(r.get("n_prompt") or 0))
            except Exception:
                n_prompt = 0
            try:
                n_gen = int(float(r.get("n_gen") or 0))
            except Exception:
                n_gen = 0

            # llm-bench often mixes prompt and generation in one row; count both
            if n_prompt > 0:
                pp_candidates.append(avg_ts)
            if n_gen > 0:
                tg_candidates.append(avg_ts)
            generic.append(avg_ts)

        # Only inject missing keys to avoid overriding good labels
        if "pp" not in metrics_map:
            metrics_map["pp"] = max(pp_candidates or generic or [0.0])
        if "tg" not in metrics_map:
            metrics_map["tg"] = max(tg_candidates or generic or [0.0])

    return metrics_map, rows


def _score(metrics_map: dict[str, float], ctx_tokens: int, weights: dict[str, float]) -> float:
    """Calculate weighted score from metrics.
    
    Balanced scoring with logarithmic context bonus:
    - TG (text generation speed) - normalized to ~100 scale
    - PP (prompt processing speed) - normalized by context to not penalize large ctx
    - CTX bonus - logarithmic scale to reward larger contexts fairly
    
    The key insight: PP naturally decreases with larger context, so we normalize it.
    A config with ctx=32K and PP=1000 t/s is actually BETTER than ctx=2K with PP=4000 t/s
    because PP/ctx ratio matters more than raw PP.
    """
    import math
    
    tg = metrics_map.get("tg") or metrics_map.get("tg128") or 0.0
    pp = metrics_map.get("pp") or metrics_map.get("pp512") or 0.0
    
    if any([tg <= 0, pp <= 0, math.isnan(tg), math.isnan(pp), math.isinf(tg), math.isinf(pp)]):
        return 0.0
    
    if not ctx_tokens or ctx_tokens <= 0:
        ctx_tokens = 2048
    
    # Normalize TG: typical range 50-150 t/s → scale to ~100
    tg_score = float(tg)
    
    # Normalize PP by context: PP efficiency = PP * ctx^0.4 / 1000
    # Using 0.4 instead of 0.5 (sqrt) to leave more room for batch/ubatch influence
    # ctx=2K, PP=4000 → 4000 * 23.5 / 1000 = 94
    # ctx=13K, PP=2000 → 2000 * 47.5 / 1000 = 95 (slightly better)
    pp_efficiency = float(pp) * math.pow(ctx_tokens, 0.4) / 1000.0
    
    # Context bonus: logarithmic scale
    # ctx=2K → log2(2) = 1, ctx=8K → 3, ctx=32K → 5, ctx=128K → 7
    ctx_bonus = math.log2(ctx_tokens / 1000.0) if ctx_tokens >= 1000 else 0.0
    ctx_score = ctx_bonus * 20  # Scale to be meaningful (~20-140 range)
    
    # Final weighted score
    w_tg = weights.get("tg", 0.333)
    w_pp = weights.get("pp", 0.333)
    w_ctx = weights.get("ctx", 0.334)
    
    score = w_tg * tg_score + w_pp * pp_efficiency + w_ctx * ctx_score
    
    return score


def _generate_range(start: int, end: int, step: int, *, max_values: int = 256) -> list[int]:
    if step <= 0:
        step = 1
    vals: list[int] = []
    if start <= end:
        cur = start
        while cur <= end and len(vals) < max_values:
            vals.append(int(cur))
            cur += step
    else:
        cur = start
        while cur >= end and len(vals) < max_values:
            vals.append(int(cur))
            cur -= step
    return vals or [start]


def _generate_candidates(cfg: HyperOptConfig) -> list[dict[str, Any]]:
    batches = _generate_range(*cfg.batch_range)
    ubatches = _generate_range(*cfg.ubatch_range)
    threads = _generate_range(*cfg.threads_range)
    ncmoe_vals = cfg.ncmoe_values if cfg.is_moe else [0]
    candidates: list[dict[str, Any]] = []
    for b in batches:
        for ub in ubatches:
            if ub > b:
                continue
            for t in threads:
                for fa in cfg.flash_values:
                    for nk in cfg.nkvo_values:
                        for nc in ncmoe_vals:
                            candidates.append(
                                {
                                    "batch": int(b),
                                    "ubatch": int(ub),
                                    "threads": int(t),
                                    "flash_attn": int(fa),
                                    "no_kv_offload": int(nk),
                                    "ncmoe": int(nc),
                                    "ngl": 99,
                                    "ctk": cfg.ctk,
                                    "ctv": cfg.ctv,
                                }
                            )
    return candidates


def _fallback_chain(cfg: dict[str, Any], is_moe: bool) -> list[dict[str, Any]]:
    fallbacks: list[dict[str, Any]] = []
    def clone(**kw):
        nc = dict(cfg)
        nc.update(kw)
        return nc
    b = int(cfg.get("batch", 1024))
    ub = int(cfg.get("ubatch", 512))
    ngl = int(cfg.get("ngl", 99))
    ncmoe = int(cfg.get("ncmoe", 0))
    fallbacks.append(clone(batch=max(256, b // 2)))
    fallbacks.append(clone(ubatch=max(64, ub // 2)))
    fallbacks.append(clone(ngl=max(0, ngl - 10)))
    if is_moe:
        fallbacks.append(clone(ncmoe=ncmoe + 5))
    fallbacks.append(clone(flash_attn=0))
    fallbacks.append(clone(batch=max(128, b // 2), ubatch=max(64, ub // 2), ngl=0, ncmoe=0))
    return fallbacks


def _build_llama_bench_cmd(cfg: dict[str, Any], *, model_container: str, n_tokens: int, repeats: int, ctx_tokens: int) -> list[str]:
    cmd = [
        "/app/llama-bench",
        "--model",
        model_container,
        "-t",
        str(int(cfg.get("threads", 1))),
        "--batch-size",
        str(int(cfg.get("batch", 1024))),
        "--ubatch-size",
        str(int(cfg.get("ubatch", 512))),
        "-ngl",
        str(int(cfg.get("ngl", 99))),
        "-n",
        str(int(max(1, n_tokens))),
        "-p",
        str(int(max(1, ctx_tokens))),
        "-r",
        str(int(max(1, repeats))),
        "-o",
        "csv",
        "--no-warmup",
        "-ctk",
        str(cfg.get("ctk", "q8_0")),
        "-ctv",
        str(cfg.get("ctv", "q8_0")),
    ]
    if cfg.get("flash_attn"):
        cmd.append("--flash-attn")
    if cfg.get("no_kv_offload"):
        cmd.append("--no-kv-offload")
    if cfg.get("ncmoe") is not None:
        cmd += ["-ncmoe", str(int(cfg["ncmoe"]))]
    return cmd


def run_trial(
    paths: RepoPaths,
    settings: AppSettings,
    *,
    model_path_host: Path,
    trial_cfg: dict[str, Any],
    ctx_tokens: int,
    n_tokens: int,
    repeats: int,
    timeout_sec: int,
    log_fn=None,
) -> TrialResult:
    model_container = host_path_to_container_path(model_path_host, paths)
    cmd = compose_base(paths, settings) + ["exec", "-T", settings.docker_service, "sh", "-c"]
    llama_cmd = _build_llama_bench_cmd(trial_cfg, model_container=model_container, n_tokens=n_tokens, repeats=repeats, ctx_tokens=ctx_tokens)
    shell_cmd = " ".join([str(part) for part in llama_cmd])
    cmd.append(shell_cmd)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=paths.repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        return TrialResult(cfg=trial_cfg, metrics={}, score=0.0, status="timeout", error=str(e), raw_output="", ctx_tokens=ctx_tokens)
    duration = time.time() - start
    raw = (proc.stdout or "") + ("\n[STDERR]\n" + (proc.stderr or ""))

    # Cleanup: kill any remaining llama processes to free GPU memory
    try:
        cleanup_cmd = compose_base(paths, settings) + ["exec", "-T", settings.docker_service, "pkill", "-9", "-f", "llama"]
        subprocess.run(cleanup_cmd, capture_output=True, timeout=5)
    except Exception:
        pass  # Ignore cleanup errors

    # Error detection using the shared module
    error_info = detect_error(proc.stdout or "", proc.stderr or "", proc.returncode)
    
    if error_info.error_type in (ErrorType.OOM_CUDA, ErrorType.OOM_SYSTEM):
        if log_fn:
            log_fn(f"⚠️ OOM detected (ctx={ctx_tokens}): {error_info.message}")
        return TrialResult(cfg=trial_cfg, metrics={}, score=0.0, status="oom", error=error_info.message, raw_output=raw, ctx_tokens=ctx_tokens)
    
    if error_info.error_type == ErrorType.LOAD_FAILED:
        if log_fn:
            log_fn(f"⚠️ Load failed (ctx={ctx_tokens}): {error_info.message}")
        return TrialResult(cfg=trial_cfg, metrics={}, score=0.0, status="failed", error=error_info.message, raw_output=raw, ctx_tokens=ctx_tokens)

    # Check for context creation failure (usually means context too large for VRAM)
    combined_output = (proc.stdout or "") + (proc.stderr or "")
    if "failed to create context" in combined_output.lower():
        err_msg = "Context creation failed - likely context too large for available VRAM"
        if log_fn:
            log_fn(f"⚠️ {err_msg} (ctx={ctx_tokens})")
        return TrialResult(cfg=trial_cfg, metrics={}, score=0.0, status="context_failed", error=err_msg, raw_output=raw, ctx_tokens=ctx_tokens)

    # Parse
    metrics_map, rows = _parse_llama_bench_stdout(raw)
    if not metrics_map:
        if log_fn:
            snippet = raw.splitlines()[-5:] if raw else []
            log_fn(f"⚠️ Parse error (ctx={ctx_tokens}) no metrics. Tail: {' | '.join(snippet)}")
        return TrialResult(cfg=trial_cfg, metrics={}, score=0.0, status="parse_error", error="No metrics parsed", raw_output=raw, ctx_tokens=ctx_tokens)

    score = _score(metrics_map, ctx_tokens=ctx_tokens, weights={"tg": trial_cfg.get("w_tg", 0.5), "pp": trial_cfg.get("w_pp", 0.3), "ctx": trial_cfg.get("w_ctx", 0.2)})
    status = "ok" if proc.returncode == 0 else "failed"
    return TrialResult(cfg=trial_cfg, metrics=metrics_map, score=score, status=status, error=None if status == "ok" else f"exit {proc.returncode}", raw_output=raw, ctx_tokens=ctx_tokens)


def best_of_trials(results: Iterable[TrialResult]) -> TrialResult | None:
    best: TrialResult | None = None
    for r in results:
        if r.status != "ok":
            continue
        if best is None or r.score > best.score:
            best = r
    return best


def run_optuna(
    cfg: HyperOptConfig,
    paths: RepoPaths,
    settings: AppSettings,
    stop_event=None,
    log_fn=None,
    progress_callback=None,
) -> tuple[list[TrialResult], TrialResult | None, Path]:
    if optuna is None or TPESampler is None:
        raise RuntimeError(f"Optuna non disponible : installe `optuna` dans l'environnement (pip install optuna). Détail: {_OPTUNA_IMPORT_ERROR}")
    run_dir = _ensure_run_dir(paths)
    log_path = run_dir / "hyperoptimus.log"
    (run_dir / "trials").mkdir(exist_ok=True)

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if log_fn:
            try:
                log_fn(line)
            except Exception:
                pass

    model_host = cfg.model_path_host
    log(f"🚀 Start HyperOptimus | trials={cfg.trials} | model={model_host}")
    log(f"📊 Model limits: max_ctx={cfg.model_max_ctx:,} experts={cfg.model_expert_count}")
    log(f"🎯 ctx_range={cfg.ctx_range} batch_range={cfg.batch_range} ubatch_range={cfg.ubatch_range}")
    log(f"⚖️ weights tg={cfg.weight_tg} pp={cfg.weight_pp} ctx={cfg.weight_ctx}")
    log(f"⏱️ timeout={cfg.timeout_sec}s (fast crash detection)")

    best_score_overall: float = 0.0
    best_result: TrialResult | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score_overall, best_result
        if stop_event and stop_event.is_set():
            raise optuna.TrialPruned("Stopped")
        
        # Report progress
        if progress_callback:
            progress_callback({
                "type": "progress",
                "trial": trial.number + 1,
                "total": cfg.trials,
                "best_score": best_score_overall
            })

        # ALL parameters are optimized by Optuna, including context!
        ctx = trial.suggest_int("ctx", cfg.ctx_range[0], cfg.ctx_range[1], step=cfg.ctx_range[2])
        batch = trial.suggest_int("batch", cfg.batch_range[0], cfg.batch_range[1], step=cfg.batch_range[2])
        # ubatch must be <= batch
        ub_max = min(batch, cfg.ubatch_range[1])
        ub_min = min(cfg.ubatch_range[0], ub_max)
        ubatch = trial.suggest_int("ubatch", ub_min, ub_max, step=cfg.ubatch_range[2])
        threads = trial.suggest_int("threads", cfg.threads_range[0], cfg.threads_range[1], step=max(1, cfg.threads_range[2]))
        flash = trial.suggest_categorical("flash_attn", cfg.flash_values or [1])
        nkvo = trial.suggest_categorical("no_kv_offload", cfg.nkvo_values or [1])
        ncmoe = trial.suggest_categorical("ncmoe", cfg.ncmoe_values if cfg.is_moe else [0])

        trial_cfg = {
            "batch": batch,
            "ubatch": ubatch,
            "threads": threads,
            "ctx": ctx,
            "flash_attn": int(flash),
            "no_kv_offload": int(nkvo),
            "ncmoe": int(ncmoe),
            "ngl": 99,
            "ctk": cfg.ctk,
            "ctv": cfg.ctv,
            "w_tg": cfg.weight_tg,
            "w_pp": cfg.weight_pp,
            "w_ctx": cfg.weight_ctx,
        }

        log(f"──────────────────────────────")
        log(f"Trial {trial.number+1}/{cfg.trials} | ctx={ctx:,} batch={batch} ubatch={ubatch} ncmoe={ncmoe}")

        res = run_trial(
            paths,
            settings,
            model_path_host=model_host,
            trial_cfg=trial_cfg,
            ctx_tokens=ctx,
            n_tokens=cfg.n_tokens,
            repeats=cfg.repeats,
            timeout_sec=cfg.timeout_sec,
            log_fn=log,
        )
        
        # Save trial log
        trial_log = run_dir / "trials" / f"trial_{trial.number}_ctx{ctx}.log"
        trial_log.write_text(res.raw_output or "", encoding="utf-8", errors="replace")
        meta = {
            "cfg": res.cfg,
            "metrics": res.metrics,
            "score": res.score,
            "status": res.status,
            "error": res.error,
            "ctx": res.ctx_tokens,
        }
        (run_dir / "trials" / f"trial_{trial.number}_ctx{ctx}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        # Log result
        # Use small penalty score (not 0) for failures to help TPE explore better
        # TPE with too many exact 0s can struggle to differentiate failure zones
        FAILURE_PENALTY_SCORE = 0.001
        
        if res.status in ("oom", "context_failed", "failed", "parse_error", "timeout"):
            log(f"Trial {trial.number+1} ❌ {res.status} (ctx={ctx:,}) - {res.error or 'unknown'}")
            score = FAILURE_PENALTY_SCORE
        else:
            tg_val = res.metrics.get("tg", 0)
            pp_val = res.metrics.get("pp", 0)
            score = res.score
            log(f"Trial {trial.number+1} ✅ score={score:.2f} | ctx={ctx:,} | TG={tg_val:.1f} t/s | PP={pp_val:.1f} t/s")

        # Track best
        if score > best_score_overall:
            best_score_overall = score
            best_result = res
            log(f"🎉 NEW BEST! score={score:.2f} ctx={ctx:,} batch={batch} ubatch={ubatch}")
            if progress_callback:
                progress_callback({
                    "type": "new_best",
                    "score": score,
                    "cfg": trial_cfg,
                    "metrics": res.metrics
                })
        
        if progress_callback:
            progress_callback({
                "type": "progress",
                "trial": trial.number + 1,
                "total": cfg.trials,
                "best_score": best_score_overall
            })
            
        return score  # Return this trial's score for Optuna to maximize

    sampler = TPESampler(multivariate=True, warn_independent_sampling=False)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=cfg.trials, n_jobs=1, gc_after_trial=True)
    log(f"Optuna finished. Best value={study.best_value if study.best_value else 0.0}")
    if study.best_params:
        log(f"Best params: {study.best_params}")

    results: list[TrialResult] = []
    csv_rows = []
    for p in (run_dir / "trials").glob("trial_*_ctx*.json"):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            tr = TrialResult(
                cfg=meta.get("cfg") or {},
                metrics=meta.get("metrics") or {},
                score=meta.get("score") or 0.0,
                status=meta.get("status") or "unknown",
                error=meta.get("error"),
                raw_output="",
                ctx_tokens=meta.get("ctx"),
            )
            results.append(tr)
            
            # Flatten for CSV
            row = {
                "ctx": tr.ctx_tokens,
                "score": tr.score,
                "status": tr.status,
                "error": tr.error or "",
            }
            row.update(tr.cfg)
            row.update(tr.metrics)
            csv_rows.append(row)
        except Exception:
            continue
            
    # Save summary CSV
    if csv_rows:
        try:
            # Collect all unique keys from all rows
            all_keys = set()
            for row in csv_rows:
                all_keys.update(row.keys())
            keys = sorted(list(all_keys))
            
            with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(csv_rows)
            log(f"💾 Summary saved to {run_dir / 'summary.csv'}")
        except Exception as e:
            log(f"⚠️ Failed to save summary CSV: {e}")

    best = best_of_trials(results)
    summary = {
        "best_params": study.best_params if study.best_params else {},
        "best_value": study.best_value if study.best_value else 0.0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Résultats enregistrés dans {run_dir}")
    return results, best, run_dir


class HyperOptimusRunner:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.is_running = False

    def start(self, cfg: HyperOptConfig, paths: RepoPaths, settings: AppSettings):
        if self.is_running:
            return
        
        self.stop_event.clear()
        self.is_running = True
        
        # Clear queues
        while not self.log_queue.empty():
            self.log_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
            
        self.thread = threading.Thread(
            target=self._run, 
            args=(cfg, paths, settings),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.is_running = False

    def _run(self, cfg: HyperOptConfig, paths: RepoPaths, settings: AppSettings):
        try:
            def log_fn(msg):
                self.log_queue.put(msg)
                
            def progress_cb(data):
                self.result_queue.put(data)

            results, best, run_dir = run_optuna(
                cfg, 
                paths, 
                settings, 
                stop_event=self.stop_event,
                log_fn=log_fn,
                progress_callback=progress_cb
            )
            
            self.result_queue.put({
                "type": "complete",
                "results": results,
                "best": best,
                "run_dir": str(run_dir)
            })
            
        except Exception as e:
            self.log_queue.put(f"❌ Fatal error: {str(e)}")
            self.result_queue.put({"type": "error", "message": str(e)})
        finally:
            self.is_running = False
