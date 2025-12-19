from __future__ import annotations

import itertools
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Deque

from .bench import BenchConfig, bench_once, finalize_bench_run
from .docker import kill_in_container
from .paths import RepoPaths
from .scoring import min_max_norm
from .settings import AppSettings
from .storage import new_id, optimizer_dir


def _parse_int_list(values: list[int] | list[str] | str, *, default: list[int]) -> list[int]:
    if isinstance(values, list):
        out: list[int] = []
        for v in values:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out or default
    if isinstance(values, str):
        parts = [p.strip() for p in values.replace(";", ",").split(",") if p.strip()]
        return _parse_int_list(parts, default=default)
    return default


@dataclass
class OptimizerConfig:
    model_path_host: Path
    is_moe: bool
    ctx_targets: list[int]
    batch_values: list[int]
    ubatch_values: list[int]
    thread_values: list[int]
    flash_values: list[int]
    nkvo_values: list[int]
    ctk_values: list[str]
    ctv_values: list[str]
    ngl_values: list[int]  # dense only (optional)
    ncmoe_values: list[int]  # MoE only
    repeats: int
    weight_tg: float
    weight_pp: float
    weight_ctx: float
    budget_runs: int
    stop_on_no_gpu: bool = True


@dataclass
class OptimizerResult:
    run_id: str
    status: str
    results: list[dict[str, Any]]
    best: dict[str, Any] | None
    logs: list[str]
    note: str | None = None


class OptimizerTask:
    """Simple sequential optimizer with ctx backtracking."""

    def __init__(self, paths: RepoPaths, settings: AppSettings, config: OptimizerConfig):
        self.paths = paths
        self.settings = settings
        self.config = config
        self.run_id = new_id("optimizer")
        self.status: str = "pending"
        self.results: list[dict[str, Any]] = []
        self.logs: Deque[str] = deque(maxlen=4000)
        self._stop_requested: bool = False
        self._current_proc = None
        self._thread: threading.Thread | None = None
        self._dir = optimizer_dir(paths, self.run_id)
        self.best: dict[str, Any] | None = None
        self.note: str | None = None
        self.result: OptimizerResult | None = None

    # Logging helpers
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        self.logs.append(line)

    def logs_text(self, last_n: int = 800) -> str:
        lines = list(self.logs)
        return "\n".join(lines[-last_n:])

    def is_running(self) -> bool:
        if self.status not in ("running", "stopping"):
            return False
        # Be defensive: if the thread died unexpectedly, don't keep UI locked in "running".
        if self._thread is not None and self._thread.is_alive():
            return True
        if self._current_proc is not None and getattr(self._current_proc, "is_running", None):
            try:
                return bool(self._current_proc.is_running())
            except Exception:
                return False
        return False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("OptimizerTask already running")

        self._dir.mkdir(parents=True, exist_ok=True)
        self.status = "running"

        def _worker() -> None:
            try:
                self.result = self.run()
            except Exception as e:  # noqa: BLE001
                self.status = "failed"
                self.note = str(e)
                self._log(f"❌ Error: {e}")
                self._persist_summary()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def run_directory(self) -> Path:
        return self._dir

    def _persist_summary(self) -> None:
        cfg_dict = asdict(self.config)
        cfg_dict["model_path_host"] = str(self.config.model_path_host)
        summary = {
            "run_id": self.run_id,
            "status": self.status,
            "config": cfg_dict,
            "results": self.results,
            "best": self.best,
            "note": self.note,
        }
        # Never silently lose a run: try strict JSON first, then fallback to string-coercion.
        summary_path = self._dir / "summary.json"
        logs_path = self._dir / "logs.txt"
        try:
            summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
        except TypeError:
            self._log("⚠️ Summary contains non-JSON values; coercing to string.")
            summary_json = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        try:
            summary_path.write_text(summary_json, encoding="utf-8")
        except Exception:
            # Keep going: logs may still be useful.
            pass
        try:
            logs_path.write_text("\n".join(self.logs), encoding="utf-8")
        except Exception:
            pass

    # Candidate generation
    def _candidates(self) -> list[dict[str, Any]]:
        cfg = self.config
        flash_vals = [int(v) for v in cfg.flash_values] or [1]
        nkvo_vals = [int(v) for v in cfg.nkvo_values] or [0]
        ctk_vals = cfg.ctk_values or ["q8_0"]
        ctv_vals = cfg.ctv_values or ["q8_0"]
        threads = cfg.thread_values or [self.settings.default_threads]
        batches = cfg.batch_values or [self.settings.default_batch]
        ubatches = cfg.ubatch_values or [self.settings.default_ubatch]

        combos = []
        for b, ub, t, fa, nk, ctk, ctv in itertools.product(
            batches, ubatches, threads, flash_vals, nkvo_vals, ctk_vals, ctv_vals
        ):
            base = {
                "batch": int(b),
                "ubatch": int(ub),
                "threads": int(t),
                "flash_attn": int(fa),
                "no_kv_offload": int(nk),
                "ctk": str(ctk),
                "ctv": str(ctv),
            }
            if self.config.is_moe:
                for nc in cfg.ncmoe_values or [0]:
                    combos.append({**base, "ncmoe": int(nc), "ngl": 99})
            else:
                # Dense: optional ngl sweep
                if cfg.ngl_values:
                    for ngl in cfg.ngl_values:
                        combos.append({**base, "ngl": int(ngl)})
                else:
                    combos.append({**base, "ngl": 99})
        return combos

    def stop(self) -> None:
        self._stop_requested = True
        if self.status == "running":
            self.status = "stopping"
            self._log("🛑 Stop requested")
            self._persist_summary()
        if self._current_proc is not None:
            try:
                self._current_proc.terminate()
            except Exception:
                pass
        try:
            kill_in_container(self.paths, self.settings, patterns=["llama-bench", "llama-optimus"])
        except Exception:
            pass

    def _score(self, runs: list[dict[str, Any]]) -> None:
        """Compute normalized score in-place."""
        needs_tg = float(self.config.weight_tg) > 0.0
        needs_pp = float(self.config.weight_pp) > 0.0

        def _metric_val(r: dict[str, Any], prefix: str) -> float | None:
            m = r.get("metrics") or {}
            bc = r.get("bench_config") or {}
            preferred_key = None
            try:
                if prefix == "tg":
                    preferred_key = f"tg{int(bc.get('n_gen') or 0)}"
                elif prefix == "pp":
                    preferred_key = f"pp{int(bc.get('n_prompt') or 0)}"
            except Exception:
                preferred_key = None
            if preferred_key and preferred_key in m and isinstance(m[preferred_key], (int, float)):
                return float(m[preferred_key])
            for k, v in m.items():
                if str(k).startswith(prefix) and isinstance(v, (int, float)):
                    return float(v)
            return None

        def _eligible_completed(r: dict[str, Any]) -> bool:
            if r.get("status") != "completed":
                return False
            tg = _metric_val(r, "tg")
            pp = _metric_val(r, "pp")
            if needs_tg and tg is None:
                return False
            if needs_pp and pp is None:
                return False
            return True

        completed = [r for r in runs if _eligible_completed(r)]
        completed_ids = {id(r) for r in completed}
        tg_vals = []
        pp_vals = []
        ctx_vals = []
        for r in completed:
            tg = _metric_val(r, "tg")
            pp = _metric_val(r, "pp")
            if tg is not None:
                tg_vals.append(float(tg))
            if pp is not None:
                pp_vals.append(float(pp))
            ctx = r.get("ctx_tokens")
            if isinstance(ctx, (int, float)):
                ctx_vals.append(float(ctx))
        tg_min, tg_max = (min(tg_vals) if tg_vals else 0.0, max(tg_vals) if tg_vals else 0.0)
        pp_min, pp_max = (min(pp_vals) if pp_vals else 0.0, max(pp_vals) if pp_vals else 0.0)
        ctx_min, ctx_max = (min(ctx_vals) if ctx_vals else 0.0, max(ctx_vals) if ctx_vals else 0.0)

        for r in runs:
            if id(r) not in completed_ids:
                r["norm"] = {"tg": 0.0, "pp": 0.0, "ctx": 0.0}
                r["score"] = None
                continue
            tg = _metric_val(r, "tg")
            pp = _metric_val(r, "pp")
            ctx = r.get("ctx_tokens")
            n_tg = min_max_norm(float(tg), min_v=tg_min, max_v=tg_max) if tg is not None else 0.0
            n_pp = min_max_norm(float(pp), min_v=pp_min, max_v=pp_max) if pp is not None else 0.0
            n_ctx = min_max_norm(float(ctx), min_v=ctx_min, max_v=ctx_max) if isinstance(ctx, (int, float)) else 0.0
            score = (
                self.config.weight_tg * n_tg
                + self.config.weight_pp * n_pp
                + self.config.weight_ctx * n_ctx
            )
            r["norm"] = {"tg": n_tg, "pp": n_pp, "ctx": n_ctx}
            r["score"] = score

        # Track best
        best = None
        best_score = None
        for r in completed:
            s = r.get("score")
            if isinstance(s, (int, float)):
                if best_score is None or float(s) > best_score:
                    best = r
                    best_score = float(s)
        self.best = best

    def _ctx_parts(self, ctx_tokens: int) -> tuple[int, int, int]:
        # Heuristic: half prompt, quarter gen, remainder depth
        p = max(32, ctx_tokens // 2)
        n = max(16, ctx_tokens // 4)
        depth = max(0, ctx_tokens - p - n)
        return p, n, depth

    def run(self) -> OptimizerResult:
        if self.status != "stopping":
            self.status = "running"
        self._log(f"Run {self.run_id} démarré")
        combos = self._candidates()
        max_runs = max(1, self.config.budget_runs)
        run_count = 0
        last_success_ctx: int | None = None
        first_fail_ctx: int | None = None

        for ctx in sorted(set(self.config.ctx_targets)):
            if self._stop_requested:
                break
            self._log(f"CTX target: {ctx}")

            for combo in combos:
                if self._stop_requested or run_count >= max_runs:
                    break

                run_count += 1
                p_tokens, n_tokens, depth_tokens = self._ctx_parts(ctx)
                cfg = BenchConfig(
                    threads=combo["threads"],
                    batch=combo["batch"],
                    ubatch=combo["ubatch"],
                    gpu_layers=combo.get("ngl", 99),
                    no_kv_offload=combo.get("no_kv_offload", 0),
                    flash_attn=combo.get("flash_attn", 1),
                    ctk=combo.get("ctk", "q8_0"),
                    ctv=combo.get("ctv", "q8_0"),
                    n_cpu_moe=combo.get("ncmoe"),
                    n_prompt=p_tokens,
                    n_gen=n_tokens,
                    n_depth=depth_tokens,
                    repeats=self.config.repeats,
                )

                # Prepare run dir
                run_dir = self._dir / f"run_{run_count:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)
                meta = bench_once(
                    self.paths,
                    self.settings,
                    model_path_host=self.config.model_path_host,
                    cfg=cfg,
                    run_dir=run_dir,
                )
                proc = meta["proc"]
                self._current_proc = proc

                # Stream logs while running
                fatal_no_gpu = False
                no_gpu_line: str | None = None
                while proc.is_running() and not self._stop_requested:
                    for ln in proc.read_new_lines(max_lines=200):
                        self.logs.append(ln)
                        if self.config.stop_on_no_gpu and int(cfg.gpu_layers) > 0:
                            low = ln.lower()
                            if (
                                "found 0 cuda devices" in low
                                or "no cuda devices" in low
                                or "cuda is not available" in low
                            ):
                                fatal_no_gpu = True
                                no_gpu_line = ln
                    if fatal_no_gpu:
                        self.status = "failed"
                        self.note = no_gpu_line or "No CUDA device detected"
                        self._log("❌ No GPU detected; stopping optimizer early")
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        try:
                            kill_in_container(self.paths, self.settings, patterns=["llama-bench"])
                        except Exception:
                            pass
                        break
                    time.sleep(0.2)

                if self._stop_requested:
                    proc.terminate()
                    break

                if fatal_no_gpu:
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass

                # Finalize
                rec = finalize_bench_run(
                    run_dir=run_dir,
                    run_meta=meta,
                    cfg=cfg,
                    model_file=self.config.model_path_host.name,
                    sweep_id=self.run_id,
                )
                rec["ctx_tokens"] = ctx
                rec["params"] = combo

                if fatal_no_gpu:
                    rec["status"] = "failed"
                    rec["error_type"] = "no_gpu"
                    rec["error_message"] = "No CUDA device detected (run forced to CPU)"
                    rec["error_details"] = no_gpu_line
                    rec["is_oom"] = False
                    try:
                        (run_dir / "run.json").write_text(
                            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                    except Exception:
                        pass

                self.results.append(rec)
                # Append logs tail
                for ln in (rec.get("parse_error") or "").splitlines():
                    if ln:
                        self.logs.append(f"[parse] {ln}")

                if rec.get("status") == "completed":
                    last_success_ctx = ctx
                    self._log(f"OK ctx={ctx} combo={combo} score pending")
                else:
                    # Consider as failure (OOM/load)
                    if first_fail_ctx is None:
                        first_fail_ctx = ctx
                    self._log(f"FAIL ctx={ctx} combo={combo} status={rec.get('status')} err={rec.get('error_message')}")
                    # Fatal: CUDA backend error -> stop early and surface
                    if rec.get("error_type") == "cuda_error":
                        self.status = "failed"
                        self.note = rec.get("error_details") or rec.get("error_message")
                        self._log("❌ CUDA error detected; stopping optimizer early")
                        break

                # Re-score after each completed run
                self._score(self.results)
                # Persist often so History stays usable even if Streamlit dies mid-run.
                self._persist_summary()
                if self.status == "failed":
                    break

            if self.status == "failed":
                break

            # Early stop on ctx if failed and we already have a success
            if first_fail_ctx is not None and last_success_ctx is not None and ctx > first_fail_ctx:
                self._log("Stop ctx exploration after failure above previous success")
                break

            if run_count >= max_runs:
                break

        if self.status not in ("failed",):
            if self._stop_requested:
                self.status = "stopped"
            else:
                self.status = "completed"

        # Persist summary
        self._persist_summary()

        return OptimizerResult(
            run_id=self.run_id,
            status=self.status,
            results=self.results,
            best=self.best,
            logs=list(self.logs),
            note=self.note,
        )
