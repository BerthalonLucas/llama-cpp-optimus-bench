from __future__ import annotations

import itertools
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque

from .bench import BenchConfig, bench_once, finalize_bench_run
from .docker import kill_in_container
from .live_process import LiveSubprocess
from .paths import RepoPaths
from .scoring import SweepScores
from .settings import AppSettings
from .storage import new_id, run_dir, save_json, sweep_dir


@dataclass
class BenchTask:
    paths: RepoPaths
    settings: AppSettings
    model_path_host: Path
    cfg: BenchConfig

    run_id: str = field(default_factory=lambda: new_id("bench"))
    sweep_id: str | None = None

    status: str = field(init=False, default="pending")
    _run_dir: Path = field(init=False)
    _meta: dict[str, Any] = field(init=False, default_factory=dict)
    _proc: LiveSubprocess | None = field(init=False, default=None)
    _thread: threading.Thread | None = field(init=False, default=None)
    _stop_requested: bool = field(init=False, default=False)

    result: dict[str, Any] | None = field(init=False, default=None)

    def start(self) -> None:
        self._run_dir = run_dir(self.paths, "bench", self.run_id)
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Persist a quick stub early (helps history).
        stub = {
            "run_type": "bench",
            "status": "running",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_file": self.model_path_host.name,
            "sweep_id": self.sweep_id,
            "bench_config": asdict(self.cfg),
        }
        save_json(self._run_dir / "run.json", stub)

        self._meta = bench_once(
            self.paths,
            self.settings,
            model_path_host=self.model_path_host,
            cfg=self.cfg,
            run_dir=self._run_dir,
        )
        self._proc = self._meta["proc"]
        self.status = "running"

        self._thread = threading.Thread(target=self._wait_and_finalize, daemon=True)
        self._thread.start()

    def _wait_and_finalize(self) -> None:
        assert self._proc is not None
        self._proc.wait()

        rec = finalize_bench_run(
            run_dir=self._run_dir,
            run_meta=self._meta,
            cfg=self.cfg,
            model_file=self.model_path_host.name,
            sweep_id=self.sweep_id,
        )

        if self._stop_requested and rec.get("status") != "completed":
            rec["status"] = "stopped"
            save_json(self._run_dir / "run.json", rec)

        self.result = rec
        self.status = rec.get("status") or "unknown"

    def proc(self) -> LiveSubprocess | None:
        return self._proc

    def run_directory(self) -> Path:
        return self._run_dir

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc is not None:
            self._proc.terminate()
        # Best-effort kill inside container (llama-bench/optimus)
        try:
            kill_in_container(self.paths, self.settings, patterns=["llama-bench", "llama-optimus"])
        except Exception:
            pass


def _product_count(lists: list[list[Any]]) -> int:
    n = 1
    for lst in lists:
        n *= max(1, len(lst))
    return n


@dataclass
class SweepTask:
    paths: RepoPaths
    settings: AppSettings
    model_path_host: Path
    base_cfg: BenchConfig
    sweep_values: dict[str, list[Any]]
    weight_tg: float
    weight_pp: float
    mode: str = "multi"  # "1d" or "multi"

    sweep_id: str = field(default_factory=lambda: new_id("sweep"))
    status: str = field(init=False, default="pending")

    _dir: Path = field(init=False)
    _thread: threading.Thread | None = field(init=False, default=None)
    _stop_requested: bool = field(init=False, default=False)
    _current_task: BenchTask | None = field(init=False, default=None)

    results: list[dict[str, Any]] = field(init=False, default_factory=list)
    logs: Deque[str] = field(init=False, default_factory=lambda: deque(maxlen=500))
    done: int = field(init=False, default=0)
    total: int = field(init=False, default=0)

    def total_combos(self) -> int:
        return _product_count(list(self.sweep_values.values()))

    def progress(self) -> float:
        if self.total == 0:
            return 0.0
        return min(1.0, self.done / self.total)

    def sweep_dir(self) -> Path:
        return self._dir

    @property
    def current_task(self) -> BenchTask | None:
        return self._current_task

    def start(self) -> None:
        # Ensure base_cfg is BenchConfig (can come from JSON)
        if isinstance(self.base_cfg, dict):
            self.base_cfg = BenchConfig(**self.base_cfg)

        self._dir = sweep_dir(self.paths, self.sweep_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.total = self.total_combos()

        meta = {
            "run_type": "sweep",
            "sweep_id": self.sweep_id,
            "status": "running",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_file": self.model_path_host.name,
            "base_config": asdict(self.base_cfg),
            "sweep_values": self.sweep_values,
            "weights": {"w_tg": self.weight_tg, "w_pp": self.weight_pp},
            "run_ids": [],
            "total": self.total,
            "done": 0,
            "mode": self.mode,
        }
        save_json(self._dir / "sweep.json", meta)

        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {msg}")

    def _run(self) -> None:
        keys = list(self.sweep_values.keys())
        value_lists = [self.sweep_values[k] for k in keys]

        if self.total > self.settings.max_combos:
            self.status = "refused"
            meta = json.loads((self._dir / "sweep.json").read_text(encoding="utf-8"))
            meta["status"] = "refused"
            meta["error"] = f"Too many combinations: {self.total} > max_combos={self.settings.max_combos}"
            save_json(self._dir / "sweep.json", meta)
            return

        run_ids: list[str] = []

        for values in itertools.product(*value_lists):
            if self._stop_requested:
                break

            overrides = dict(zip(keys, values))
            self._log(f"Starting combo: {overrides}")

            cfg_dict = asdict(self.base_cfg)
            for k, v in overrides.items():
                if k not in cfg_dict:
                    continue
                cfg_dict[k] = v
            cfg = BenchConfig(**cfg_dict)

            task = BenchTask(
                paths=self.paths,
                settings=self.settings,
                model_path_host=self.model_path_host,
                cfg=cfg,
                sweep_id=self.sweep_id,
            )
            self._current_task = task
            task.start()
            run_ids.append(task.run_id)

            # Wait bench completion
            while task.status == "running" and not self._stop_requested:
                time.sleep(0.2)

            if self._stop_requested:
                task.stop()
                # Wait a bit to let docker compose exit
                time.sleep(0.5)

            # task.result becomes available when finalize done
            while task.result is None and task.status == "running":
                time.sleep(0.2)

            self.done += 1

            if task.result is not None:
                rec = dict(task.result)
                rec["overrides"] = overrides
                self.results.append(rec)
                
                # Enhanced logging with error detection
                status = rec.get("status")
                error_type = rec.get("error_type", "none")
                is_oom = rec.get("is_oom", False)
                
                if is_oom:
                    self._log(f"⚠️ OOM: {overrides} -> {error_type}")
                elif status == "completed":
                    metrics = rec.get("metrics", {})
                    tg = metrics.get("tg128", metrics.get("tg", 0))
                    self._log(f"✅ OK: {overrides} -> tg={tg:.1f}")
                else:
                    error_msg = rec.get("error_message", "unknown")
                    self._log(f"❌ FAIL: {overrides} -> {error_msg}")

                # Recompute scores over current successful/failed runs
                scorer = SweepScores(w_tg=self.weight_tg, w_pp=self.weight_pp)
                scorer.score_runs(self.results)

                # Persist sweep.json frequently so History page stays live
                meta = {
                    "run_type": "sweep",
                    "sweep_id": self.sweep_id,
                    "status": "running",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model_file": self.model_path_host.name,
                    "base_config": asdict(self.base_cfg),
                    "sweep_values": self.sweep_values,
                    "weights": {"w_tg": self.weight_tg, "w_pp": self.weight_pp},
                    "run_ids": run_ids,
                    "results": self.results,
                    "total": self.total,
                    "done": self.done,
                }
                # Best run so far
                best = None
                best_score = None
                for r in self.results:
                    s = r.get("score")
                    if isinstance(s, (int, float)):
                        if best_score is None or float(s) > best_score:
                            best = r
                            best_score = float(s)
                if best is not None:
                    meta["best_run_id"] = best.get("run_id") or best.get("id") or best.get("_run_id")
                    if best.get("llama_server_cmd"):
                        meta["best_llama_server_cmd"] = best.get("llama_server_cmd")
                        meta["best_ctx_size_tokens"] = best.get("ctx_size_tokens")
                save_json(self._dir / "sweep.json", meta)

        # Final status
        final = json.loads((self._dir / "sweep.json").read_text(encoding="utf-8"))
        if self._stop_requested:
            final["status"] = "stopped"
            self.status = "stopped"
        else:
            final["status"] = "completed"
            self.status = "completed"
        save_json(self._dir / "sweep.json", final)
        
        # Create sweep_results.csv for easy analysis
        self._save_results_csv()

    def _save_results_csv(self) -> None:
        """Save sweep results to CSV for easy analysis."""
        if not self.results:
            return
        
        import pandas as pd
        
        rows = []
        for r in self.results:
            row = {}
            # Add overrides (sweep parameters)
            overrides = r.get("overrides", {})
            for k, v in overrides.items():
                row[k] = v
            
            # Add metrics
            metrics = r.get("metrics", {})
            for k, v in metrics.items():
                row[k] = v
            
            # Add status, score, and error info
            row["status"] = r.get("status", "?")
            row["score"] = r.get("score")
            row["run_id"] = r.get("run_id")
            row["wall_time_sec"] = r.get("wall_time_sec")
            row["error_type"] = r.get("error_type", "none")
            row["is_oom"] = r.get("is_oom", False)
            row["error_message"] = r.get("error_message")
            row["ctx_size_tokens"] = r.get("ctx_size_tokens")
            row["llama_server_cmd"] = r.get("llama_server_cmd")
            
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = self._dir / "sweep_results.csv"
            df.to_csv(csv_path, index=False)

    def stop(self) -> None:
        self._stop_requested = True
        if self._current_task is not None:
            self._current_task.stop()
        try:
            kill_in_container(self.paths, self.settings, patterns=["llama-bench", "llama-optimus"])
        except Exception:
            pass
