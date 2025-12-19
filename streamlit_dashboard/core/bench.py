from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .docker import compose_run
from .error_detection import ErrorInfo, ErrorType, detect_error
from .paths import RepoPaths, host_path_to_container_path
from .settings import AppSettings
from .validation import ValidationError, validate_choice


CACHE_TYPES = ("f16", "q8_0", "q4_0")


def _as_int(val: Any) -> int:
    try:
        return int(val)
    except Exception:
        return 0


def recommended_ctx_tokens(cfg: "BenchConfig") -> int:
    """Approximate ctx-size to cover prompt + generation + prefilling."""
    tokens = _as_int(cfg.n_prompt) + _as_int(cfg.n_gen) + _as_int(cfg.n_depth)
    return max(128, tokens)


def format_llama_server_command(model_path_container: str, params: Any, *, ctx_size: int | None = None) -> str:
    """Build a llama-server command from bench/optimus parameters."""
    if hasattr(params, "__dataclass_fields__"):
        params = asdict(params)
    else:
        params = dict(params)

    def _get(key: str, *aliases: str) -> Any:
        for k in (key, *aliases):
            if k in params and params[k] is not None:
                return params[k]
        return None

    parts: list[str] = ["./llama.sh", "server", "--model", model_path_container]

    threads = _get("threads", "n_threads")
    if threads is not None:
        parts += ["-t", str(int(threads))]

    batch = _get("batch", "n_batch")
    if batch is not None:
        parts += ["--batch-size", str(int(batch))]

    ubatch = _get("ubatch", "u_batch", "n_ubatch")
    if ubatch is not None:
        parts += ["--ubatch-size", str(int(ubatch))]

    ngl = _get("gpu_layers", "ngl", "n_gpu_layers")
    if ngl is not None:
        parts += ["-ngl", str(int(ngl))]

    nkvo = _get("no_kv_offload", "nkvo")
    if nkvo is not None:
        # llama-server expects `--no-kv-offload` as a flag (no value),
        # while llama-bench uses `-nkvo 0|1`.
        try:
            nkvo_i = int(nkvo)
        except Exception:
            nkvo_i = 0
        if nkvo_i != 0:
            parts += ["--no-kv-offload"]

    flash = _get("flash_attn", "flash", "fa")
    if flash is not None:
        # llama-server expects `--flash-attn` as a flag (no value),
        # while llama-bench uses `-fa 0|1`.
        try:
            flash_i = int(flash)
        except Exception:
            flash_i = 0
        if flash_i != 0:
            parts += ["--flash-attn"]

    ctk = _get("ctk", "cache_type_k")
    ctv = _get("ctv", "cache_type_v")
    if ctk is not None:
        parts += ["-ctk", str(ctk)]
    if ctv is not None:
        parts += ["-ctv", str(ctv)]

    ncmoe = _get("n_cpu_moe", "ncmoe")
    if ncmoe is not None:
        parts += ["-ncmoe", str(int(ncmoe))]

    override_tensor = _get("override_tensor")
    if override_tensor:
        parts += ["--override-tensor", str(override_tensor)]

    if ctx_size is not None:
        parts += ["--ctx-size", str(int(ctx_size))]

    return " ".join(parts)


@dataclass
class BenchConfig:
    threads: int
    batch: int
    ubatch: int
    gpu_layers: int
    no_kv_offload: int
    flash_attn: int
    ctk: str
    ctv: str
    n_cpu_moe: int | None = None
    n_prompt: int = 512
    n_gen: int = 128
    n_depth: int = 0  # Pre-filled context depth (simulates existing KV cache)
    repeats: int = 3

    def as_llama_bench_args(self, *, llama_bin_dir: str, model_path_container: str) -> list[str]:
        # Validate restrictive choices
        validate_choice(self.ctk, allowed=CACHE_TYPES, label="ctk")
        validate_choice(self.ctv, allowed=CACHE_TYPES, label="ctv")

        args: list[str] = [
            f"{llama_bin_dir.rstrip('/')}/llama-bench",
            "-m",
            model_path_container,
            "-o",
            "csv",
            "-t",
            str(int(self.threads)),
            "-b",
            str(int(self.batch)),
            "-ub",
            str(int(self.ubatch)),
            "-ngl",
            str(int(self.gpu_layers)),
            "-nkvo",
            str(int(self.no_kv_offload)),
            "-fa",
            str(int(self.flash_attn)),
            "-ctk",
            self.ctk,
            "-ctv",
            self.ctv,
            "-p",
            str(int(self.n_prompt)),
            "-n",
            str(int(self.n_gen)),
            "-d",
            str(int(self.n_depth)),
            "-r",
            str(int(self.repeats)),
        ]
        if self.n_cpu_moe is not None:
            args += ["-ncmoe", str(int(self.n_cpu_moe))]
        return args


def _derive_test_label(row: dict[str, Any]) -> str:
    # llama-bench output formats may vary. Prefer 'test' if present.
    test = row.get("test")
    if test:
        return str(test)

    def _as_int(k: str) -> int:
        v = row.get(k)
        try:
            return int(v)
        except Exception:
            return 0

    n_prompt = _as_int("n_prompt")
    n_gen = _as_int("n_gen")

    if n_prompt > 0 and n_gen == 0:
        return f"pp{n_prompt}"
    if n_prompt == 0 and n_gen > 0:
        return f"tg{n_gen}"
    if n_prompt > 0 and n_gen > 0:
        return f"pp{n_prompt}_tg{n_gen}"
    return "unknown"


def parse_llama_bench_csv(stdout_text: str) -> list[dict[str, Any]]:
    """Parse the CSV output from llama-bench.

    We try hard to locate the CSV header inside stdout, since some builds may
    print extra messages.
    """

    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    if not lines:
        raise ValidationError("No stdout received from llama-bench")

    header_idx = None
    for i, ln in enumerate(lines):
        # Heuristic: the CSV header always contains 'build_commit' and 'avg_ts'.
        if "build_commit" in ln and "avg_ts" in ln and "," in ln:
            header_idx = i
            break
    if header_idx is None:
        raise ValidationError(
            "Could not find llama-bench CSV header in stdout. "
            "Try running without other flags that might change output."
        )

    csv_text = "\n".join(lines[header_idx:]) + "\n"
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict[str, Any]] = []
    for r in reader:
        if not r:
            continue
        rows.append(dict(r))

    if not rows:
        raise ValidationError("Parsed 0 CSV rows from llama-bench")
    return rows


def extract_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for r in rows:
        label = _derive_test_label(r)
        try:
            v = float(r.get("avg_ts") or 0.0)
        except Exception:
            v = 0.0
        metrics[label] = v
    return metrics


def bench_once(
    paths: RepoPaths,
    settings: AppSettings,
    *,
    model_path_host: Path,
    cfg: BenchConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """Run a single llama-bench invocation (via docker compose)."""

    # Convert host path to container path
    model_path_host = Path(model_path_host).absolute()
    model_path_container = host_path_to_container_path(model_path_host, paths)

    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = cfg.as_llama_bench_args(
        llama_bin_dir=settings.llama_bin_dir,
        model_path_container=model_path_container,
    )

    started_at = time.time()
    proc = compose_run(paths, settings, command=cmd, log_dir=run_dir)

    return {
        "proc": proc,
        "started_at": started_at,
        "model_path_container": model_path_container,
        "cmd": cmd,
    }


def finalize_bench_run(
    *,
    run_dir: Path,
    run_meta: dict[str, Any],
    cfg: BenchConfig,
    model_file: str,
    sweep_id: str | None = None,
) -> dict[str, Any]:
    """Collect outputs and persist run.json + parsed artifacts."""

    proc = run_meta["proc"]
    assert proc is not None

    stdout = proc.stdout_text()
    stderr = proc.stderr_text()
    exit_code = proc.returncode
    wall_time_sec = proc.wall_time_sec()
    model_path_container = run_meta.get("model_path_container")

    # Build recommended llama-server command (mirrors bench params)
    ctx_tokens = recommended_ctx_tokens(cfg)
    llama_server_cmd = None
    if model_path_container:
        llama_server_cmd = format_llama_server_command(
            model_path_container=model_path_container,
            params=cfg,
            ctx_size=ctx_tokens,
        )

    status = "completed" if exit_code == 0 else "failed"
    
    # Detect specific error type (OOM, CUDA errors, etc.)
    error_info: ErrorInfo = detect_error(stdout, stderr, exit_code)

    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    parse_error: str | None = None
    parse_failed: bool = False

    if exit_code == 0:
        try:
            rows = parse_llama_bench_csv(stdout)
            metrics = extract_metrics(rows)
        except Exception as e:  # noqa: BLE001
            status = "failed"
            parse_error = str(e)
            parse_failed = True

    # Base error fields
    error_type = error_info.error_type.value
    error_message = error_info.message if error_info.error_type != ErrorType.NONE else None
    error_details = error_info.details
    is_oom = error_info.is_oom

    # If parsing failed despite exit_code==0, surface it as an explicit error type.
    if parse_failed:
        error_type = "parse_error"
        error_message = parse_error or "Failed to parse llama-bench output"
        error_details = None
        is_oom = False

    run_record = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_type": "bench",
        "status": status,
        "exit_code": exit_code,
        "wall_time_sec": wall_time_sec,
        "model_file": model_file,
        "sweep_id": sweep_id,
        "bench_config": asdict(cfg),
        "cmd": run_meta.get("cmd"),
        "metrics": metrics,
        "parse_error": parse_error,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_meta["started_at"])),
        # Error detection fields
        "error_type": error_type,
        "error_message": error_message,
        "error_details": error_details,
        "is_oom": is_oom,
        # Convenience: ready-to-run llama-server command
        "ctx_size_tokens": ctx_tokens,
        "llama_server_cmd": llama_server_cmd,
    }

    (run_dir / "run.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Parsed artifacts
    if rows:
        (run_dir / "results_rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Clean CSV (for exports)
        csv_path = run_dir / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # The raw stdout/stderr are already stored in stdout.log/stderr.log by LiveSubprocess.
    (run_dir / "stderr_tail.txt").write_text("\n".join(stderr.splitlines()[-80:]), encoding="utf-8")

    return run_record
