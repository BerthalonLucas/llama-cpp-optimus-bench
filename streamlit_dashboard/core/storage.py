from __future__ import annotations

import json
import random
import string
import time
from pathlib import Path
from typing import Any, Iterable

from .paths import RepoPaths


def _rand_suffix(k: int = 6) -> str:
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(k))


def new_id(prefix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{prefix}_{ts}_{_rand_suffix()}"


def ensure_run_dirs(paths: RepoPaths) -> None:
    paths.ensure_dirs()
    (paths.runs_dir / "bench").mkdir(parents=True, exist_ok=True)
    (paths.runs_dir / "sweeps").mkdir(parents=True, exist_ok=True)
    (paths.runs_dir / "optimus").mkdir(parents=True, exist_ok=True)
    (paths.runs_dir / "optimizer").mkdir(parents=True, exist_ok=True)


def list_local_models(paths: RepoPaths, external_dirs: list[Path] | None = None) -> list[Path]:
    """List local GGUF models from ./models and external directories.
    
    Args:
        paths: RepoPaths instance
        external_dirs: Additional directories to scan for GGUF files (e.g., LM Studio)
    
    Returns:
        List of unique GGUF model paths, preferring local files over external,
        and filtering out mmproj/vision files.
    """
    paths.ensure_dirs()
    
    # Collect all GGUF files from local models directory
    all_models: list[tuple[Path, str]] = []  # (path, source)
    
    for p in paths.models_dir.glob("**/*.gguf"):
        all_models.append((p, "local"))
    
    # Add external directories
    if external_dirs:
        for ext_dir in external_dirs:
            if ext_dir.exists() and ext_dir.is_dir():
                for p in ext_dir.glob("**/*.gguf"):
                    all_models.append((p, str(ext_dir)))
    
    # Filter out mmproj files (vision model projectors, not usable standalone)
    all_models = [(p, src) for p, src in all_models if "mmproj" not in p.name.lower()]
    
    # Filter out split model parts (keep only first part or standalone)
    # Files like: model-00001-of-00003.gguf should keep only 00001
    filtered = []
    for p, src in all_models:
        name = p.name.lower()
        # Check if it's a split file (contains -NNNNN-of-NNNNN pattern)
        if "-of-" in name:
            # Only keep the first part (00001)
            if "-00001-of-" in name:
                filtered.append((p, src))
        else:
            filtered.append((p, src))
    all_models = filtered
    
    # Group by filename to detect duplicates
    by_name: dict[str, list[tuple[Path, str]]] = {}
    for p, src in all_models:
        name = p.name.lower()
        if name not in by_name:
            by_name[name] = []
        by_name[name].append((p, src))
    
    # For duplicates, prefer local files, then shorter paths
    result = []
    for name, paths_list in by_name.items():
        if len(paths_list) == 1:
            result.append(paths_list[0][0])
        else:
            # Sort: local first, then by path length
            paths_list.sort(key=lambda x: (0 if x[1] == "local" else 1, len(x[0].parts)))
            result.append(paths_list[0][0])
    
    return sorted(result, key=lambda p: p.name.lower())


def list_models_by_source(paths: RepoPaths, external_dirs: list[Path] | None = None) -> dict[str, list[Path]]:
    """List models grouped by source directory.
    
    Returns dict with keys like "local", "/home/user/.lmstudio/models", etc.
    """
    paths.ensure_dirs()
    
    result: dict[str, list[Path]] = {"local": []}
    
    # Local models
    for p in paths.models_dir.glob("**/*.gguf"):
        if "mmproj" not in p.name.lower():
            result["local"].append(p)
    
    # External directories
    if external_dirs:
        for ext_dir in external_dirs:
            if ext_dir.exists() and ext_dir.is_dir():
                key = str(ext_dir)
                result[key] = []
                for p in ext_dir.glob("**/*.gguf"):
                    if "mmproj" not in p.name.lower():
                        result[key].append(p)
    
    # Sort each list
    for key in result:
        result[key] = sorted(result[key], key=lambda p: p.name.lower())
    
    return result


def run_dir(paths: RepoPaths, run_type: str, run_id: str) -> Path:
    ensure_run_dirs(paths)
    return paths.runs_dir / run_type / run_id


def sweep_dir(paths: RepoPaths, sweep_id: str) -> Path:
    ensure_run_dirs(paths)
    return paths.runs_dir / "sweeps" / sweep_id


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_run_json(paths: RepoPaths) -> Iterable[Path]:
    if not paths.runs_dir.exists():
        return []
    for p in (paths.runs_dir / "bench").glob("*/run.json"):
        yield p


def load_all_runs(paths: RepoPaths) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for p in iter_run_json(paths):
        try:
            rec = load_json(p)
            rec["_run_dir"] = str(p.parent)
            runs.append(rec)
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs


def load_sweep(paths: RepoPaths, sweep_id: str) -> dict[str, Any] | None:
    p = sweep_dir(paths, sweep_id) / "sweep.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def list_sweeps(paths: RepoPaths) -> list[Path]:
    """Return list of sweep directories."""
    d = paths.runs_dir / "sweeps"
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def list_runs(paths: RepoPaths) -> list[Path]:
    """Return list of bench run directories."""
    d = paths.runs_dir / "bench"
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def load_run_json(run_dir: Path) -> dict[str, Any] | None:
    """Load run.json from a run directory."""
    p = run_dir / "run.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def delete_run(run_dir: Path) -> None:
    """Delete a run directory."""
    import shutil
    if run_dir.exists():
        shutil.rmtree(run_dir)


def load_sweep_results(sweep_dir: Path) -> dict[str, Any] | None:
    """Load sweep results."""
    p = sweep_dir / "sweep_results.csv"
    if not p.exists():
        return None
    try:
        import pandas as pd
        return pd.read_csv(p)
    except Exception:
        return None


def delete_sweep(sweep_dir: Path) -> None:
    """Delete a sweep directory."""
    import shutil
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)


def avg_wall_time_for_model(paths: RepoPaths, model_file: str, *, limit: int = 10) -> float | None:
    """Return average wall_time_sec for successful bench runs for that model."""

    runs = [r for r in load_all_runs(paths) if r.get("model_file") == model_file and r.get("status") == "completed"]
    times: list[float] = []
    for r in runs[:limit]:
        t = r.get("wall_time_sec")
        if isinstance(t, (int, float)) and t > 0:
            times.append(float(t))
    if not times:
        return None
    return sum(times) / len(times)


# ---- Optimus runs ----

def optimus_dir(paths: RepoPaths, run_id: str) -> Path:
    ensure_run_dirs(paths)
    d = paths.runs_dir / "optimus"
    d.mkdir(parents=True, exist_ok=True)
    return d / run_id


def save_optimus_run(paths: RepoPaths, run_id: str, data: dict[str, Any]) -> Path:
    """Save an Optimus run result."""
    d = optimus_dir(paths, run_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "run.json"
    save_json(p, data)
    return d


def list_optimus_runs(paths: RepoPaths) -> list[Path]:
    """Return list of optimus run directories."""
    d = paths.runs_dir / "optimus"
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def load_all_optimus_runs(paths: RepoPaths) -> list[dict[str, Any]]:
    """Load all Optimus runs."""
    runs: list[dict[str, Any]] = []
    d = paths.runs_dir / "optimus"
    if not d.exists():
        return runs
    for p in d.glob("*/run.json"):
        try:
            rec = load_json(p)
            rec["_run_dir"] = str(p.parent)
            runs.append(rec)
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs


def delete_optimus_run(run_dir: Path) -> None:
    """Delete an optimus run directory."""
    import shutil
    if run_dir.exists():
        shutil.rmtree(run_dir)


# ---- Optimizer runs (custom orchestrator) ----

def optimizer_dir(paths: RepoPaths, run_id: str) -> Path:
    ensure_run_dirs(paths)
    d = paths.runs_dir / "optimizer"
    d.mkdir(parents=True, exist_ok=True)
    return d / run_id


def list_optimizer_runs(paths: RepoPaths) -> list[Path]:
    """Return list of optimizer run directories."""
    d = paths.runs_dir / "optimizer"
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def load_optimizer_summary(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def delete_optimizer_run(run_dir: Path) -> None:
    """Delete an optimizer run directory."""
    import shutil
    if run_dir.exists():
        shutil.rmtree(run_dir)


def delete_all_failed_runs(paths: RepoPaths, run_type: str = "all") -> int:
    """Delete all failed runs. Returns count of deleted runs."""
    import shutil
    count = 0
    
    if run_type in ("bench", "all"):
        for run_dir in list_runs(paths):
            data = load_run_json(run_dir)
            if data and data.get("status") == "failed":
                shutil.rmtree(run_dir)
                count += 1
    
    if run_type in ("optimus", "all"):
        for run_dir in list_optimus_runs(paths):
            data = load_run_json(run_dir)
            if data and data.get("status") == "failed":
                shutil.rmtree(run_dir)
                count += 1
    
    if run_type in ("sweeps", "all"):
        for sweep_d in list_sweeps(paths):
            p = sweep_d / "sweep.json"
            if p.exists():
                try:
                    data = load_json(p)
                    if data.get("status") == "failed":
                        shutil.rmtree(sweep_d)
                        count += 1
                except Exception:
                    pass
    
    return count
