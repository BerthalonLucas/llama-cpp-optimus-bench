from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Mapping of external host directories to container paths
EXTERNAL_DIR_MAPPINGS: dict[str, str] = {
    str(Path.home() / ".lmstudio" / "models"): "/external/lmstudio",
}


@dataclass(frozen=True)
class RepoPaths:
    """Canonical paths used by the dashboard.

    Runtime artifacts stay inside the repo:
    - ./models (GGUF files + HF cache)
    - ./runs   (bench + sweeps history)
    """

    repo_root: Path
    models_dir: Path
    runs_dir: Path
    compose_file: Path
    legacy_dir: Path

    @staticmethod
    def detect() -> "RepoPaths":
        # This file lives at: <repo_root>/streamlit_dashboard/core/paths.py
        repo_root = Path(__file__).resolve().parents[2]
        return RepoPaths(
            repo_root=repo_root,
            models_dir=repo_root / "models",
            runs_dir=repo_root / "runs",
            compose_file=repo_root / "compose.yml",
            legacy_dir=repo_root / "legacy",
        )

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "hf-cache").mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_dir.mkdir(parents=True, exist_ok=True)


def host_path_to_container_path(host_path: Path, paths: RepoPaths) -> str:
    """Convert a host file path to the corresponding container path.
    
    Args:
        host_path: Absolute path on the host system
        paths: RepoPaths instance
    
    Returns:
        Path string inside the container
    """
    host_path = host_path.resolve()
    host_str = str(host_path)
    
    # Check if it's in the local models directory
    try:
        rel = host_path.relative_to(paths.models_dir)
        return f"/models/{rel}"
    except ValueError:
        pass
    
    # Check external directories
    for host_dir, container_dir in EXTERNAL_DIR_MAPPINGS.items():
        if host_str.startswith(host_dir):
            rel = host_str[len(host_dir):].lstrip("/")
            return f"{container_dir}/{rel}"
    
    # Fallback: assume it's relative to models dir somehow
    return f"/models/{host_path.name}"


def get_model_source_label(model_path: Path, paths: RepoPaths) -> str:
    """Get a human-readable label for the model's source."""
    try:
        model_path.relative_to(paths.models_dir)
        return "local"
    except ValueError:
        pass
    
    for host_dir in EXTERNAL_DIR_MAPPINGS.keys():
        if str(model_path).startswith(host_dir):
            if "lmstudio" in host_dir.lower():
                return "🔷 LM Studio"
            return f"📁 {Path(host_dir).name}"
    
    return "external"
