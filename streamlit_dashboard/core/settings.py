from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Default external model directories (common locations)
DEFAULT_EXTERNAL_MODEL_DIRS = [
    Path.home() / ".lmstudio" / "models",
    # Ollama uses a blob format, not directly usable GGUF files
    # Path.home() / ".ollama" / "models",
]


@dataclass
class AppSettings:
    # Sweeps
    max_combos: int = 200

    # Docker / llama.cpp
    llama_bin_dir: str = "/app"
    docker_compose_file: str = "compose.yml"
    docker_service: str = "llama-cpp"

    # Defaults for llama-bench
    default_threads: int = 8
    default_batch: int = 2048
    default_ubatch: int = 512
    default_gpu_layers: int = 99
    default_no_kv_offload: int = 1  # -nkvo 1 => KV cache stays on GPU by défaut
    default_flash_attn: int = 1
    default_ctk: str = "q8_0"
    default_ctv: str = "q8_0"
    default_prompt: int = 512
    default_gen: int = 128
    default_repeats: int = 3

    # Scoring (sweep-only)
    default_w_tg: float = 0.7
    default_w_pp: float = 0.3
    
    # External model directories (in addition to ./models)
    external_model_dirs: list[Path] = field(default_factory=lambda: [
        d for d in DEFAULT_EXTERNAL_MODEL_DIRS if d.exists()
    ])


SESSION_SETTINGS_KEY = "_llama_dash_settings"
SESSION_SELECTED_MODEL_KEY = "selected_model"
SESSION_MODEL_META_CACHE_KEY = "_model_meta_cache"
SESSION_ACTIVE_TASK_KEY = "_active_task"
