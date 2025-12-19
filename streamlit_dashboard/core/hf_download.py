from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, hf_hub_download

from .paths import RepoPaths
from .validation import ValidationError, safe_filename, validate_hf_repo_filename, validate_quant, validate_repo_id


@dataclass(frozen=True)
class HFSpec:
    repo_id: str
    quant: str | None = None


def _log(log_fn: Callable[[str], None] | None, msg: str) -> None:
    if log_fn is not None:
        log_fn(msg)


def parse_hf_spec(raw: str) -> HFSpec:
    """Parse the UI input for a Hugging Face GGUF repo.

    Supported:
      - "hf.co/<org>/<repo>:<quant>"
      - "<org>/<repo>:<quant>"
      - "<org>/<repo>" (no quant)

    Notes:
      - This function only validates the *repo id* and optional *quant*. It does not
        validate file existence.
    """

    if raw is None:
        raise ValidationError("Empty Hugging Face input")

    s = str(raw).strip()
    if not s:
        raise ValidationError("Empty Hugging Face input")

    # Allow a minimal 'hf.co/' prefix.
    if s.startswith("hf.co/"):
        s = s[len("hf.co/") :]

    # Allow a full URL. Keep this intentionally strict.
    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :]

    if ":" in s:
        repo_part, quant_part = s.rsplit(":", 1)
        repo_id = validate_repo_id(repo_part)
        quant = validate_quant(quant_part)
        return HFSpec(repo_id=repo_id, quant=quant)

    repo_id = validate_repo_id(s)
    return HFSpec(repo_id=repo_id, quant=None)


def get_default_hf_token() -> str | None:
    # Common environment variables.
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or None
    )


def list_gguf_files(repo_id: str, *, token: str | None = None) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    ggufs.sort(key=str.lower)
    return ggufs


def match_gguf_by_quant(gguf_files: list[str], quant: str) -> tuple[str | None, list[str]]:
    """Return (best_match, candidates).

    Heuristic:
      1) Exact suffix match: ".<quant>.gguf" or "-<quant>.gguf"
      2) Substring match anywhere (case-insensitive)
    """

    q = quant.lower()

    exact: list[str] = []
    for f in gguf_files:
        fl = f.lower()
        if fl.endswith(f".{q}.gguf") or fl.endswith(f"-{q}.gguf"):
            exact.append(f)

    if exact:
        exact.sort(key=lambda x: (len(x), x.lower()))
        return exact[0], exact

    contains = [f for f in gguf_files if q in f.lower()]
    contains.sort(key=lambda x: (len(x), x.lower()))
    if contains:
        return contains[0], contains

    return None, []


def download_gguf(
    paths: RepoPaths,
    *,
    repo_id: str,
    filename: str,
    token: str | None = None,
    overwrite: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> Path:
    """Download a single GGUF file directly into ./models/<filename>.

    Downloads to a temp location first, then moves to final destination.
    No HF cache is kept - the file goes directly to ./models/.
    """
    import tempfile

    paths.ensure_dirs()

    hf_filename = validate_hf_repo_filename(filename)
    dest_name = safe_filename(Path(hf_filename).name)
    
    # Ensure filename ends with .gguf
    if not dest_name.lower().endswith('.gguf'):
        dest_name = dest_name + '.gguf'
    
    dest_path = paths.models_dir / dest_name

    if dest_path.exists() and not overwrite:
        raise ValidationError(
            f"File already exists: {dest_path.name}. Enable overwrite to replace it."
        )

    _log(log_fn, f"🔎 Downloading {repo_id}/{hf_filename} ...")
    _log(log_fn, f"📁 Destination: {dest_path}")

    # Use a temporary directory for download (will be cleaned up automatically)
    with tempfile.TemporaryDirectory(prefix="hf_download_") as tmp_dir:
        tmp_cache = Path(tmp_dir) / "cache"
        
        src_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=hf_filename,
            cache_dir=str(tmp_cache),
            token=token,
            force_download=overwrite,
        )

        _log(log_fn, f"📦 Downloaded to temp: {src_path}")
        _log(log_fn, f"📁 Moving to: {dest_path}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file (follow symlinks to get actual content)
        shutil.copy2(src_path, dest_path, follow_symlinks=True)
    
    # Temp directory is automatically cleaned up here

    _log(log_fn, "✅ Download complete")
    return dest_path
