from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_QUANT_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")


class ValidationError(ValueError):
    """User input failed strict validation."""


def parse_list_tokens(raw: str) -> list[str]:
    """Parse a user list input.

    Accepts:
      - "2,4,8,16"
      - "2 4 8 16"
      - "2\n4\n8\n16"

    Returns a list of non-empty string tokens.
    """

    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []

    parts = re.split(r"[\s,]+", s)
    return [p for p in (p.strip() for p in parts) if p]


def parse_int_list(raw: str, *, min_value: int | None = None, max_value: int | None = None) -> list[int]:
    vals: list[int] = []
    for tok in parse_list_tokens(raw):
        try:
            v = int(tok)
        except Exception as e:  # noqa: BLE001
            raise ValidationError(f"Invalid integer: {tok!r}") from e
        if min_value is not None and v < min_value:
            raise ValidationError(f"Value {v} < min {min_value}")
        if max_value is not None and v > max_value:
            raise ValidationError(f"Value {v} > max {max_value}")
        vals.append(v)
    return vals


def parse_float_list(raw: str, *, min_value: float | None = None, max_value: float | None = None) -> list[float]:
    vals: list[float] = []
    for tok in parse_list_tokens(raw):
        try:
            v = float(tok)
        except Exception as e:  # noqa: BLE001
            raise ValidationError(f"Invalid float: {tok!r}") from e
        if min_value is not None and v < min_value:
            raise ValidationError(f"Value {v} < min {min_value}")
        if max_value is not None and v > max_value:
            raise ValidationError(f"Value {v} > max {max_value}")
        vals.append(v)
    return vals


def parse_bool01_list(raw: str) -> list[int]:
    vals = parse_int_list(raw, min_value=0, max_value=1)
    out: list[int] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def validate_repo_id(repo_id: str) -> str:
    rid = repo_id.strip()
    if not _REPO_ID_RE.match(rid):
        raise ValidationError(
            "Invalid Hugging Face repo id. Expected format 'org/repo' (letters/digits/._- only)."
        )
    return rid


def validate_quant(quant: str) -> str:
    q = quant.strip()
    if not _QUANT_RE.match(q):
        raise ValidationError(
            "Invalid quant. Expected something like 'Q4_K_M' (letters/digits/underscore)."
        )
    return q


def ensure_within_dir(path: Path, parent_dir: Path) -> Path:
    """Ensure *path* is located within *parent_dir* (prevents path traversal)."""

    p = path.resolve()
    root = parent_dir.resolve()
    try:
        p.relative_to(root)
    except Exception as e:  # noqa: BLE001
        raise ValidationError(f"Path {p} is outside of {root}") from e
    return p


def safe_filename(name: str) -> str:
    """Return a filesystem-safe filename (no path separators)."""

    base = Path(name).name
    if base in {"", ".", ".."}:
        raise ValidationError("Invalid filename")
    if "/" in base or "\\" in base:
        raise ValidationError("Invalid filename")
    return base


def validate_choice(value: str, *, allowed: Iterable[str], label: str = "value") -> str:
    v = value.strip()
    allowed_set = set(allowed)
    if v not in allowed_set:
        raise ValidationError(f"Invalid {label}: {v!r}. Allowed: {sorted(allowed_set)}")
    return v


def validate_hf_repo_filename(filename: str) -> str:
    """Validate a Hugging Face repo file path.

    We allow sub-directories (e.g. "GGUF/model.Q4_K_M.gguf") but forbid:
    - absolute paths
    - path traversal ("..")
    - backslashes

    This function is intentionally strict because the filename will later
    be used as a subprocess argument and a filesystem destination (basename).
    """
    if not filename or "\x00" in filename:
        raise ValidationError("Invalid filename")
    if "\\" in filename:
        raise ValidationError("Backslashes are not allowed")
    if filename.startswith("/"):
        raise ValidationError("Absolute paths are not allowed")
    # Normalize separators to forward slashes only
    parts = filename.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValidationError("Invalid path segments in filename")
    if not filename.lower().endswith(".gguf"):
        raise ValidationError("Selected file is not a .gguf")
    return filename
