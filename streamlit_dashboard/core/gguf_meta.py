from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import RepoPaths


# GGUF value types (per gguf spec)
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


_TYPE_SIZES: dict[int, int] = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


@dataclass
class GGUFMeta:
    path: Path
    architecture: str | None
    is_moe: bool
    expert_count: int | None
    expert_used_count: int | None
    layer_count: int | None  # n_layer / block_count - used for ncmoe bounds
    context_length: int | None  # max context size supported by the model
    extra: dict[str, Any]


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF")
    return b


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_i32(f) -> int:
    return struct.unpack("<i", _read_exact(f, 4))[0]


def _read_i64(f) -> int:
    return struct.unpack("<q", _read_exact(f, 8))[0]


def _read_f32(f) -> float:
    return struct.unpack("<f", _read_exact(f, 4))[0]


def _read_f64(f) -> float:
    return struct.unpack("<d", _read_exact(f, 8))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    if n == 0:
        return ""
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_str(f) -> None:
    n = _read_u64(f)
    if n:
        f.seek(n, 1)


def _skip_value(f, vtype: int) -> None:
    if vtype == GGUF_TYPE_STRING:
        _skip_str(f)
        return

    if vtype == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        n = _read_u64(f)
        if elem_type == GGUF_TYPE_STRING:
            for _ in range(n):
                _skip_str(f)
            return

        size = _TYPE_SIZES.get(elem_type)
        if size is None:
            raise ValueError(f"Unsupported GGUF array element type: {elem_type}")
        f.seek(size * n, 1)
        return

    size = _TYPE_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"Unsupported GGUF value type: {vtype}")
    f.seek(size, 1)


def _read_value(f, vtype: int):
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack("<B", _read_exact(f, 1))[0]
    if vtype == GGUF_TYPE_INT8:
        return struct.unpack("<b", _read_exact(f, 1))[0]
    if vtype == GGUF_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(f, 2))[0]
    if vtype == GGUF_TYPE_INT16:
        return struct.unpack("<h", _read_exact(f, 2))[0]
    if vtype == GGUF_TYPE_UINT32:
        return _read_u32(f)
    if vtype == GGUF_TYPE_INT32:
        return _read_i32(f)
    if vtype == GGUF_TYPE_UINT64:
        return _read_u64(f)
    if vtype == GGUF_TYPE_INT64:
        return _read_i64(f)
    if vtype == GGUF_TYPE_FLOAT32:
        return _read_f32(f)
    if vtype == GGUF_TYPE_FLOAT64:
        return _read_f64(f)
    if vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<?", _read_exact(f, 1))[0]
    if vtype == GGUF_TYPE_STRING:
        return _read_str(f)
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        n = _read_u64(f)
        if elem_type == GGUF_TYPE_STRING:
            return [_read_str(f) for _ in range(n)]
        size = _TYPE_SIZES.get(elem_type)
        if size is None:
            raise ValueError(f"Unsupported GGUF array element type: {elem_type}")
        # Avoid loading huge arrays (tokenizer). Skip.
        f.seek(size * n, 1)
        return {"__gguf_array__": True, "elem_type": elem_type, "n": n}

    raise ValueError(f"Unsupported GGUF value type: {vtype}")


def read_gguf_meta_fast(path: Path, *, max_kv: int | None = None) -> GGUFMeta:
    """Read a small subset of GGUF metadata.

    We focus on MoE detection (expert_count), layer_count (for ncmoe bounds),
    and a couple of general keys.
    """

    architecture: str | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    layer_count: int | None = None
    context_length: int | None = None
    extra: dict[str, Any] = {}

    with path.open("rb") as f:
        magic = _read_exact(f, 4)
        if magic != b"GGUF":
            raise ValueError("Not a GGUF file (missing GGUF magic)")
        _version = _read_u32(f)
        _n_tensors = _read_u64(f)
        n_kv = _read_u64(f)

        if max_kv is None:
            max_kv = n_kv

        for _ in range(min(n_kv, max_kv)):
            key = _read_str(f)
            vtype = _read_u32(f)

            key_l = key.lower()
            interested = (
                key_l in {"general.architecture", "general.name", "general.description"}
                or key_l.endswith(".expert_count")
                or key_l.endswith(".expert_used_count")
                or key_l.endswith(".context_length")  # Matches both <arch>.context_length and .context_length
                or key_l.endswith(".block_count")
            )

            if interested:
                val = _read_value(f, vtype)
                extra[key] = val

                if key_l == "general.architecture" and isinstance(val, str):
                    architecture = val

                if key_l.endswith(".expert_count") and isinstance(val, (int, bool)):
                    expert_count = int(val)

                if key_l.endswith(".expert_used_count") and isinstance(val, (int, bool)):
                    expert_used_count = int(val)

                if key_l.endswith(".block_count") and isinstance(val, (int, bool)):
                    layer_count = int(val)

                if key_l.endswith(".context_length") and isinstance(val, (int, bool)):
                    context_length = int(val)

                # Early exit: once we have all key info, we can stop.
                if (
                    architecture is not None
                    and expert_count is not None
                    and layer_count is not None
                ):
                    break
            else:
                _skip_value(f, vtype)

    is_moe = expert_count is not None and expert_count > 0
    return GGUFMeta(
        path=path,
        architecture=architecture,
        is_moe=is_moe,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        layer_count=layer_count,
        context_length=context_length,
        extra=extra,
    )


def _cache_path(paths: RepoPaths) -> Path:
    return paths.runs_dir / ".cache" / "gguf_meta.json"


def load_meta_cache(paths: RepoPaths) -> dict[str, Any]:
    p = _cache_path(paths)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_meta_cache(paths: RepoPaths, cache: dict[str, Any]) -> None:
    p = _cache_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def get_gguf_meta(paths: RepoPaths, gguf_path: Path) -> GGUFMeta:
    """Cached GGUF metadata reader.

    Cache key is (resolved path, mtime, size).
    """

    paths.ensure_dirs()

    gguf_path = gguf_path.resolve()
    st = gguf_path.stat()
    cache = load_meta_cache(paths)

    key = str(gguf_path)
    entry = cache.get(key)
    if isinstance(entry, dict):
        meta = entry.get("meta", {})
        # Validate cache: must have same mtime/size AND include context_length field
        # (older cache versions may not have context_length, force re-read in that case)
        is_valid = (
            entry.get("mtime") == st.st_mtime 
            and entry.get("size") == st.st_size
            and "context_length" in meta  # Force re-read if context_length missing from old cache
        )
        if is_valid:
            return GGUFMeta(
                path=gguf_path,
                architecture=meta.get("architecture"),
                is_moe=bool(meta.get("is_moe")),
                expert_count=meta.get("expert_count"),
                expert_used_count=meta.get("expert_used_count"),
                layer_count=meta.get("layer_count"),
                context_length=meta.get("context_length"),
                extra=dict(meta.get("extra") or {}),
            )

    # Cache miss or invalid: re-read from file
    meta_obj = read_gguf_meta_fast(gguf_path)

    cache[key] = {
        "mtime": st.st_mtime,
        "size": st.st_size,
        "meta": {
            "architecture": meta_obj.architecture,
            "is_moe": meta_obj.is_moe,
            "expert_count": meta_obj.expert_count,
            "expert_used_count": meta_obj.expert_used_count,
            "layer_count": meta_obj.layer_count,
            "context_length": meta_obj.context_length,
            "extra": meta_obj.extra,
        },
    }
    save_meta_cache(paths, cache)
    return meta_obj


def get_cached_meta(paths: RepoPaths, gguf_path: Path) -> GGUFMeta:
    """Alias for get_gguf_meta for backward compatibility."""
    return get_gguf_meta(paths, gguf_path)
