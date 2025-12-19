"""Detect specific errors like OOM from llama.cpp output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    """Types of errors we can detect."""
    NONE = "none"
    OOM_CUDA = "oom_cuda"       # CUDA out of memory
    OOM_SYSTEM = "oom_system"   # System RAM out of memory
    LOAD_FAILED = "load_failed" # Generic model load failure
    CUDA_ERROR = "cuda_error"   # Other CUDA errors
    UNKNOWN = "unknown"         # Failed but unknown reason


@dataclass
class ErrorInfo:
    """Information about a detected error."""
    error_type: ErrorType
    message: str
    details: str | None = None
    
    @property
    def is_oom(self) -> bool:
        return self.error_type in (ErrorType.OOM_CUDA, ErrorType.OOM_SYSTEM)
    
    @property
    def is_recoverable(self) -> bool:
        """OOM errors are recoverable - we can try with different params."""
        return self.error_type in (
            ErrorType.OOM_CUDA, 
            ErrorType.OOM_SYSTEM,
            ErrorType.LOAD_FAILED,  # Might be OOM in disguise
        )
    
    @property
    def user_message(self) -> str:
        """User-friendly message."""
        if self.error_type == ErrorType.OOM_CUDA:
            return "⚠️ GPU Out of Memory (CUDA OOM) - Essayez de réduire ngl ou augmenter ncmoe"
        elif self.error_type == ErrorType.OOM_SYSTEM:
            return "⚠️ System RAM insuffisante - Essayez de réduire batch/ubatch"
        elif self.error_type == ErrorType.LOAD_FAILED:
            return "❌ Échec du chargement du modèle (possiblement OOM)"
        elif self.error_type == ErrorType.CUDA_ERROR:
            return f"❌ Erreur CUDA: {self.message}"
        elif self.error_type == ErrorType.UNKNOWN:
            return f"❌ Erreur inconnue: {self.message}"
        return "✅ Pas d'erreur détectée"
    
    @property
    def short_label(self) -> str:
        """Short label for tables/lists."""
        labels = {
            ErrorType.NONE: "OK",
            ErrorType.OOM_CUDA: "OOM_GPU",
            ErrorType.OOM_SYSTEM: "OOM_RAM", 
            ErrorType.LOAD_FAILED: "LOAD_FAIL",
            ErrorType.CUDA_ERROR: "CUDA_ERR",
            ErrorType.UNKNOWN: "ERROR",
        }
        return labels.get(self.error_type, "?")


# Patterns to detect CUDA OOM
OOM_CUDA_PATTERNS = [
    # Direct CUDA OOM messages
    r"out of memory",
    r"CUDA_ERROR_OUT_OF_MEMORY",
    r"cudaErrorMemoryAllocation",
    r"cudaMalloc failed",
    r"cuMemCreate failed",
    r"failed to allocate.*on device",
    r"can't allocate.*Bytes.*on device",
    r"CUBLAS_STATUS_ALLOC_FAILED",
    # HIP/ROCm equivalent
    r"hipErrorOutOfMemory",
    r"HIP out of memory",
    # GGML allocation failures that often mean OOM
    r"ggml_backend_cuda_buffer_type_alloc_buffer.*failed",
    r"ggml_cuda_device_malloc.*failed",
]

# Patterns for system RAM OOM
OOM_SYSTEM_PATTERNS = [
    r"failed to allocate.*MB",
    r"std::bad_alloc",
    r"Cannot allocate memory",
    r"ggml_malloc.*failed",
    r"ggml_calloc.*failed",
]

# Patterns for generic load failures
LOAD_FAILED_PATTERNS = [
    r"failed to load model",
    r"error: failed to load model",
    r"error loading model",
    r"unable to load model",
    # Common llama.cpp message when the model loads but context init fails
    r"failed to create context",
]

# Other CUDA errors
CUDA_ERROR_PATTERNS = [
    r"CUDA error",
    r"CUBLAS error",
    r"cuBLAS error",
    r"GGML_CUDA.*error",
    r"cudaGetLastError",
]


def _match_patterns(text: str, patterns: list[str]) -> str | None:
    """Check if any pattern matches, return the matched line."""
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            # Try to extract the full line containing the match
            start = text_lower.rfind('\n', 0, match.start()) + 1
            end = text_lower.find('\n', match.end())
            if end == -1:
                end = len(text)
            return text[start:end].strip()
    return None


def detect_error(stdout: str, stderr: str, exit_code: int) -> ErrorInfo:
    """Analyze stdout/stderr to detect specific errors.
    
    Args:
        stdout: Standard output from llama-bench
        stderr: Standard error from llama-bench  
        exit_code: Process exit code
        
    Returns:
        ErrorInfo with detected error type and details
    """
    if exit_code == 0:
        return ErrorInfo(ErrorType.NONE, "Success")
    
    # Combine stdout and stderr for analysis (stderr is more likely to have errors)
    combined = f"{stderr}\n{stdout}"
    
    # Check for CUDA OOM first (most specific)
    match = _match_patterns(combined, OOM_CUDA_PATTERNS)
    if match:
        return ErrorInfo(
            ErrorType.OOM_CUDA,
            "CUDA out of memory",
            details=match
        )
    
    # Check for system OOM
    match = _match_patterns(combined, OOM_SYSTEM_PATTERNS)
    if match:
        return ErrorInfo(
            ErrorType.OOM_SYSTEM,
            "System memory allocation failed",
            details=match
        )
    
    # Check for generic load failures (might be OOM in disguise)
    match = _match_patterns(combined, LOAD_FAILED_PATTERNS)
    if match:
        return ErrorInfo(
            ErrorType.LOAD_FAILED,
            "Model load failed",
            details=match
        )
    
    # Check for other CUDA errors
    match = _match_patterns(combined, CUDA_ERROR_PATTERNS)
    if match:
        return ErrorInfo(
            ErrorType.CUDA_ERROR,
            "CUDA error",
            details=match
        )
    
    # Unknown failure
    # Try to extract last meaningful line from stderr
    last_lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    last_msg = last_lines[-1] if last_lines else f"Exit code {exit_code}"
    
    return ErrorInfo(
        ErrorType.UNKNOWN,
        last_msg,
        details=None
    )


def analyze_sweep_failures(results: list[dict]) -> dict:
    """Analyze failures in sweep results to provide recommendations.
    
    Returns a summary dict with:
    - total_runs
    - successful_runs  
    - oom_runs (includes load_failed which is often OOM in disguise)
    - other_failures
    - recommendation
    """
    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "completed")
    
    # Explicit OOM
    oom_explicit = sum(1 for r in results if r.get("error_type") in ("oom_cuda", "oom_system"))
    
    # load_failed is very often OOM in disguise with llama-bench
    # (llama-bench doesn't propagate the actual CUDA error message)
    load_fail = sum(1 for r in results if r.get("error_type") == "load_failed")
    
    # Count load_failed as "probable OOM" for UI purposes
    oom_probable = oom_explicit + load_fail
    
    other_failures = total - successful - oom_explicit - load_fail
    
    recommendation = None
    if oom_probable > 0:
        if successful == 0:
            recommendation = "⚠️ Tous les runs ont échoué. Le modèle est probablement trop gros pour cette config. Augmentez ncmoe ou réduisez ngl."
        elif oom_probable > total / 2:
            recommendation = f"⚠️ {oom_probable} échecs (probables OOM). Les configs avec ncmoe plus élevé ou ngl plus bas devraient fonctionner."
        else:
            recommendation = f"💡 {oom_probable} échecs probables OOM. Regardez les configs qui ont réussi pour trouver la limite."
    elif other_failures > 0 and successful == 0:
        recommendation = "❌ Tous les runs ont échoué. Vérifiez le modèle et la configuration de base."
    elif successful > 0 and successful < total:
        recommendation = f"✅ {successful}/{total} runs réussis. Analysez les configs gagnantes."
    
    return {
        "total_runs": total,
        "successful_runs": successful,
        "oom_runs": oom_probable,  # Include load_failed as probable OOM
        "load_failures": load_fail,
        "other_failures": other_failures,
        "recommendation": recommendation,
    }
