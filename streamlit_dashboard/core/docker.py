from __future__ import annotations

import shutil
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from .live_process import LiveSubprocess
from .paths import RepoPaths
from .settings import AppSettings


class DockerNotFoundError(RuntimeError):
    pass


def _docker_cmd() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise DockerNotFoundError(
            "Docker binary not found. Install docker and ensure 'docker' is on PATH."
        )
    return docker


def check_container_running(container_name: str) -> bool:
    """Check if a Docker container is running."""
    try:
        docker = _docker_cmd()
        result = subprocess.run(
            [docker, "compose", "ps", "-q", container_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def start_container(container_name: str = "llama-cpp", cwd: Path | None = None) -> tuple[bool, str]:
    """Start a Docker container using docker compose up -d.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        docker = _docker_cmd()
        work_dir = cwd or Path.cwd()
        
        result = subprocess.run(
            [docker, "compose", "up", "-d", container_name],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(work_dir)
        )
        
        if result.returncode == 0:
            return True, f"Container '{container_name}' started successfully"
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Failed to start container: {error_msg}"
    except subprocess.TimeoutExpired:
        return False, "Timeout starting container (120s)"
    except Exception as e:
        return False, f"Error starting container: {str(e)}"


def compose_base(paths: RepoPaths, settings: AppSettings) -> list[str]:
    docker = _docker_cmd()
    compose_file = paths.repo_root / settings.docker_compose_file
    return [docker, "compose", "-f", str(compose_file)]


def compose_build(paths: RepoPaths, settings: AppSettings) -> LiveSubprocess:
    args = compose_base(paths, settings) + ["build", "--pull", settings.docker_service]
    p = LiveSubprocess(args=args, cwd=paths.repo_root, log_dir=None)
    p.start()
    return p


def service_up(paths: RepoPaths, settings: AppSettings) -> tuple[bool, str]:
    """docker compose up -d for the configured service."""
    args = compose_base(paths, settings) + ["up", "-d", settings.docker_service]
    try:
        result = subprocess.run(args, cwd=paths.repo_root, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, f"Service '{settings.docker_service}' démarré"
        return False, result.stderr or result.stdout or "Erreur inconnue"
    except subprocess.TimeoutExpired:
        return False, "Timeout (120s) au démarrage du service"
    except Exception as e:
        return False, str(e)


def service_stop(paths: RepoPaths, settings: AppSettings) -> tuple[bool, str]:
    """docker compose stop for the configured service."""
    args = compose_base(paths, settings) + ["stop", settings.docker_service]
    try:
        result = subprocess.run(args, cwd=paths.repo_root, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True, f"Service '{settings.docker_service}' arrêté"
        return False, result.stderr or result.stdout or "Erreur inconnue"
    except subprocess.TimeoutExpired:
        return False, "Timeout (60s) à l'arrêt du service"
    except Exception as e:
        return False, str(e)


def service_restart(paths: RepoPaths, settings: AppSettings) -> tuple[bool, str]:
    """docker compose restart for the configured service."""
    args = compose_base(paths, settings) + ["restart", settings.docker_service]
    try:
        result = subprocess.run(args, cwd=paths.repo_root, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True, f"Service '{settings.docker_service}' redémarré"
        return False, result.stderr or result.stdout or "Erreur inconnue"
    except subprocess.TimeoutExpired:
        return False, "Timeout (120s) au redémarrage du service"
    except Exception as e:
        return False, str(e)


def compose_run(
    paths: RepoPaths,
    settings: AppSettings,
    *,
    service: str | None = None,
    command: list[str],
    log_dir: Path | None = None,
) -> LiveSubprocess:
    """Execute a command in the running container using docker compose exec."""
    svc = service or settings.docker_service
    # Use exec instead of run to use the already running container
    # -T disables pseudo-tty allocation (needed for non-interactive use)
    args = compose_base(paths, settings) + ["exec", "-T", svc] + command
    p = LiveSubprocess(args=args, cwd=paths.repo_root, log_dir=log_dir)
    p.start()
    return p


def kill_in_container(
    paths: RepoPaths,
    settings: AppSettings,
    *,
    patterns: Iterable[str],
    service: str | None = None,
) -> None:
    """Best-effort pkill inside the container for given patterns (safety scoped to known binaries)."""
    svc = service or settings.docker_service
    safe_patterns = [p for p in patterns if p]
    if not safe_patterns:
        return

    # Use pkill/pgrep if available, otherwise fall back to parsing `ps` output.
    # We run each pattern independently (no short-circuit) to ensure all targets are killed.
    script_parts: list[str] = ["set +e"]
    for pat in safe_patterns:
        qpat = shlex.quote(str(pat))
        script_parts.append(
            "if command -v pkill >/dev/null 2>&1; then "
            f"pkill -f {qpat} >/dev/null 2>&1 || true; "
            "elif command -v pgrep >/dev/null 2>&1; then "
            f"for pid in $(pgrep -f {qpat} 2>/dev/null); do kill -9 \"$pid\" 2>/dev/null || true; done; "
            "else "
            f"ps ax -o pid= -o command= 2>/dev/null | grep -F {qpat} | grep -v grep | "
            "awk '{print $1}' | while read -r pid; do kill -9 \"$pid\" 2>/dev/null || true; done; "
            "fi"
        )
    script = "; ".join(script_parts) + "; true"
    args = compose_base(paths, settings) + ["exec", "-T", svc, "sh", "-c", script]
    try:
        subprocess.run(args, cwd=paths.repo_root, timeout=10)
    except Exception:
        pass
