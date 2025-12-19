from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque


@dataclass
class LiveSubprocess:
    """Run a subprocess and stream stdout/stderr lines.

    - No `shell=True`.
    - Creates a new process group so we can stop it reliably.
    """

    args: list[str]
    cwd: Path | None = None
    env: dict[str, str] | None = None
    log_dir: Path | None = None
    max_lines: int = 4000

    proc: subprocess.Popen | None = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _lines: Deque[str] = field(init=False)
    _stdout: list[str] = field(init=False, default_factory=list)
    _stderr: list[str] = field(init=False, default_factory=list)
    _q: queue.Queue[str] = field(init=False, default_factory=queue.Queue)
    _threads: list[threading.Thread] = field(init=False, default_factory=list)
    start_time: float = field(init=False, default_factory=time.time)

    def __post_init__(self) -> None:
        self._lines = deque(maxlen=self.max_lines)

    def start(self) -> None:
        if self.proc is not None:
            raise RuntimeError("Process already started")

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = (self.log_dir / "stdout.log") if self.log_dir else None
        stderr_path = (self.log_dir / "stderr.log") if self.log_dir else None

        self.proc = subprocess.Popen(
            self.args,
            cwd=str(self.cwd) if self.cwd else None,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )

        def _reader(stream, sink_list: list[str], prefix: str, out_path: Path | None) -> None:
            f_out = out_path.open("w", encoding="utf-8") if out_path else None
            try:
                assert stream is not None
                for line in iter(stream.readline, ""):
                    if line == "":
                        break
                    line = line.rstrip("\n")
                    stamp = time.strftime("%H:%M:%S")
                    msg = f"[{stamp}] {prefix} {line}"
                    with self._lock:
                        sink_list.append(line)
                        self._lines.append(msg)
                    self._q.put(msg)
                    if f_out is not None:
                        f_out.write(line + "\n")
                        f_out.flush()
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
                if f_out is not None:
                    f_out.close()

        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        t_out = threading.Thread(
            target=_reader,
            args=(self.proc.stdout, self._stdout, "STDOUT", stdout_path),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_reader,
            args=(self.proc.stderr, self._stderr, "STDERR", stderr_path),
            daemon=True,
        )
        t_out.start()
        t_err.start()
        self._threads.extend([t_out, t_err])

    def poll(self) -> int | None:
        if self.proc is None:
            return None
        return self.proc.poll()

    @property
    def returncode(self) -> int | None:
        return self.poll()

    def is_running(self) -> bool:
        rc = self.poll()
        return rc is None

    def terminate(self, *, grace_sec: float = 2.0) -> None:
        """Try to stop the process group (SIGINT then SIGKILL)."""

        if self.proc is None:
            return

        if self.proc.poll() is not None:
            return

        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass

        deadline = time.time() + grace_sec
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return
            time.sleep(0.1)

        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def wait(self, timeout: float | None = None) -> int | None:
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def read_lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def lines_text(self, last_n: int | None = None) -> str:
        """Return lines as a single string."""
        lines = self.read_lines()
        if last_n is not None:
            lines = lines[-last_n:]
        return "\n".join(lines)

    def read_new_lines(self, *, max_lines: int = 200) -> list[str]:
        out: list[str] = []
        for _ in range(max_lines):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def stdout_text(self) -> str:
        with self._lock:
            return "\n".join(self._stdout)

    def stderr_text(self) -> str:
        with self._lock:
            return "\n".join(self._stderr)

    def wall_time_sec(self) -> float:
        end = time.time()
        return max(0.0, end - self.start_time)
