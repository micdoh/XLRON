"""Subprocess management for XLRON GUI.

Launches training/eval processes detached so they survive browser close / SSH drop.
Tracks processes via a JSON registry in /tmp/xlron_runs/.
"""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUNS_DIR = Path("/tmp/xlron_runs")
REGISTRY_PATH = RUNS_DIR / "registry.json"


@dataclass
class RunInfo:
    run_id: str
    pid: int
    command: str
    log_path: str
    status: str = "running"  # running | finished | failed | stopped
    return_code: int | None = None
    started_at: float = field(default_factory=time.time)


def _ensure_dirs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict[str, dict]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_registry(registry: dict[str, dict]):
    _ensure_dirs()
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def launch_run(command: str) -> RunInfo:
    """Spawn a detached subprocess for the given command string."""
    _ensure_dirs()
    run_id = f"run_{int(time.time())}_{os.getpid()}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(run_dir / "output.log")

    log_file = open(log_path, "w")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.Popen(
        [sys.executable, "-u"] + command.split()[1:],  # -u for unbuffered, skip "python" prefix
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    info = RunInfo(
        run_id=run_id,
        pid=proc.pid,
        command=command,
        log_path=log_path,
    )

    registry = _load_registry()
    registry[run_id] = asdict(info)
    _save_registry(registry)
    return info


def stop_run(run_id: str) -> bool:
    """Send SIGTERM to the process group of a run. Returns True if signal sent."""
    registry = _load_registry()
    entry = registry.get(run_id)
    if not entry:
        return False
    try:
        os.killpg(os.getpgid(entry["pid"]), signal.SIGTERM)
        entry["status"] = "stopped"
        _save_registry(registry)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def tail_log(log_path: str, offset: int = 0) -> tuple[str, int]:
    """Read log file from offset. Returns (new_text, new_offset)."""
    try:
        with open(log_path, "r") as f:
            f.seek(offset)
            text = f.read()
            return text, f.tell()
    except (FileNotFoundError, OSError):
        return "", offset


def refresh_registry():
    """Update status of finished processes."""
    registry = _load_registry()
    changed = False
    for run_id, entry in registry.items():
        if entry["status"] == "running":
            try:
                os.kill(entry["pid"], 0)  # check if alive
            except ProcessLookupError:
                # Process gone — check log for clues
                entry["status"] = "finished"
                changed = True
            except PermissionError:
                pass  # still alive, just can't signal
    if changed:
        _save_registry(registry)


def get_active_runs() -> list[dict]:
    """Return list of currently running entries."""
    refresh_registry()
    registry = _load_registry()
    return [e for e in registry.values() if e["status"] == "running"]


def get_all_runs() -> list[dict]:
    """Return all runs, most recent first."""
    refresh_registry()
    registry = _load_registry()
    runs = list(registry.values())
    runs.sort(key=lambda r: r["started_at"], reverse=True)
    return runs
