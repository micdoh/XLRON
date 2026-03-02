"""Subprocess management for XLRON GUI.

Launches training/eval processes detached so they survive browser close / SSH drop.
Tracks processes via a JSON registry in /tmp/xlron_runs/.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
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


def _extract_flag(command: str, flag: str) -> str | None:
    """Extract `--flag=value` or `--flag value` from a raw CLI command."""
    pat_eq = re.compile(rf"--{re.escape(flag)}=([^\s]+)", flags=re.IGNORECASE)
    m_eq = pat_eq.search(command)
    if m_eq:
        return m_eq.group(1).strip()
    pat_sp = re.compile(rf"--{re.escape(flag)}\s+([^\s]+)", flags=re.IGNORECASE)
    m_sp = pat_sp.search(command)
    if m_sp:
        return m_sp.group(1).strip()
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _build_run_id(command: str) -> str:
    """Build human-readable run id: timestamp + mode + env + topology."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode = "train"
    if re.search(r"--eval_model\b", command, flags=re.IGNORECASE):
        mode = "model-eval"
    elif re.search(r"--eval_heuristic\b", command, flags=re.IGNORECASE):
        heur = _extract_flag(command, "path_heuristic")
        mode = f"heur-eval-{_slug(heur)}" if heur else "heur-eval"
    env = _slug(_extract_flag(command, "env_type") or "env")
    topo = _slug(_extract_flag(command, "topology_name") or "topology")
    return f"run_{ts}_{mode}_{env}_{topo}"


def launch_run(command: str) -> RunInfo:
    """Spawn a detached subprocess for the given command string."""
    cmd_lower = command.lower()
    is_eval = "--eval_heuristic" in cmd_lower or "--eval_model" in cmd_lower
    if is_eval:
        # GUI runs should always render to file (for in-GUI playback), not live popups.
        if "--render_eval_mode=" in cmd_lower:
            command = re.sub(
                r"--RENDER_EVAL_MODE=\S+",
                "--RENDER_EVAL_MODE=save",
                command,
                flags=re.IGNORECASE,
            )
        elif "--render_eval_mode" in cmd_lower:
            command = re.sub(
                r"--RENDER_EVAL_MODE(\s+\S+)?",
                "--RENDER_EVAL_MODE=save",
                command,
                flags=re.IGNORECASE,
            )

    _ensure_dirs()
    run_id = _build_run_id(command)
    # Ensure uniqueness if launched multiple times within the same second.
    base_run_id = run_id
    suffix = 1
    while (RUNS_DIR / run_id).exists():
        suffix += 1
        run_id = f"{base_run_id}_{suffix}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(run_dir / "output.log")

    log_file = open(log_path, "w")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # GUI is file-playback oriented for renders.
    env["MPLBACKEND"] = "Agg"
    env["XLRON_RUN_DIR"] = str(run_dir)
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
                # On macOS/Linux a zombie can still pass kill(pid, 0); treat it as finished.
                ps = subprocess.run(
                    ["ps", "-p", str(entry["pid"]), "-o", "stat="],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                stat = ps.stdout.strip()
                if (not stat) or stat.startswith("Z"):
                    entry["status"] = "finished"
                    changed = True
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
