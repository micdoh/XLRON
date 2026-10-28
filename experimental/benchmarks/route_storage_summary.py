"""Tabulate route_storage_bench.py results: one row per run, markdown output.

Usage: python -m experimental.benchmarks.route_storage_summary results/*.json
"""

import json
import re
import sys
from pathlib import Path


def gb(x):
    return f"{x / 1e9:.2f}"


def row(path):
    r = json.loads(Path(path).read_text())
    setup = r.get("setup", {})
    progs = r.get("programs", [])
    compile_s = sum(p["compile_s"] for p in progs)
    lower_s = setup.get("warmup_trace_lower", 0) + setup.get("training_trace_lower", 0)
    hlo = sum(p.get("stablehlo_bytes", 0) for p in progs)
    code = sum(p.get("generated_code_size_in_bytes") or 0 for p in progs)
    # Runs before the FPS regex fix recorded garbage; read the run's log instead.
    log = Path(path).parent.parent / "logs" / (Path(path).stem + ".log")
    found = re.findall(r"^FPS:\s+([\d.eE+]+)", log.read_text(), re.M) if log.exists() else []
    fps = float(found[-1]) if found else float("nan")
    return [
        Path(path).stem,
        "hoisted" if r.get("jax_simplified_constants") else "embedded",
        r["status"][:20],
        f"{setup.get('environment', 0):.1f}",
        f"{lower_s:.1f}",
        f"{compile_s:.1f}",
        f"{hlo / 1e6:.1f}",
        f"{code / 1e6:.1f}",
        gb(r.get("peak_rss_bytes", 0)),
        gb(r.get("peak_device_bytes", 0)),
        f"{fps:,.0f}",
    ]


def main():
    header = [
        "run",
        "constants",
        "status",
        "env setup s",
        "trace+lower s",
        "XLA compile s",
        "StableHLO MB",
        "GPU code MB",
        "peak RSS GB",
        "peak device GB",
        "steps/s",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for path in sorted(sys.argv[1:]):
        try:
            print("| " + " | ".join(row(path)) + " |")
        except Exception as e:  # noqa: BLE001
            print(f"| {Path(path).stem} | unreadable: {e!r} |")


if __name__ == "__main__":
    main()
