"""Measure what the routing tables cost each compiled program (issue #80).

Runs ``xlron.train.train`` unchanged and records, for every program it compiles
(warmup, training/eval, eval-during-training): trace+lower time, compile time,
serialized executable size, XLA's buffer estimate, and process RSS around the
compile. Also reports peak host RSS, peak device memory and the train/eval fps
that train.py prints.

Usage (all train.py flags pass through):
  python -m experimental.benchmarks.route_storage_bench --bench_out=out.json \
      [--bench_hoist_constants] -- --topology_name=usa100_directed ...

--bench_hoist_constants turns on ``jax_use_simplified_jaxpr_constants``, which
passes closed-over device arrays to each executable as arguments instead of
embedding them as HLO literals (JAX's "constants as arguments" mode).
"""

import json
import os
import re
import resource
import runpy
import sys
import threading
import time

import psutil


def _rss():
    return psutil.Process().memory_info().rss


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        own, train_args = argv[:split], argv[split + 1 :]
    else:
        own, train_args = [], argv
    out_path = None
    hoist = False
    for a in own:
        if a.startswith("--bench_out="):
            out_path = a.split("=", 1)[1]
        elif a == "--bench_hoist_constants":
            hoist = True
        else:
            raise SystemExit(f"unknown bench flag {a}")

    # JAX reads this once, when jax.Array is registered as a literal type, so it has
    # to be set before jax is imported (xlron/__init__.py defaults it to on).
    os.environ["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] = "1" if hoist else "0"
    import io

    import jax
    import jax.stages

    programs = []
    peak = {"device": 0, "rss": 0}

    def watch():
        while True:
            try:
                s = jax.devices()[0].memory_stats() or {}
                peak["device"] = max(peak["device"], s.get("peak_bytes_in_use", 0))
            except Exception:
                pass
            peak["rss"] = max(peak["rss"], _rss())
            time.sleep(0.5)

    # Start sampling at the first compile: touching jax.devices() earlier would
    # initialise the backend before train.py has set XLA_FLAGS.
    watcher = threading.Thread(target=watch, daemon=True)

    orig_compile = jax.stages.Lowered.compile

    def timed_compile(self, *args, **kwargs):
        if not watcher.is_alive():
            watcher.start()
        pre = {}
        try:
            buf = io.BytesIO()
            self.compiler_ir("stablehlo").operation.write_bytecode(buf)
            pre["stablehlo_bytes"] = buf.tell()
        except Exception as e:  # noqa: BLE001
            pre["stablehlo_error"] = repr(e)
        consts = getattr(self._lowering, "const_args", None) or []
        pre["hoisted_const_args"] = len(consts)
        pre["hoisted_const_bytes"] = int(sum(getattr(c, "nbytes", 0) for c in consts))
        rss0 = _rss()
        t0 = time.perf_counter()
        compiled = orig_compile(self, *args, **kwargs)
        elapsed = time.perf_counter() - t0
        rec = {"compile_s": elapsed, "rss_before": rss0, "rss_after": _rss(), **pre}
        try:
            from jax.experimental.serialize_executable import serialize

            payload, _, _ = serialize(compiled)
            rec["serialized_bytes"] = len(payload)
        except Exception as e:  # noqa: BLE001
            rec["serialized_error"] = repr(e)
        try:
            ma = compiled.memory_analysis()
            for f in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
            ):
                rec[f] = getattr(ma, f, None)
        except Exception as e:  # noqa: BLE001
            rec["memory_analysis_error"] = repr(e)
        programs.append(rec)
        print(f"BENCH compiled program #{len(programs)}: {rec}", flush=True)
        return compiled

    jax.stages.Lowered.compile = timed_compile

    # Tee stdout so we can parse train.py's timing lines.
    captured = []
    real_write = sys.stdout.write

    def tee(s):
        captured.append(s)
        return real_write(s)

    sys.stdout.write = tee
    t_start = time.perf_counter()
    sys.argv = ["xlron.train.train"] + train_args
    status = "ok"
    try:
        runpy.run_module("xlron.train.train", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        if e.code not in (None, 0):
            status = f"exit {e.code}"
    except Exception as e:  # noqa: BLE001
        status = f"error {e!r}"
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.write = real_write
    wall = time.perf_counter() - t_start

    text = "".join(captured)
    setup = {m.group(1): float(m.group(2)) for m in re.finditer(r"SETUP (\S+): ([\d.]+)s", text)}
    fps = [float(x) for x in re.findall(r"^FPS:\s+([\d.eE+]+)", text, re.MULTILINE)]
    metrics = {}
    for name in ("service_blocking_probability", "bitrate_blocking_probability"):
        m = re.search(rf"{name}: ([\d.eE+-]+)", text)
        if m:
            metrics[name] = float(m.group(1))
    result = {
        "status": status,
        "hoist_constants": hoist,
        "jax_simplified_constants": bool(jax.config.jax_use_simplified_jaxpr_constants),
        "train_args": train_args,
        "wall_s": wall,
        "setup": setup,
        "fps_reported": fps,
        "metrics": metrics,
        "programs": programs,
        "peak_device_bytes": peak["device"],
        "peak_rss_bytes": max(peak["rss"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "git": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
    }
    print("BENCH_RESULT " + json.dumps(result), flush=True)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
    sys.stdout.flush()
    os._exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
