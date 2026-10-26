"""Opt-in host diagnostics for experimental continuous-operation GN campaigns."""

import json
from dataclasses import fields
from pathlib import Path

import numpy as np


def summarize_replicas(record):
    """Expose catastrophic tails without dropping them from reported means."""
    blocking = np.asarray(record["service_blocking_probability"], dtype=float)
    snr = np.asarray(record.get("snr_blocking_probability", np.zeros_like(blocking)), dtype=float)
    high_snr = snr > 0.1
    return {
        "replica_count": int(blocking.size),
        "service_blocking_mean": float(np.mean(blocking)),
        "service_blocking_median": float(np.median(blocking)),
        "service_blocking_min": float(np.min(blocking)),
        "service_blocking_max": float(np.max(blocking)),
        "high_snr_blocking_count": int(np.count_nonzero(high_snr)),
        "high_snr_share_of_snr_blocks": float(
            np.sum(snr[high_snr]) / np.sum(snr) if np.sum(snr) > 0 else 0.0
        ),
    }


def per_env_record(metrics, num_envs, measured_steps):
    """Extract cumulative post-warmup counters without averaging away replicas."""
    result = {"measured_steps_per_env": measured_steps}
    for key in (
        "accepted_services",
        "accepted_bitrate",
        "total_bitrate",
        "blocked_spectrum",
        "blocked_snr",
        "blocked_power",
    ):
        if key in metrics:
            values = np.asarray(metrics[key])
            result[key] = values.reshape(-1, num_envs)[-1].tolist()
    accepted = np.asarray(result["accepted_services"])
    result["service_blocking_probability"] = (1 - accepted / measured_steps).tolist()
    total_br = np.asarray(result["total_bitrate"])
    result["bitrate_blocking_probability"] = (
        1 - np.asarray(result["accepted_bitrate"]) / np.maximum(total_br, 1)
    ).tolist()
    for cause in ("spectrum", "snr", "power"):
        key = f"blocked_{cause}"
        if key in result:
            result[f"{cause}_blocking_probability"] = (
                np.asarray(result[key]) / measured_steps
            ).tolist()
    return result


def existing_path_health(state, params):
    """Fresh physical health of one replica; predicates repeat paths over slots/hops."""
    import jax.numpy as jnp

    from xlron.environments import env_funcs

    fresh = state.replace(link_snr_array=env_funcs.get_snr_link_array(state, params))
    snr = env_funcs.get_lightpath_snr(fresh, params)
    required = env_funcs.get_required_snr_se_kurtosis_array(
        state.modulation_format_index_array, 2, params
    )
    occupied = state.channel_centre_bw_array > 0
    return {
        "active_link_slot_predicates": jnp.count_nonzero(occupied),
        "failed_link_slot_predicates": jnp.count_nonzero(occupied & ~(snr >= required)),
        "min_margin_db": jnp.min(jnp.where(occupied, snr - required, jnp.inf)),
    }


def install_diagnostics(train_utils, output_path, *, physical_audit=None):
    """Wrap host logging; keep diagnostics outside compiled rollouts/gradients."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = train_utils.log_metrics
    calls = 0
    measured_steps = 0
    captured = False

    def log_metrics(config, out, *args, **kwargs):
        nonlocal calls, measured_steps, captured
        # This experimental recorder deliberately supports one learner only.
        if config.NUM_LEARNERS != 1 or not config.continuous_operation:
            raise ValueError("GN campaign diagnostics require one learner and continuous operation")
        calls += 1
        measured_steps += config.STEPS_PER_INCREMENT // config.NUM_ENVS
        record = per_env_record(out["metrics"], config.NUM_ENVS, measured_steps)
        record["increment"] = calls
        record["replica_summary"] = summarize_replicas(record)
        if physical_audit is not None:
            health = physical_audit(out["runner_state"][1].env_state)
            record["physical_health"] = {
                key: [float(x) if np.isfinite(x) else None for x in np.asarray(value).reshape(-1)]
                for key, value in health.items()
            }
        snr = np.asarray(record.get("snr_blocking_probability", [0]))
        if not captured and np.max(snr) > 0.1:
            inner = out["runner_state"][1].env_state
            index = int(np.argmax(snr))
            arrays = {}
            for field in fields(inner):
                value = getattr(inner, field.name)
                if hasattr(value, "dtype"):
                    array = np.asarray(value)
                    arrays[field.name] = array[index] if config.NUM_ENVS > 1 else array
            snapshot = path.with_suffix(".anomaly.npz")
            np.savez_compressed(snapshot, **arrays)
            snapshot.with_suffix(".json").write_text(
                json.dumps(
                    {"config": config.to_dict(), "env_index": index, "metrics": record}, indent=2
                )
                + "\n"
            )
            record["anomaly_snapshot"] = str(snapshot)
            captured = True
            print(f"Captured high-SNR-blocking state: env {index}, {snapshot}", flush=True)
            if config.WANDB:
                train_utils.wandb.save(str(snapshot))
                train_utils.wandb.save(str(snapshot.with_suffix(".json")))
        with path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        result = original(config, out, *args, **kwargs)
        if config.WANDB:
            run = train_utils.wandb.run
            run.summary["replicas/max_snr_blocking"] = float(np.max(snr))
            run.summary["replicas/high_snr_blocking_count"] = int(np.count_nonzero(snr > 0.1))
            for name, value in record["replica_summary"].items():
                run.summary[f"replicas/{name}"] = value
            if "physical_health" in record:
                health = record["physical_health"]
                failed = np.asarray(health["failed_link_slot_predicates"])
                run.summary["diagnostics/existing_snr_outage_replicas"] = int(np.sum(failed > 0))
                margins = [x for x in health["min_margin_db"] if x is not None]
                if margins:
                    run.summary["diagnostics/existing_snr_min_margin_db"] = min(margins)
            if calls == config.NUM_INCREMENTS:
                columns = ["env_index", "service_blocking", "bitrate_blocking", "snr_blocking"]
                rows = [
                    [
                        i,
                        record["service_blocking_probability"][i],
                        record["bitrate_blocking_probability"][i],
                        float(snr[i]),
                    ]
                    for i in range(config.NUM_ENVS)
                ]
                run.summary["replicas/final_blocking"] = train_utils.wandb.Table(
                    columns=columns, data=rows
                )
                train_utils.wandb.save(str(path))
        return result

    train_utils.log_metrics = log_metrics
