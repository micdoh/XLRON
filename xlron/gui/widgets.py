"""Widget builder functions for XLRON GUI.

Each function renders a UI section and returns a dict of {flag_name: value}
for flags that differ from their defaults. Help text for each widget is
auto-populated from parameter_flags.py so it stays in sync with the CLI.
"""

import ast
import functools
from pathlib import Path

import streamlit as st

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FLAGS_PATH = Path(__file__).resolve().parent.parent / "parameter_flags.py"


@functools.cache
def _parse_flag_help() -> dict[str, str]:
    """Parse parameter_flags.py and return {flag_name: help_text}.

    Uses the Python AST so it works without importing absl/JAX.
    """
    help_map: dict[str, str] = {}
    try:
        source = _FLAGS_PATH.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return help_map

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match flags.DEFINE_*(name, default, help_text, ...)
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "flags"
            and func.attr.startswith("DEFINE_")
        ):
            continue
        if not node.args:
            continue
        # First arg is the flag name
        name_node = node.args[0]
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            continue
        flag_name = name_node.value
        # Help text is the last positional string arg (typically arg index 2)
        help_text = None
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                help_text = arg.value
            elif isinstance(arg, ast.JoinedStr):
                # f-string — skip
                pass
            elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                # Concatenated string literals: "foo " + "bar"
                parts = []
                _collect_str_parts(arg, parts)
                if parts:
                    help_text = "".join(parts)
        # Also check keyword arguments for 'help' (rare but possible)
        # The standard pattern is positional, but just in case
        if help_text is None:
            for kw in node.keywords:
                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                    help_text = kw.value.value
        if help_text:
            help_map[flag_name] = help_text
    return help_map


def _collect_str_parts(node: ast.expr, parts: list[str]):
    """Recursively collect string literal parts from BinOp(Add) chains."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parts.append(node.value)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        _collect_str_parts(node.left, parts)
        _collect_str_parts(node.right, parts)


def _h(flag_name: str) -> str | None:
    """Get the help text for a flag, or None if not found."""
    return _parse_flag_help().get(flag_name)


def _none_or_float(text: str | None) -> float | None:
    """Parse optional float text input. Empty string -> None."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped == "":
        return None
    return float(stripped)


def _scan_topologies() -> list[str]:
    """Scan topology JSON files and return sorted list of names."""
    topo_dir = _DATA_DIR / "topologies"
    if not topo_dir.exists():
        return []
    return sorted(p.stem for p in topo_dir.glob("*.json"))


def _scan_modulations() -> list[str]:
    """Scan modulation CSV files and return sorted list of paths."""
    mod_dir = _DATA_DIR / "modulations"
    if not mod_dir.exists():
        return []
    return sorted(str(p) for p in mod_dir.glob("*.csv"))


def _scan_band_data() -> list[str]:
    """Scan band data CSV files and return sorted list of paths."""
    band_dir = _DATA_DIR / "gn_model" / "band_data"
    if not band_dir.exists():
        return []
    return sorted(str(p) for p in band_dir.glob("*.csv"))


def _scan_noise_data() -> list[str]:
    """Scan transceiver/amplifier noise data CSV files and return sorted list of paths."""
    noise_dir = _DATA_DIR / "gn_model" / "transceiver_amplifier_data"
    if not noise_dir.exists():
        return []
    return sorted(str(p) for p in noise_dir.glob("*.csv"))


# ---------------------------------------------------------------------------
# Defaults — must match parameter_flags.py
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Environment
    "env_type": "rmsa",
    "topology_name": "4node",
    "link_resources": 5,
    "k": 5,
    "slot_size": 12.5,
    "guardband": 1,
    # Traffic
    "load": 250,
    "mean_service_holding_time": 25,
    "continuous_operation": False,
    "ENV_WARMUP_STEPS": 0,
    "max_requests": 4,
    "reward_type": "service",
    "incremental_loading": False,
    "end_first_blocking": False,
    "truncate_holding_time": False,
    # Execution
    "TOTAL_TIMESTEPS": 1e6,
    "NUM_ENVS": 1,
    "ROLLOUT_LENGTH": 150,
    "STEPS_PER_INCREMENT": 100000,
    "NUM_MINIBATCHES": 1,
    "UPDATE_EPOCHS": 1,
    "SEED": 42,
    # Architecture
    "NUM_LAYERS": 2,
    "NUM_UNITS": 64,
    "ACTIVATION": "tanh",
    "USE_GNN": False,
    "USE_TRANSFORMER": False,
    # GNN
    "message_passing_steps": 3,
    "edge_embedding_size": 128,
    "node_embedding_size": 16,
    "attn_mlp_layers": 1,
    # Transformer
    "transformer_num_layers": 1,
    "transformer_num_heads": 4,
    "transformer_embedding_size": 128,
    "transformer_obs_type": "departure",
    "transformer_share_layers": False,
    "transformer_actor_mlp_width": 128,
    "transformer_critic_mlp_width": 128,
    "transformer_intermediate_size": 256,
    "transformer_actor_mlp_depth": 1,
    "transformer_critic_mlp_depth": 2,
    # PPO
    "LR": 5e-4,
    "GAMMA": 0.999,
    "GAE_LAMBDA": None,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    # Schedules
    "LR_SCHEDULE": "cosine",
    "LR_END_FRACTION": 0.1,
    "LR_SCHEDULE_MULTIPLIER": 1.0,
    "WARMUP_STEPS_FRACTION": 0.2,
    "WARMUP_MULTIPLIER": 1.0,
    "ENT_SCHEDULE": "constant",
    "ENT_END_FRACTION": 0.1,
    "ENT_SCHEDULE_MULTIPLIER": 1.0,
    "VML_SCHEDULE": "constant",
    "VML_END_FRACTION": 10.0,
    "VML_SCHEDULE_MULTIPLIER": 1.0,
    "LAMBDA_SCHEDULE_MULTIPLIER": 1.0,
    "VF_SCHEDULE_MULTIPLIER": 1.0,
    "INITIAL_LAMBDA": 0.9,
    "FINAL_LAMBDA": 0.98,
    "STEP_ON_GRADIENT": False,
    # Advanced
    "REWARD_CENTERING": False,
    "OFF_POLICY_IAM": False,
    "VALID_MASS_LOSS_COEF": 0.0,
    "PRIO_ALPHA": 0.0,
    "PRIO_BETA0": 1.0,
    "RHO_CLIP": -1.0,
    "C_CLIP": -1.0,
    "REWARD_SCALE": 1.0,
    "SEPARATE_VF_OPTIMIZER": False,
    # Path sorting
    "path_sort_criteria": "spectral_resources",
    # Heuristics
    "path_heuristic": "ksp_ff",
    "EVAL_HEURISTIC": False,
    "EVAL_MODEL": False,
    "RETRAIN_MODEL": False,
    "KEEP_VF": False,
    # Logging
    "WANDB": False,
    "PROJECT": "",
    "EXPERIMENT_NAME": "",
    "SAVE_MODEL": False,
    "MODEL_PATH": None,
    "DATA_OUTPUT_FILE": None,
    "EPISODE_DATA_OUTPUT_FILE": None,
    "DOWNSAMPLE_FACTOR": 1,
    "PLOTTING": False,
    "PROFILE": False,
    "EVAL_DURING_TRAINING": False,
    "RENDER_EVAL_MODE": "off",
    "RENDER_FPS": 2.0,
    "RENDER_SCALE": 0.6,
    "RENDER_OUTPUT_FILE": None,
    "RENDER_MAX_STEPS": 100,
    "RENDER_CLICK_THROUGH": False,
    # Physical layer (GN model)
    "modulations_csv_filepath": "./xlron/data/modulations/modulations_deeprmsa.csv",
    "calc_minimum_osnr": True,
    "beta_fec": 1.5e-2,
    "fec_rate": 0.8,
    "band_data_filepath": None,
    "noise_data_filepath": None,
    "band_preference": None,
    "slots_per_band": None,
    "alpha": 0.2,
    "attenuation": 4.605111673e-5,
    "attenuation_bar": 4.605111673e-5,
    "beta_2": -21.7,
    "dispersion_coeff": 17e-6,
    "dispersion_slope": 60.7,
    "nonlinear_coefficient": 1.2e-3,
    "gamma": 1.2,
    "span_length": 100,
    "span_lumped_loss_db": None,
    "snr_margin": 0.5,
    "launch_power_type": "fixed",
    "max_power_per_fibre": 13.0,
    "power_per_channel": None,
    "power_per_channel_per_band": None,
    "launch_power_csv": None,
    "inter_band_gap_ghz": 25.0,
    "num_subchannels": 1,
    "use_raman_amp": False,
    "raman_gain_slope": 2.8e-17,
    "raman_pump_power_fw": None,
    "raman_pump_power_bw": None,
    "raman_pump_freq_fw": None,
    "raman_pump_freq_bw": None,
    "raman_max_bandwidth_thz": 15.0,
    # Differentiable
    "differentiable": False,
    "temperature": 1.0,
    # Aggregate
    "aggregate_slots": 1,
    # Capacity bounds
    "num_trials": 10,
    "cutset_link_selection_mode": "least_congested",
    "CUTSET_EXHAUSTIVE": False,
    "CUTSET_TOP_K": 256,
    "CUTSET_BATCH_SIZE": 512,
    "CUTSET_ITERATIONS": 32,
    "CUTSET_PARALLEL_PROCESSES": 1,
    "COMPILE_RR_BOUNDS": True,
}

ENV_TYPES = [
    "rwa",
    "rsa",
    "rmsa",
    "deeprmsa",
    "rwa_lightpath_reuse",
    "rsa_gn_model",
    "rmsa_gn_model",
]

PATH_HEURISTICS = [
    "ksp_ff",
    "ksp_lf",
    "ksp_bf",
    "ksp_mu",
    "ff_ksp",
    "lf_ksp",
    "bf_ksp",
    "mu_ksp",
    "kmc_ff",
    "kmf_ff",
    "kme_ff",
    "kca_ff",
]


def _emit(flags: dict, key: str, value):
    """Add key to flags dict only if it differs from the default."""
    default = DEFAULTS.get(key)
    if value != default:
        flags[key] = value


def _get_preset_val(key):
    """Get the value for a key from the loaded preset, falling back to default."""
    preset = st.session_state.get("_loaded_preset", {})
    return preset.get(key, DEFAULTS.get(key))


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


CUTSET_LINK_SELECTION_MODES = [
    "least_congested",
    "most_congested",
    "best_fit",
    "random",
]


def execution_mode_section() -> dict:
    """Radio selector for RL / Heuristic / Model Eval / Capacity Bounds."""
    flags = {}
    modes = [
        "RL Training",
        "Heuristic Evaluation",
        "Model Evaluation",
        "Capacity Bound Estimation",
    ]
    preset = st.session_state.get("_loaded_preset", {})

    if preset.get("_bounds_method"):
        default_idx = 3
    elif preset.get("EVAL_HEURISTIC"):
        default_idx = 1
    elif preset.get("EVAL_MODEL"):
        default_idx = 2
    else:
        default_idx = 0

    mode = st.radio("Execution Mode", modes, index=default_idx, horizontal=True)

    # Clear bounds session state when not in bounds mode
    if mode != "Capacity Bound Estimation":
        st.session_state["_bounds_mode"] = False
        st.session_state.pop("_bounds_method", None)

    if mode == "RL Training":
        retrain = st.checkbox(
            "Retrain from saved model",
            value=bool(_get_preset_val("RETRAIN_MODEL")),
            help=_h("RETRAIN_MODEL"),
        )
        if retrain:
            flags["RETRAIN_MODEL"] = True
            model_path = st.text_input(
                "Model Path (to load for retraining)",
                value=_get_preset_val("MODEL_PATH") or "",
                help=_h("MODEL_PATH"),
            )
            if model_path:
                flags["MODEL_PATH"] = model_path
            keep_vf = st.checkbox(
                "Keep pre-trained value function",
                value=bool(_get_preset_val("KEEP_VF")),
                help=_h("KEEP_VF"),
            )
            if keep_vf:
                flags["KEEP_VF"] = True

    elif mode == "Heuristic Evaluation":
        flags["EVAL_HEURISTIC"] = True
        heuristic = st.selectbox(
            "Path Heuristic",
            PATH_HEURISTICS,
            index=PATH_HEURISTICS.index(_get_preset_val("path_heuristic")),
            help=_h("path_heuristic"),
        )
        _emit(flags, "path_heuristic", heuristic)

    elif mode == "Model Evaluation":
        flags["EVAL_MODEL"] = True
        model_path = st.text_input(
            "Model Path", value=_get_preset_val("MODEL_PATH") or "", help=_h("MODEL_PATH")
        )
        if model_path:
            flags["MODEL_PATH"] = model_path

    elif mode == "Capacity Bound Estimation":
        st.session_state["_bounds_mode"] = True

        preset_method = _get_preset_val("_bounds_method") or "cutsets"
        methods = ["Cut-Sets Method", "Reconfigurable Routing (Defragmentation)"]
        method_idx = 1 if preset_method == "reconfigurable" else 0
        method = st.radio("Bounds Method", methods, index=method_idx, horizontal=True)
        bounds_method = "cutsets" if method == methods[0] else "reconfigurable"
        st.session_state["_bounds_method"] = bounds_method

        # Shared control
        num_trials = st.number_input(
            "Number of Trials",
            min_value=1,
            value=int(_get_preset_val("num_trials")),
            help=_h("num_trials"),
        )
        _emit(flags, "num_trials", int(num_trials))

        if bounds_method == "cutsets":
            st.markdown("**Cut-Set Discovery**")
            col1, col2 = st.columns(2)
            with col1:
                exhaustive = st.checkbox(
                    "Exhaustive Search",
                    value=bool(_get_preset_val("CUTSET_EXHAUSTIVE")),
                    help=_h("CUTSET_EXHAUSTIVE"),
                )
                _emit(flags, "CUTSET_EXHAUSTIVE", exhaustive)

                top_k = st.number_input(
                    "Top K Cut-Sets",
                    min_value=1,
                    value=int(_get_preset_val("CUTSET_TOP_K")),
                    help=_h("CUTSET_TOP_K"),
                )
                _emit(flags, "CUTSET_TOP_K", int(top_k))

            with col2:
                if exhaustive:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        value=int(_get_preset_val("CUTSET_BATCH_SIZE")),
                        help=_h("CUTSET_BATCH_SIZE"),
                    )
                    _emit(flags, "CUTSET_BATCH_SIZE", int(batch_size))

                    iterations = st.number_input(
                        "Iterations",
                        min_value=1,
                        value=int(_get_preset_val("CUTSET_ITERATIONS")),
                        help=_h("CUTSET_ITERATIONS"),
                    )
                    _emit(flags, "CUTSET_ITERATIONS", int(iterations))

                    parallel = st.number_input(
                        "Parallel Processes",
                        min_value=1,
                        value=int(_get_preset_val("CUTSET_PARALLEL_PROCESSES")),
                        help=_h("CUTSET_PARALLEL_PROCESSES"),
                    )
                    _emit(flags, "CUTSET_PARALLEL_PROCESSES", int(parallel))

            st.markdown("**Simulation**")
            link_modes = CUTSET_LINK_SELECTION_MODES
            default_lsm = _get_preset_val("cutset_link_selection_mode")
            lsm_idx = link_modes.index(default_lsm) if default_lsm in link_modes else 0
            link_sel = st.selectbox(
                "Link Selection Mode",
                link_modes,
                index=lsm_idx,
                help=_h("cutset_link_selection_mode"),
            )
            _emit(flags, "cutset_link_selection_mode", link_sel)

        else:  # reconfigurable
            heuristic = st.selectbox(
                "Path Heuristic",
                PATH_HEURISTICS,
                index=PATH_HEURISTICS.index(_get_preset_val("path_heuristic")),
                help=_h("path_heuristic"),
            )
            _emit(flags, "path_heuristic", heuristic)

            compile_rr = st.checkbox(
                "Compile Main Loop (AOT)",
                value=bool(_get_preset_val("COMPILE_RR_BOUNDS")),
                help=_h("COMPILE_RR_BOUNDS"),
            )
            _emit(flags, "COMPILE_RR_BOUNDS", compile_rr)

    return flags


def environment_section() -> dict:
    """Environment type, topology, link resources, k, slot size, guardband."""
    flags = {}

    col1, col2 = st.columns(2)
    with col1:
        env_type = st.selectbox(
            "Environment Type",
            ENV_TYPES,
            index=ENV_TYPES.index(_get_preset_val("env_type")),
            help=_h("env_type"),
        )
        _emit(flags, "env_type", env_type)

        topologies = _scan_topologies()
        default_topo = _get_preset_val("topology_name")
        topo_idx = topologies.index(default_topo) if default_topo in topologies else 0
        topology = st.selectbox("Topology", topologies, index=topo_idx, help=_h("topology_name"))
        _emit(flags, "topology_name", topology)

        link_res = st.number_input(
            "Link Resources (slots)",
            min_value=1,
            value=int(_get_preset_val("link_resources")),
            help=_h("link_resources"),
        )
        _emit(flags, "link_resources", int(link_res))

    with col2:
        k = st.number_input(
            "K Shortest Paths", min_value=1, value=int(_get_preset_val("k")), help=_h("k")
        )
        _emit(flags, "k", int(k))

        slot_size = st.number_input(
            "Slot Size (GHz)",
            min_value=0.1,
            value=float(_get_preset_val("slot_size")),
            step=0.5,
            help=_h("slot_size"),
        )
        _emit(flags, "slot_size", slot_size)

        guardband = st.number_input(
            "Guardband (slots)",
            min_value=0,
            value=int(_get_preset_val("guardband")),
            help=_h("guardband"),
        )
        _emit(flags, "guardband", int(guardband))

        agg = st.number_input(
            "Aggregate Slots",
            min_value=1,
            value=int(_get_preset_val("aggregate_slots")),
            help=_h("aggregate_slots"),
        )
        _emit(flags, "aggregate_slots", int(agg))

    sort_options = ["spectral_resources", "hops", "distance", "hops_distance", "capacity"]
    default_sort = _get_preset_val("path_sort_criteria")
    sort_idx = sort_options.index(default_sort) if default_sort in sort_options else 0
    path_sort = st.selectbox(
        "Path Sort Criteria", sort_options, index=sort_idx, help=_h("path_sort_criteria")
    )
    _emit(flags, "path_sort_criteria", path_sort)

    # Store env_type in session state for other sections to read
    st.session_state["_env_type"] = env_type
    return flags


def traffic_section() -> dict:
    """Load, holding time, bandwidth, continuous operation, warmup."""
    flags = {}

    col1, col2 = st.columns(2)
    with col1:
        load = st.number_input(
            "Load (Erlangs)",
            min_value=0.1,
            value=float(_get_preset_val("load")),
            step=10.0,
            help=_h("load"),
        )
        _emit(flags, "load", load)

        mht = st.number_input(
            "Mean Service Holding Time",
            min_value=1.0,
            value=float(_get_preset_val("mean_service_holding_time")),
            help=_h("mean_service_holding_time"),
        )
        _emit(flags, "mean_service_holding_time", mht)

        values_bw = st.text_input(
            "Bandwidth Values (comma-separated, leave blank for default)",
            value=str(_get_preset_val("values_bw") or ""),
            help=_h("values_bw"),
        )
        if values_bw.strip():
            flags["values_bw"] = values_bw.strip()

        max_req = st.number_input(
            "Max Requests",
            min_value=1,
            value=int(_get_preset_val("max_requests")),
            step=1,
            help=_h("max_requests"),
        )
        _emit(flags, "max_requests", int(max_req))

    with col2:
        cont_op = st.checkbox(
            "Continuous Operation",
            value=bool(_get_preset_val("continuous_operation")),
            help=_h("continuous_operation"),
        )
        _emit(flags, "continuous_operation", cont_op)

        trunc_ht = st.checkbox(
            "Truncate Holding Time",
            value=bool(_get_preset_val("truncate_holding_time")),
            help=_h("truncate_holding_time"),
        )
        _emit(flags, "truncate_holding_time", trunc_ht)

        inc_load = st.checkbox(
            "Incremental Loading",
            value=bool(_get_preset_val("incremental_loading")),
            help=_h("incremental_loading"),
        )
        _emit(flags, "incremental_loading", inc_load)

        end_fb = st.checkbox(
            "End on First Blocking",
            value=bool(_get_preset_val("end_first_blocking")),
            help=_h("end_first_blocking"),
        )
        _emit(flags, "end_first_blocking", end_fb)

    warmup = st.number_input(
        "ENV Warmup Steps",
        min_value=0,
        value=int(_get_preset_val("ENV_WARMUP_STEPS")),
        step=1000,
        help=_h("ENV_WARMUP_STEPS"),
    )
    _emit(flags, "ENV_WARMUP_STEPS", int(warmup))

    reward_type = st.selectbox(
        "Reward Type",
        ["service", "bitrate", "utilisation"],
        index=0,
        help=_h("reward_type"),
    )
    _emit(flags, "reward_type", reward_type)

    return flags


def execution_section() -> dict:
    """Total timesteps, num envs, rollout length, steps per increment, etc."""
    flags = {}

    col1, col2 = st.columns(2)
    with col1:
        total = st.number_input(
            "Total Timesteps",
            min_value=1,
            value=int(_get_preset_val("TOTAL_TIMESTEPS")),
            step=1_000_000,
            format="%d",
            help=_h("TOTAL_TIMESTEPS"),
        )
        _emit(flags, "TOTAL_TIMESTEPS", int(total))

        num_envs = st.number_input(
            "Num Environments",
            min_value=1,
            value=int(_get_preset_val("NUM_ENVS")),
            step=100,
            help=_h("NUM_ENVS"),
        )
        _emit(flags, "NUM_ENVS", int(num_envs))

        rollout = st.number_input(
            "Rollout Length",
            min_value=1,
            value=int(_get_preset_val("ROLLOUT_LENGTH")),
            step=10,
            help=_h("ROLLOUT_LENGTH"),
        )
        _emit(flags, "ROLLOUT_LENGTH", int(rollout))

    with col2:
        spi = st.number_input(
            "Steps Per Increment",
            min_value=1,
            value=int(_get_preset_val("STEPS_PER_INCREMENT")),
            step=10000,
            format="%d",
            help=_h("STEPS_PER_INCREMENT"),
        )
        _emit(flags, "STEPS_PER_INCREMENT", int(spi))

        minibatches = st.number_input(
            "Num Minibatches",
            min_value=1,
            value=int(_get_preset_val("NUM_MINIBATCHES")),
            help=_h("NUM_MINIBATCHES"),
        )
        _emit(flags, "NUM_MINIBATCHES", int(minibatches))

        epochs = st.number_input(
            "Update Epochs",
            min_value=1,
            value=int(_get_preset_val("UPDATE_EPOCHS")),
            help=_h("UPDATE_EPOCHS"),
        )
        _emit(flags, "UPDATE_EPOCHS", int(epochs))

    seed = st.number_input("Seed", min_value=0, value=int(_get_preset_val("SEED")), help=_h("SEED"))
    _emit(flags, "SEED", int(seed))

    return flags


def architecture_section() -> dict:
    """MLP / GNN / Transformer architecture selection and params."""
    flags = {}

    preset = st.session_state.get("_loaded_preset", {})
    if preset.get("USE_GNN"):
        arch_default = 1
    elif preset.get("USE_TRANSFORMER"):
        arch_default = 2
    else:
        arch_default = 0

    arch = st.radio(
        "Architecture", ["MLP", "GNN", "Transformer"], index=arch_default, horizontal=True
    )

    if arch == "MLP":
        col1, col2, col3 = st.columns(3)
        with col1:
            layers = st.number_input(
                "Num Layers",
                min_value=1,
                value=int(_get_preset_val("NUM_LAYERS")),
                help=_h("NUM_LAYERS"),
            )
            _emit(flags, "NUM_LAYERS", int(layers))
        with col2:
            units = st.number_input(
                "Hidden Units",
                min_value=1,
                value=int(_get_preset_val("NUM_UNITS")),
                help=_h("NUM_UNITS"),
            )
            _emit(flags, "NUM_UNITS", int(units))
        with col3:
            activation = st.selectbox(
                "Activation",
                ["tanh", "relu", "gelu", "silu"],
                index=["tanh", "relu", "gelu", "silu"].index(_get_preset_val("ACTIVATION")),
                help=_h("ACTIVATION"),
            )
            _emit(flags, "ACTIVATION", activation)

    elif arch == "GNN":
        flags["USE_GNN"] = True
        col1, col2 = st.columns(2)
        with col1:
            mp_steps = st.number_input(
                "Message Passing Steps",
                min_value=1,
                value=int(_get_preset_val("message_passing_steps")),
                help=_h("message_passing_steps"),
            )
            _emit(flags, "message_passing_steps", int(mp_steps))

            edge_emb = st.number_input(
                "Edge Embedding Size",
                min_value=1,
                value=int(_get_preset_val("edge_embedding_size")),
                help=_h("edge_embedding_size"),
            )
            _emit(flags, "edge_embedding_size", int(edge_emb))
        with col2:
            node_emb = st.number_input(
                "Node Embedding Size",
                min_value=1,
                value=int(_get_preset_val("node_embedding_size")),
                help=_h("node_embedding_size"),
            )
            _emit(flags, "node_embedding_size", int(node_emb))

            attn_layers = st.number_input(
                "Attention MLP Layers",
                min_value=1,
                value=int(_get_preset_val("attn_mlp_layers")),
                help=_h("attn_mlp_layers"),
            )
            _emit(flags, "attn_mlp_layers", int(attn_layers))

    elif arch == "Transformer":
        flags["USE_TRANSFORMER"] = True
        col1, col2 = st.columns(2)
        with col1:
            t_layers = st.number_input(
                "Num Layers",
                min_value=1,
                value=int(_get_preset_val("transformer_num_layers")),
                help=_h("transformer_num_layers"),
            )
            _emit(flags, "transformer_num_layers", int(t_layers))

            t_heads = st.number_input(
                "Num Heads",
                min_value=1,
                value=int(_get_preset_val("transformer_num_heads")),
                help=_h("transformer_num_heads"),
            )
            _emit(flags, "transformer_num_heads", int(t_heads))

            t_emb = st.number_input(
                "Embedding Size",
                min_value=1,
                value=int(_get_preset_val("transformer_embedding_size")),
                help=_h("transformer_embedding_size"),
            )
            _emit(flags, "transformer_embedding_size", int(t_emb))

            t_inter = st.number_input(
                "Intermediate Size",
                min_value=1,
                value=int(_get_preset_val("transformer_intermediate_size")),
                help=_h("transformer_intermediate_size"),
            )
            _emit(flags, "transformer_intermediate_size", int(t_inter))

            t_obs = st.selectbox(
                "Observation Type",
                ["departure", "occupancy", "capacity"],
                index=["departure", "occupancy", "capacity"].index(
                    _get_preset_val("transformer_obs_type")
                ),
                help=_h("transformer_obs_type"),
            )
            _emit(flags, "transformer_obs_type", t_obs)
        with col2:
            t_actor_w = st.number_input(
                "Actor MLP Width",
                min_value=1,
                value=int(_get_preset_val("transformer_actor_mlp_width")),
                help=_h("transformer_actor_mlp_width"),
            )
            _emit(flags, "transformer_actor_mlp_width", int(t_actor_w))

            t_actor_d = st.number_input(
                "Actor MLP Depth",
                min_value=1,
                value=int(_get_preset_val("transformer_actor_mlp_depth")),
                help=_h("transformer_actor_mlp_depth"),
            )
            _emit(flags, "transformer_actor_mlp_depth", int(t_actor_d))

            t_critic_w = st.number_input(
                "Critic MLP Width",
                min_value=1,
                value=int(_get_preset_val("transformer_critic_mlp_width")),
                help=_h("transformer_critic_mlp_width"),
            )
            _emit(flags, "transformer_critic_mlp_width", int(t_critic_w))

            t_critic_d = st.number_input(
                "Critic MLP Depth",
                min_value=1,
                value=int(_get_preset_val("transformer_critic_mlp_depth")),
                help=_h("transformer_critic_mlp_depth"),
            )
            _emit(flags, "transformer_critic_mlp_depth", int(t_critic_d))

        share = st.checkbox(
            "Share encoder layers between actor and critic",
            value=bool(_get_preset_val("transformer_share_layers")),
            help=_h("transformer_share_layers"),
        )
        _emit(flags, "transformer_share_layers", share)

    return flags


def ppo_section() -> dict:
    """PPO hyperparameters."""
    flags = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        lr = st.number_input(
            "Learning Rate",
            min_value=0.0,
            value=float(_get_preset_val("LR")),
            format="%.1e",
            step=1e-4,
            help=_h("LR"),
        )
        _emit(flags, "LR", lr)

        gamma = st.number_input(
            "Discount (GAMMA)",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("GAMMA")),
            step=0.001,
            format="%.4f",
            help=_h("GAMMA"),
        )
        _emit(flags, "GAMMA", gamma)

    with col2:
        clip = st.number_input(
            "Clip Epsilon",
            min_value=0.0,
            value=float(_get_preset_val("CLIP_EPS")),
            step=0.05,
            help=_h("CLIP_EPS"),
        )
        _emit(flags, "CLIP_EPS", clip)

        ent = st.number_input(
            "Entropy Coef",
            min_value=0.0,
            value=float(_get_preset_val("ENT_COEF")),
            step=0.001,
            format="%.4f",
            help=_h("ENT_COEF"),
        )
        _emit(flags, "ENT_COEF", ent)

    with col3:
        vf = st.number_input(
            "VF Coef",
            min_value=0.0,
            value=float(_get_preset_val("VF_COEF")),
            step=0.1,
            help=_h("VF_COEF"),
        )
        _emit(flags, "VF_COEF", vf)

        grad_norm = st.number_input(
            "Max Grad Norm",
            min_value=0.0,
            value=float(_get_preset_val("MAX_GRAD_NORM")),
            step=0.1,
            help=_h("MAX_GRAD_NORM"),
        )
        _emit(flags, "MAX_GRAD_NORM", grad_norm)

    use_gae = st.checkbox("Set GAE Lambda (default: auto-anneal)", help=_h("GAE_LAMBDA"))
    if use_gae:
        gae = st.number_input(
            "GAE Lambda",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.01,
            help=_h("GAE_LAMBDA"),
        )
        flags["GAE_LAMBDA"] = gae

    return flags


def schedule_section() -> dict:
    """LR schedule, entropy schedule, VML schedule, end fractions, multipliers, warmup."""
    flags = {}

    schedules = ["cosine", "warmup_cosine", "linear", "constant"]

    # --- LR Schedule ---
    st.markdown("**Learning Rate**")
    col1, col2, col3 = st.columns(3)
    with col1:
        lr_sched = st.selectbox(
            "LR Schedule",
            schedules,
            index=schedules.index(_get_preset_val("LR_SCHEDULE")),
            help=_h("LR_SCHEDULE"),
        )
        _emit(flags, "LR_SCHEDULE", lr_sched)
    with col2:
        lr_end = st.number_input(
            "LR End Fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("LR_END_FRACTION")),
            step=0.01,
            format="%.3f",
            help=_h("LR_END_FRACTION"),
        )
        _emit(flags, "LR_END_FRACTION", lr_end)
    with col3:
        lr_mult = st.number_input(
            "LR Schedule Multiplier",
            min_value=0.01,
            value=float(_get_preset_val("LR_SCHEDULE_MULTIPLIER")),
            step=0.1,
            format="%.2f",
            help=_h("LR_SCHEDULE_MULTIPLIER"),
        )
        _emit(flags, "LR_SCHEDULE_MULTIPLIER", lr_mult)

    # --- Warmup (visible when warmup_cosine selected) ---
    if lr_sched == "warmup_cosine":
        wc1, wc2 = st.columns(2)
        with wc1:
            warmup_frac = st.number_input(
                "Warmup Steps Fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(_get_preset_val("WARMUP_STEPS_FRACTION")),
                step=0.05,
                format="%.2f",
                help=_h("WARMUP_STEPS_FRACTION"),
            )
            _emit(flags, "WARMUP_STEPS_FRACTION", warmup_frac)
        with wc2:
            warmup_mult = st.number_input(
                "Warmup Peak Multiplier",
                min_value=0.01,
                value=float(_get_preset_val("WARMUP_MULTIPLIER")),
                step=0.1,
                format="%.2f",
                help=_h("WARMUP_MULTIPLIER"),
            )
            _emit(flags, "WARMUP_MULTIPLIER", warmup_mult)

    # --- Entropy Schedule ---
    st.markdown("**Entropy Coefficient**")
    ent_scheds = ["constant", "linear", "cosine"]
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        ent_sched = st.selectbox(
            "Entropy Schedule",
            ent_scheds,
            index=ent_scheds.index(_get_preset_val("ENT_SCHEDULE")),
            help=_h("ENT_SCHEDULE"),
        )
        _emit(flags, "ENT_SCHEDULE", ent_sched)
    with ec2:
        ent_end = st.number_input(
            "Entropy End Fraction",
            min_value=0.0,
            value=float(_get_preset_val("ENT_END_FRACTION")),
            step=0.01,
            format="%.3f",
            help=_h("ENT_END_FRACTION"),
        )
        _emit(flags, "ENT_END_FRACTION", ent_end)
    with ec3:
        ent_mult = st.number_input(
            "Entropy Schedule Multiplier",
            min_value=0.01,
            value=float(_get_preset_val("ENT_SCHEDULE_MULTIPLIER")),
            step=0.1,
            format="%.2f",
            help=_h("ENT_SCHEDULE_MULTIPLIER"),
        )
        _emit(flags, "ENT_SCHEDULE_MULTIPLIER", ent_mult)

    # --- VML Schedule ---
    st.markdown("**Valid Mass Loss Coefficient**")
    vml_scheds = ["constant", "linear", "cosine"]
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        vml_sched = st.selectbox(
            "VML Schedule",
            vml_scheds,
            index=vml_scheds.index(_get_preset_val("VML_SCHEDULE")),
            help=_h("VML_SCHEDULE"),
        )
        _emit(flags, "VML_SCHEDULE", vml_sched)
    with vc2:
        vml_end = st.number_input(
            "VML End Fraction",
            min_value=0.0,
            value=float(_get_preset_val("VML_END_FRACTION")),
            step=0.1,
            format="%.2f",
            help=_h("VML_END_FRACTION"),
        )
        _emit(flags, "VML_END_FRACTION", vml_end)
    with vc3:
        vml_mult = st.number_input(
            "VML Schedule Multiplier",
            min_value=0.01,
            value=float(_get_preset_val("VML_SCHEDULE_MULTIPLIER")),
            step=0.1,
            format="%.2f",
            help=_h("VML_SCHEDULE_MULTIPLIER"),
        )
        _emit(flags, "VML_SCHEDULE_MULTIPLIER", vml_mult)

    # --- GAE Lambda Anneal ---
    st.markdown("**GAE Lambda Anneal**")
    st.caption("When GAE Lambda is not set explicitly (in PPO section), it anneals between these values.")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        init_lam = st.number_input(
            "Initial Lambda",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("INITIAL_LAMBDA")),
            step=0.01,
            format="%.3f",
            help=_h("INITIAL_LAMBDA"),
        )
        _emit(flags, "INITIAL_LAMBDA", init_lam)
    with lc2:
        final_lam = st.number_input(
            "Final Lambda",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("FINAL_LAMBDA")),
            step=0.01,
            format="%.3f",
            help=_h("FINAL_LAMBDA"),
        )
        _emit(flags, "FINAL_LAMBDA", final_lam)
    with lc3:
        lam_mult = st.number_input(
            "Lambda Schedule Multiplier",
            min_value=0.01,
            value=float(_get_preset_val("LAMBDA_SCHEDULE_MULTIPLIER")),
            step=0.1,
            format="%.2f",
            help=_h("LAMBDA_SCHEDULE_MULTIPLIER"),
        )
        _emit(flags, "LAMBDA_SCHEDULE_MULTIPLIER", lam_mult)

    # --- VF Schedule Multiplier ---
    vf_s_mult = st.number_input(
        "VF Schedule Multiplier",
        min_value=0.01,
        value=float(_get_preset_val("VF_SCHEDULE_MULTIPLIER")),
        step=0.1,
        format="%.2f",
        help=_h("VF_SCHEDULE_MULTIPLIER"),
    )
    _emit(flags, "VF_SCHEDULE_MULTIPLIER", vf_s_mult)

    # --- Step on Gradient ---
    step_grad = st.checkbox(
        "Step Schedule on Gradient Update",
        value=bool(_get_preset_val("STEP_ON_GRADIENT")),
        help=_h("STEP_ON_GRADIENT"),
    )
    _emit(flags, "STEP_ON_GRADIENT", step_grad)

    return flags


def advanced_training_section() -> dict:
    """Advanced training flags: reward centering, off-policy IAM, etc."""
    flags = {}

    col1, col2 = st.columns(2)
    with col1:
        rc = st.checkbox(
            "Reward Centering",
            value=bool(_get_preset_val("REWARD_CENTERING")),
            help=_h("REWARD_CENTERING"),
        )
        _emit(flags, "REWARD_CENTERING", rc)

        opiam = st.checkbox(
            "Off-Policy IAM",
            value=bool(_get_preset_val("OFF_POLICY_IAM")),
            help=_h("OFF_POLICY_IAM"),
        )
        _emit(flags, "OFF_POLICY_IAM", opiam)

        sep_vf = st.checkbox(
            "Separate VF Optimizer",
            value=bool(_get_preset_val("SEPARATE_VF_OPTIMIZER")),
            help=_h("SEPARATE_VF_OPTIMIZER"),
        )
        _emit(flags, "SEPARATE_VF_OPTIMIZER", sep_vf)

    with col2:
        vml = st.number_input(
            "Valid Mass Loss Coef",
            min_value=0.0,
            value=float(_get_preset_val("VALID_MASS_LOSS_COEF")),
            step=0.01,
            format="%.3f",
            help=_h("VALID_MASS_LOSS_COEF"),
        )
        _emit(flags, "VALID_MASS_LOSS_COEF", vml)

        prio = st.number_input(
            "Priority Alpha",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("PRIO_ALPHA")),
            step=0.1,
            help=_h("PRIO_ALPHA"),
        )
        _emit(flags, "PRIO_ALPHA", prio)

        prio_beta = st.number_input(
            "Priority Beta0",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("PRIO_BETA0")),
            step=0.1,
            help=_h("PRIO_BETA0"),
        )
        _emit(flags, "PRIO_BETA0", prio_beta)

        rscale = st.number_input(
            "Reward Scale",
            min_value=0.0,
            value=float(_get_preset_val("REWARD_SCALE")),
            step=0.1,
            help=_h("REWARD_SCALE"),
        )
        _emit(flags, "REWARD_SCALE", rscale)

    st.markdown("**VTrace / Importance Ratio Clipping**")
    col3, col4 = st.columns(2)
    with col3:
        rho = st.number_input(
            "RHO Clip",
            value=float(_get_preset_val("RHO_CLIP")),
            step=0.5,
            help=_h("RHO_CLIP"),
        )
        _emit(flags, "RHO_CLIP", rho)
    with col4:
        c_clip = st.number_input(
            "C Clip",
            value=float(_get_preset_val("C_CLIP")),
            step=0.5,
            help=_h("C_CLIP"),
        )
        _emit(flags, "C_CLIP", c_clip)

    diff = st.checkbox(
        "Differentiable Mode",
        value=bool(_get_preset_val("differentiable")),
        help=_h("differentiable"),
    )
    _emit(flags, "differentiable", diff)
    if diff:
        temp = st.number_input(
            "Temperature",
            min_value=0.01,
            value=float(_get_preset_val("temperature")),
            step=0.1,
            help=_h("temperature"),
        )
        _emit(flags, "temperature", temp)

    return flags


def physical_layer_section() -> dict:
    """GN model physical layer params. Only shown for *gn_model env types."""
    flags = {}

    mod_files = _scan_modulations()
    default_mod = _get_preset_val("modulations_csv_filepath")
    # Show relative paths for display
    display_files = [str(p) for p in mod_files]
    if default_mod in display_files:
        mod_idx = display_files.index(default_mod)
    else:
        mod_idx = 0
    if display_files:
        mod_path = st.selectbox(
            "Modulations CSV", display_files, index=mod_idx, help=_h("modulations_csv_filepath")
        )
        _emit(flags, "modulations_csv_filepath", mod_path)

    calc_osnr = st.checkbox(
        "Calculate Minimum OSNR from Spectral Efficiency",
        value=bool(_get_preset_val("calc_minimum_osnr")),
        help=_h("calc_minimum_osnr"),
    )
    _emit(flags, "calc_minimum_osnr", calc_osnr)
    if calc_osnr:
        beta_fec = st.number_input(
            "Pre-FEC BER Target (beta_fec)",
            min_value=1e-6,
            max_value=0.5,
            value=float(_get_preset_val("beta_fec")),
            format="%.1e",
            step=1e-4,
            help=_h("beta_fec"),
        )
        _emit(flags, "beta_fec", beta_fec)

    if st.session_state.get("_env_type", "").endswith("gn_model"):
        fec_rate = st.number_input(
            "FEC Code Rate",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_preset_val("fec_rate")),
            step=0.01,
            format="%.2f",
            help=_h("fec_rate"),
        )
        _emit(flags, "fec_rate", fec_rate)

    st.subheader("Band Configuration")

    # --- Band Data CSV dropdown ---
    band_files = _scan_band_data()
    preset_band = _get_preset_val("band_data_filepath") or ""
    band_options = ["(default)"] + band_files + ["Custom path..."]
    if preset_band and preset_band not in band_files:
        band_default_idx = len(band_options) - 1  # Custom path...
    elif preset_band in band_files:
        band_default_idx = band_files.index(preset_band) + 1
    else:
        band_default_idx = 0
    band_sel = st.selectbox(
        "Band Data CSV",
        band_options,
        index=band_default_idx,
        help=_h("band_data_filepath"),
    )
    if band_sel == "Custom path...":
        band_custom = st.text_input(
            "Band Data CSV path",
            value=preset_band if preset_band not in band_files else "",
        )
        if band_custom.strip():
            flags["band_data_filepath"] = band_custom.strip()
    elif band_sel != "(default)":
        flags["band_data_filepath"] = band_sel

    # --- Noise / Transceiver-Amplifier Data CSV dropdown ---
    noise_files = _scan_noise_data()
    preset_noise = _get_preset_val("noise_data_filepath") or ""
    noise_options = ["(default)"] + noise_files + ["Custom path..."]
    if preset_noise and preset_noise not in noise_files:
        noise_default_idx = len(noise_options) - 1  # Custom path...
    elif preset_noise in noise_files:
        noise_default_idx = noise_files.index(preset_noise) + 1
    else:
        noise_default_idx = 0
    noise_sel = st.selectbox(
        "Transceiver & Amplifier Noise Data CSV",
        noise_options,
        index=noise_default_idx,
        help=_h("noise_data_filepath"),
    )
    if noise_sel == "Custom path...":
        noise_custom = st.text_input(
            "Noise Data CSV path",
            value=preset_noise if preset_noise not in noise_files else "",
        )
        if noise_custom.strip():
            flags["noise_data_filepath"] = noise_custom.strip()
    elif noise_sel != "(default)":
        flags["noise_data_filepath"] = noise_sel

    available_bands = ["C", "L", "S", "U", "E", "O"]
    preset_pref = _get_preset_val("band_preference")
    default_selection = [b.strip().upper() for b in preset_pref.split(",")] if preset_pref else []
    band_prefs = st.multiselect(
        "Band Preference Order (drag to reorder)",
        available_bands,
        default=[b for b in default_selection if b in available_bands],
        help=(
            "Order in which first-fit/last-fit heuristics fill bands. "
            "Selected bands are tried first (in listed order); "
            "unselected bands are appended afterwards. "
            "Leave empty for default frequency-order allocation."
        ),
    )
    if band_prefs:
        flags["band_preference"] = ",".join(band_prefs)

    preset_spb = _get_preset_val("slots_per_band")
    slots_per_band = st.text_input(
        "Slots Per Band",
        value=preset_spb if preset_spb else "",
        help=(
            "Comma-separated number of slots per band (e.g. '45,45'). "
            "Must match the number of selected bands. Leave blank to fill "
            "each band's full spectral width."
        ),
    )
    if slots_per_band.strip():
        flags["slots_per_band"] = slots_per_band.strip()

    st.subheader("Physical Parameters")
    col1, col2 = st.columns(2)
    with col1:
        attenuation = st.number_input(
            "Attenuation [1/m]",
            min_value=0.0,
            value=float(_get_preset_val("attenuation")),
            step=1e-6,
            format="%.3e",
            help=_h("attenuation"),
        )
        _emit(flags, "attenuation", attenuation)

        attenuation_bar = st.number_input(
            "Attenuation Bar [1/m]",
            min_value=0.0,
            value=float(_get_preset_val("attenuation_bar")),
            step=1e-6,
            format="%.3e",
            help=_h("attenuation_bar"),
        )
        _emit(flags, "attenuation_bar", attenuation_bar)

        dispersion_coeff = st.number_input(
            "Dispersion Coefficient [s/m^2]",
            value=float(_get_preset_val("dispersion_coeff")),
            step=1e-6,
            format="%.3e",
            help=_h("dispersion_coeff"),
        )
        _emit(flags, "dispersion_coeff", dispersion_coeff)

        dispersion_slope = st.number_input(
            "Dispersion Slope [s/m^3]",
            value=float(_get_preset_val("dispersion_slope")),
            step=0.1,
            help=_h("dispersion_slope"),
        )
        _emit(flags, "dispersion_slope", dispersion_slope)

        nonlinear_coeff = st.number_input(
            "Nonlinear Coefficient [1/W^2]",
            value=float(_get_preset_val("nonlinear_coefficient")),
            step=1e-4,
            format="%.3e",
            help=_h("nonlinear_coefficient"),
        )
        _emit(flags, "nonlinear_coefficient", nonlinear_coeff)

        raman_gain_slope = st.number_input(
            "Raman Gain Slope [1/(W*m*Hz)]",
            min_value=0.0,
            value=float(_get_preset_val("raman_gain_slope")),
            step=1e-18,
            format="%.2e",
            help=_h("raman_gain_slope"),
        )
        _emit(flags, "raman_gain_slope", raman_gain_slope)

    with col2:
        span = st.number_input(
            "Span Length (km)",
            value=float(_get_preset_val("span_length")),
            step=1.0,
            help=_h("span_length"),
        )
        _emit(flags, "span_length", span)

        span_lumped = st.text_input(
            "Span Lumped Loss (dB, optional)",
            value=(
                ""
                if _get_preset_val("span_lumped_loss_db") is None
                else str(_get_preset_val("span_lumped_loss_db"))
            ),
            help=_h("span_lumped_loss_db"),
        )
        _emit(flags, "span_lumped_loss_db", _none_or_float(span_lumped))

        snr_m = st.number_input(
            "SNR Margin (dB)",
            value=float(_get_preset_val("snr_margin")),
            step=0.1,
            help=_h("snr_margin"),
        )
        _emit(flags, "snr_margin", snr_m)

        gap_ghz = st.number_input(
            "Inter-Band Gap (GHz)",
            value=float(_get_preset_val("inter_band_gap_ghz")),
            step=1.0,
            format="%.1f",
            help=_h("inter_band_gap_ghz"),
        )
        _emit(flags, "inter_band_gap_ghz", gap_ghz)

    num_sub = st.number_input(
        "Nyquist Subchannels per Slot",
        min_value=1,
        max_value=64,
        value=int(_get_preset_val("num_subchannels")),
        step=1,
        help=_h("num_subchannels"),
    )
    _emit(flags, "num_subchannels", num_sub)

    st.subheader("Launch Power")
    power_types = ["fixed", "tabular", "rl", "scaled"]
    preset_pt = _get_preset_val("launch_power_type")
    pt_idx = power_types.index(preset_pt) if preset_pt in power_types else 0
    lp_type = st.selectbox(
        "Launch Power Type",
        power_types,
        index=pt_idx,
        help=_h("launch_power_type"),
    )
    _emit(flags, "launch_power_type", lp_type)

    col_a, col_b = st.columns(2)
    with col_a:
        max_pf = st.number_input(
            "Max Power per Fibre (dBm)",
            value=float(_get_preset_val("max_power_per_fibre")),
            step=0.5,
            format="%.1f",
            help=_h("max_power_per_fibre"),
        )
        _emit(flags, "max_power_per_fibre", max_pf)

    with col_b:
        preset_ppc = _get_preset_val("power_per_channel")
        ppc = st.number_input(
            "Power per Channel (dBm, blank = auto)",
            value=float(preset_ppc) if preset_ppc is not None else 0.0,
            step=0.5,
            format="%.1f",
            help=_h("power_per_channel"),
        )
        if preset_ppc is not None or ppc != 0.0:
            _emit(flags, "power_per_channel", ppc)

    preset_ppc_band = _get_preset_val("power_per_channel_per_band")
    ppc_band = st.text_input(
        "Power per Channel per Band (dBm, comma-separated)",
        value=str(preset_ppc_band) if preset_ppc_band is not None else "",
        help="Comma-separated per-channel launch power values in dBm, one per band "
        "in band_preference order (e.g. '2.3,2.5' for C,L). "
        "Overrides Power per Channel when set.",
    )
    if ppc_band.strip():
        _emit(flags, "power_per_channel_per_band", ppc_band.strip())

    preset_lp_csv = _get_preset_val("launch_power_csv")
    lp_csv = st.text_input(
        "Launch Power CSV (path)",
        value=str(preset_lp_csv) if preset_lp_csv is not None else "",
        help="Path to a CSV file with columns 'slot_index', 'freq_ghz', 'power_dbm' "
        "specifying per-slot launch power in dBm. "
        "Slots absent from the file keep the default power. "
        "Overrides Power per Channel and Power per Channel per Band when set.",
    )
    if lp_csv.strip():
        _emit(flags, "launch_power_csv", lp_csv.strip())

    st.subheader("Distributed Raman Amplification")
    raman_enabled = st.checkbox(
        "Enable Raman Amplification",
        value=bool(_get_preset_val("use_raman_amp")),
        help="Enable Distributed Raman Amplification model for NLI calculation. "
        "Uses triangular Raman gain approximation to derive fit parameters.",
    )
    _emit(flags, "use_raman_amp", raman_enabled)

    if raman_enabled:
        raman_max_bw = st.number_input(
            "Max Modulated Bandwidth (THz)",
            min_value=1.0,
            max_value=100.0,
            value=float(_get_preset_val("raman_max_bandwidth_thz")),
            step=1.0,
            format="%.1f",
            help="Maximum modulated bandwidth for triangular Raman approximation validity. "
            "Bands are trimmed to fit within this limit when band_preference is set.",
        )
        _emit(flags, "raman_max_bandwidth_thz", raman_max_bw)

        def _list_val(key):
            """Get preset value as a comma-separated string for text_input."""
            v = _get_preset_val(key)
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            return ",".join(str(x) for x in v)

        col_fw, col_bw = st.columns(2)
        with col_fw:
            st.markdown("**Forward Pumps**")
            fw_pow = st.text_input(
                "Pump Powers (W, comma-sep)",
                value=_list_val("raman_pump_power_fw"),
                help="Forward Raman pump powers in Watts, comma-separated.",
                key="dra_fw_pow",
            )
            if fw_pow.strip():
                flags["raman_pump_power_fw"] = fw_pow.strip()

            fw_freq = st.text_input(
                "Pump Frequencies (Hz, comma-sep)",
                value=_list_val("raman_pump_freq_fw"),
                help="Forward Raman pump frequencies in Hz, comma-separated.",
                key="dra_fw_freq",
            )
            if fw_freq.strip():
                flags["raman_pump_freq_fw"] = fw_freq.strip()

        with col_bw:
            st.markdown("**Backward Pumps**")
            bw_pow = st.text_input(
                "Pump Powers (W, comma-sep)",
                value=_list_val("raman_pump_power_bw"),
                help="Backward Raman pump powers in Watts, comma-separated.",
                key="dra_bw_pow",
            )
            if bw_pow.strip():
                flags["raman_pump_power_bw"] = bw_pow.strip()

            bw_freq = st.text_input(
                "Pump Frequencies (Hz, comma-sep)",
                value=_list_val("raman_pump_freq_bw"),
                help="Backward Raman pump frequencies in Hz, comma-separated.",
                key="dra_bw_freq",
            )
            if bw_freq.strip():
                flags["raman_pump_freq_bw"] = bw_freq.strip()

    return flags


def logging_section() -> dict:
    """Logging, saving, plotting, eval-during-training."""
    flags = {}

    col1, col2 = st.columns(2)
    with col1:
        wandb = st.checkbox(
            "Enable W&B Logging", value=bool(_get_preset_val("WANDB")), help=_h("WANDB")
        )
        _emit(flags, "WANDB", wandb)

        if wandb:
            project = st.text_input(
                "W&B Project", value=_get_preset_val("PROJECT") or "", help=_h("PROJECT")
            )
            if project:
                _emit(flags, "PROJECT", project)

            exp_name = st.text_input(
                "Experiment Name",
                value=_get_preset_val("EXPERIMENT_NAME") or "",
                help=_h("EXPERIMENT_NAME"),
            )
            if exp_name:
                _emit(flags, "EXPERIMENT_NAME", exp_name)

            dsf = st.number_input(
                "Downsample Factor",
                min_value=1,
                value=int(_get_preset_val("DOWNSAMPLE_FACTOR")),
                step=1,
                help=_h("DOWNSAMPLE_FACTOR"),
            )
            _emit(flags, "DOWNSAMPLE_FACTOR", int(dsf))

        save = st.checkbox(
            "Save Model", value=bool(_get_preset_val("SAVE_MODEL")), help=_h("SAVE_MODEL")
        )
        _emit(flags, "SAVE_MODEL", save)

        if save:
            mpath = st.text_input(
                "Model Save Path", value=_get_preset_val("MODEL_PATH") or "", help=_h("MODEL_PATH")
            )
            if mpath:
                flags["MODEL_PATH"] = mpath

    with col2:
        plotting = st.checkbox(
            "Enable Plotting", value=bool(_get_preset_val("PLOTTING")), help=_h("PLOTTING")
        )
        _emit(flags, "PLOTTING", plotting)

        profile = st.checkbox(
            "Enable Profiler",
            value=bool(_get_preset_val("PROFILE")),
            help=_h("PROFILE"),
        )
        _emit(flags, "PROFILE", profile)

        edt = st.checkbox(
            "Eval During Training",
            value=bool(_get_preset_val("EVAL_DURING_TRAINING")),
            help=_h("EVAL_DURING_TRAINING"),
        )
        _emit(flags, "EVAL_DURING_TRAINING", edt)

        data_out = st.text_input(
            "Run Summary Output File (JSONL)",
            value=_get_preset_val("DATA_OUTPUT_FILE") or "",
            help=_h("DATA_OUTPUT_FILE"),
        )
        if data_out.strip():
            flags["DATA_OUTPUT_FILE"] = data_out.strip()

        episode_data_out = st.text_input(
            "Episode Data Output File (CSV)",
            value=_get_preset_val("EPISODE_DATA_OUTPUT_FILE") or "",
            help=_h("EPISODE_DATA_OUTPUT_FILE"),
        )
        if episode_data_out.strip():
            flags["EPISODE_DATA_OUTPUT_FILE"] = episode_data_out.strip()

    st.markdown("### Evaluation Rendering")
    st.caption(
        "Only applies to Heuristic Evaluation or Model Evaluation runs. "
        "GUI currently supports file playback (save) rather than live pop-up rendering."
    )
    colr1, colr2 = st.columns(2)
    with colr1:
        render_mode_labels = {
            "Off": "off",
            "Save Recording": "save",
        }
        current_mode = str(_get_preset_val("RENDER_EVAL_MODE") or "off").lower()
        reverse = {v: k for k, v in render_mode_labels.items()}
        selected_label = st.selectbox(
            "Render Mode",
            list(render_mode_labels.keys()),
            index=list(render_mode_labels.keys()).index(reverse.get(current_mode, "Off")),
            help=_h("RENDER_EVAL_MODE"),
        )
        _emit(flags, "RENDER_EVAL_MODE", render_mode_labels[selected_label])

        render_fps = st.number_input(
            "Render FPS",
            min_value=0.1,
            value=float(_get_preset_val("RENDER_FPS")),
            step=0.1,
            help=_h("RENDER_FPS"),
        )
        _emit(flags, "RENDER_FPS", render_fps)
        render_scale = st.number_input(
            "Render Scale",
            min_value=0.4,
            max_value=2.0,
            value=float(_get_preset_val("RENDER_SCALE")),
            step=0.1,
            help=_h("RENDER_SCALE"),
        )
        _emit(flags, "RENDER_SCALE", render_scale)

    with colr2:
        render_max_steps = st.number_input(
            "Max Render Steps",
            min_value=1,
            value=int(_get_preset_val("RENDER_MAX_STEPS")),
            step=10,
            help=_h("RENDER_MAX_STEPS"),
        )
        _emit(flags, "RENDER_MAX_STEPS", int(render_max_steps))

    render_out = st.text_input(
        "Render Output File (optional .gif/.mp4)",
        value=_get_preset_val("RENDER_OUTPUT_FILE") or "",
        help=_h("RENDER_OUTPUT_FILE"),
    )
    if render_out.strip():
        flags["RENDER_OUTPUT_FILE"] = render_out.strip()

    return flags
