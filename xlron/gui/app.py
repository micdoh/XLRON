"""XLRON GUI — Streamlit-based command builder and process manager."""

import time
from pathlib import Path

import streamlit as st

from xlron.gui.presets import PRESETS
from xlron.gui.process import get_active_runs, get_all_runs, launch_run, stop_run, tail_log
from xlron.gui.widgets import (
    DEFAULTS,
    _emit,
    _get_preset_val,
    _h,
    advanced_training_section,
    architecture_section,
    environment_section,
    execution_mode_section,
    execution_section,
    logging_section,
    physical_layer_section,
    ppo_section,
    schedule_section,
    traffic_section,
)

_LOGO_PATH = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "images" / "xlron_nobackground.png"
)
_DOCS_URL = "https://micdoh.github.io/XLRON/"
_PRESETS_PATH = Path(__file__).resolve().parent / "presets.py"


# ---------------------------------------------------------------------------
# Preset saving helpers
# ---------------------------------------------------------------------------


def _format_preset_value(v) -> str:
    """Format a Python value for the presets source file."""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, int):
        return f"{v:_}" if abs(v) >= 10_000 else repr(v)
    if isinstance(v, float):
        if v != 0.0 and abs(v) < 0.01:
            s = f"{v:e}"
            mantissa, exp = s.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            return f"{mantissa}e{int(exp)}"
        return repr(v)
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def _write_presets_file() -> None:
    """Rewrite presets.py with the current in-memory PRESETS dict."""
    lines = ['"""Named preset configurations for common XLRON experiments."""\n\n']
    lines.append("PRESETS = {\n")
    for name, config in PRESETS.items():
        if not config:
            lines.append(f'    "{name}": {{}},\n')
        else:
            lines.append(f'    "{name}": {{\n')
            for k, v in config.items():
                lines.append(f'        "{k}": {_format_preset_value(v)},\n')
            lines.append("    },\n")
    lines.append("}\n")
    _PRESETS_PATH.write_text("".join(lines))


def _save_preset(name: str, config: dict) -> None:
    """Save a preset to presets.py and update the in-memory dict."""
    PRESETS[name] = config
    _write_presets_file()


st.set_page_config(page_title="XLRON", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def build_command(all_flags: dict) -> str:
    """Build a CLI command string from the collected flags, omitting defaults."""
    bounds_method = st.session_state.get("_bounds_method")
    if bounds_method == "cutsets":
        parts = ["python", "-m", "xlron.bounds.cutsets_bounds"]
    elif bounds_method == "reconfigurable":
        parts = ["python", "xlron/bounds/reconfigurable_routing_bounds.py"]
    else:
        parts = ["python", "-m", "xlron.train.train"]
    for key, value in sorted(all_flags.items()):
        if value is None or value is False:
            continue
        if value is True:
            parts.append(f"--{key}")
        elif isinstance(value, list):
            parts.append(f"--{key}={','.join(str(v) for v in value)}")
        elif isinstance(value, float):
            if value == int(value):
                # Whole floats as integers for readability (100.0 → 100)
                parts.append(f"--{key}={int(value)}")
            elif abs(value) >= 1e6 or (0 < abs(value) < 1e-3):
                # Scientific notation with enough precision (use g to trim trailing zeros)
                parts.append(f"--{key}={value:.6g}")
            else:
                parts.append(f"--{key}={value}")
        else:
            parts.append(f"--{key}={value}")
    return " ".join(parts)


def _format_command_multiline(cmd: str) -> str:
    """Format command with backslash line continuations for readability."""
    parts = cmd.split(" --")
    if len(parts) <= 1:
        return cmd
    lines = [parts[0]]
    for part in parts[1:]:
        lines.append(f"    --{part}")
    return " \\\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width="stretch")
    else:
        st.title("XLRON")
    st.caption("Optical Network RL Training GUI")
    st.markdown(f"[Documentation]({_DOCS_URL})")

    # Preset selector
    st.subheader("Presets")
    preset_name = st.selectbox("Load a preset", list(PRESETS.keys()))
    if st.button("Load Preset", width="stretch"):
        st.session_state["_loaded_preset"] = PRESETS[preset_name]
        st.rerun()

    if st.button("Save Current as Preset", width="stretch"):
        st.session_state["_show_save_preset"] = True

    st.divider()

    # Run controls — placeholder, filled after command is built
    st.subheader("Run Controls")
    cmd_placeholder = st.empty()
    copy_btn_placeholder = st.empty()
    run_btn_col, stop_btn_col = st.columns(2)
    run_btn_placeholder = run_btn_col.empty()
    stop_btn_placeholder = stop_btn_col.empty()
    status_placeholder = st.empty()

    st.divider()

    # Run history
    st.subheader("Run History")
    history_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Main area — two-column layout: widgets (left) + output (right)
# ---------------------------------------------------------------------------

all_flags: dict = {}

# Initialize preset if not set
if "_loaded_preset" not in st.session_state:
    st.session_state["_loaded_preset"] = {}

# Seed with preset values so flags without widgets are still included in the command
all_flags.update(st.session_state.get("_loaded_preset", {}))

col_widgets, col_output = st.columns([3, 2])

# ---- Left column: configuration widgets ----
with col_widgets:
    tab_setup, tab_model, tab_physical, tab_logging = st.tabs(
        ["Setup", "Model & Training", "Physical Layer", "Logging & Output"]
    )

    with tab_setup:
        st.header("Execution Mode")
        mode_flags = execution_mode_section()
        all_flags.update(mode_flags)

        st.header("Environment")
        env_flags = environment_section()
        all_flags.update(env_flags)

        st.header("Traffic")
        traffic_flags = traffic_section()
        all_flags.update(traffic_flags)

        if not st.session_state.get("_bounds_mode", False):
            st.header("Execution")
            exec_flags = execution_section()
            all_flags.update(exec_flags)
        else:
            st.header("Execution")
            bounds_method = st.session_state.get("_bounds_method")
            exec_flags = {}
            if bounds_method == "reconfigurable":
                total = st.number_input(
                    "Total Timesteps (requests per trial)",
                    min_value=1,
                    value=int(_get_preset_val("TOTAL_TIMESTEPS")),
                    step=10000,
                    format="%d",
                    help=_h("TOTAL_TIMESTEPS"),
                )
                _emit(exec_flags, "TOTAL_TIMESTEPS", int(total))
            elif bounds_method == "cutsets":
                max_req = st.number_input(
                    "Requests per Trial",
                    min_value=1,
                    value=int(_get_preset_val("max_requests")),
                    step=10000,
                    format="%d",
                    help=_h("max_requests"),
                )
                _emit(exec_flags, "max_requests", int(max_req))

            seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(_get_preset_val("SEED")),
                help=_h("SEED"),
            )
            _emit(exec_flags, "SEED", int(seed))
            all_flags.update(exec_flags)

    is_bounds_mode = st.session_state.get("_bounds_mode", False)
    is_rl_mode = (
        "EVAL_HEURISTIC" not in all_flags and "EVAL_MODEL" not in all_flags and not is_bounds_mode
    )
    env_type = st.session_state.get("_env_type", "rmsa")

    with tab_model:
        if is_rl_mode:
            st.header("Architecture")
            arch_flags = architecture_section()
            all_flags.update(arch_flags)

            st.header("PPO Hyperparameters")
            ppo_flags = ppo_section()
            all_flags.update(ppo_flags)

            with st.expander("Schedules", expanded=False):
                sched_flags = schedule_section()
                all_flags.update(sched_flags)

            with st.expander("Advanced Training", expanded=False):
                adv_flags = advanced_training_section()
                all_flags.update(adv_flags)
        else:
            st.info("Model & Training options are only available in RL Training mode.")

    with tab_physical:
        if "gn_model" in env_type:
            st.header("GN Model Physical Layer")
            phys_flags = physical_layer_section()
            all_flags.update(phys_flags)
        else:
            st.info(
                "Physical layer options are only available for GN model environment types "
                "(rsa_gn_model, rmsa_gn_model)."
            )

    with tab_logging:
        st.header("Logging & Output")
        log_flags = logging_section()
        all_flags.update(log_flags)


# ---------------------------------------------------------------------------
# Build command and populate sidebar controls
# ---------------------------------------------------------------------------

command = build_command(all_flags)
command_multiline = _format_command_multiline(command)

with st.sidebar:
    cmd_placeholder.code(command_multiline, language="bash")

    # Copy Command button — shows the command in a modal/popover for easy copy
    if copy_btn_placeholder.button("Copy Command", width="stretch"):
        st.session_state["_show_command"] = True

    if run_btn_placeholder.button("Run", type="primary", width="stretch"):
        info = launch_run(command)
        st.session_state["_active_run_id"] = info.run_id
        st.session_state["_log_offset"] = 0
        st.rerun()

    # Stop button
    active_run = st.session_state.get("_active_run_id")
    if active_run:
        active_runs = get_active_runs()
        active_ids = {r["run_id"] for r in active_runs}
        if active_run in active_ids:
            if stop_btn_placeholder.button("Stop", type="secondary", width="stretch"):
                stop_run(active_run)
                st.session_state.pop("_active_run_id", None)
                st.rerun()
            status_placeholder.success(f"Running: {active_run}")
        else:
            status_placeholder.info("No active run")
            st.session_state.pop("_active_run_id", None)

    # Run history
    runs = get_all_runs()
    if runs:
        for run in runs[:5]:
            col_id, col_status = history_placeholder.container().columns([3, 1])
            with col_id:
                label = run["run_id"]
                if st.button(label, key=f"reconnect_{run['run_id']}"):
                    st.session_state["_active_run_id"] = run["run_id"]
                    st.session_state["_log_offset"] = 0
                    st.rerun()
            with col_status:
                if run["status"] == "running":
                    st.markdown("🟢")
                elif run["status"] == "finished":
                    st.markdown("⬜")
                else:
                    st.markdown("🔴")
    else:
        history_placeholder.caption("No runs yet")


# ---------------------------------------------------------------------------
# Copy command dialog
# ---------------------------------------------------------------------------

if st.session_state.get("_show_command"):

    @st.dialog("Copy Command")
    def _show_copy_dialog():
        st.markdown("Copy this command to run XLRON from the terminal:")
        st.code(command_multiline, language="bash")
        if st.button("Close"):
            st.session_state["_show_command"] = False
            st.rerun()

    _show_copy_dialog()


# ---------------------------------------------------------------------------
# Save preset dialog
# ---------------------------------------------------------------------------

if st.session_state.get("_show_save_preset"):

    @st.dialog("Save as Preset")
    def _show_save_preset_dialog():
        name = st.text_input("Preset Name", placeholder="e.g., My Custom Config")
        if name.strip() and name.strip() in PRESETS:
            st.warning(f'A preset named "{name.strip()}" already exists and will be overwritten.')
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("Save", type="primary", width="stretch"):
                if not name.strip():
                    st.error("Please enter a name.")
                else:
                    _save_preset(name.strip(), dict(all_flags))
                    st.session_state["_show_save_preset"] = False
                    st.rerun()
        with col_cancel:
            if st.button("Cancel", width="stretch"):
                st.session_state["_show_save_preset"] = False
                st.rerun()

    _show_save_preset_dialog()


# ---------------------------------------------------------------------------
# Right column: output area — live log stream
# ---------------------------------------------------------------------------

with col_output:
    st.subheader("Output")

    active_run_id = st.session_state.get("_active_run_id")
    if active_run_id:
        runs = get_all_runs()
        run_entry = next((r for r in runs if r["run_id"] == active_run_id), None)
        if run_entry:
            log_path = run_entry["log_path"]

            # Read the full log every render (fast for files on disk)
            full_text, new_offset = tail_log(log_path, 0)
            st.session_state["_log_offset"] = new_offset

            if full_text:
                lines = full_text.splitlines()
                if len(lines) > 500:
                    display_text = f"... ({len(lines) - 500} lines above) ...\n"
                    display_text += "\n".join(lines[-500:])
                else:
                    display_text = full_text
                st.code(display_text, language="text")
            else:
                st.caption("Waiting for output...")

            # Check for plot images saved by the subprocess
            run_dir = Path(log_path).parent
            plot_files = sorted(run_dir.glob("*.png"))
            if plot_files:
                st.subheader("Plots")
                for pf in plot_files:
                    st.image(str(pf), caption=pf.stem, width="stretch")

            # Auto-refresh while process is running
            if run_entry["status"] == "running":
                time.sleep(1)
                st.rerun()
        else:
            st.caption("Run not found in registry.")
    else:
        st.caption("No active run. Configure and click **Run** to start.")
