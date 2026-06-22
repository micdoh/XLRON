"""XLRON GUI — Streamlit-based command builder and process manager."""

import base64
import time
from pathlib import Path

import streamlit as st

from xlron.gui.presets import PRESETS
from xlron.gui.process import get_all_runs, launch_run, stop_run, tail_log
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

# Flags that execution_mode_section() emits to mark the selected execution mode.
# That section is the single source of truth for them: each render it re-emits
# exactly the discriminators matching the mode radio. They must therefore NOT be
# seeded from a loaded preset — otherwise a stale preset value (e.g.
# EVAL_MODEL=True) survives after the user switches the radio to a mode that
# doesn't set it, leaving the wrong tabs hidden and a stale flag in the command.
_MODE_DISCRIMINATOR_FLAGS = frozenset(
    {"EVAL_HEURISTIC", "EVAL_MODEL", "ACTION_OPTIMIZATION", "differentiable"}
)


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

# Theme tweaks beyond what .streamlit/config.toml can express:
#   * keep the sidebar white (config's secondaryBackgroundColor stays mint
#     so input widgets pick it up, but applying that mint to the whole
#     left rail looks too heavy)
#   * recolor st.info alerts from default blue to a teal-tinted theme color
st.markdown(
    """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div {
    /* Match Streamlit's default code/output-box grey */
    background-color: #F0F2F6 !important;
}
/* st.info → single solid mint box, no internal borders.
 * Streamlit wraps the alert in several nested divs and at least one of
 * them keeps a default blue background.  Make every descendant inside an
 * info alert transparent so only the outer mint shows, and strip all
 * internal borders/box-shadows.  Use :has() to restrict to info alerts
 * so warning/error alerts keep their default colors. */
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]),
[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {
    background-color: #E1F6F2 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0.5rem !important;
    color: #000000 !important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) *,
[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) * {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #000000 !important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) svg,
[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) svg {
    fill: #1D605B !important;
}
</style>
<script>
(function() {
  const KEY = "xlron_sidebar_scroll_top";
  const getSidebar = () => document.querySelector('[data-testid="stSidebarContent"]');
  const restore = () => {
    const el = getSidebar();
    if (!el) return;
    const saved = window.sessionStorage.getItem(KEY);
    if (saved !== null) el.scrollTop = parseInt(saved, 10) || 0;
  };
  const attach = () => {
    const el = getSidebar();
    if (!el || el.dataset.scrollBound === "1") return;
    el.dataset.scrollBound = "1";
    el.addEventListener("scroll", () => {
      window.sessionStorage.setItem(KEY, String(el.scrollTop));
    }, { passive: true });
  };
  restore();
  attach();
  setTimeout(() => { restore(); attach(); }, 50);
})();
</script>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def build_command(all_flags: dict) -> str:
    """Build a CLI command string from the collected flags, omitting defaults."""
    bounds_method = st.session_state.get("_bounds_method")
    is_diff_sim = st.session_state.get("_diff_sim_mode", False)
    if bounds_method == "cutsets":
        parts = ["python", "-m", "xlron.bounds.cutsets_bounds"]
    elif bounds_method == "reconfigurable":
        parts = ["python", "xlron/bounds/reconfigurable_routing_bounds.py"]
    elif is_diff_sim:
        parts = ["python", "-m", "xlron.diff_sim.run_direct_optimization"]
    else:
        parts = ["python", "-m", "xlron.train.train"]
    for key, value in sorted(all_flags.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
            elif DEFAULTS.get(key) is True:
                parts.append(f"--no{key}")
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
    st.markdown(
        '<p style="color: #000000; font-size: 0.95rem; margin: 0.25rem 0 0.75rem 0;">'
        "Optical Network Simulation and RL Training GUI"
        "</p>",
        unsafe_allow_html=True,
    )
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
    run_btn_col, stop_btn_col = st.columns(2)
    run_btn_placeholder = run_btn_col.empty()
    stop_btn_placeholder = stop_btn_col.empty()

    st.divider()
    st.subheader("Run History")
    history_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Main area — two-column layout: widgets (left) + output (right)
# ---------------------------------------------------------------------------

all_flags: dict = {}

# Initialize preset if not set
if "_loaded_preset" not in st.session_state:
    st.session_state["_loaded_preset"] = {}

# Seed with preset values so flags without widgets are still included in the
# command. Mode-discriminator flags are excluded: execution_mode_section() owns
# them and re-emits the ones matching the current mode, so seeding them here
# would let a stale preset value survive a mode switch (see issue #22).
all_flags.update(
    {
        k: v
        for k, v in st.session_state.get("_loaded_preset", {}).items()
        if k not in _MODE_DISCRIMINATOR_FLAGS
    }
)

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

        is_diff_sim = st.session_state.get("_diff_sim_mode", False)
        if not st.session_state.get("_bounds_mode", False) and not is_diff_sim:
            st.header("Execution")
            exec_flags = execution_section()
            all_flags.update(exec_flags)
        elif is_diff_sim:
            st.header("Execution")
            exec_flags = {}
            col1, col2 = st.columns(2)
            with col1:
                max_req = st.number_input(
                    "Number of Requests to Optimise",
                    min_value=1,
                    value=int(_get_preset_val("max_requests")),
                    step=10,
                    format="%d",
                    help=_h("max_requests"),
                )
                _emit(exec_flags, "max_requests", int(max_req))
            with col2:
                total = st.number_input(
                    "Total Timesteps",
                    min_value=1,
                    value=int(_get_preset_val("TOTAL_TIMESTEPS")),
                    step=1000,
                    format="%d",
                    help=_h("TOTAL_TIMESTEPS"),
                )
                _emit(exec_flags, "TOTAL_TIMESTEPS", int(total))
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(_get_preset_val("SEED")),
                help=_h("SEED"),
            )
            _emit(exec_flags, "SEED", int(seed))
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
    is_diff_sim_mode = st.session_state.get("_diff_sim_mode", False)
    is_rl_mode = (
        "EVAL_HEURISTIC" not in all_flags
        and "EVAL_MODEL" not in all_flags
        and not is_bounds_mode
        and not is_diff_sim_mode
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
            # GN-model physical params use attenuation/attenuation_bar; drop legacy alpha if loaded from old presets.
            all_flags.pop("alpha", None)
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

# MODEL_PATH is a single CLI flag bound to two widgets: the load path in the
# execution-mode section (Model Evaluation / Retrain) and the save path in the
# logging section. The logging section renders last, so without this its
# "Model Save Path" (often a stale preset value) would clobber a load path the
# user just typed. Give the load-model path priority when that mode is active.
if "MODEL_PATH" in mode_flags:
    all_flags["MODEL_PATH"] = mode_flags["MODEL_PATH"]


# ---------------------------------------------------------------------------
# Build command and populate sidebar controls
# ---------------------------------------------------------------------------

command = build_command(all_flags)
command_multiline = _format_command_multiline(command)

with st.sidebar:
    cmd_placeholder.code(command_multiline, language="bash")

    if run_btn_placeholder.button("Run", type="primary", width="stretch"):
        info = launch_run(command)
        st.session_state["_active_run_id"] = info.run_id
        st.session_state["_log_offset"] = 0
        st.rerun()

    # Stop button (no live sidebar status polling to avoid UI jumpiness)
    active_run = st.session_state.get("_active_run_id")
    runs = get_all_runs()
    active_entry = (
        next((r for r in runs if r["run_id"] == active_run), None) if active_run else None
    )
    if active_run:
        if stop_btn_placeholder.button("Stop", type="secondary", width="stretch"):
            stop_run(active_run)
            st.session_state.pop("_active_run_id", None)
            st.rerun()
        st.caption(f"Selected run: {active_run}")
        if active_entry is not None:
            st.caption(f"Status: {active_entry['status']}")
            with st.expander("Current Command", expanded=False):
                st.code(_format_command_multiline(active_entry["command"]), language="bash")

    if runs:
        for run in runs[:8]:
            col_id, col_status = history_placeholder.container().columns([5, 1])
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
    runs = get_all_runs()
    if not active_run_id and runs:
        # Fall back to the newest run so playback/logs are visible after page reloads.
        active_run_id = runs[0]["run_id"]
        st.session_state["_active_run_id"] = active_run_id

    if active_run_id:
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

            gif_files = sorted(run_dir.glob("*.gif"))
            mp4_files = sorted(run_dir.glob("*.mp4"))
            if gif_files or mp4_files:
                with st.expander("Render Playback", expanded=True):
                    for gf in gif_files:
                        try:
                            size = gf.stat().st_size
                            if run_entry["status"] == "running" and size == 0:
                                continue
                        except OSError:
                            continue
                        size_mb = size / (1024 * 1024)
                        st.caption(f"{gf.name} ({size_mb:.1f} MB)")
                        try:
                            if size_mb <= 20.0:
                                gif_b64 = base64.b64encode(gf.read_bytes()).decode("ascii")
                                st.markdown(
                                    f'<img src="data:image/gif;base64,{gif_b64}" style="width:100%;height:auto;" />',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.image(str(gf), width="stretch")
                                st.caption("Large GIF previewed statically in Streamlit.")
                        except Exception:
                            if run_entry["status"] == "running":
                                st.caption(f"Render file is still being written: {gf.name}")
                            else:
                                st.warning(f"Could not load render file: {gf.name}")
                    for mf in mp4_files:
                        try:
                            size = mf.stat().st_size
                            if run_entry["status"] == "running" and size == 0:
                                continue
                        except OSError:
                            continue
                        size_mb = size / (1024 * 1024)
                        st.caption(f"{mf.name} ({size_mb:.1f} MB)")
                        try:
                            st.video(str(mf))
                            st.caption(mf.name)
                        except Exception:
                            if run_entry["status"] == "running":
                                st.caption(f"Render file is still being written: {mf.name}")
                            else:
                                st.warning(f"Could not load render file: {mf.name}")

            # Auto-refresh while process is running
            if run_entry["status"] == "running":
                time.sleep(1)
                st.rerun()
        else:
            st.caption("Run not found in registry.")
    else:
        st.caption("No active run. Configure and click **Run** to start.")
