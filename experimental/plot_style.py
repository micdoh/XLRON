"""Global plot style configuration for XLRON benchmark figures.

Import and call configure_style() at the top of any plotting script to ensure
consistent publication-quality formatting across all figures.

Style based on JOCN2024 plots (Helvetica/Arial, large fonts for readability).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer

# -- Color Palettes -----------------------------------------------------------

# Core palette
PRIMARY_COLORS = ["#30A08E", "#E1F6F2", "#A5E5D7", "#1D605B"]
ACCENT_COLORS = ["#8064A2", "#E75D72", "#69D3BE", "#FF9A56"]

# Combined sequence for general use (primary dark/mid + all accents)
PALETTE = [
    PRIMARY_COLORS[0],  # teal
    ACCENT_COLORS[1],   # coral
    ACCENT_COLORS[0],   # purple
    ACCENT_COLORS[3],   # orange
    PRIMARY_COLORS[3],  # dark teal
    ACCENT_COLORS[2],   # seafoam
    PRIMARY_COLORS[2],  # medium mint
    ACCENT_COLORS[1],   # coral (repeat for >7 series)
]

# Table styling
TABLE_HEADER_COLOR = PRIMARY_COLORS[3]   # dark teal
TABLE_ROW_ALT_COLOR = PRIMARY_COLORS[1]  # light mint

# Heatmap colormaps (built lazily by configure_style or on first access)
HEATMAP_GPU_COLORS = [
    PRIMARY_COLORS[1],  # light mint
    ACCENT_COLORS[2],   # seafoam
    PRIMARY_COLORS[0],  # teal
    PRIMARY_COLORS[3],  # dark teal
]
HEATMAP_CPU_COLORS = [
    "#F3E8FF",          # pale lavender
    ACCENT_COLORS[0],   # purple
    ACCENT_COLORS[1],   # coral
    "#9E2B3A",          # deep berry
]

ENV_TYPE_COLORS = {
    "rwa": PRIMARY_COLORS[0],          # teal
    "rmsa": ACCENT_COLORS[3],          # orange
    "rsa_gn_model": ACCENT_COLORS[0],  # purple
    "rmsa_gn_model": ACCENT_COLORS[1], # coral
    "rwa_lightpath_reuse": ACCENT_COLORS[2],  # seafoam
}

TOPOLOGY_COLORS = {
    "5node_directed": PRIMARY_COLORS[0],          # teal
    "cost239_deeprmsa_directed": ACCENT_COLORS[3], # orange
    "cost239_ptrnet_published_directed": ACCENT_COLORS[2],  # seafoam
    "cost239_ptrnet_real_directed": PRIMARY_COLORS[3],      # dark teal
    "nsfnet_deeprmsa_directed": ACCENT_COLORS[1],  # coral
    "german17_directed": ACCENT_COLORS[0],         # purple
    "usnet_gcnrnn_directed": PRIMARY_COLORS[2],    # medium mint
    "usnet_ptrnet_directed": "#C4487A",            # rose (extra)
    "jpn48_directed": "#5B9BD5",                   # steel blue (extra)
    "conus_directed": PRIMARY_COLORS[3],           # dark teal
}

TOPOLOGY_DISPLAY = {
    "5node_directed": "5-node",
    "cost239_deeprmsa_directed": "COST239",
    "cost239_ptrnet_published_directed": "COST239-A",
    "cost239_ptrnet_real_directed": "COST239-B",
    "nsfnet_deeprmsa_directed": "NSFNET",
    "german17_directed": "German17",
    "usnet_gcnrnn_directed": "USNET-A",
    "usnet_ptrnet_directed": "USNET",
    "jpn48_directed": "JPN48",
    "conus_directed": "CONUS",
}

ENV_TYPE_DISPLAY = {
    "rwa": "RWA",
    "rmsa": "RMSA",
    "rsa_gn_model": "RSA-GN",
    "rmsa_gn_model": "RMSA-GN",
    "rwa_lightpath_reuse": "RWA-LR",
}

DEVICE_COLORS = {
    "cpu": ACCENT_COLORS[0],   # purple
    "gpu": ACCENT_COLORS[1],   # coral
}

BAND_COLORS = {
    "C": PRIMARY_COLORS[0],    # teal
    "C,L": ACCENT_COLORS[3],   # orange
    "C,L,S": ACCENT_COLORS[0], # purple
}

COMPARISON_COLORS = {
    "deeprmsa_original": PRIMARY_COLORS[0],  # teal
    "xlron": ACCENT_COLORS[3],               # orange
    "optical_rl_gym": ACCENT_COLORS[0],      # purple
}

REFERENCE_LINE_COLOR = ACCENT_COLORS[1]  # coral

BAND_DISPLAY = {
    "C": "C (43 x 100 GHz)",
    "C,L": "C+L (115 x 100 GHz)",
    "C,L,S": "C+L+S (209 x 100 GHz)",
}


# -- Style Configuration -----------------------------------------------------


def configure_style(
    font_size: int = 32,
    axes_label_size: int = 34,
    tick_size: int = 24,
    legend_size: int = 20,
    figure_dpi: int = 150,
):
    """Apply global matplotlib style for publication-quality figures."""
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": axes_label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "figure.dpi": figure_dpi,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )


# -- Utilities ----------------------------------------------------------------


def get_topology_order() -> list[str]:
    """Return directed topologies ordered roughly by network size."""
    return [
        "cost239_deeprmsa_directed",
        "nsfnet_deeprmsa_directed",
        "german17_directed",
        "usnet_ptrnet_directed",
        "jpn48_directed",
        "conus_directed",
    ]


def format_fps(fps: float) -> str:
    """Format FPS value for human-readable display."""
    if fps >= 1e6:
        return f"{fps / 1e6:.1f}M"
    elif fps >= 1e3:
        return f"{fps / 1e3:.1f}K"
    else:
        return f"{fps:.0f}"


def increase_legend_line_thickness(legend, line_width=3, marker_size=10):
    """Increase line thickness and marker size in legend handles.

    Works with Line2D, LineCollection and ErrorbarContainer handles.
    """
    for n, handle in enumerate(legend.legend_handles):
        if isinstance(handle, mlines.Line2D):
            handle.set_linewidth(line_width)
            handle.set_markersize(25 if n < 2 else marker_size)
        elif isinstance(handle, mcollections.LineCollection):
            handle.set_linewidth(line_width)
        elif isinstance(handle, mcontainer.ErrorbarContainer):
            handle.lines[0].set_linewidth(line_width)
            if len(handle.lines) > 1:
                handle.lines[1].set_linewidth(line_width)
                handle.lines[2].set_linewidth(line_width)
            handle.lines[0].set_markersize(marker_size)


# -- Example plots (run with `python plot_style.py`) --------------------------


def _example_line_chart():
    """Line chart with error bands -- typical blocking probability plot."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    loads = np.arange(100, 350, 25)
    for name, color in list(TOPOLOGY_COLORS.items())[:4]:
        display = TOPOLOGY_DISPLAY[name]
        base = np.random.RandomState(hash(name) % 2**31).uniform(0.5, 2.0)
        mean = base * np.exp((loads - 100) / 120)
        std = mean * 0.15
        ax.plot(loads, mean, marker="o", color=color, label=display)
        ax.fill_between(loads, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Traffic Load (Erlang)")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.set_title("Line Chart with Error Bands")
    ax.legend()
    plt.tight_layout()
    return fig


def _example_bar_chart():
    """Grouped bar chart -- typical heuristic comparison."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["NSFNET", "COST239", "USNET", "JPN48"]
    x = np.arange(len(categories))
    width = 0.25
    rng = np.random.RandomState(42)
    for i, (label, color) in enumerate(
        [("KSP-FF", PALETTE[0]), ("FF-KSP", PALETTE[1]), ("KSP-MU", PALETTE[2])]
    ):
        vals = rng.uniform(0.5, 5.0, len(categories))
        err = vals * 0.1
        ax.bar(
            x + i * width, vals, width, label=label, color=color,
            yerr=err, capsize=4,
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Topology")
    ax.set_ylabel("Blocking Probability (%)")
    ax.set_title("Grouped Bar Chart")
    ax.legend()
    plt.tight_layout()
    return fig


def _example_errorbar_scatter():
    """Scatter / dot plot with error bars -- path length vs hops."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.RandomState(7)
    markers = ["o", "s", "^", "D"]
    for i, (name, color) in enumerate(list(TOPOLOGY_COLORS.items())[:4]):
        display = TOPOLOGY_DISPLAY[name]
        cx, cy = rng.uniform(500, 2500), rng.uniform(2, 7)
        xerr, yerr = rng.uniform(100, 400), rng.uniform(0.3, 1.5)
        ax.errorbar(
            cx, cy, xerr=xerr, yerr=yerr, fmt=markers[i],
            color=color, capsize=5, markersize=14, markeredgewidth=2,
            label=display,
        )
    ax.set_xlabel("Path Length (km)")
    ax.set_ylabel("Path Length (hops)")
    ax.set_title("Scatter Plot with Error Bars")
    ax.legend()
    plt.tight_layout()
    return fig


def _example_multiline_log():
    """Multi-panel line plot with log y-axis -- env type comparison."""
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    loads = np.arange(50, 300, 25)
    for ax, topo in zip(axes, ["NSFNET", "COST239", "USNET"]):
        rng = np.random.RandomState(hash(topo) % 2**31)
        for env, color in list(ENV_TYPE_COLORS.items())[:3]:
            display = ENV_TYPE_DISPLAY[env]
            base = rng.uniform(0.01, 0.5)
            mean = base * np.exp((loads - 50) / 80)
            ax.plot(loads, mean, marker="o", color=color, label=display)
        ax.set_yscale("log")
        ax.set_title(topo)
        ax.set_xlabel("Traffic Load (Erlang)")
        ax.yaxis.grid(True)
    axes[0].set_ylabel("Blocking Probability (%)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    configure_style()
    print("Showing example plots with JOCN2024 style presets...")
    print("Close each figure window to see the next one.\n")

    figs = [
        ("Line chart with error bands", _example_line_chart),
        ("Grouped bar chart", _example_bar_chart),
        ("Scatter with error bars", _example_errorbar_scatter),
        ("Multi-panel log-scale lines", _example_multiline_log),
    ]
    for title, fn in figs:
        print(f"  -> {title}")
        fn()

    plt.show()
