"""Global plot style configuration for XLRON benchmark figures.

Import and call configure_style() at the top of any plotting script to ensure
consistent publication-quality formatting across all figures.

Style based on JOCN2024 plots (Helvetica/Arial, large fonts for readability).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# -- Color Palettes -----------------------------------------------------------

ENV_TYPE_COLORS = {
    "rwa": "#1f77b4",
    "rmsa": "#ff7f0e",
    "rsa_gn_model": "#2ca02c",
    "rmsa_gn_model": "#d62728",
    "rwa_lightpath_reuse": "#9467bd",
}

TOPOLOGY_COLORS = {
    "5node_directed": "#1f77b4",
    "cost239_deeprmsa_directed": "#ff7f0e",
    "cost239_ptrnet_published_directed": "#bcbd22",
    "cost239_ptrnet_real_directed": "#17becf",
    "nsfnet_deeprmsa_directed": "#2ca02c",
    "german17_directed": "#d62728",
    "usnet_gcnrnn_directed": "#9467bd",
    "usnet_ptrnet_directed": "#8c564b",
    "jpn48_directed": "#e377c2",
    "conus_directed": "#7f7f7f",
}

TOPOLOGY_DISPLAY = {
    "5node_directed": "5-node",
    "cost239_deeprmsa_directed": "COST239",
    "cost239_ptrnet_published_directed": "COST239-A",
    "cost239_ptrnet_real_directed": "COST239-B",
    "nsfnet_deeprmsa_directed": "NSFNET",
    "german17_directed": "German17",
    "usnet_gcnrnn_directed": "USNET-A",
    "usnet_ptrnet_directed": "USNET-B",
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
    "cpu": "#1f77b4",
    "gpu": "#d62728",
}

BAND_COLORS = {
    "C": "#1f77b4",
    "C,L": "#ff7f0e",
    "C,L,S": "#2ca02c",
}

BAND_DISPLAY = {
    "C": "C (43 × 100 GHz)",
    "C,L": "C+L (115 × 100 GHz)",
    "C,L,S": "C+L+S (209 × 100 GHz)",
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
        "5node_directed",
        "cost239_deeprmsa_directed",
        "cost239_ptrnet_published_directed",
        "cost239_ptrnet_real_directed",
        "nsfnet_deeprmsa_directed",
        "german17_directed",
        "usnet_gcnrnn_directed",
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
