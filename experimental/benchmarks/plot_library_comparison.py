"""Library throughput comparison bar chart for the JOCN paper (replaces Table 2).

Log-scale SPS bars across the surveyed libraries for the common NSFNET
100-FSU scenario. XLRON bars are highlighted in the brand teal; other
libraries are neutral grey; GN-model-inclusive entries are hatched.

Values are recorded here as constants with provenance comments (mirroring the
old Table 2 caption): FUSION / ON-Gym / Flex Net Sim derive from wall-clock
times reported by Borquez-Paredes et al. 2026 (Table 1) and Natalino et al.
2024; GNPy and XLRON are measured in this work (Apple M1 Pro CPU / NVIDIA
H100 GPU). UPDATE THE XLRON/GNPy CONSTANTS when benchmark data is refreshed.

Run: uv run python experimental/benchmarks/plot_library_comparison.py
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_style import configure_paper_style, PAPER_COL_WIDTH_IN  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures_20260719")

TEAL = "#30A08E"
DARK_TEAL = "#1D605B"
GREY = "#B7BEC4"

# (label, SPS, is_xlron, inc_gn) — order = plot order, slowest-ish to fastest.
# Sources: FUSION & Flex Net Sim from Borquez-Paredes 2026 Table 1 wall-clocks
# (self-reported); ON-Gym exc-GN same source, inc-GN from Natalino 2024
# per-request cost; GNPy & XLRON measured in this work.
ENTRIES = [
    ("ON-Gym", 2.7e1, False, False),
    ("GNPy (GN)", 3.7e1, False, True),
    ("FUSION", 1.0e2, False, False),
    ("ON-Gym (GN)", 1.3e2, False, True),
    ("Flex Net Sim", 2.0e4, False, False),
    # --- XLRON (measured, this work) ---
    ("XLRON CPU (GN)", 3.9e2, True, True),     # RSA-GN, 1 env, M1 Pro core
    ("XLRON GPU (GN)", 1.8e3, True, True),     # RSA-GN, 1 env, H100
    ("XLRON CPU", 4.6e4, True, False),         # RMSA, 1 env, M1 Pro core
    ("XLRON GPU 4096", 1.18e7, True, False),   # RMSA, 4096 envs, H100 (mixed)
]


def main():
    configure_paper_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(PAPER_COL_WIDTH_IN, 2.4), constrained_layout=True)

    labels = [e[0] for e in ENTRIES]
    values = [e[1] for e in ENTRIES]
    colors = [TEAL if e[2] else GREY for e in ENTRIES]
    hatches = ["//" if e[3] else "" for e in ENTRIES]

    x = range(len(ENTRIES))
    bars = ax.bar(x, values, color=colors, width=0.72, zorder=3,
                  edgecolor=[DARK_TEAL if e[2] else "0.45" for e in ENTRIES],
                  linewidth=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Value labels above each bar
    for bar, v in zip(bars, values):
        if v >= 1e6:
            txt = f"{v / 1e6:.1f}M"
        elif v >= 1e3:
            txt = f"{v / 1e3:.0f}K"
        else:
            txt = f"{v:.0f}"
        ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color="0.15")

    ax.set_yscale("log")
    ax.set_ylim(1e1, 6e7)
    ax.set_ylabel("Throughput (SPS)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6.2, rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.grid(True, axis="y", which="major", alpha=0.25, lw=0.4, zorder=0)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0)

    # Legend: hatch = GN physical layer in the loop
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=TEAL, edgecolor=DARK_TEAL, lw=0.6, label="XLRON (this work)"),
        Patch(facecolor=GREY, edgecolor="0.45", lw=0.6, label="Other libraries"),
        Patch(facecolor="white", edgecolor="0.45", lw=0.6, hatch="//",
              label="incl. GN model"),
    ], loc="upper left", fontsize=6.2, borderpad=0.35, handlelength=1.4,
        handletextpad=0.5, labelspacing=0.3, framealpha=0.9)

    out = os.path.join(OUT_DIR, "library_comparison.png")
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
