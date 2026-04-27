"""
Plot the 5-node test-case topology diagram for the differentiable RSA paper.

Single integrated figure:
  (a) The 5-node topology with the K=3 candidate paths drawn directly on the
      graph. Each path has its own colour and is offset on its own "lane" so
      shared edges show parallel coloured stripes and divergences are obvious.
      Two pending traffic requests are shown as coral pills below.
  (b) An action-space grid (paths x slots) in the same colours, with the
      a = p * N_slot + s mapping spelled out.

Outputs (in experimental/differentiable/figures/):
  - topology_test_case.pdf  -- vector, drop into \\includegraphics
  - topology_test_case.png  -- 300 dpi preview

Usage:
  uv run python experimental/differentiable/plot_topology_diagram.py
"""

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_style import configure_style  # noqa: E402

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# --- Data ---------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 0)]

PATHS = {
    "Path 0": [0, 1],            # direct
    "Path 1": [0, 4, 3, 1],      # via 4, 3
    "Path 2": [0, 4, 3, 2, 1],   # via 4, 3, 2
}

# Topology layout in the topology panel's local coordinates.
# Spread out horizontally so the topology fills panel (a) without dead space.
NODE_POS = {
    0: (0.7, 2.40),
    1: (5.00, 2.40),
    2: (3.70, 0.55),
    3: (2.80, 1.40),
    4: (0.7, 0.55),
}

NUM_SLOTS = 3

# --- Style --------------------------------------------------------------

TEXT_DARK = "#1B2026"
TEXT_DIM = "#5B6770"

NODE_FACE = "#1D605B"  # dark teal
EDGE_GRAY = "#D5D9DD"

PANEL_BG = "#FAFBFC"
PANEL_EDGE = "#E5E9ED"

# Three perceptually-distinct colours for the three candidate paths.
PATH_COLORS = ["#30A08E", "#8064A2", "#FF9A56"]  # teal, purple, orange

REQUEST_COLOR = "#E75D72"  # coral
REQUEST_BG = "#FDECEF"

NODE_R = 0.23

# --- Helpers ------------------------------------------------------------


def _lighten(c, factor=0.85):
    """Lighten hex colour `c` toward white. factor=0 -> unchanged, 1 -> white."""
    rgb = np.array(mcolors.to_rgb(c))
    return tuple(rgb + (1.0 - rgb) * factor)


def _draw_panel(ax, x0, y0, w, h, label=None, label_y_pad=0.18):
    """Soft rounded background panel for visual grouping.

    The panel `label` is drawn ABOVE the panel (not inside) so it never
    overlaps content like nodes or column headers.
    """
    rect = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.14",
        linewidth=1.0, edgecolor=PANEL_EDGE, facecolor=PANEL_BG, zorder=0,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x0 + 0.2, y0 + h + label_y_pad, label,
                fontsize=15, fontweight="bold", color=TEXT_DIM,
                ha="left", va="bottom", zorder=1)


def _draw_node(ax, x, y, label, radius=NODE_R):
    """Node with a soft drop shadow and a thin outer ring.

    Drawn at high zorder (>=10) so the node is always on top of any path lane
    that passes through it — the node ID label must remain readable.
    """
    # Shadow
    ax.add_patch(Circle((x + 0.025, y - 0.03), radius,
                        facecolor="black", alpha=0.13, zorder=10))
    # Outer halo (separates the node from any path lane underneath)
    ax.add_patch(Circle((x, y), radius + 0.035,
                        facecolor="white", edgecolor=PANEL_EDGE,
                        linewidth=1.0, zorder=11))
    # Inner coloured disk
    ax.add_patch(Circle((x, y), radius, facecolor=NODE_FACE,
                        edgecolor="white", linewidth=2.0, zorder=12))
    ax.text(x, y, label, fontsize=18, fontweight="bold",
            ha="center", va="center", color="white", zorder=13)


def _draw_path_lane(ax, path_nodes, color, lane_offset, zorder_base):
    """Draw a path as straight per-edge segments offset perpendicular to each edge.

    Each segment uses the perpendicular of its OWN edge (not a bisector at the
    vertex), so every coloured segment is mathematically parallel to the
    underlying grey edge it shadows. The small misalignment that this creates
    at vertices is hidden by the node circle drawn on top (high zorder).
    """
    pts = [np.array(NODE_POS[n], dtype=float) for n in path_nodes]
    # First pass: white halo under each segment.
    # Second pass: coloured line on top. Splitting passes keeps halos cleanly
    # below all coloured segments along the path.
    for halo, lw, c, z in [
        (True,  5.5, "white", zorder_base),
        (False, 3.5, color,   zorder_base + 1),
    ]:
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            d = p1 - p0
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-9:
                continue
            d_unit = d / d_norm
            perp = np.array([-d_unit[1], d_unit[0]])
            offset = perp * lane_offset
            s0 = p0 + offset
            s1 = p1 + offset
            ax.plot([s0[0], s1[0]], [s0[1], s1[1]],
                    color=c, lw=lw, solid_capstyle="round", zorder=z)


def _draw_topology_section(ax):
    """Underlying graph + coloured path lanes + nodes."""
    # Underlying edges -- thick, soft grey so the colour overlays pop.
    for u, v in EDGES:
        x0, y0 = NODE_POS[u]
        x1, y1 = NODE_POS[v]
        ax.plot([x0, x1], [y0, y1], color=EDGE_GRAY, lw=7.0,
                solid_capstyle="round", zorder=2)

    # Coloured path lanes -- per-edge perpendicular offset keeps each lane
    # exactly parallel to its underlying edge.
    lane_offsets = [0.0, -0.085, 0.085]  # path 0, 1, 2
    for path_idx, ((_, nodes), color, off) in enumerate(
        zip(PATHS.items(), PATH_COLORS, lane_offsets)
    ):
        _draw_path_lane(ax, nodes, color, off, zorder_base=3 + path_idx * 2)

    # Nodes drawn last (zorder >= 10) so they sit on top of every lane.
    for nid, (x, y) in NODE_POS.items():
        _draw_node(ax, x, y, str(nid))


def _draw_pending_requests(ax, x_center, y):
    """Two coral pills, each labelled '0 -> 1'."""
    ax.text(x_center, y + 0.55, "Pending requests",
            fontsize=14, fontweight="bold", ha="center", va="center",
            color=REQUEST_COLOR)

    pill_w = 1.55
    pill_h = 0.50
    spacing = 0.28
    total = 2 * pill_w + spacing
    x0 = x_center - total / 2

    for i in range(2):
        x = x0 + i * (pill_w + spacing)
        rect = FancyBboxPatch(
            (x, y - pill_h / 2), pill_w, pill_h,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            linewidth=1.5, edgecolor=REQUEST_COLOR, facecolor=REQUEST_BG,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + pill_w / 2, y, r"0  $\rightarrow$  1",
                fontsize=16, fontweight="bold", ha="center", va="center",
                color=REQUEST_COLOR, zorder=3)


def _draw_action_grid(ax, x_left, y_top):
    """3x3 grid: rows = paths, cols = slots, cells labelled with action index."""
    cell_w = 0.70
    cell_h = 0.65
    row_gap = 0.18
    col_gap = 0.14

    grid_w = NUM_SLOTS * cell_w + (NUM_SLOTS - 1) * col_gap

    # Column headers (use variable names that match the equation footer)
    for s in range(NUM_SLOTS):
        cx = x_left + s * (cell_w + col_gap) + cell_w / 2
        ax.text(cx, y_top + 0.30, f"$s = {s}$",
                fontsize=14, ha="center", va="center",
                color=TEXT_DIM)

    # Rows
    for p in range(3):
        y_top_row = y_top - p * (cell_h + row_gap)
        y_bot_row = y_top_row - cell_h

        # Path label + colour chip on the left of the row.
        # "p = 0/1/2" is shorter than "Path 0" and ties directly to the
        # equation a = p * N_slot + s shown below the grid.
        chip_x = x_left - 1.10
        chip_y = (y_top_row + y_bot_row) / 2
        ax.add_patch(Circle((chip_x, chip_y), 0.10,
                            color=PATH_COLORS[p], zorder=2))
        ax.text(chip_x + 0.20, chip_y, f"$p = {p}$",
                fontsize=14, fontweight="bold", ha="left", va="center",
                color=PATH_COLORS[p])

        # Cells
        for s in range(NUM_SLOTS):
            action = p * NUM_SLOTS + s
            cx = x_left + s * (cell_w + col_gap)
            cy = y_bot_row
            face = _lighten(PATH_COLORS[p], 0.86)
            rect = FancyBboxPatch(
                (cx, cy), cell_w, cell_h,
                boxstyle="round,pad=0.0,rounding_size=0.09",
                linewidth=1.6, edgecolor=PATH_COLORS[p], facecolor=face,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(cx + cell_w / 2, cy + cell_h / 2, str(action),
                    fontsize=19, fontweight="bold",
                    ha="center", va="center",
                    color=PATH_COLORS[p], zorder=3)

    # Equation footer -- left-aligned with the colour chips above so the
    # left margin of the action-grid block reads as a clean vertical line.
    chip_x = x_left - 1.10
    grid_bottom = y_top - 3 * cell_h - 2 * row_gap
    ax.text(chip_x, grid_bottom - 0.42,
            r"$a \;=\; p \cdot N_{\mathrm{slot}} + s$",
            fontsize=17, ha="left", va="center",
            color=TEXT_DIM, style="italic")


# --- Main ---------------------------------------------------------------


def plot_topology_diagram():
    """Single integrated figure: topology + paths + action grid."""
    configure_style(font_size=12, axes_label_size=12, tick_size=10,
                    legend_size=10, figure_dpi=150)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(-0.6, 10.5)
    ax.set_ylim(-1.30, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title row -- centred on the figure, not the panel split.
    # Slightly larger gap between the title and the subtitle (3.85 -> 3.35).
    title_x = 4.95
    ax.text(title_x, 3.85, "5-node test-case: topology and action space",
            fontsize=19, fontweight="600", ha="center",
            color=TEXT_DARK)
    ax.text(title_x, 3.35,
            r"$N_{\mathrm{slot}} = 3$ frequency slots per link  ·  "
            r"$K = 3$ candidate paths from node 0 $\rightarrow$ node 1",
            fontsize=14, ha="center", color=TEXT_DIM, style="italic")

    # Background panels (labels drawn ABOVE each panel by _draw_panel).
    # Panel (a) is given more horizontal space than (b) so the topology can
    # breathe; panel (b) is sized just to fit the 3x3 action grid + labels.
    _draw_panel(ax, -0.35, -1.10, 5.85, 3.85, label="(a)  Topology")
    _draw_panel(ax,  5.70, -1.10, 4.70, 3.85, label="(b)  Action space")

    _draw_topology_section(ax)
    _draw_pending_requests(ax, x_center=2.55, y=-0.50)

    # Action grid: place so the grid + row labels are centred within panel (b).
    _draw_action_grid(ax, x_left=7.40, y_top=1.95)

    pdf_path = os.path.join(FIGURES_DIR, "topology_test_case.pdf")
    png_path = os.path.join(FIGURES_DIR, "topology_test_case.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return fig


def plot_topology_diagram_vertical():
    """Stacked variant: topology on top, action grid below. Useful for column figures."""
    configure_style(font_size=12, axes_label_size=12, tick_size=10,
                    legend_size=10, figure_dpi=150)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 8.7))
    ax.set_xlim(-0.6, 6.4)
    ax.set_ylim(-5.00, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    title_x = 2.90
    ax.text(title_x, 3.85, "5-node test-case",
            fontsize=18, fontweight="600", ha="center", color=TEXT_DARK)
    ax.text(title_x, 3.35,
            r"$N_{\mathrm{slot}} = 3$ slots per link  ·  "
            r"$K = 3$ paths, node 0 $\rightarrow$ node 1",
            fontsize=12.5, ha="center", color=TEXT_DIM, style="italic")

    # Panel bottoms raised; action panel slid up to sit just below the
    # tightened topology panel.
    _draw_panel(ax, -0.35, -1.10, 6.40, 3.85, label="(a)  Topology")
    _draw_panel(ax, -0.35, -4.80, 6.40, 3.55, label="(b)  Action space")

    _draw_topology_section(ax)
    _draw_pending_requests(ax, x_center=2.85, y=-0.50)

    _draw_action_grid(ax, x_left=2.30, y_top=-1.75)

    pdf_path = os.path.join(FIGURES_DIR, "topology_test_case_vertical.pdf")
    png_path = os.path.join(FIGURES_DIR, "topology_test_case_vertical.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return fig


if __name__ == "__main__":
    plot_topology_diagram()
    plot_topology_diagram_vertical()
