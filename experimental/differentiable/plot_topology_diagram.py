"""
Plot the 5-node test case topology diagram with traffic requests and FSU state.

Shows:
  - 5-node undirected topology
  - Two traffic requests (node 0 -> node 1) indicated by text
  - FSU (frequency slot unit) state per link (3 slots, 2 FSU per request)

Usage:
  uv run python experimental/differentiable/plot_topology_diagram.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_style import configure_style, PRIMARY_COLORS, ACCENT_COLORS

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# 5-node topology edges (undirected)
EDGES = [
    (0, 1, 1000),
    (1, 2, 1),
    (1, 3, 1),
    (2, 3, 1),
    (3, 4, 1000),
    (4, 0, 1),
]

# K=3 shortest paths from node 0 to node 1
PATHS = {
    "Path 0": [0, 1],           # direct: 0->1
    "Path 1": [0, 4, 3, 1],     # via 4,3
    "Path 2": [0, 4, 3, 2, 1],  # via 4,3,2
}

# Node positions (manual layout for nice visualization)
NODE_POS = {
    0: (0.0, 1.0),
    1: (2.0, 1.0),
    2: (1.5, 0.0),
    3: (1.0, 0.5),
    4: (0.0, 0.0),
}

NODE_COLOR = PRIMARY_COLORS[0]
NODE_EDGE_COLOR = PRIMARY_COLORS[3]
LINK_COLOR = '#888888'
REQUEST_COLOR = ACCENT_COLORS[1]  # coral

PATH_COLORS = [
    PRIMARY_COLORS[0],   # teal
    ACCENT_COLORS[0],    # purple
    ACCENT_COLORS[3],    # orange
]

NUM_SLOTS = 3


def _draw_topology(ax, show_title=True):
    """Draw the 5-node topology with traffic request labels."""
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-0.5, 1.7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw edges (no distance labels)
    for u, v, dist in EDGES:
        x0, y0 = NODE_POS[u]
        x1, y1 = NODE_POS[v]
        lw = 2.0 if dist == 1 else 3.0
        ax.plot([x0, x1], [y0, y1], '-', color=LINK_COLOR, lw=lw,
                alpha=0.6, zorder=1)

    # Draw nodes
    node_radius = 0.15
    for node_id, (x, y) in NODE_POS.items():
        circle = plt.Circle((x, y), node_radius, facecolor=NODE_COLOR,
                             edgecolor=NODE_EDGE_COLOR, linewidth=2.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(node_id), fontsize=20, fontweight='bold',
                ha='center', va='center', color='white', zorder=4)

    if show_title:
        ax.set_title('5-Node Topology:\n3 FSU per Link, 3 Candidate Paths',
                     fontsize=18, pad=14)

    # Traffic requests
    ax.text(1.0, 1.55, 'Request 1:  0  $\\rightarrow$  1',
            fontsize=16, ha='center', va='center', color=REQUEST_COLOR,
            fontweight='bold')
    ax.text(1.0, 1.35, 'Request 2:  0  $\\rightarrow$  1',
            fontsize=16, ha='center', va='center', color=REQUEST_COLOR,
            fontweight='bold')


def _draw_paths(ax, text_size=16, fsu_label_size=14, action_label_size=15):
    """Draw K-shortest paths with FSU state and action-index labels."""
    ax.set_xlim(-0.8, 3.5)
    ax.set_ylim(-0.6, 4.6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend at top
    legend_y = 4.35
    legend_x_offset = -0.57
    ax.add_patch(plt.Rectangle((legend_x_offset, legend_y - 0.04), 0.2, 0.12,
                                facecolor='#e8f5e9', edgecolor='k', lw=0.8))
    ax.text(legend_x_offset + 0.3, legend_y + 0.02, '= Empty FSU (available)',
            fontsize=text_size, va='center')
    ax.text(legend_x_offset, legend_y - 0.25,
            'Action = path_index $\\times$ num_slots + slot_index',
            fontsize=text_size, va='center', color='#555555', style='italic')

    # Draw each path
    path_y_positions = [3.3, 1.85, 0.4]
    node_r = 0.11

    for path_idx, (path_name, path_nodes) in enumerate(PATHS.items()):
        y_base = path_y_positions[path_idx]
        color = PATH_COLORS[path_idx]

        # Path header
        path_str = ' $\\rightarrow$ '.join(str(n) for n in path_nodes)
        ax.text(-0.2, y_base + 0.35, f'{path_name}:  {path_str}',
                fontsize=text_size, fontweight='bold', color=color, va='center')

        # Action indices for this path
        action_start = path_idx * NUM_SLOTS
        action_str = f'Actions {action_start}\u2013{action_start + NUM_SLOTS - 1}'
        ax.text(-0.2, y_base + 0.1, action_str,
                fontsize=action_label_size, color='#666666', va='center')

        # Draw the path as linked nodes with FSU blocks on each link
        num_links = len(path_nodes) - 1
        x_spacing = 2.8 / max(num_links, 1)

        for link_idx in range(num_links):
            n_from = path_nodes[link_idx]
            n_to = path_nodes[link_idx + 1]

            x_from = 0.2 + link_idx * x_spacing
            x_to = 0.2 + (link_idx + 1) * x_spacing

            # Draw link line
            ax.plot([x_from + node_r, x_to - node_r],
                    [y_base - 0.15, y_base - 0.15],
                    '-', color=color, lw=2.5, alpha=0.7)

            # Draw node circles
            for nx_pos, n_id in [(x_from, n_from), (x_to, n_to)]:
                circ = plt.Circle((nx_pos, y_base - 0.15), node_r,
                                   facecolor=NODE_COLOR, edgecolor=NODE_EDGE_COLOR,
                                   linewidth=2, zorder=3)
                ax.add_patch(circ)
                ax.text(nx_pos, y_base - 0.15, str(n_id), fontsize=text_size,
                        fontweight='bold', ha='center', va='center',
                        color='white', zorder=4)

            # Draw FSU block below the link — label with action indices
            slot_w = 0.15
            slot_h = 0.18
            fsu_x = (x_from + x_to) / 2 - NUM_SLOTS * slot_w / 2
            fsu_y = y_base - 0.5

            for s in range(NUM_SLOTS):
                rect = plt.Rectangle(
                    (fsu_x + s * slot_w, fsu_y - slot_h / 2),
                    slot_w * 0.9, slot_h,
                    facecolor='#e8f5e9', edgecolor='k', linewidth=0.8, zorder=4,
                )
                ax.add_patch(rect)
                action_idx = path_idx * NUM_SLOTS + s
                ax.text(fsu_x + s * slot_w + slot_w * 0.45, fsu_y,
                        str(action_idx), fontsize=fsu_label_size,
                        ha='center', va='center', zorder=5)

            # FSU label
            ax.text((x_from + x_to) / 2, fsu_y - slot_h * 0.9,
                    f'Link {n_from}\u2013{n_to}', fontsize=fsu_label_size - 1,
                    ha='center', va='top', color='#777777')


def plot_topology_diagram():
    """Side-by-side layout: topology left, paths right."""
    configure_style(font_size=18, axes_label_size=18, tick_size=14,
                    legend_size=14, figure_dpi=150)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={'width_ratios': [1, 1.2],
                                           'wspace': -0.42})
    _draw_topology(axes[0], show_title=True)
    _draw_paths(axes[1])

    plt.subplots_adjust(wspace=-0.42)
    fname = "topology_test_case.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname))
    print(f"Saved {fname}")
    return fig


def plot_topology_diagram_vertical():
    """Stacked layout: topology on top, paths below."""
    configure_style(font_size=18, axes_label_size=18, tick_size=14,
                    legend_size=14, figure_dpi=150)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12),
                              gridspec_kw={'height_ratios': [1, 1.3],
                                           'hspace': -0.08})
    _draw_topology(axes[0], show_title=True)
    _draw_paths(axes[1])

    plt.subplots_adjust(hspace=-0.08)
    fname = "topology_test_case_vertical.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname))
    print(f"Saved {fname}")
    return fig


if __name__ == "__main__":
    plot_topology_diagram()
    plot_topology_diagram_vertical()
