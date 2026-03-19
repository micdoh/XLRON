"""Plot a network topology from the XLRON topologies directory.

Usage:
    uv run python experimental/topology_visualization/plot_topology.py --topology_name=nsfnet_deeprmsa_directed
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.layout import spring_layout

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import configure_style

TOPOLOGIES_DIR = Path(__file__).resolve().parents[2] / "xlron" / "data" / "topologies"


def load_topology(topology_name: str) -> nx.Graph:
    """Load a topology JSON file and return a NetworkX graph."""
    path = TOPOLOGIES_DIR / f"{topology_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Topology file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    G = nx.Graph()

    for node in data["nodes"]:
        kwargs = {}
        if "latitude" in node and "longitude" in node:
            kwargs["pos"] = (node["longitude"], node["latitude"])
        label = node.get("name", str(node["id"]))
        G.add_node(node["id"], label=label, **kwargs)

    for link in data["links"]:
        distance = link.get("distance", 1)
        G.add_edge(link["source"], link["target"], weight=distance)

    return G


def plot_topology(G: nx.Graph, title: str, ax=None, show_edge_labels: bool | None = None):
    """Plot a single topology graph.

    Args:
        show_edge_labels: Whether to show edge distance labels.
            None = auto (hide for graphs with > 50 nodes).
    """
    n_nodes = G.number_of_nodes()

    if ax is None:
        figsize = (16, 12) if n_nodes > 40 else (12, 10)
        fig, ax = plt.subplots(figsize=figsize)

    # Determine node positions
    initial_pos = nx.get_node_attributes(G, "pos")
    has_coords = len(initial_pos) == len(G.nodes())

    if has_coords:
        # Use geographic coordinates directly
        pos = dict(initial_pos)
    else:
        # No coordinates — use spring layout from scratch
        pos = spring_layout(G, k=2.0, iterations=200, seed=42)

    # Scale node size and font based on graph density
    if n_nodes <= 20:
        node_size, font_size, edge_width = 1500, 20, 2.5
    elif n_nodes <= 50:
        node_size, font_size, edge_width = 800, 14, 2.0
    else:
        node_size, font_size, edge_width = 300, 8, 1.5

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="black", width=edge_width, ax=ax)

    # Draw nodes — use IDs for dense graphs, names for small ones
    node_labels = nx.get_node_attributes(G, "label")
    if n_nodes > 30:
        # Use numeric IDs to avoid overlap
        node_labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color="white",
        edgecolors="black",
        linewidths=2.0 if n_nodes > 30 else 2.5,
        alpha=1,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=font_size,
        font_family="Arial",
        font_weight="bold",
        ax=ax,
    )

    # Add edge labels (distance) — skip for dense graphs by default.
    # Labels are drawn manually at edge midpoints to avoid a networkx 3.6 /
    # matplotlib 3.10 CurvedArrowText compatibility bug.
    if show_edge_labels is None:
        show_edge_labels = n_nodes <= 50
    if show_edge_labels:
        label_font_size = max(8, 13 - n_nodes // 15)
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 0)
            if w >= 10:
                label = f"{round(w, -1):.0f}"
            elif w > 0:
                label = f"{w:.1f}"
            else:
                continue
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(
                x, y, label,
                fontsize=label_font_size,
                fontfamily="Arial",
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            )

    ax.set_axis_off()


def print_stats(G: nx.Graph, name: str):
    """Print basic graph statistics."""
    print(f"Graph: {name}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    if nx.is_connected(G):
        print(f"  Diameter: {nx.diameter(G)}")
        print(f"  Avg shortest path length: {nx.average_shortest_path_length(G):.2f}")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    weights = [d["weight"] for _, _, d in G.edges(data=True) if "weight" in d]
    if weights:
        print(f"  Mean edge length: {sum(weights) / len(weights):.2f}")
        print(f"  Min edge length: {min(weights)}")
        print(f"  Max edge length: {max(weights)}")


def main():
    parser = argparse.ArgumentParser(description="Plot a network topology")
    parser.add_argument(
        "--topology_name",
        type=str,
        required=True,
        help="Topology name (filename without .json in xlron/data/topologies/)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save plot to this file path instead of showing interactively",
    )
    parser.add_argument(
        "--show_edge_labels",
        action="store_true",
        default=None,
        help="Force showing edge distance labels (auto-hidden for large graphs)",
    )
    args = parser.parse_args()

    configure_style()

    G = load_topology(args.topology_name)
    print_stats(G, args.topology_name)

    n = G.number_of_nodes()
    figsize = (18, 14) if n > 40 else (12, 10)
    fig, ax = plt.subplots(figsize=figsize)
    plot_topology(G, args.topology_name, ax=ax, show_edge_labels=args.show_edge_labels)
    plt.tight_layout(pad=0.1)

    if args.save:
        plt.savefig(args.save)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
