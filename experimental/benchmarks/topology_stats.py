#!/usr/bin/env python3
"""Compute graph statistics for all XLRON topologies.

Loads each topology JSON via make_graph(), computes NetworkX graph metrics,
and writes results to CSV for use in benchmark plots.

Usage:
    python benchmarks/topology_stats.py
    python benchmarks/topology_stats.py --output topology_stats.csv
    python benchmarks/topology_stats.py --topologies nsfnet_deeprmsa_directed,cost239_deeprmsa_directed
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from experimental.plot_style import (
    TABLE_HEADER_COLOR,
    TABLE_ROW_ALT_COLOR,
    TOPOLOGY_DISPLAY,
    configure_style,
)

from xlron.environments.env_funcs import make_graph

_BENCHMARKS_DIR = Path(__file__).resolve().parent
TOPOLOGY_DIR = Path(__file__).resolve().parents[2] / "xlron" / "data" / "topologies"
print(f"Topology dir {TOPOLOGY_DIR}")


def compute_topology_stats(
    topology_names: list[str] | None = None,
    output_file: str = "experimental/benchmarks/results/topology_stats.csv",
) -> list[dict]:
    """Compute graph statistics for topologies and write to CSV.

    Args:
        topology_names: List of topology names (without .json). If None, discovers all.
        output_file: Path for the output CSV.

    Returns:
        List of dicts, one per topology.
    """
    if topology_names is None:
        topology_names = sorted(f.stem for f in TOPOLOGY_DIR.glob("*.json"))

    rows = []
    for name in topology_names:
        graph = make_graph(name, topology_directory=str(TOPOLOGY_DIR))

        is_directed = graph.is_directed()
        ug = graph.to_undirected() if is_directed else graph

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        num_undirected_edges = ug.number_of_edges()
        avg_degree = float(np.mean([d for _, d in ug.degree()]))

        if nx.is_connected(ug):
            diameter = nx.diameter(ug)
            avg_path_length = nx.average_shortest_path_length(ug)
        else:
            diameter = float("inf")
            avg_path_length = float("inf")

        avg_clustering = nx.average_clustering(ug)
        edge_connectivity = nx.edge_connectivity(ug)
        density = nx.density(ug)

        distances = [d.get("distance", 1) for _, _, d in graph.edges(data=True)]
        avg_distance = float(np.mean(distances)) if distances else 0.0
        total_distance = float(np.sum(distances))

        rows.append(
            {
                "topology_name": name,
                "directed": is_directed,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_undirected_edges": num_undirected_edges,
                "avg_degree": round(avg_degree, 2),
                "diameter": diameter,
                "avg_path_length": round(avg_path_length, 3),
                "avg_clustering": round(avg_clustering, 3),
                "edge_connectivity": edge_connectivity,
                "density": round(density, 3),
                "avg_link_distance_km": round(avg_distance, 1),
                "total_distance_km": round(total_distance, 1),
            }
        )

    # Write CSV
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Topology stats for {len(rows)} topologies written to {output_file}")
    return rows


def plot_topology_stats(
    topo_stats_path: str | Path,
    output_dir: str | Path = _BENCHMARKS_DIR / "figures",
):
    """Render a publication-quality table of topology stats from CSV."""
    topo_stats_path = Path(topo_stats_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not topo_stats_path.exists():
        print(f"  No topology_stats.csv found at {topo_stats_path}, skipping topology table")
        return

    topo_stats = pd.read_csv(topo_stats_path)

    # Only keep directed topologies
    merged = topo_stats[topo_stats["directed"] == True].copy()
    if merged.empty:
        print("  No directed topologies found")
        return

    # Sort by number of edges
    merged = merged.sort_values("num_edges")

    # Build table data
    col_labels = ["Topology", "Nodes", "Links", "Avg Degree",
                  "Avg Shortest Path Length (hops)", "Avg Shortest Path Length (km)"]
    cell_text = []
    for _, row in merged.iterrows():
        name = row["topology_name"]
        fallback = name.replace("_directed", "").replace("_", " ").upper()
        display = TOPOLOGY_DISPLAY.get(name, fallback)
        avg_path_km = row["avg_path_length"] * row["avg_link_distance_km"]
        cell_text.append([
            display,
            str(int(row["num_nodes"])),
            str(int(row["num_edges"])),
            f"{row['avg_degree']:.2f}",
            f"{row['avg_path_length']:.2f}",
            f"{avg_path_km:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 1.0 + 0.22 * len(cell_text)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    # Custom column widths: narrow for Nodes/Links, wider for the rest
    col_widths = [0.14, 0.08, 0.08, 0.14, 0.22, 0.22]
    n_rows = len(cell_text) + 1  # +1 for header
    for j, w in enumerate(col_widths):
        for i in range(n_rows):
            table[i, j].set_width(w)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(TABLE_HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(cell_text)):
        color = TABLE_ROW_ALT_COLOR if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout()
    out_path = output_dir / "topology_table.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_heatmap_topology_stats(
    topo_stats_path: str | Path,
    output_dir: str | Path = _BENCHMARKS_DIR / "figures",
):
    """Render a topology table for only the topologies shown in the heatmap."""
    from experimental.plot_style import get_topology_order

    topo_stats_path = Path(topo_stats_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not topo_stats_path.exists():
        print(f"  No topology_stats.csv found at {topo_stats_path}, skipping heatmap topology table")
        return

    topo_stats = pd.read_csv(topo_stats_path)

    # Filter to heatmap topologies in their display order
    heatmap_topos = get_topology_order()
    merged = topo_stats[topo_stats["topology_name"].isin(heatmap_topos)].copy()
    if merged.empty:
        print("  No heatmap topologies found in stats")
        return

    # Preserve heatmap order
    merged["_order"] = merged["topology_name"].map({t: i for i, t in enumerate(heatmap_topos)})
    merged = merged.sort_values("_order")

    # Build table data
    col_labels = ["Topology", "Nodes", "Links", "Avg Deg.",
                  "Avg Shortest\nPath Length (hops)", "Avg Shortest\nPath Length (km)"]
    cell_text = []
    for _, row in merged.iterrows():
        name = row["topology_name"]
        fallback = name.replace("_directed", "").replace("_", " ").upper()
        display = TOPOLOGY_DISPLAY.get(name, fallback)
        avg_path_km = row["avg_path_length"] * row["avg_link_distance_km"]
        cell_text.append([
            display,
            str(int(row["num_nodes"])),
            str(int(row["num_edges"])),
            f"{row['avg_degree']:.2f}",
            f"{row['avg_path_length']:.2f}",
            f"{avg_path_km:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(6, 1.4 + 0.22 * len(cell_text)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    # Compact column widths
    col_widths = [0.18, 0.10, 0.10, 0.14, 0.18, 0.18]
    n_rows = len(cell_text) + 1  # +1 for header
    for j, w in enumerate(col_widths):
        for i in range(n_rows):
            table[i, j].set_width(w)

    # Make header row taller for two-line labels
    for j in range(len(col_labels)):
        table[0, j].set_height(table[0, j].get_height() * 1.8)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(TABLE_HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(cell_text)):
        color = TABLE_ROW_ALT_COLOR if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout()
    out_path = output_dir / "heatmap_topology_table.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_large_topology_table(
    topo_stats_path: str | Path,
    output_dir: str | Path = _BENCHMARKS_DIR / "figures",
):
    """Render a topology table for the large-topology experiments."""
    topo_stats_path = Path(topo_stats_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not topo_stats_path.exists():
        print(f"  No topology_stats.csv found at {topo_stats_path}, skipping large topology table")
        return

    topo_stats = pd.read_csv(topo_stats_path)

    # Fixed topology list in desired order
    large_topos = [
        "nsfnet_deeprmsa_directed",
        "cost239_deeprmsa_directed",
        "usnet_ptrnet_directed",
        "jpn48_directed",
        "usa100_directed",
        "tataind_directed",
    ]
    merged = topo_stats[topo_stats["topology_name"].isin(large_topos)].copy()
    if merged.empty:
        print("  No large topologies found in stats")
        return

    # Preserve specified order
    merged["_order"] = merged["topology_name"].map({t: i for i, t in enumerate(large_topos)})
    merged = merged.sort_values("_order")

    # Build table data
    col_labels = ["Topology", "Nodes", "Directed\nLinks", "Avg\nDegree",
                  "Avg Shortest\nPath Length\n(hops)", "Avg Shortest\nPath Length\n(km)"]
    cell_text = []
    for _, row in merged.iterrows():
        name = row["topology_name"]
        fallback = name.replace("_directed", "").replace("_", " ").upper()
        display = TOPOLOGY_DISPLAY.get(name, fallback)
        avg_path_km = row["avg_path_length"] * row["avg_link_distance_km"]
        cell_text.append([
            display,
            str(int(row["num_nodes"])),
            str(int(row["num_edges"])),
            f"{row['avg_degree'] / 2:.2f}",
            f"{row['avg_path_length']:.2f}",
            f"{avg_path_km:.0f}",
        ])

    fig, ax = plt.subplots(figsize=(6, 1.4 + 0.22 * len(cell_text)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    # Compact column widths
    col_widths = [0.14, 0.08, 0.10, 0.10, 0.18, 0.18]
    n_rows = len(cell_text) + 1  # +1 for header
    for j, w in enumerate(col_widths):
        for i in range(n_rows):
            table[i, j].set_width(w)

    # Make header row taller for three-line labels
    for j in range(len(col_labels)):
        table[0, j].set_height(table[0, j].get_height() * 2.4)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(TABLE_HEADER_COLOR)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(len(cell_text)):
        color = TABLE_ROW_ALT_COLOR if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout()
    out_path = output_dir / "large_topology_table.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute XLRON topology graph statistics")
    parser.add_argument("--output", default="experimental/benchmarks/results/topology_stats.csv", help="Output CSV path")
    parser.add_argument(
        "--topologies", default=None, help="Comma-separated topology names (default: all)"
    )
    parser.add_argument(
        "--figures_dir", default=str(_BENCHMARKS_DIR / "figures"),
        help="Output directory for figures (default: benchmarks/figures/)",
    )
    args = parser.parse_args()

    topo_names = args.topologies.split(",") if args.topologies else None
    compute_topology_stats(topology_names=topo_names, output_file=args.output)

    configure_style()
    plot_topology_stats(topo_stats_path=args.output, output_dir=args.figures_dir)
    plot_heatmap_topology_stats(topo_stats_path=args.output, output_dir=args.figures_dir)
    plot_large_topology_table(topo_stats_path=args.output, output_dir=args.figures_dir)
