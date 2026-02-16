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

import networkx as nx
import numpy as np

from xlron.environments.env_funcs import make_graph

TOPOLOGY_DIR = Path(__file__).resolve().parents[1] / "xlron" / "data" / "topologies"


def compute_topology_stats(
    topology_names: list[str] | None = None,
    output_file: str = "benchmarks/topology_stats.csv",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute XLRON topology graph statistics")
    parser.add_argument("--output", default="benchmarks/results/topology_stats.csv", help="Output CSV path")
    parser.add_argument(
        "--topologies", default=None, help="Comma-separated topology names (default: all)"
    )
    args = parser.parse_args()

    topo_names = args.topologies.split(",") if args.topologies else None
    compute_topology_stats(topology_names=topo_names, output_file=args.output)
