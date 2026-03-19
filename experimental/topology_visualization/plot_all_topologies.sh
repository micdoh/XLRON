#!/usr/bin/env bash
# Plot all undirected topologies and save figures to the figures/ directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIGURES_DIR="$SCRIPT_DIR/figures"
TOPO_DIR="$REPO_ROOT/xlron/data/topologies"

mkdir -p "$FIGURES_DIR"

for topo_file in "$TOPO_DIR"/*_undirected.json; do
    topo_name="$(basename "$topo_file" .json)"
    output="$FIGURES_DIR/${topo_name}.png"
    echo "Plotting $topo_name ..."
    uv run python "$SCRIPT_DIR/plot_topology.py" \
        --topology_name="$topo_name" \
        --show_edge_labels \
        --save="$output"
done

echo "Done. Figures saved to $FIGURES_DIR/"
