"""Convert TopologyBench topology files to XLRON JSON format.

TopologyBench (https://github.com/TopologyBench/Real-Topologies) provides
119 real-world network topologies. This script converts them from the
TopologyBench Python format to the XLRON JSON format (NetworkX node-link).

Usage:
    # Convert a single topology (produces both directed and undirected)
    python topology_bench_to_xlron_conversion.py TOP_59_NSFNET13.py

    # Convert all TopologyBench .py files in a directory
    python topology_bench_to_xlron_conversion.py topology_bench/

    # Convert all, specifying output directory
    python topology_bench_to_xlron_conversion.py topology_bench/ --output_dir .

    # List available topologies without converting
    python topology_bench_to_xlron_conversion.py topology_bench/ --list

    # Download TopologyBench topologies from GitHub first
    python topology_bench_to_xlron_conversion.py --download

Output:
    For each input file TOP_XX_NAME.py, two JSON files are created:
        - name_undirected.json  (undirected graph)
        - name_directed.json    (directed graph, with reverse links added)

    where 'name' is the topology name in lowercase (e.g. 'nsfnet13', 'geant').
"""

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path


# All TopologyBench v1 filenames (119 topologies)
TOPOLOGY_BENCH_FILES = [
    "TOP_01_GEANT.py",
    "TOP_02_LAMBDARAIL.py",
    "TOP_03_JAPAN25.py",
    "TOP_04_PORTUGAL.py",
    "TOP_05_PIONIER21.py",
    "TOP_06_CONUS30.py",
    "TOP_07_CONUS100.py",
    "TOP_08_CONUS6077.py",
    "TOP_09_CONUS6079.py",
    "TOP_10_CONUS75.py",
    "TOP_11_OMNICOM.py",
    "TOP_12_NEWNET.py",
    "TOP_13_MZIMA.py",
    "TOP_14_METRONA.py",
    "TOP_15_MEMOREX.py",
    "TOP_16_GEANT2.py",
    "TOP_17_EON.py",
    "TOP_18_CANARIE19.py",
    "TOP_19_BREN.py",
    "TOP_20_ARPANET.py",
    "TOP_21_ARNES.py",
    "TOP_22_JAPAN48.py",
    "TOP_23_JAPAN12.py",
    "TOP_24_REDCLARA.py",
    "TOP_25_COST37.py",
    "TOP_26_ABILENE.py",
    "TOP_27_CORONET.py",
    "TOP_28_GERMANY50.py",
    "TOP_29_JANOS_US.py",
    "TOP_30_NOBEL_EU.py",
    "TOP_31_NOBEL_GERMANY.py",
    "TOP_32_NOBEL_US.py",
    "TOP_33_POLSKA.py",
    "TOP_34_LONI.py",
    "TOP_35_VIA.py",
    "TOP_37_DARKSTRAND.py",
    "TOP_38_FUNET.py",
    "TOP_39_HIBERNIA-CANADA.py",
    "TOP_40_HIBERNIA_IRELAND.py",
    "TOP_41_HIBERNIA_NIRELAND.py",
    "TOP_42_HIBERNIA_UK.py",
    "TOP_43_HIBERNIA_US.py",
    "TOP_44_HOSTWAYINTERNATIONAL.py",
    "TOP_45_IBM.py",
    "TOP_46_INTEGRA.py",
    "TOP_47_IRIS.py",
    "TOP_48_ISTAR.py",
    "TOP_49_JGN2PLUS.py",
    "TOP_50_KAREN.py",
    "TOP_51_KENTMANFEB2008.py",
    "TOP_52_KENTMANJUL2005.py",
    "TOP_53_LAYER42.py",
    "TOP_54_MARWAN.py",
    "TOP_55_NETRAIL.py",
    "TOP_56_NETWORKUSA.py",
    "TOP_57_NEXTGEN.py",
    "TOP_58_NOEL.py",
    "TOP_59_NSFNET13.py",
    "TOP_60_OXFORD.py",
    "TOP_61_PACKETEXCHANGE.py",
    "TOP_62_PALMETTO.py",
    "TOP_63_PIONIER27_L3.py",
    "TOP_64_PSINET.py",
    "TOP_65_REDIRIS19.py",
    "TOP_66_RENATER1999.py",
    "TOP_67_RENATER2001.py",
    "TOP_68_RENATER2004.py",
    "TOP_69_RENATER2006.py",
    "TOP_70_RENATER2008.py",
    "TOP_71_RENATER2010.py",
    "TOP_72_SAGO.py",
    "TOP_73_SANREN.py",
    "TOP_74_SAVVIS.py",
    "TOP_75_SPIRALIGHT.py",
    "TOP_76_SUNET.py",
    "TOP_77_TATAIND.py",
    "TOP_78_TELECOMSERBIA.py",
    "TOP_79_UNIC.py",
    "TOP_80_VTLWAVENET2008.py",
    "TOP_81_VTLWAVENET2011.py",
    "TOP_82_YORK.py",
    "TOP_83_AARNET.py",
    "TOP_84_ANS.py",
    "TOP_85_ATMNET.py",
    "TOP_86_BBNPLANET.py",
    "TOP_87_BELNET2009.py",
    "TOP_88_BELNET2010.py",
    "TOP_89_BEYOND_THE_NETWORK.py",
    "TOP_90_BICS.py",
    "TOP_91_BIZNET.py",
    "TOP_92_CANARIE24.py",
    "TOP_93_CLARANET.py",
    "TOP_94_CRL_NETWORK_SERVICES.py",
    "TOP_95_CWIX.py",
    "TOP_96_DIGEX.py",
    "TOP_97_ELIBACKBONE.py",
    "TOP_98_EPOCH.py",
    "TOP_99_ERNET.py",
    "TOP_100_GAMBIA.py",
    "TOP_101_GARR201201.py",
    "TOP_102_GBLNET.py",
    "TOP_103_GETNET.py",
    "TOP_104_GRENA.py",
    "TOP_105_GRNET.py",
    "TOP_106_GTS_CZECH_REPUBLIC.py",
    "TOP_107_GTS_POLAND.py",
    "TOP_108_HIBERNIA_GLOBAL.py",
    "TOP_109_USA100.py",
    "TOP_110_CERNET.py",
    "TOP_111_CESNET.py",
    "TOP_112_DT17.py",
    "TOP_113_ITALY.py",
    "TOP_114_OPTUNET_SWEDEN.py",
    "TOP_115_RAILTEL_INDIA.py",
    "TOP_117_TELENET_BE.py",
    "TOP_118_RNP_BRAZIL.py",
    "TOP_119_INTERNET2.py",
    "TOP_120_RENATER.py",
    "TOP_121_SANET.py",
]

RAW_URL_BASE = "https://raw.githubusercontent.com/TopologyBench/Real-Topologies/main/Code/v1/"


def download_topology_bench_files(output_dir: Path) -> None:
    """Download all TopologyBench .py files from GitHub."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(TOPOLOGY_BENCH_FILES)} TopologyBench files to {output_dir}/")
    for filename in TOPOLOGY_BENCH_FILES:
        url = RAW_URL_BASE + filename
        dest = output_dir / filename
        if dest.exists():
            continue
        try:
            subprocess.run(
                ["curl", "-sL", "-o", str(dest), url],
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print(f"  FAILED: {filename}")
            if dest.exists():
                dest.unlink()
            continue
    downloaded = list(output_dir.glob("TOP_*.py"))
    print(f"Downloaded {len(downloaded)} files.")


class _NanToNone(ast.NodeTransformer):
    """Replace bare `nan` identifiers with None for ast.literal_eval."""

    def visit_Name(self, node):
        if node.id == "nan":
            return ast.copy_location(ast.Constant(value=None), node)
        return node


def parse_topology_file(filepath: Path) -> dict:
    """Parse a TopologyBench .py file and extract node/edge data.

    Uses AST parsing (no exec/eval) for safety. Extracts the
    node_attributes and edge_attributes dicts from the source code.

    Returns:
        dict with keys 'nodes' and 'edges', where:
            nodes: {int_id: {'lat': float, 'long': float,
                             'location': str, 'country': str}}
            edges: list of (source_int, dest_int, fiber_length_km)
    """
    source = filepath.read_text()
    tree = ast.parse(source)
    # Replace bare `nan` identifiers so ast.literal_eval works
    tree = _NanToNone().visit(tree)

    node_attrs = None
    edge_attrs = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id == "node_attributes":
                    node_attrs = ast.literal_eval(node.value)
                elif target.id == "edge_attributes":
                    edge_attrs = ast.literal_eval(node.value)

    if node_attrs is None or edge_attrs is None:
        raise ValueError(f"Could not parse node_attributes or edge_attributes from {filepath}")

    # Normalise node keys to int
    nodes = {}
    for k, v in node_attrs.items():
        node_id = int(k)
        nodes[node_id] = {
            "lat": v.get("lat", 0.0),
            "long": v.get("long", 0.0),
            "location": v.get("location", ""),
            "country": v.get("country", ""),
        }
        # Handle nan/None values
        for field in ("lat", "long"):
            val = nodes[node_id][field]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                nodes[node_id][field] = 0.0
        for field in ("location", "country"):
            val = nodes[node_id][field]
            if val is None or not isinstance(val, str):
                nodes[node_id][field] = ""

    # Extract edges
    edges = []
    for _edge_id, v in edge_attrs.items():
        src = int(v["source"])
        dst = int(v["destination"])
        fiber_length = round(v["fiber_length"])
        edges.append((src, dst, fiber_length))

    return {"nodes": nodes, "edges": edges}


def topology_name_from_filename(filename: str) -> str:
    """Extract a clean topology name from a TopologyBench filename.

    TOP_59_NSFNET13.py -> nsfnet13
    TOP_39_HIBERNIA-CANADA.py -> hibernia_canada
    TOP_106_GTS_CZECH_REPUBLIC.py -> gts_czech_republic
    """
    name = filename.replace(".py", "")
    # Remove the TOP_XX_ prefix
    name = re.sub(r"^TOP_\d+_", "", name)
    # Replace hyphens with underscores
    name = name.replace("-", "_")
    return name.lower()


def to_xlron_json(parsed: dict, directed: bool) -> dict:
    """Convert parsed topology data to XLRON JSON format.

    Args:
        parsed: Output of parse_topology_file()
        directed: If True, create directed graph (each undirected edge
                  becomes two directed links). If False, undirected.

    Returns:
        dict in XLRON JSON format (NetworkX node-link format).
    """
    nodes_data = parsed["nodes"]
    edges_data = parsed["edges"]

    # Build nodes list, sorted by ID
    nodes = []
    for node_id in sorted(nodes_data.keys()):
        attrs = nodes_data[node_id]
        node = {"id": node_id}
        if attrs.get("location"):
            node["name"] = attrs["location"]
        if attrs.get("lat", 0.0) != 0.0 or attrs.get("long", 0.0) != 0.0:
            node["latitude"] = attrs["lat"]
            node["longitude"] = attrs["long"]
        nodes.append(node)

    # Build links list
    links = []
    for src, dst, dist in edges_data:
        # Ensure distance is at least 1 km
        dist = max(dist, 1)
        links.append({"distance": dist, "source": src, "target": dst})

    if directed:
        # Add reverse links
        reverse_links = []
        for src, dst, dist in edges_data:
            dist = max(dist, 1)
            reverse_links.append({"distance": dist, "source": dst, "target": src})
        links.extend(reverse_links)

    return {
        "directed": directed,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": links,
    }


def convert_file(
    filepath: Path,
    output_dir: Path,
    directed: bool = True,
    undirected: bool = True,
) -> list[str]:
    """Convert a single TopologyBench file to XLRON JSON.

    Returns list of output filenames created.
    """
    topo_name = topology_name_from_filename(filepath.name)
    parsed = parse_topology_file(filepath)

    created = []

    if undirected:
        data = to_xlron_json(parsed, directed=False)
        outfile = output_dir / f"{topo_name}_undirected.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=4)
            f.write("\n")
        created.append(outfile.name)

    if directed:
        data = to_xlron_json(parsed, directed=True)
        outfile = output_dir / f"{topo_name}_directed.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=4)
            f.write("\n")
        created.append(outfile.name)

    return created


def list_topologies(input_path: Path) -> None:
    """List available TopologyBench topologies."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("TOP_*.py"))

    print(f"{'TopologyBench File':<45} {'XLRON Name':<30} {'Nodes':>5} {'Edges':>5}")
    print("-" * 90)
    for f in files:
        topo_name = topology_name_from_filename(f.name)
        try:
            parsed = parse_topology_file(f)
            n_nodes = len(parsed["nodes"])
            n_edges = len(parsed["edges"])
            print(f"{f.name:<45} {topo_name:<30} {n_nodes:>5} {n_edges:>5}")
        except Exception as e:
            print(f"{f.name:<45} {'ERROR: ' + str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TopologyBench topologies to XLRON JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a TopologyBench .py file or directory of .py files. "
        "If omitted, looks for topology_bench/ in the same directory "
        "as this script.",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for JSON files. Defaults to the same "
        "directory as this script (xlron/data/topologies/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available topologies and their properties without converting.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download TopologyBench .py files from GitHub to topology_bench/ subdirectory.",
    )
    parser.add_argument(
        "--undirected-only",
        action="store_true",
        help="Only generate undirected topology files.",
    )
    parser.add_argument(
        "--directed-only",
        action="store_true",
        help="Only generate directed topology files.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Handle download
    if args.download:
        tb_dir = script_dir / "topology_bench"
        download_topology_bench_files(tb_dir)
        if not args.input and not args.list:
            print("\nTo convert, run:")
            print(f"  python {__file__} {tb_dir}")
            return

    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = script_dir / "topology_bench"

    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        print("Run with --download to fetch TopologyBench files first.")
        sys.exit(1)

    # Handle list mode
    if args.list:
        list_topologies(input_path)
        return

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine directed/undirected
    gen_directed = not args.undirected_only
    gen_undirected = not args.directed_only

    # Get files to convert
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("TOP_*.py"))

    if not files:
        print(f"No TopologyBench .py files found in {input_path}")
        sys.exit(1)

    print(f"Converting {len(files)} topologies to {output_dir}/")

    total_created = 0
    errors = 0
    for f in files:
        try:
            created = convert_file(
                f,
                output_dir,
                directed=gen_directed,
                undirected=gen_undirected,
            )
            print(f"  {f.name} -> {', '.join(created)}")
            total_created += len(created)
        except Exception as e:
            print(f"  ERROR: {f.name}: {e}")
            errors += 1

    print(f"\nDone: {total_created} files created, {errors} errors.")


if __name__ == "__main__":
    main()
