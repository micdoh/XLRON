# Topologies

XLRON ships with **all 119 real-world optical network topologies from [TopologyBench](https://github.com/TopologyBench/Real-Topologies)** in addition to the historically standard research topologies (NSFNET, COST239, USNET, JPN48, German17, CONUS, …). Both directed (each fibre direction is a separate link) and undirected variants are pre-converted and cached, so you can run on any topology by just changing one flag:

```bash
--topology_name=coronet_directed
--topology_name=geant_undirected
--topology_name=germany50_undirected
--topology_name=usa100_directed     # 100 nodes, 342 directed links
--topology_name=tataind_directed    # 143 nodes, 362 directed links
```

To list all available TopologyBench topologies (or regenerate from source):

```bash
python xlron/data/topologies/topology_bench_to_xlron_conversion.py --list
python xlron/data/topologies/topology_bench_to_xlron_conversion.py --download  # re-download from upstream
```

If your topology JSON includes `latitude` and `longitude` per node, the [`render`](../understanding_xlron.md) visualisation will use them to seed geographic node placement.

---

## Bundled research topologies

| Topology | Nodes | Directed links | Avg degree | Avg SP (hops) | Avg SP (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| COST239 | 11 | 52 | 4.73 | 1.56 | 1,810 |
| NSFNET | 14 | 44 | 3.14 | 2.12 | 2,054 |
| German17 | 17 | 48 | 2.82 | 2.85 | 799 |
| USNET | 24 | 86 | 3.58 | 2.99 | 2,993 |
| JPN48 | 48 | 164 | 3.42 | 5.21 | 1,201 |
| CONUS | 75 | 198 | 2.64 | 6.45 | 2,687 |
| USA100 (TopologyBench) | 100 | 342 | 1.71 | 6.52 | 2,684 |
| TataInd (TopologyBench) | 143 | 362 | 1.26 | 9.87 | 1,972 |

---

## Why this matters

Most published RL-for-RMSA studies use a small handful of topologies (NSFNET, COST239, USNET, JPN48). Conclusions drawn on those tiny topologies do not always generalise — both because the action space is small and because the path diversity is limited. The largest topology any prior dynamic-RMSA RL study had attempted before the [Graph Transformer paper](transformer.md) was JPN48 (48 nodes); USA100 and TataInd are 2–3× larger and have substantially more candidate paths per node-pair.

XLRON makes those big topologies a one-flag change. The same Graph Transformer architecture and training script that is benchmarked on NSFNET also trains successfully on TataInd in 5 hours on a single H100 — see the [reproduction guide](../reproduce_jocn_transformer.md#section-4-large-scale-experiments-usa100-and-tataind).

---

## Adding your own topology

Drop a JSON file into `xlron/data/topologies/` describing the graph in node-link format with `distance` (km) on each edge, then refer to it via `--topology_name=<filename without .json>`. Optional fields:

- `latitude` / `longitude` per node (used by `render` for geographic layouts).
- The `xlron/data/topologies/topology_bench_to_xlron_conversion.py` script is a worked example of converting an external topology source to the XLRON format.

K-shortest paths are pre-computed and cached under `xlron/data/topologies/ksp/` per `(topology, k, sort_criteria)`.

---

## Credit

The 119-topology bundle is a redistribution of [**TopologyBench: A Comprehensive Repository of Real-World Optical Network Topologies**](https://github.com/TopologyBench/Real-Topologies) by Robin Matzner *et al.* If you use these topologies in published work, please cite TopologyBench:

```bibtex
@article{matzner_topology_2024,
  author  = {Matzner, Robin and others},
  title   = {{TopologyBench}: A Comprehensive Repository of Real-World Optical Network Topologies},
  year    = {2024}
}
```
