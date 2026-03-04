"""Generate KSP cache files for all K-sensitivity runs.

Computes paths once at K_max=100 per topology, then trims and saves
cache files for all smaller K values. Skips any topology+K that
already has a cache file.
"""

import time

import numpy as np

from config import (
    K_SENSITIVITY_MIN_NODES,
    K_SENSITIVITY_VALUES,
    SHARED_FLAGS,
    get_topology_list,
)

# Import XLRON internals directly (no subprocess needed)
from xlron.environments.env_funcs import (
    _ksp_cache_key,
    init_modulations_array,
    init_path_link_array,
    make_graph,
)
from xlron.environments.make_env import KSP_CACHE_DIR


K_MAX = max(K_SENSITIVITY_VALUES)
PATH_SORT_CRITERIA = "spectral_resources"


def get_cache_path(graph, topology_name, k, modulations_array):
    """Build the cache file path for a given topology+K (matching make_env conventions)."""
    params_hash = _ksp_cache_key(
        graph, k, disjoint=False, path_sort_criteria=PATH_SORT_CRITERIA,
        directed=graph.is_directed(), modulations_array=modulations_array,
        rwa_lr=False, scale_factor=1.0, path_snr=False,
    )
    return KSP_CACHE_DIR / f"{topology_name}_k{k}_{PATH_SORT_CRITERIA}_{params_hash}.npz"


def trim_path_link_array(arr, k_from, k_to):
    """Trim a path-link array from k_from paths per node-pair to k_to.

    The array has shape (num_pairs * k_from, num_edges). Each consecutive
    block of k_from rows belongs to one node pair; we keep the first k_to
    rows of each block.
    """
    num_edges = arr.shape[1]
    num_pairs = arr.shape[0] // k_from
    reshaped = arr.reshape(num_pairs, k_from, num_edges)
    return reshaped[:, :k_to, :].reshape(num_pairs * k_to, num_edges)


def main():
    KSP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    topologies = get_topology_list()
    large_topos = [t for t in topologies if t["num_nodes"] > K_SENSITIVITY_MIN_NODES]

    modulations_array = init_modulations_array(SHARED_FLAGS["modulations_csv_filepath"])
    mod_np = np.array(modulations_array)

    print(f"Warming KSP cache: {len(large_topos)} topologies, K values {K_SENSITIVITY_VALUES}")
    print(f"Strategy: compute K={K_MAX} once, trim for smaller K values\n")

    for i, topo in enumerate(large_topos, 1):
        name = topo["topology_name"]

        # Check which K values still need cache files
        graph = make_graph(name)
        missing_k = [
            k for k in K_SENSITIVITY_VALUES
            if not get_cache_path(graph, name, k, mod_np).exists()
        ]

        if not missing_k:
            print(f"[{i}/{len(large_topos)}] {name} (nodes={topo['num_nodes']}) -> all cached, skipping")
            continue

        print(f"[{i}/{len(large_topos)}] {name} (nodes={topo['num_nodes']}), need K={missing_k}")
        t0 = time.time()

        # If K_MAX is among the missing, compute it; otherwise find the largest
        # missing K and compute that (still need to compute at least once)
        k_compute = max(missing_k)
        k_max_cache = get_cache_path(graph, name, K_MAX, mod_np)

        if k_compute == K_MAX or not k_max_cache.exists():
            # Compute at K_MAX so we can trim all smaller values
            k_compute = K_MAX
            print(f"  Computing K={K_MAX}...", end=" ", flush=True)
            arr_max = init_path_link_array(
                graph, K_MAX,
                disjoint=False,
                path_sort_criteria=PATH_SORT_CRITERIA,
                directed=graph.is_directed(),
                modulations_array=mod_np,
                topology_name=name,
                cache_dir=KSP_CACHE_DIR,
            )
            print(f"shape={arr_max.shape}, {time.time() - t0:.1f}s")
        else:
            # K_MAX already cached, load it
            print(f"  Loading K={K_MAX} from cache...", end=" ", flush=True)
            arr_max = np.load(k_max_cache)["arr"]
            print(f"shape={arr_max.shape}")

        # Trim and save for each missing smaller K
        for k_val in sorted(missing_k):
            if k_val == K_MAX:
                continue  # already saved above
            cache_path = get_cache_path(graph, name, k_val, mod_np)
            arr_trimmed = trim_path_link_array(arr_max, K_MAX, k_val)
            np.savez_compressed(cache_path, arr=arr_trimmed)
            size_mb = cache_path.stat().st_size / 1e6
            print(f"  K={k_val}: trimmed to {arr_trimmed.shape}, saved ({size_mb:.1f} MB)")

        print(f"  Done in {time.time() - t0:.1f}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
