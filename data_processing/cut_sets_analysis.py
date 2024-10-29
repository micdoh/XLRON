import networkx as nx
import numpy as np
import jax.numpy as jnp
import jax
import itertools
import math
from functools import partial
from xlron.environments.env_funcs import make_graph
from absl import flags, app
from xlron.environments.rsa import make_rsa_env
import xlron.train.parameter_flags
from xlron.train.train_utils import define_env
from xlron.environments.wrappers import TimeIt, HashableArrayWrapper
from xlron.environments.env_funcs import get_paths, get_paths_se, normalise_traffic_matrix

FLAGS = flags.FLAGS


@partial(jax.jit, static_argnums=(1,2))
def calculate_cut_congestion(partition_mask, adjacency_matrix, traffic_matrix):
    """
    Calculate congestion for a single cut defined by a binary partition mask.
    
    Args:
        partition_mask: binary array of shape (n_nodes,) where 1 indicates node is in set S
        n_nodes: static number of nodes for compilation
        adjacency_matrix: (n_nodes, n_nodes) binary matrix
        traffic_matrix: (n_nodes, n_nodes) float matrix
    
    Returns:
        congestion: scalar congestion value
        cut_size: number of edges in the cut
    """
    # Create masks for sets S and V-S
    S_mask = partition_mask
    V_minus_S_mask = 1 - partition_mask
    
    # Find edges in cut set (S to V-S)
    cut_matrix = (S_mask[:, None] * V_minus_S_mask[None, :]) * adjacency_matrix.val
    cut_size = jnp.sum(cut_matrix)
    
    # Calculate total traffic across cut
    traffic_across_cut = jnp.sum(
        traffic_matrix.val * (S_mask[:, None] * V_minus_S_mask[None, :])
    )
    
    # # Avoid division by zero
    # congestion = jnp.where(
    #     cut_size > 0,
    #     traffic_across_cut / cut_size,
    #     0.0
    # )
    congestion = traffic_across_cut / cut_size
    
    return congestion, cut_size


def generate_balanced_partition_masks(n_nodes, min_partition_size=1, max_batch_size=10000):
    """
    Generate balanced partition masks using Gray code to minimize bit flips.
    Only generates partitions where smaller set has size >= min_partition_size.
    """
    import numpy as np
    
    def gray_code(n):
        return n ^ (n >> 1)
    
    max_num = 2**n_nodes
    masks = []
    
    for i in range(max_num):
        gray = gray_code(i)
        mask = np.array([int(b) for b in format(gray, f'0{n_nodes}b')])
        ones = np.sum(mask)
        
        # Only keep masks with balanced partitions
        if min_partition_size <= ones <= n_nodes - min_partition_size:
            masks.append(mask)
            
        if len(masks) >= max_batch_size:  # Limit batch size for memory
            yield np.stack(masks)
            masks = []
            
    if masks:
        yield np.stack(masks)

def find_congested_cuts(graph, traffic_matrix, n_nodes, min_partition_size=1,
                       congestion_threshold=None, top_k=100, max_batch_size=10000):
    """
    Find the most congested cuts in the graph.
    
    Args:
        graph: networkx graph (used only for adjacency information)
        traffic_matrix: (n_nodes, n_nodes) traffic demand matrix
        n_nodes: number of nodes (must be static for JIT)
        min_partition_size: minimum size of smaller partition
        congestion_threshold: minimum congestion to keep (if None, keep top_k)
        top_k: number of top cuts to keep if no threshold specified
    """
    # Convert inputs to JAX arrays
    # We use an adjanency matrix with unit weights and a traffic matrix weighted by inverse SE of shortest path between node-pairs
    adjacency_matrix = HashableArrayWrapper(jnp.array(nx.adjacency_matrix(graph, weight=None).todense()))
    traffic_matrix = HashableArrayWrapper(jnp.array(traffic_matrix))
    
    # Storage for results
    all_congestions = []
    all_masks = []

    # vmap for evaluating congestion of multiple partitions
    evaluate_partition_batch = jax.vmap(calculate_cut_congestion, in_axes=(0, None, None))
    
    # Process partitions in batches
    for partition_masks_batch in generate_balanced_partition_masks(n_nodes, min_partition_size, max_batch_size):
        
        # Convert batch to JAX array
        partition_masks_batch = jnp.array(partition_masks_batch)
        
        # Evaluate batch
        congestions, cut_sizes = evaluate_partition_batch(
            partition_masks_batch, adjacency_matrix, traffic_matrix
        )
        
        # Filter based on threshold or collect all for later top-k
        if congestion_threshold is not None:
            mask = congestions >= congestion_threshold
            all_congestions.extend(congestions[mask].tolist())
            all_masks.extend(partition_masks_batch[mask].tolist())
        else:
            all_congestions.extend(congestions.tolist())
            all_masks.extend(partition_masks_batch.tolist())
    
    # Sort and return top results
    sorted_indices = np.argsort(all_congestions)[-top_k:][::-1]
    
    results = []
    for idx in sorted_indices:
        mask = all_masks[idx]
        S = {i for i in range(n_nodes) if mask[i]}
        V_minus_S = {i for i in range(n_nodes) if not mask[i]}
        cut_set = {(u, v) for u in S for v in V_minus_S if adjacency_matrix[u, v]}
        results.append((all_congestions[idx], S, V_minus_S, cut_set))
    
    return results


def find_minimum_cut(graph, weight=""):
    """Find the minimum cut-set of a given graph."""
    cut_value, partition = nx.stoer_wagner(graph, weight=weight)
    cut_set = set()

    # Extract the nodes in the two partitions
    S, T = partition
    for u in S:
        for v in T:
            if graph.has_edge(u, v):
                cut_set.add((u, v))

    return cut_set, (S, T)


def find_cut_sets(graph):
    """Find all cut-sets of a given graph."""
    cut_sets = []
    nodes = list(graph.nodes())

    # Generate all possible non-trivial partitions of the node set
    for i in range(
        1,#len(nodes) - min(5, len(nodes)//2), 
        len(nodes)
    ):
        for partition in itertools.combinations(nodes, i):
            S = set(partition)
            V_minus_S = set(nodes) - S

            cut_set = set()
            for u in S:
                for v in V_minus_S:
                    if graph.has_edge(u, v):
                        cut_set.add((u, v))

            if cut_set:
                cut_sets.append((S, V_minus_S, cut_set))
                
    return cut_sets


def calculate_congestion_levels(graph, traffic_matrix):
    """Calculate the congestion levels of all cut-sets."""
    cut_sets = find_cut_sets(graph)
    congestion_levels = []

    for S, V_minus_S, cut_set in cut_sets:
        w_n = 0
        for s in S:
            for d in V_minus_S:
                w_n += traffic_matrix[s][d]
        w_n /= len(cut_set)
        congestion_levels.append((w_n, S, V_minus_S, cut_set))

    return congestion_levels


def find_heavy_cut_sets(graph, traffic_matrix, top_n=1):
    """Find the top_n heavy cut-sets based on congestion levels."""
    congestion_levels = calculate_congestion_levels(graph, traffic_matrix)
    # Sort cut-sets by congestion level in descending order
    congestion_levels.sort(key=lambda x: x[0], reverse=True)
    # Return the top_n heavy cut-sets
    return congestion_levels[:top_n]


def find_less_congested_edges(self, traffic_matrix, bottom_n=1):
    """Find the bottom_n less congested edges based on congestion levels."""
    congestion_levels = self.calculate_congestion_levels(traffic_matrix)
    # Sort cut-sets by congestion level in ascending order
    congestion_levels.sort(key=lambda x: x[0])
    # Return the bottom_n less congested cut-sets
    return congestion_levels[:bottom_n]


def get_weighted_traffic_matrix(graph, params, se_measure='shortest'):
    n_nodes = len(graph.nodes())
    traffic_matrix = jnp.zeros((n_nodes, n_nodes))
    for s in range(n_nodes):
        for d in range(n_nodes):
            if s != d:
                nodes = jnp.array([s, d])
                se = get_paths_se(params, nodes)
                traffic_matrix = jax.lax.dynamic_update_slice(
                    traffic_matrix,
                    jnp.array(1/se[0]).reshape((1,1)) if se_measure == 'shortest' else jnp.array(1/se.mean()).reshape((1,1)),
                    (s, d)
                )
            else:
                traffic_matrix = jax.lax.dynamic_update_slice(traffic_matrix, jnp.array(0.).reshape((1,1)), (s, d))
    return traffic_matrix


def main_numpy(argv):
    # default_weight = FLAGS.weight
    # # Set weight = 'weight'
    # FLAGS.weight = "weight"
    # _, weight_params = define_env(FLAGS)
    # FLAGS.weight = None
    # env, unweight_params = define_env(FLAGS)
    # length_paths = weight_params.path_link_array
    # hops_paths = unweight_params.path_link_array
    # path_se_array_length = weight_params.path_se_array
    # path_se_array_hops = unweight_params.path_se_array

    graph = make_graph(FLAGS.topology_name, FLAGS.topology_directory)
    adj_matrix = nx.adjacency_matrix(graph, weight=None).todense()
    _, params = define_env(FLAGS)
    traffic_matrix = get_weighted_traffic_matrix(graph, params, se_measure='shortest')
    traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    print(f"Traffic Matrix:\n{traffic_matrix}\n")
    
    # Find the heavy cut-sets
    with TimeIt("Heavy Cut-Set Calculation:"):
        heavy_cut_sets = find_heavy_cut_sets(graph, traffic_matrix, top_n=10)

    print("Heavy cut-sets with their congestion levels:")
    for congestion_level, S, V_minus_S, cut_set in heavy_cut_sets:
      print(f"Congestion Level: {congestion_level}, Cut-set: {cut_set}, Min. Partition Size: {min(len(S), len(V_minus_S))}")


def main_jax(argv):
    if FLAGS.DISABLE_JIT:
        jax.config.update("jax_disable_jit", True)
    graph = make_graph(FLAGS.topology_name, FLAGS.topology_directory)
    adj_matrix = nx.adjacency_matrix(graph, weight=None).todense()
    _, params = define_env(FLAGS)
    traffic_matrix = get_weighted_traffic_matrix(graph, params, se_measure='shortest')
    traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    print(f"Traffic Matrix:\n{traffic_matrix}\n")

    # TODO - find reasonable congestion threshold (find minimum cut then get congestion)

    # Find the heavy cut-sets
    with TimeIt("Heavy Cut-Set Calculation:"):
        heavy_cut_sets = find_congested_cuts(
            graph,
            traffic_matrix,
            n_nodes=params.num_nodes,
            min_partition_size=math.ceil(params.num_nodes / 5),  # Avoid trivial cuts
            congestion_threshold=0,  # Optional: only keep cuts above this congestion
            top_k=100,  # Optional: keep top 100 most congested cuts
            max_batch_size=10000,
        )

    print("Heavy cut-sets with their congestion levels:")
    for congestion_level, S, V_minus_S, cut_set in heavy_cut_sets:
        print(
            f"Congestion Level: {congestion_level}, Cut-set: {cut_set}, Min. Partition Size: {min(len(S), len(V_minus_S))}")


if __name__ == "__main__":
    app.run(main_jax)
