"""Unit tests for `cutsets_bounds.py`."""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from absl.testing import absltest

from xlron.bounds.cutsets_bounds import (
    compute_exhaustive_starts,
    find_congested_cuts_exhaustive,
    run_single_trial,
)
from xlron.environments.dataclasses import HashableArrayWrapper
from xlron.environments.make_env import make


def _gray_to_index(g: int) -> int:
    """Invert the gray code g = i ^ (i >> 1)."""
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def _masks_to_indices(masks: np.ndarray) -> list[int]:
    """Recover combination indices from gray-code partition masks (MSB first)."""
    n = masks.shape[1]
    powers = 2 ** np.arange(n - 1, -1, -1)
    gray_numbers = masks @ powers
    return [_gray_to_index(int(g)) for g in gray_numbers]


class ExhaustiveCutsetCoverageTest(absltest.TestCase):
    def test_parallel_processes_tile_search_space(self):
        """Starts from compute_exhaustive_starts must make the per-process slices
        of find_congested_cuts_exhaustive exactly tile [0, 2^num_nodes)."""
        num_nodes = 6
        total_combinations = 2**num_nodes
        parallel_processes = 2
        batch_size = 4
        # top_k == batch_size and 1 batch/iteration => every evaluated mask is returned
        top_k = batch_size
        iterations_per_process = total_combinations // (parallel_processes * batch_size)
        batches_per_iteration = 1

        graph = nx.cycle_graph(num_nodes)
        edges = sorted(graph.edges())
        adj_matrix_haw = HashableArrayWrapper(jnp.array(nx.adjacency_matrix(graph).todense()))
        traffic_matrix_haw = HashableArrayWrapper(jnp.ones((num_nodes, num_nodes)))
        source_nodes_haw = HashableArrayWrapper(jnp.array([e[0] for e in edges]))
        dest_nodes_haw = HashableArrayWrapper(jnp.array([e[1] for e in edges]))

        starts = compute_exhaustive_starts(parallel_processes, iterations_per_process)
        evaluated = []
        for start in starts:
            _, partition1, _ = find_congested_cuts_exhaustive(
                start,
                iterations_per_process,
                batches_per_iteration,
                adj_matrix_haw,
                traffic_matrix_haw,
                num_nodes,
                top_k,
                batch_size,
                source_nodes_haw,
                dest_nodes_haw,
                False,
            )
            evaluated.extend(_masks_to_indices(np.asarray(partition1)))

        self.assertLen(evaluated, total_combinations)
        self.assertEqual(set(evaluated), set(range(total_combinations)))


class ServiceTableOverflowTest(absltest.TestCase):
    def _run(self, max_services, num_requests=200, seed=0):
        settings = dict(
            load=100,
            k=2,
            topology_name="4node",
            link_resources=5,
            max_requests=num_requests,
            mean_service_holding_time=10,
            env_type="rsa",
            values_bw=[1],
            slot_size=1,
            guardband=0,
        )
        env, params = make(settings, log_wrapper=False)
        rng = jax.random.PRNGKey(seed)
        _, initial_state = env.reset(rng, params)

        # Single cut-set separating node 0 from the rest of the 4-node ring
        partition1 = jnp.array([[1, 0, 0, 0]], dtype=jnp.int32)
        partition2 = 1 - partition1
        cutset_sizes = jnp.array([2], dtype=jnp.int32)  # edges (0,1) and (0,3)
        best_se_matrix = jnp.ones((4, 4), dtype=jnp.int32)

        return run_single_trial(
            rng,
            initial_state,
            params,
            partition1,
            partition2,
            cutset_sizes,
            cutset_sizes,
            best_se_matrix,
            num_requests,
            max_services,
        )

    def test_overflow_detected_when_table_too_small(self):
        results = self._run(max_services=1)
        overflow_count = int(results[-1])
        self.assertGreater(overflow_count, 0)

    def test_no_overflow_when_table_sized_correctly(self):
        results = self._run(max_services=500)
        overflow_count = int(results[-1])
        self.assertEqual(overflow_count, 0)


if __name__ == "__main__":
    absltest.main()
