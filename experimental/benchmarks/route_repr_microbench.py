"""Micro-benchmark of routing-table representations and storage (issue #80).

Builds the real k-shortest-path table for a topology, then times the per-step
route lookups an RMSA env and the transformer policy do, for each combination of

  storage:        "const" (closed over, embedded in the program as today) or
                  "arg" (passed to the executable as a device array)
  representation: "dense" (num_paths x num_links int8 incidence, today's default),
                  "packed" (bit-packed rows, --pack_path_bits),
                  "index" (num_paths x max_hops link indices, padded with num_links)

Ops, each vmapped over a batch of envs and run in a lax.scan of fresh requests:
  occupancy  per-path slot occupancy, (k, L) @ (L, S) > 0   [mask_slots]
  pool       per-path sum of link features, (k, L) x (L, D)   [transformer actor]
  row        the chosen path's dense link vector, (L,)          [implement_path_action]

Results are checked equal across representations. Usage:
  python -m experimental.benchmarks.route_repr_microbench --topology usa100_directed --k 70
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np


def build_table(topology, k):
    from xlron.environments.make_env import make

    config = {
        "env_type": "rmsa",
        "topology_name": topology,
        "link_resources": 100,
        "k": k,
        "load": 100,
        "continuous_operation": True,
        "TOTAL_TIMESTEPS": 1000,
        "NUM_ENVS": 1,
        "ROLLOUT_LENGTH": 1,
    }
    t0 = time.perf_counter()
    _, params = make(config, log_wrapper=False)
    build_s = time.perf_counter() - t0
    dense = np.asarray(params.path_link_array).astype(np.int8)
    return dense, params.num_links, params.num_nodes, build_s


def to_index(dense, num_links):
    hops = dense.sum(axis=1)
    max_hops = int(hops.max())
    rows, cols = np.nonzero(dense)
    pos = np.arange(len(rows)) - np.repeat(np.cumsum(hops) - hops, hops)
    idx = np.full((dense.shape[0], max_hops), num_links, dtype=np.int16)
    idx[rows, pos] = cols
    return idx


def make_ops(rep, num_links, k, S, D):
    """Return (occupancy, pool, row) closures taking (table, first_path_index, ...)."""

    def rows(table, first):
        return jax.lax.dynamic_slice_in_dim(table, first, k, axis=0)

    if rep == "dense":

        def dense_rows(table, first):
            return rows(table, first)

    elif rep == "packed":

        def dense_rows(table, first):
            return jnp.unpackbits(rows(table, first), axis=1)[:, :num_links]

    if rep in ("dense", "packed"):

        def occupancy(table, first, occ):
            p = dense_rows(table, first)
            return jnp.matmul(p, occ.astype(jnp.int8), preferred_element_type=jnp.int32) > 0

        def pool(table, first, h):
            p = dense_rows(table, first).astype(h.dtype)
            return p @ h

        def row(table, first, i):
            return dense_rows(table, first)[i]

    else:

        def occupancy(table, first, occ):
            idx = rows(table, first).astype(jnp.int32)  # (k, H)
            occ_pad = jnp.concatenate([occ, jnp.zeros((1, S), occ.dtype)], axis=0)
            return jnp.any(occ_pad[idx], axis=1)

        def pool(table, first, h):
            idx = rows(table, first).astype(jnp.int32)
            h_pad = jnp.concatenate([h, jnp.zeros((1, D), h.dtype)], axis=0)
            return jnp.sum(h_pad[idx], axis=1)

        def row(table, first, i):
            idx = rows(table, first)[i].astype(jnp.int32)
            return jnp.zeros(num_links + 1, jnp.int8).at[idx].set(1)[:num_links]

    return {"occupancy": occupancy, "pool": pool, "row": row}


def run(args):
    dense, num_links, num_nodes, build_s = build_table(args.topology, args.k)
    k = args.k
    num_pairs = dense.shape[0] // k
    tables = {
        "dense": dense,
        "packed": np.packbits(dense, axis=1),
        "index": to_index(dense, num_links),
    }
    info = {
        "topology": args.topology,
        "k": k,
        "num_paths": int(dense.shape[0]),
        "num_links": int(num_links),
        "max_hops": int(tables["index"].shape[1]),
        "mean_hops": float(dense.sum(axis=1).mean()),
        "table_bytes": {r: int(t.nbytes) for r, t in tables.items()},
        "env_build_s": build_s,
        "device": str(jax.devices()[0]),
        "batch": args.batch,
        "steps": args.steps,
        "slots": args.slots,
        "feat": args.feat,
    }
    print(json.dumps(info), flush=True)

    S, D, B, T = args.slots, args.feat, args.batch, args.steps
    key = jax.random.PRNGKey(0)
    occ = jax.random.bernoulli(key, 0.5, (B, num_links, S))
    h = jax.random.normal(key, (B, num_links, D), jnp.float32)
    pairs = jax.random.randint(jax.random.PRNGKey(1), (T, B), 0, num_pairs)
    choice = jax.random.randint(jax.random.PRNGKey(2), (T, B), 0, k)

    results = []
    reference = {}
    for rep in args.reps:
        ops = make_ops(rep, num_links, k, S, D)
        dev_table = jax.device_put(tables[rep])
        for storage in args.storage:
            for op_name in args.ops:
                op = ops[op_name]

                def step(carry, xs, table, op=op, op_name=op_name):
                    pair, ch = xs
                    first = pair * k

                    def one(f, o, hh, c):
                        if op_name == "occupancy":
                            return op(table, f, o).sum(dtype=jnp.int32)
                        if op_name == "pool":
                            return op(table, f, hh).sum()
                        return op(table, f, c).astype(jnp.int32).sum() + jnp.argmax(
                            op(table, f, c)
                        )

                    out = jax.vmap(one)(first, occ, h, ch)
                    return carry + out.astype(jnp.float32).sum(), out

                if storage == "const":

                    def prog(pairs, choice, step=step):
                        return jax.lax.scan(
                            lambda c, x: step(c, x, dev_table), 0.0, (pairs, choice)
                        )

                    call_args = (pairs, choice)
                else:

                    def prog(pairs, choice, table, step=step):
                        return jax.lax.scan(lambda c, x: step(c, x, table), 0.0, (pairs, choice))

                    call_args = (pairs, choice, dev_table)

                t0 = time.perf_counter()
                lowered = jax.jit(prog).lower(*call_args)
                lower_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                compiled = lowered.compile()
                compile_s = time.perf_counter() - t0
                out = compiled(*call_args)
                jax.block_until_ready(out)
                times = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    jax.block_until_ready(compiled(*call_args))
                    times.append(time.perf_counter() - t0)
                per_step_us = 1e6 * min(times) / T
                ref = reference.setdefault(op_name, np.asarray(out[1]))
                same = bool(np.allclose(np.asarray(out[1]), ref, rtol=1e-4, atol=1e-3))
                rec = {
                    "rep": rep,
                    "storage": storage,
                    "op": op_name,
                    "lower_s": lower_s,
                    "compile_s": compile_s,
                    "per_step_us": per_step_us,
                    "matches_dense": same,
                }
                results.append(rec)
                print(json.dumps(rec), flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"info": info, "results": results}, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default="usa100_directed")
    p.add_argument("--k", type=int, default=70)
    p.add_argument("--slots", type=int, default=320)
    p.add_argument("--feat", type=int, default=128)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--reps", nargs="+", default=["dense", "packed", "index"])
    p.add_argument("--storage", nargs="+", default=["const", "arg"])
    p.add_argument("--ops", nargs="+", default=["occupancy", "pool", "row"])
    p.add_argument("--output")
    run(p.parse_args())


if __name__ == "__main__":
    main()
