# GN kernel experiment records

See [the report](../../gn_kernel_poc.md) for workload definitions, numerical
limitations, unsuccessful experiments and comparisons. Prefixes are experiment
identifiers, not release versions. JSON reports contain synchronized execution
medians and separate compilation times; JSONL files contain real PPO summaries.
Campaign `.meta.json` files save commands, physical GPU, compiler flags, exit
status and process wall time. Paths refer to the isolated Turin tree.

## Main matched comparisons

- `v17-original-420-env64.json`, `v23-control-420-jax.json` and
  `v22-final-420-pallas_inline.json`: original `2256456`, final ordinary-JAX
  optimization and current kernels. Same physical GPU 2, JAX 0.12.0.1, library
  compiler profile, 420 slots, 64 independent envs, 128 real warmup requests,
  16-step rollout. All recorded discrete outputs match exactly.
- `v10-control-1376-jax.jsonl` and `v19-transformer-1376-pallas_reuse.jsonl`:
  corrected transformer PPO, ordinary JAX versus row reuse/precomputed full
  path SNR. Normal PPO compiler settings, 1,376 slots, 16 envs, four epochs,
  four minibatches, 4,096 training steps. These short runs test execution and
  admission metrics; they do not establish convergence.
- `v25-transformer-420-pallas_reuse.jsonl` and `v27-control-420-jax.jsonl`:
  corrected 420-slot transformer PPO on the same physical GPU 3, 64 envs,
  8,192 training steps. The v25 JAX run on GPU 2 is an additional repeat.
- `v26-final-1376-pallas_inline.json`: final precomputed path-factor environment
  result at 1,376 slots / 16 envs, matched against the prior full-path snapshot.
- `v5-original-100.json` and `v5-current-originalworkload-100.json`: exact same
  archived single-env workload script on old and current production modules.
- `v15-original-420.json` and `v16-current-420.json`: library profile fixes the
  original synthetic single-env 420-slot compilation. `v16-original-420-auto0`
  isolates disabling GPU autotuning alone.

## Interpretation of other records

- Earlier fixed-power **MLP** timings in the v2 directory used a one-logit head
  and are historical evidence only. The `v4-mlp` runs use the corrected head.
- Early `v4-transformer` runs precede the wrapper's JAX-initialization ordering
  fix. Use `v10-control` for the matched normal-compiler transformer control.
- Both `v6-long-transformer` runs use fused path SNR; their difference measures
  the XPM backend, not ordinary JAX versus all custom kernels.
- `v8-cache-cold` includes an unsuccessful jitted model-initialization experiment
  that was reverted. `v10-cache-cold/warm` are the comparable cache measurements.
- `v11-noautotune`, `v12-heuristic-gemm` and `v13-library-gemm` are whole-PPO
  compiler experiments. They are separate from the default-compiler kernel result.
- `v20-grouped` and `v20-transformer` preserve multiplicity but did not improve
  the tested end-to-end performance. The environment JSON's profile field says
  `default` due to a campaign forwarding bug fixed afterward; its effective
  `xla_flags` and metadata correctly record the library compiler profile.
- `v4-warm-1376-mosaic` and `v5b-fusedpath` record unsuccessful Mosaic and
  gather-only experiments. The latter was replaced by full path-SNR fusion.
- Nsight CSVs describe captured kernel/API times, not unprofiled throughput or
  hardware-counter measurements. Nsight Compute could not acquire profiling
  resources; no bandwidth/occupancy conclusion is claimed.

Snapshot and trace binaries remain on Turin and are not included here. New reports
save their snapshot comparison results in `accuracy`. Source files for the original
comparison were verified against Git objects at `2256456` before measurement.
