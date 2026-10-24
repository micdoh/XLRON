# GN XPM kernel proof of concept

This experiment lives on `perf/gn-kernel-poc` in a separate managed worktree.
It is based on the first GN optimization commit, `804c53a`; it is not part of
PR #62. Production defaults stay on ordinary JAX in this branch. The explicit
benchmark wrapper enables an alternative before tracing warmup, masks and steps.

## What the kernel changes

`xlron/kernels/gn_xpm.py` implements the existing uniform-span, uncorrected XPM
coefficient in float32 using Pallas. Each GPU program computes 16 output slots
against tiles of 64 interfering slots, accumulates the reduction in registers,
and writes one result per output slot. Four warps per program won the tested
12-configuration sweep. This is a measured configuration, not a universal optimum.

The first version retained every input slot. Despite avoiding a global pair matrix,
its best runtime was essentially equal to XLA's dense fusion. The useful change was
packing nonzero interferers and giving each program a dynamic reduction bound.
Zero-power or zero-bandwidth interferers contribute exactly zero to the existing
formula, so omitting them removes arithmetic without approximating interactions.
All output rows remain, repeated centre frequencies remain separate entries, and
Raman tilt still uses the original total power. No duplicate-channel aggregation,
changed SPM formula, approximate arctangent or reduced precision is introduced.

A pure-JAX packed/tiled reduction is included as a control. Under ordinary nested
`vmap`, dynamic loops run to the largest live count in the mapped batch. The
Pallas programs can use their own link's live count inside the kernel. This and
explicit tiling provide opportunities beyond just allocating smaller arrays.

## Measurements

A100 80 GB PCIe on Turin, JAX/JAXlib 0.12.0.1, float32. Timings are synchronized
medians after three warmup executions. Full environment cases use seven timed
samples, directed NSFNET, five paths, four formats, independent traffic keys and
128 real warmup admissions per environment. Initial occupancy is about 10% for
420 slots and 5% for 1,376 slots; these are not saturation experiments.

| Slots / environments | Dense JAX mask | Packed JAX mask | Packed Pallas mask | Dense JAX 16-step rollout | Packed JAX rollout | Packed Pallas rollout |
|---|---:|---:|---:|---:|---:|---:|
| 420 / 64 | 23.436 ms | 21.189 ms | 11.427 ms | 422.086 ms | 389.448 ms | 232.282 ms |
| 1,376 / 16 | 47.263 ms | 16.261 ms | 9.611 ms | 889.610 ms | 394.077 ms | 280.848 ms |

The Pallas rollout gains are **1.82×** and **3.17×** relative to the first optimized
dense-JAX implementation; they are **1.68×** and **1.40×** relative to packed JAX.
Aggregate rollout throughput reaches approximately **4,408** and **912 steps/s**.
These are environment rollouts, not full PPO training throughput. The ordinary
candidate-compaction follow-up is not included in these comparisons.

For 64 independent synthetic links, 1,376 slots and 25% random occupancy, the isolated
XPM reduction measured 1.207 ms in dense JAX, 0.608 ms in packed JAX and 0.448 ms
in the best packed Pallas variant: 1.98× and 2.69× improvements. This microbenchmark
uses nine samples and is separate from the real-admission environment workload.
The first uncompressed kernel sweep's best speedup was 1.002×, within timing noise.

Temporary-buffer estimates for complete rollouts remain about 361 MiB at 420 slots
and 296 MiB at 1,376 slots for all three backends. The speedup does not imply a
similar reduction in total process VRAM or in the state carried between steps.
Compilation, warmup and Python logging are excluded from execution timings. GPU
clocks were not locked; comparisons are within a run on the same device.

## Validation and limitations

- All 12 kernel tests pass on the A100, including interpret-mode cases, padded
  tails, an empty link, different live counts, heterogeneous powers/bandwidths,
  repeated centres, batched full SNR and both compressed/uncompressed CUDA kernels.
- Recorded masks, modulation selections, rewards and final occupancy match dense
  JAX exactly in all four environment comparisons. Link-SNR differences are at
  float32 rounding levels, with maximum recorded absolute error 1.53e-5 in linear SNR.
- Coefficient comparisons use `rtol=5e-5`; the packed microbenchmark's maximum
  relative error is 3.26e-7. This validates implementation equivalence to the current
  formula, not independent physical accuracy. The coefficient discrepancies noted
  in [the main report](gn_model_performance.md) still need resolution.
- Only the uniform-span, no-modulation-correction XPM path is accelerated. SPM,
  ASE, generic span scans, DRA and modulation-dependent correction kernels are
  unchanged. No private DRA implementation is used or included.
- Reverse-mode AD through the custom call is not implemented. Standard PPO does
  not differentiate through environment dynamics; differentiable physical-model
  optimization must retain a supported JAX path.
- This POC explicitly selects Pallas' Triton backend, which JAX has deprecated.
  The [Pallas changelog](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html) also
  records Ampere primitives added to Mosaic GPU in JAX 0.11. A supported-backend
  port is a follow-up requirement before production adoption; Mosaic is not
  categorically restricted to Hopper. No extra dependency was installed for this POC.
- Hardware counters remain unavailable on this host. Faster execution after
  removing arithmetic does not establish whether ALU/SFU throughput, instruction
  latency, caches or HBM bandwidth limits the remaining kernel.

## Reproduction

On a CUDA host with the locked GPU dependency group installed:

```bash
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python -m xlron.kernels.gn_xpm_benchmark \
  --channels 1376 --links 64 --iterations 9 --compact --output /tmp/xpm.json

CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python -m xlron.kernels.gn_environment_benchmark \
  --xpm-backend jax --slots 1376 --envs 16 --warmup-steps 128 \
  --independent-streams --kernels link mask rollout --iterations 7 \
  --output /tmp/dense.json --snapshot /tmp/dense.npz

CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python -m xlron.kernels.gn_environment_benchmark \
  --xpm-backend pallas_compact --slots 1376 --envs 16 --warmup-steps 128 \
  --independent-streams --kernels link mask rollout --iterations 7 \
  --output /tmp/pallas.json --snapshot /tmp/pallas.npz --reference /tmp/dense.npz

uv run --no-sync pytest xlron/kernels/gn_xpm_test.py -q
```

Use `--xpm-backend jax_compact` for the ordinary-JAX control. The wrapper overrides
the function before tracing; it does not change the default environment API.
Raw JSONs, including every autotune result and numerical assertions, are saved in
`docs/benchmarks/gn_kernel_20261024`. Remote scripts, logs and binary snapshots
remain in `/tmp/xlron-gn-kernel-20261024` on Turin. Its existing project and Python
environment were not modified.
