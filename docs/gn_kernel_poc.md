# GN kernels: performance, accuracy and reproducible comparisons

This branch continues the ordinary-JAX optimizations in PR #62 (`97373df`), which
has now merged. Custom backends remain opt-in. The supported differentiable path
and the normal environment default remain ordinary JAX. All simulations below
retain float32 physics, modulation thresholds, new-path margin, existing-path
checks and per-fibre power checks.

## Total progress from the original code

Do not multiply rows in this table: runtimes, spectra, traffic and parallelism
vary. The journal reference is environment throughput, not PPO throughput.

| Reference or matched comparison | Workload | Result |
|---|---|---|
| [Journal](https://preprints.opticaopen.org/articles/preprint/XLRON_A_Framework_for_Hardware-Accelerated_and_Differentiable_Simulation_of_Optical_Networks/33317361?file=67772697), v1.6.0 / `2256456` | H100, JAXlib 0.9.1, 1 env, 100 slots, **100 GHz slots**, NSFNET, k=5, KSP-FF, load 250, no warmup, 50,000 steps | About 3,200 environment steps/s; five-repeat reference, not a matched comparison with this branch |
| Original → first optimization `804c53a` | Same A100 / JAX 0.9.1, 420 slots, synthetic loaded single env | Mask 4.320 → 1.151 ms (3.75×); 16-step rollout 77.073 → 17.362 ms (4.44×); mask temporaries 126.505 → 5.168 MiB |
| Original → current kernels | Same A100 model / JAX 0.12.0.1, archived identical synthetic 100-slot script, 1 env, 12.5 GHz slots | 16-step rollout 8.814 → 5.373 ms (1.64×); every recorded output exactly equal |
| Original → current kernels | Same A100 model / JAX 0.12.0.1, identical synthetic 420-slot script, 1 env, library compiler profile | 16-step rollout 60.801 → 18.255 ms (3.33×); mask 4.303 → 1.238 ms; recorded masks/formats/rewards/occupancy exactly equal |
| Original → current kernels | Same physical A100 / JAX 0.12.0.1, 420 slots, 64 independent envs, 128 real warmup requests, library compiler profile | 16-step rollout 3,738.863 → 150.687 ms (**24.81×**, 274 → 6,796 steps/s); mask 230.502 → 5.968 ms; mask temporaries 7,663.2 → 235.4 MiB |
| First optimization → final PR #62 | Same JAX 0.12.0.1, real warmup, 420 slots / 64 envs | 422.405 → 390.717 ms per rollout (1.08×) |
| First optimization → final PR #62 | Same JAX 0.12.0.1, real warmup, 1,376 slots / 16 envs | 890.309 → 413.241 ms (2.15×) |

The new real-warmup original-code comparison uses the **same workload script and
configuration**, not scaled single-env timings. Initial mean occupancy is 10.06%.
The final `v17-original` / `v22-final` pair both ran on physical GPU 2.
The four original production files (`env_funcs.py`, `make_env.py`, `dataclasses.py`
and `isrs_gn_model.py`) were SHA-256 matched to Git objects at `2256456`.
Snapshots are checked against the same JAX control. Original and current masks,
modulation selections, rewards and final occupancy match; link SNR differs by at
most 1.53e-5 in linear units. These are low-occupancy runs, not saturation studies.

The final same-GPU, same-library-profile progression is **3,738.863 ms original →
389.602 ms optimized ordinary JAX (`v23-control`) → 150.687 ms custom kernels**.
That is 9.60× from ordinary-JAX changes and another 2.59× from kernels on this
specific workload, measured directly rather than multiplying unrelated runs.

## Making the 420-slot baseline compile

The original wide mask stalled beyond 120 seconds in the current GPU autotuning
path. This was a compiler problem, not a reason to reduce slots, paths or physics.
Disabling autotuning alone (`--xla_gpu_autotune_level=0`) makes the unchanged
original mask compile in 1.73 s and its rollout in 3.61 s. The explicit **library**
profile also disables XLA Triton GEMMs and gives 1.42 s / 3.09 s compilation.
It produces the same recorded outputs; its single-env rollout is 60.8 ms versus
73.2 ms with autotuning disabled alone. Custom Pallas kernels remain enabled.

Use `--compiler-profile library` on the environment benchmark. The wrapper sets
flags **before importing modules that initialize JAX**, and records the profile
and effective `XLA_FLAGS`. Use the same profile for controls and variants.
Production compiler defaults are unchanged by this option. The old 120-second
timeout remains a recorded unsuccessful experiment, superseded by these runs.

## What was changed and tested

- Packed XPM columns omit only exactly zero contributions. The original total
  power still drives Raman tilt, and repeated interferers retain multiplicity.
- `pallas_sparse` also packs nonzero-bandwidth output rows. Zero-power rows with
  positive bandwidth remain for coefficient equivalence.
- `pallas_reuse` evaluates consecutive exactly identical power/frequency/bandwidth
  output rows once and restores every output. It does not aggregate interferers.
- `pallas_grouped` also combines consecutive identical interferers using their
  exact multiplicities. The original power sum and same-centre exclusions remain.
  It passed native parity tests but did not improve the tested end-to-end workloads.
- `pallas_skip` branches around empty contiguous output tiles. `pallas_inline`
  also skips empty interferer tiles inside the kernel, avoiding external packing.
- `--path-snr-backend pallas_fused` fuses referenced-hop gathers, ROADM ASE,
  transceiver NSR and dB conversion. It avoids large multidimensional gather-index
  tensors. `pallas_precomputed` additionally precomputes fixed per-slot amplifier
  factors. All hop contributions and output checks remain.
- A Mosaic GPU port works on A100 with approximate math disabled. It remains an
  unsuccessful performance experiment at the tested 128×128 layout.

## Successive environment measurements

A100 80 GB PCIe, JAX/JAXlib 0.12.0.1, directed NSFNET, k=5, four formats,
12.5 GHz slots, independent traffic, synchronized medians after warmup. Compilation,
state warmup and logging are excluded. GPU clocks are not locked.

| Backend | 420 slots / 64 envs / 16 steps | 1,376 slots / 16 envs / 16 steps |
|---|---:|---:|
| Final PR #62 JAX | 389.685 ms | 407.422 ms |
| Packed-column Pallas | 231.178 ms | 294.110 ms |
| Row compaction | — | 237.914 ms |
| Exact consecutive-row reuse | — | 233.538 ms |
| Empty output-tile skipping | — | 228.935 ms |
| Inline empty row/column tiles | 175.117 ms | 228.239 ms |
| Inline tiles + fused full path SNR | **160.919 ms** | **209.261 ms** |
| Mosaic port | — | 949.815 ms |

The last default-compiler row reaches 6,363 and 1,223 aggregate environment steps/s:
**2.42× and 1.95×** over PR #62. With 512 warmup requests and 64 measured steps,
the 1,376-slot case falls from 2,310.030 to **940.466 ms** (2.46×, 1,089 steps/s).
Recorded masks/formats/rewards/occupancy match exactly, including this longer run.

The library-profile precomputed-factor variant reaches 150.687 ms at 420 slots /
64 envs (`v22-final`); repeats on GPU 3 gave 151.905 and 151.731 ms. The same-GPU
non-precomputed variant was 160.059 ms. Grouping identical interferers was slower
at 162.922 ms, including packing costs (`v20-grouped`). These are direct timings,
not multipliers for the historical comparisons.

The gather-only path kernel was slower: 237.0 ms versus 228.9 ms at 1,376 slots,
and increased mask temporaries from 270 to 329 MiB. Full fusion removed that
regression and reduced mask temporaries to 186 MiB. Buffer estimates are not
process VRAM. The isolated synthetic 10%-occupancy XPM microbenchmark's 0.239 ms
result is likewise not an environment throughput result.

## Real transformer and MLP PPO

A correctness issue was found during real transformer warmup: both GN environments
reported one action even for fixed-power routing. The transformer therefore built
zero slot logits, while the MLP's single logit silently broadcast across every
masked action. Both now report the complete aggregated path-slot dimension for
routing and retain the scalar contract for RL launch-power policies.

**Earlier fixed-power MLP PPO timings used the single-logit head and should not be
presented as representative learned-routing baselines.** They remain saved as
historical execution evidence. New tests check actual logit dimensions, and real
parallel warmup/PPO tests cover both GN environments with a transformer.

Matched current-runtime runs use 1,376 slots, 16 envs, 128 policy-warmup requests,
16-step rollouts, 4 minibatches, 4 epochs and 4,096 training steps:

| Policy / backend | Compiled training throughput |
|---|---:|
| Transformer / ordinary JAX (`v10-control`) | 412.4 steps/s |
| Transformer / inline XPM + fused path SNR (`v10`) | 808.0 steps/s |
| Transformer / exact row reuse + fused path SNR (`v12`) | **1,016.1 steps/s (2.46×)** |
| Transformer / row reuse + precomputed path factors (`v19`) | **1,044.9 steps/s (2.53×)** |
| Transformer / grouped interferers + precomputed factors (`v20`) | 1,033.9 steps/s |
| Corrected MLP / JAX (`v4`) | 451.4 steps/s |
| Corrected MLP / row compaction (`v4`) | 950.2 steps/s |
| Corrected MLP / empty output tiles (`v4`) | 956.6 steps/s |

The row-reuse kernel is better for the policy's dispersed placements; inline tiles
win in the compact heuristic workload. There is no universal best backend.
Admission counts, mean accepted bitrate and blocking causes agree in the transformer
comparison. Bitrate standard deviations differ at rounding level; placement metrics
(utilisation/fragmentation) differ slightly, so PPO trajectories are not claimed
bit-identical. These short runs do not establish learning convergence.

## Profiling and accuracy

The wide empty-tile trace had only ~19% of kernel time in XPM; path-SNR gathers and
index construction consumed roughly half. Full path fusion removed the two large
index-building kernels. A later trace assigned ~36% to XPM and 25% to fused path
SNR, motivating the fixed-factor experiment. The 420-slot precomputed trace assigns
38% to candidate XPM and 18% to full path SNR. Names alone do not identify the
limiting resource: `input_multiply_reduce_fusion` included path gathers and noise
math, not just multiplication. Saved Nsight CSVs accompany the raw reports.

Hardware-counter collection was unavailable, including a new LaunchStats/Occupancy
attempt with CUDA 13 Nsight Compute: the driver reported profiling resources busy. Kernel
activity near 100% does not establish compute versus HBM saturation. No clocks,
services or other users' jobs were changed.

Coefficient checks include independent NumPy float64 controls, padding, empty
links, repeated centres, heterogeneous powers/bandwidths, different live counts,
nested batching and CUDA/Mosaic execution. Full path-SNR tests include dummy
paths, padded hops, on-path zero SNR, ROADM and transceiver noise. Environment
snapshots validate admission behavior separately from coefficient comparisons.
Native admission tests also compare all three mask outputs on an occupied network
with rejecting modulation thresholds, new-path margin, fibre-power limits and
insufficient existing-path SNR. HLO assertions verify the tested mask really uses
both custom kernels rather than cached ordinary-JAX traces.

Final checks: CPU GN/kernel/cache suite 128 passed, 30 skipped and two expected
issue-64 failures; native A100 coefficient/path suite 51 passed and two expected
failures; native admission tests 3 passed; ordinary-JAX GN performance tests on
A100 21 passed. Model, training utility and real PPO smoke checks passed 71 tests.
Pre-commit/type checks pass. Documentation builds; strict mode remains blocked by
122 existing API-docstring warnings.

These establish implementation equivalence to the current equations, not independent
physical validation. [Issue #64](https://github.com/micdoh/XLRON/issues/64)'s existing
higher-order dispersion underflow is deliberately **not fixed**. Binary-rescaled
accuracy controls are experimental only; two strict expected failures document the
legacy limitation. Existing SPM grouping/generic XPM-axis concerns also remain.
No reduced precision, approximate physics, interactions or admission checks were
removed. Private DRA code is excluded; the approved public stub remains.

Pallas Triton is deprecated; the slower Mosaic port is not yet a replacement for
performance purposes. Custom-call reverse-mode AD is not implemented. Differentiable
physical-model optimization must use the supported JAX path.

## Reproduction and evidence

Raw reports are under `docs/benchmarks/gn_kernel_v2_20261024/` and
`docs/benchmarks/gn_kernel_v3_20261024/`. Prefixes `v2`–`v24` identify experiments,
not released versions. Remote traces, HLO and snapshots remain in the isolated
`/tmp/xlron-gn-kernel-v2-20261024` tree on Turin. The existing department checkout
and environment were not modified.

```bash
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python -m xlron.kernels.gn_environment_benchmark \
  --compiler-profile library --xpm-backend pallas_inline \
  --path-snr-backend pallas_precomputed --slots 420 --envs 64 \
  --warmup-steps 128 --independent-streams --kernels link mask rollout \
  --iterations 7 --output /tmp/current.json --snapshot /tmp/current.npz

uv run --no-sync python -m experimental.benchmarks.gn_kernel_campaign \
  --mode train --gpu 2 --backends pallas_reuse --transformer \
  --path-snr-backend pallas_precomputed \
  --slots 1376 --envs 16 --prefix /tmp/transformer

uv run --no-sync pytest xlron/kernels -q
```

`--reference-script PATH` runs an archived workload script unchanged, with its
production imports selected by `PYTHONPATH`. This enabled the genuine original-code
comparisons. The campaign saves exact commands, exit codes and wall times; each
workload report separates compilation from synchronized execution.
