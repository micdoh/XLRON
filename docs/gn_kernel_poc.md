# GN kernels: performance, accuracy and reproducible comparisons

This branch continues the ordinary-JAX optimizations in PR #62 (`97373df`), which
has now merged. Custom backends remain opt-in. The supported differentiable path
and the normal environment default remain ordinary JAX. All simulations below
retain float32 physics, modulation thresholds, new-path margin, existing-path
checks and per-fibre power checks.

## Latest profiling iteration: generic step XPM and dense C+L+S

The previous mask optimizations exposed a separate dense reduction in the actual
environment step: 65.1% of GPU kernel time in a 1,376-slot rollout and 54.4% in
real transformer PPO. The generic GN model has a different existing 1-D broadcast
convention from the mask model. `gn_xpm_legacy.py` implements that convention
explicitly; it does **not** substitute the mask's equations or fix the discrepancy.
Frequency-only output-row reuse retains even zero-bandwidth output rows, whose
generic XPM coefficients can be nonzero. Optional interferer grouping preserves
exact multiplicity and the original total-power sum.

An additional path kernel returns the two admission predicates directly: the
minimum SNR on the new footprint and whether **every** existing channel passes
its threshold. NaN/+infinity thresholds retain rejection behavior. Encoding
inactive old predicates as -infinity saves one large intermediate array compared
with the first fusion attempt. This remains an explicit experimental override.

| Same light-load workload as the earlier measurements | Previous kernels | New kernels |
|---|---:|---:|
| 420 slots, 64 envs, 128 warmup, 16 measured steps, library profile | 6,796 env steps/s | **9,911 env steps/s** (`v30`) |
| 1,376 slots, 16 envs, 128 warmup, 16 measured steps | 1,268 env steps/s | **3,369 env steps/s** (`v29`); **3,931** after admission fusion/tuning (`v38`) |
| 420-slot transformer, 4 epochs / 4 minibatches | 4,801 training steps/s | **6,169 training steps/s** (`v32`) |
| 1,376-slot transformer, 4 epochs / 4 minibatches | 1,045 training steps/s | **2,259 training steps/s** (`v39`) |

The matched 420-slot original-code rollout is now 3,738.863 → 103.320 ms,
**36.19× overall** (274 → 9,911 env steps/s), on the same physical A100 and
compiler profile. The additional gain over ordinary optimized JAX is 3.77×.
These remain low-occupancy throughput experiments, not full-band training results.

Full **C+L+S** at 12.5 GHz resolves to **1,668 slots**, reference wavelength
1.5381051432646564e-6 m and single-slot gaps at indices 568 and 916. The historical
1,376-slot tests are not this complete band layout. A load-5,000, coherent,
16-env workload after 3,000 real warmup requests has 64.4% occupied link slots.
The warmup is used to build a dense test state, not claimed to reach steady state.
All following variants reuse that complete state and traffic RNG archive:

| Dense full-band backend | Mask median | 1,024-request rollout | Aggregate env steps/s |
|---|---:|---:|---:|
| Ordinary JAX (`v43`) | 102.310 ms | 7,315.448 ms | 139.98 |
| Inline XPM + generic step kernel + fused evidence (`v42`) | 68.268 ms | 4,461.131 ms | 229.54 |
| Grouped mask XPM (`v42`) | 7.175 ms | 520.742 ms | 1,966.42 |
| Grouped mask and generic step XPM (`v44`) | 7.172 ms | 481.597 ms | **2,126.26** |

This is **15.19×** over matched ordinary JAX on the dense full-band workload.
Masks, formats, rewards and final occupancy agree exactly across the measured
1,024 requests. Link SNR maximum relative error is 5.57e-7 (maximum absolute
1.23e-4 linear units). Separate 3,000-step warmups are **not** bit-identical:
grouping changed one admission among 48,000 requests. The same-state comparisons
separate that float32 trajectory sensitivity from kernel-output validation;
no thresholds or comparison tolerances in the simulator were relaxed.

The dense profile after grouped mask XPM assigns 19.4% of GPU time to fused
admission evidence, 12.0% to generic step XPM, 10.1% to candidate XPM and 8.6%
to candidate construction. Grouping generic step XPM subsequently saves another
8.1% end to end. Path layouts 256 slots/4 warps, 128/1 and 64/4 were slower;
64/1 gave only ~1%, so 128/4 remains the reference. Many packing/scatter/window
kernels now account for the remaining cost. Nsight launch records show ~100
registers/thread for XPM, no reported local-memory allocation; these records do
not establish achieved occupancy or compute/HBM saturation.

Raw reports and Nsight CSVs are in `docs/benchmarks/gn_kernel_v4_20261024/`.
`v44-dense-validation.json` records exact comparisons and the warmup caveat.
The initial v44 reference invocation requested a link output absent from its
inline reference archive and raised KeyError; the separate validation uses the
ordinary-JAX archive with all matching keys. State-cache save/load was also
checked for exact masks, rewards and occupancy on CPU.

### Transformer training with SNR observations

GN transformers now support `snr`, `occupancy_snr` and `departure_snr` observations.
The cache is refreshed after departures outside NN forward/backward passes;
empty channels are zero and finite dB features are normalized by `max_snr`.
Physical SNR values and admission comparisons are unchanged. Observation shape,
forward gradients, real batched PPO, separate critic optimization, best-checkpoint
save and model reload are tested.

A two-layer, 128-wide, four-head transformer with `occupancy_snr`, 16 environments,
64-step rollouts, four epochs/minibatches, 3,000 heuristic warmup requests and
32,768 training steps reaches **1,477.6 training steps/s** on full C+L+S (`v48`).
This is a different, denser workload from the historical 1,376-slot policy tests.
Training compile takes 15.15 s; synchronized training execution takes 22.18 s.
Neither warmup nor compilation is included in the throughput figure.

An opt-in actor experiment commutes only the final affine slot projection with
path summation, retaining hidden nonlinearities and multiplying bias by path
weight sum. Logits and parameter/input gradients pass parity checks, including
dummy paths. It gives **1,466.8 steps/s**, so it is **not selected** for the campaign.
It remains available as `--pool-before-projection` for reproducible experiments.

`experimental.benchmarks.gn_cls_training_campaign` runs four matched heuristics
(KSP-FF, FF-KSP, KSP-MSCL, MSCL-KSP) and two transformer candidates with online
W&B logging, source fingerprints, exact command manifests and checkpointed
evaluation. Each PPO candidate is configured for 10,485,760 training steps;
continuous-operation warmup is 25,000 requests per environment. Final model
evaluations use held-out traffic seeds 4242/4243, with matched heuristic seed 4242.
These are experiments, not evidence of convergence or beating a heuristic.

Latest CPU GN/kernel/observation/projection checks: **177 passed, 70 skipped,
two expected issue-64 failures**. The two SNR PPO smoke tests and a separate
campaign optimizer/checkpoint round trip also pass. Native suite results are
**130 passed, two expected issue-64 failures**. Pre-commit/type checks pass and
the documentation builds with the existing API-docstring warnings.

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
  It did not improve the early low-occupancy workloads; it is substantially faster
  in the dense full-band workload above.
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
| Inline tiles + precomputed full path SNR | — | **201.892 ms** |
| Mosaic port | — | 949.815 ms |

Fused path SNR reaches 6,363 aggregate environment steps/s at 420 slots (2.42×).
Precomputed path factors reach **1,268 steps/s at 1,376 slots (2.02×)** over PR #62;
the new snapshot matches every recorded output exactly, including link SNR.
With 512 warmup requests and 64 measured steps,
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

A new 420-slot / 64-env transformer comparison uses 8,192 training steps with the
same 128 warmup requests, 16-step rollouts, four epochs and four minibatches.
Row reuse plus precomputed path SNR reaches **4,801.4 training steps/s**, versus
1,795.1 for ordinary JAX on the same physical GPU 3 (2.67×; `v25` / `v27`).
The JAX repeat on GPU 2 gave 1,809.2 steps/s. Inline XPM reaches 2,782.7 steps/s,
confirming that row reuse is better for these policy placements. Every recorded
summary metric agrees exactly between the JAX and row-reuse runs.

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

Earlier-phase checks: CPU GN/kernel/cache suite 128 passed, 30 skipped and two expected
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

Raw reports are under `docs/benchmarks/gn_kernel_v2_20261024/`,
`docs/benchmarks/gn_kernel_v3_20261024/` and `docs/benchmarks/gn_kernel_v4_20261024/`.
Prefixes identify experiments,
not released versions. Remote traces, HLO and snapshots remain in the isolated
`/tmp/xlron-gn-kernel-v2-20261024` tree on Turin. The existing department checkout
and environment were not modified.

```bash
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync python -m xlron.kernels.gn_environment_benchmark \
  --compiler-profile library --xpm-backend pallas_inline \
  --step-xpm-backend pallas \
  --path-snr-backend pallas_precomputed --slots 420 --envs 64 \
  --warmup-steps 128 --independent-streams --kernels link mask rollout \
  --iterations 7 --output /tmp/current.json --snapshot /tmp/current.npz

uv run --no-sync python -m experimental.benchmarks.gn_kernel_campaign \
  --mode train --gpu 2 --backends pallas_reuse --transformer \
  --path-snr-backend pallas_precomputed \
  --step-xpm-backend pallas --candidate-snr-backend pallas --xpm-tile 16 32 4 \
  --slots 1376 --envs 16 --prefix /tmp/transformer

uv run --no-sync pytest xlron/kernels -q
```

`--reference-script PATH` runs an archived workload script unchanged, with its
production imports selected by `PYTHONPATH`. This enabled the genuine original-code
comparisons. The campaign saves exact commands, exit codes and wall times; each
workload report separates compilation from synchronized execution.
