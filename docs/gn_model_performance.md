# GN simulation performance investigation — 24 October 2026

The implemented changes reduce repeated GN calculations and bound candidate-state
memory while retaining the existing admission checks. On directed NSFNET with 420
slots, code changes alone made a 16-step RMSA rollout **4.44× faster** and reduced
action-mask temporary buffers **24.48×**. Further ordinary-JAX work removes infeasible/duplicate candidate evaluations and
zero-interferer arithmetic. Against the first optimized version, a 1,376-slot
16-environment rollout improves another **2.15×**, reaching **619 requests/s**
with real admitted warmup traffic and full SNR checks.

These are performance and implementation-equivalence results. The independent
formula audit below identifies pre-existing discrepancies that need physical-model
validation before making stronger multiband accuracy claims.

## Repository and PR status

The local workspace initially contained only `.venv`. The checkout was restored
from `https://github.com/micdoh/XLRON.git`, and the performance branch was based on
remote `main`, commit `2256456` (`Bump version to 1.6.0`). A subsequent fetch confirmed
the same remote head. During final validation, PR #61 landed on main as `dce10f9`;
the performance branch was rebased onto that commit. Performance changes are on
`perf/gn-multiband`, published as draft [PR #62](https://github.com/micdoh/XLRON/pull/62).
Its first commit, `804c53a`, passed all GitHub checks; the further work below
is recorded separately from those initial results.

One PR was open: [#61, Add ARBR adaptive-routing heuristic](https://github.com/micdoh/XLRON/pull/61).
It combines static path resource cost with quantized maximum link utilization,
exposes `arbr_alpha` (default 0.8), and uses first-fit spectrum assignment. Its
single-fibre adaptation omits the original algorithm's constrained-transceiver and
regeneration terms. It also changes comparison plots and figure styling. GitHub
initially reported it mergeable, with the Python 3.12 build passing and pre-commit
failing. A separate checkout reproduced the failure: Ruff reformatted one expression
in `wrappers.py`, while environment setup refreshed stale lockfile metadata. Commit
`3bf64db` fixes both and makes CI dependency synchronization require the committed
lockfile. It was pushed only to the public PR branch, `arbr-heuristic`.

All six original PR commits and the fix were checked for changes to the withheld
DRA implementation. They retain the public stub. A new GitHub public-release check
examines every incoming commit, requires the approved stub, and rejects additional
DRA-named Python/notebook source files. Its six regression tests include a change
that adds private content and removes it in a later commit. This guard cannot
identify private code copied under unrelated names. Existing public DRA
documentation is retained at the user's request; the private GitLab checkout was
not changed. Local PR validation passed all pre-commit hooks and 666 heuristic and
dispatch tests (157 skipped), separately from the GN test results below.
GitHub CI on that commit passed pre-commit, public-release and build checks; the
build reported **2,328 passed, 498 skipped**. The PR subsequently merged as
`dce10f9`; its DRA guard also passes on the rebased performance branch.

## Implemented architecture

### Masking semantics retained by the optimization

`rsa_gn_model` inherits the spectrum-only RSA mask and accepts requests based on
spectrum availability. GN SNR is evaluated separately for its throughput objective;
it does not enforce per-request SNR or fibre-power admission checks. A misleading
comment claiming otherwise has been corrected.

`rmsa_gn_model` computes a separate footprint and required SNR for every path and
modulation format. It tests first-fit and last-fit placements for each pair, not
every interior placement. Each tested candidate must satisfy spectrum and power
constraints, the new connection's SNR threshold plus admission margin, and existing
connections' stored modulation thresholds. The existing-connection check uses the
base thresholds without reapplying the admission margin. If several formats pass
at the same path/start slot, the mask stores the highest-SE format. The step reads
that stored format and uses its SE to determine the allocated width. Modulation
correction, when enabled, also uses the hypothetical candidate's format.

This is implicit format selection. An agent that explicitly chooses modulation
needs a joint or conditional mask that retains that dimension. A compact list of
physically evaluated `(path, format, placement)` candidates could expose format
choice without expanding to every path/format/slot combination. A conditional
policy alone does not remove the cost of checking whether its earlier choices
have any physically valid completion.

### Occupied-path SNR accumulation

Previously, `get_lightpath_snr` formed SNR arrays for every possible route and every
slot, then gathered values for the paths actually referenced by the state. Directed
NSFNET with `k=5` has 911 route rows, including the dummy row, and 44 links.

Environment creation now precomputes a padded `(paths, maximum_hops)` table of
constituent link indices. SNR accumulation gathers only the links of each path
referenced by the `(links, slots)` state. Its intermediate work scales with
`links × slots × maximum_hops`, independent of the number of unused route rows.
This also avoids zero-times-infinity contamination from off-path zero SNR entries.
ROADM and transceiver noise retain their path-level treatment. Empty paths, packed
path bits and single-slot shapes have regression coverage.

### Shared baseline and selective link refresh

RMSA masking evaluates first-fit and last-fit placements for each route/modulation
pair: 40 candidates for five routes and four formats. Previously each candidate
recomputed link SNR over the whole network.

Masking now refreshes the current network once, then each candidate recalculates
only the links on its route. Unchanged links reuse that shared baseline. Importantly,
the baseline is recomputed rather than trusting `state.link_snr_array`: departures
can leave that cache stale. Both generic and fused link-SNR helpers accept selected
link indices, retaining their ordering and duplicates.

Existing lightpaths are still checked across their complete routes. Selective GN
refresh does not mean that disruption checks inspect only the new lightpath.

### Bounded candidate batches (first optimization stage)

Candidate states are constructed inside `jax.lax.map`, rather than materializing
every hypothetical network simultaneously. The measured default is up to 40
candidates per batch for at most 128 slots, and eight for wider spectra. Batching
one candidate at a time substantially reduced memory but slowed small workloads;
32- and 40-candidate experiments motivated the small-spectrum policy.

Invalid candidates remain masked. A batched `lax.cond` may execute invalid lanes,
so this change does **not** claim that every invalid candidate avoids GN work.
The storage bound comes from batching. See the [JAX map documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html).

## Measurements

Runs used Turin's NVIDIA A100 80 GB PCIe GPUs, with JAX preallocation disabled.
GPUs 2 and 3 were available; jobs on GPUs 0 and 1 were left alone. Each result
separates compilation from synchronized execution, performs three warmups, and
reports the median of 7–20 measurements. Rollouts contain 16 actual environment
steps. Measurements exclude neural-network training, compilation and host logging.

**Memory columns are XLA compiled temporary-buffer estimates, not total VRAM or
process peak memory.** Arguments, outputs, constants and allocator/compiler memory
are additional. Small timings near the dispatch floor show noise and should not be
interpreted as precise microsecond-level comparisons.

### Code changes isolated from dependency changes

These two 420-slot runs both use JAX 0.9.1, directed NSFNET, `k=5`, four modulation
formats, 12.5 GHz slots and approximately 25% synthetic occupancy.

| Measurement | Original | Eight-candidate wideband path | Improvement |
|---|---:|---:|---:|
| Action mask | 4.320 ms | 1.151 ms | 3.75× |
| 16-step rollout | 77.073 ms | 17.362 ms | 4.44× |
| Mask temporary buffers | 126.505 MiB | 5.168 MiB | 24.48× smaller |
| Rollout temporary buffers | 126.413 MiB | 5.086 MiB | 24.86× smaller |
| Standalone mask compilation | 866.33 s | 4.35 s | About 199× in this case |

The compilation pathology was an NVIDIA `ptxas` invocation for an input-reduction
fusion. The original source under JAX 0.12.0.1 also failed to finish mask compilation
within a 240-second cap. Its result file contains only the completed link/path
measurements; there is no inferred runtime for the missing mask or rollout.

### Final runtime and small-spectrum policy

The following small-spectrum comparisons all use JAX 0.12.0.1, full-precision
matrix accumulation and 100 slots. The optimized column uses batches of 40.

| Execution mode | Original 16 steps | Optimized 16 steps | Original mask temporaries | Optimized mask temporaries |
|---|---:|---:|---:|---:|
| Uniform spans | 8.828 ms | 5.783 ms | 33.18 MiB | 8.25 MiB |
| Modulation correction | 11.019 ms | 6.680 ms | 75.40 MiB | 8.92 MiB |
| Non-uniform span execution | 15.104 ms | 13.792 ms | 35.18 MiB | 8.93 MiB |

The non-uniform benchmark exercises the span-scan implementation; the factory's
topology-derived spans remain equal within each link. Separate regression tests
cover span padding and non-uniform inputs.

Wider final-runtime results:

| Configuration | 16-step time | Aggregate requests/s | Mask temporaries |
|---|---:|---:|---:|
| 420 slots, one environment, synthetic occupancy | 16.882 ms | 948 | 5.34 MiB |
| 420 slots, 16 environments, synthetic occupancy | 135.158 ms | 1,894 | 73.98 MiB |
| 1,376 slots, one environment, 128 real warmup steps | 79.076 ms | 202 | 16.03 MiB |

The real-warmup 1,376-slot state had 5.33% occupancy and 128 accepted requests;
the following 16 requests were accepted. This demonstrates a functioning 17.2 THz
calculation with admission and disruption checks, not near-saturation performance
or validation of the model's accuracy over 17.2 THz. The synthetic 25% wideband
stress state includes physically unvalidated placements and rejects all 16 subsequent
requests; its timing is recorded separately and is not an admitted-traffic result.

Those initial parallel comparisons use replicated initial states and common traffic
to make before/after comparisons reproducible. The independent-stream results below
measure different traffic in each environment.

### Parallel rollouts and PPO training

These additional runs use independent reset/traffic keys, 128 real warmup admissions
per environment, uniform spans, no modulation correction, and the final JAX runtime.
The physics and admission checks run for every tested placement. These are not
near-saturation traffic experiments: initial mean occupancy is about 9–10% at 420
slots and 4–5% at 1,376 slots. Times are synchronized medians of five 16-step rollouts.

| Slots | Parallel environments | Aggregate steps/s | Rollout temporary buffers |
|---:|---:|---:|---:|
| 420 | 1 | 944 | 5.10 MiB |
| 420 | 16 | 1,907 | 72.45 MiB |
| 420 | 64 | 2,430 | 360.99 MiB |
| 420 | 256 | 2,521 | 1,732.70 MiB |
| 1,376 | 1 | 201 | 15.25 MiB |
| 1,376 | 16 | 288 | 295.64 MiB |
| 1,376 | 64 | 306 | 1,064.31 MiB |

The 420-slot workload gains only about 4% throughput from 64 to 256 environments.
The wider workload gains about 6% from 16 to 64. Larger batches fit, but are not
automatically useful: there is already parallel work across links, channel pairs
and candidate placements within one environment. More environments add independent
GN work; they do not make that work reusable.

An Nsight Systems node-level CUDA graph trace of one warmed-up 64-environment,
420-slot mask recorded 148 kernels with 22.65 ms of summed kernel time in a
22.69 ms kernel window. The union of kernel intervals occupies 99.8% of that
window. Host launch gaps are therefore small in this case. The largest fusion,
`input_scatter_fusion`, takes 63.6% of kernel time. Its optimized HLO includes the
candidate GN SNR computation, including XPM `atan` arithmetic, fused into the
selected-link scatter; its name does not mean the time is just array scattering.

This establishes a GPU execution bottleneck, not whether arithmetic throughput,
instruction latency, cache traffic or HBM bandwidth dominates that fusion. Nsight
Compute counter collection failed with a driver-resource-unavailable error; a DCGM
profiling-field read also failed. Neither tool's failed counter collection provides
evidence of bandwidth saturation. Dense slot-pair arithmetic (including empty slots)
and repeated candidate evaluations remain the strongest algorithmic targets. The
[kernel summary](benchmarks/gn_20261024/nsys-node-mask-420-env64_cuda_gpu_kern_sum.csv)
and [CUDA API summary](benchmarks/gn_20261024/nsys-node-mask-420-env64_cuda_api_sum.csv)
are saved; the full trace and HLO remain on Turin. See NVIDIA's
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) for the
distinction between timing traces and utilization/roofline counters.

Real MLP routing-policy PPO runs also completed with fixed launch power, policy
warmup, 16-step rollouts, four minibatches and four update epochs:

**Subsequent correction:** these initial fixed-power MLP runs used a single actor
logit broadcast over the path-slot mask. The GN environment's reported action
dimension has since been corrected, and the transformer warmup/PPO path is now
tested too. Retain the table below as historical execution evidence; see
[Pallas kernel backend](#pallas-kernel-backend) for current full-mask throughput.

| Slots | Environments | Training steps | Compiled training steps/s | Compilation | Sampled process VRAM |
|---:|---:|---:|---:|---:|---:|
| 420 | 64 | 8,192 | 2,367 | 30.13 s | 1,250 MiB |
| 1,376 | 16 | 4,096 | 286 | 49.47 s | 2,134 MiB |

Training throughput includes policy inference, environment stepping and PPO
optimization; it excludes compilation, warmup and metric logging. Process VRAM is
the largest `nvidia-smi` reading sampled once per second, not an exact high-water
mark. It is a different measurement from XLA's temporary-buffer estimate above.
These short runs verify execution and measure throughput, not learning convergence.
The raw `ppo-*-execution.json` files record commands, exit status, sampling and wall
time; the corresponding JSONL files contain the training timing/metrics.

The first fixed-power MLP trial exposed an obsolete warmup guard that raised an
exception for this supported routing-policy configuration. Removing that guard
allows both GN variants to complete parallel PPO updates. New end-to-end regression
tests cover both variants with two independent environments and policy warmup.

### RSA GN variant

At 420 slots on JAX 0.12.0.1, RSA GN rollouts measured 6.563 ms before and 6.554 ms
after: essentially unchanged. Standalone path-SNR temporaries fell from 3.06 MiB to
0.21 MiB. Its Shannon-throughput calculation measured approximately 0.41 ms on both
versions with unchanged temporary storage.

`rsa_gn_model` intentionally defers physical-layer refresh/capacity evaluation in
places where `rmsa_gn_model` performs admission checks. These modes should not be
treated as equivalent measurements of per-request physical feasibility. The shared
path-SNR change benefits both, while the candidate rearchitecture primarily targets
RMSA GN masking. The RSA capacity routine remains a possible subsequent optimization.

## Further reductions in wasted work

These follow-up changes build on `804c53a`. Comparisons in this section use that
first optimized version as the baseline, with the same JAX 0.12.0.1 runtime and
benchmark harness. They must not be multiplied into the earlier 4.44× comparison:
the traffic, occupancy, parallelism and runtime differ.

### Compact feasible candidates and reuse physical footprints

For non-differentiable spectra wider than 128 slots, availability and fibre-power
checks run before GN. A fixed-capacity index buffer packs the feasible candidates;
a dynamic loop evaluates only the live batches of eight. Identical placements
share SNR evidence when their route, start, width and launch power match. The key
also includes modulation when modulation correction or Raman amplification is
enabled, so format-dependent physics is not shared incorrectly.

The evidence records endpoint SNR, minimum SNR on newly assigned slots, and whether
all old connections pass. Each candidate then applies its own modulation threshold
and margin. Equal-width formats therefore retain different admission requirements.
Small spectra and differentiable mode keep the previous static traversal. A first
attempt added about 6% rollout overhead at 100 slots; keeping the original validity
calculation on that path recovered essentially unchanged performance.

Under an outer environment `vmap`, the dynamic loop still runs to the largest live
candidate count in the batch. Padding in its final batch may execute. This removes
whole unused batches; it does not guarantee zero work for every invalid lane.

Candidate compaction alone on synthetic states at approximately 46% occupancy:

| Slots / environments | First optimized mask | Compacted mask | First optimized rollout | Compacted rollout |
|---|---:|---:|---:|---:|
| 420 / 64 | 22.198 ms | 6.922 ms | 427.737 ms | 244.093 ms |
| 1,376 / 16 | 44.746 ms | 20.972 ms | 902.457 ms | 610.502 ms |

These placements are physically unvalidated synthetic workloads, not admitted
near-saturation traffic. All 16 requests are rejected in the wider case; the
420-slot case accepts 12. Masks, rewards and occupancy match exactly. Raw files
are `optimized-crowded*.json` and `reduced-crowded*.json`.

### Skip zero interfering channels in uniform-span XPM

The uniform-span, no-modulation-correction fused model now packs interferers with
nonzero power and bandwidth, and traverses them in 128-column tiles using ordinary
JAX. All output slots remain. The reduction uses the original total power for
Raman tilt; repeated-centre occupied slots remain separate contributions. Only
exactly zero pair contributions are removed. This is not channel aggregation or a
physical approximation, and the existing physical formulas are unchanged.

The default enables this at 420 slots and above, the measured wide-spectrum range,
and retains dense fusion below that threshold and for differentiable environments.
Generic/non-uniform and modulation-corrected GN calculations keep their previous
physical kernels. Active counts may differ per link; a vmapped JAX loop stops at
the largest live count in the mapped batch. Reduction order can change float32
rounding, so equivalence is numerical rather than bitwise for SNR.

Both ordinary-JAX follow-ups together, synchronized medians of seven samples:

| Slots / environments / state | First optimized mask | Final mask | First optimized 16-step rollout | Final rollout | Final requests/s |
|---|---:|---:|---:|---:|---:|
| 100 / 1 / real warmup | 0.353 ms | 0.357 ms | 5.951 ms | 5.736 ms | 2,789 |
| 420 / 64 / independent real warmup | 23.424 ms | 21.087 ms | 422.405 ms | 390.717 ms | 2,621 |
| 1,376 / 16 / independent real warmup | 47.304 ms | 17.022 ms | 890.309 ms | 413.241 ms | 619 |
| 1,376 / 16 / synthetic 46% occupancy | 44.784 ms | 20.837 ms | 901.299 ms | 565.166 ms | 453 |

The real-warmup cases use 128 admitted warmup steps, initial occupancy 36.5%, 10.1%
and 4.65% respectively, and the same independent streams as their baselines. The
last row is the physically unvalidated synthetic case described above. The final
wide-spectrum rollout gain is 2.15× in the real-admission workload and 1.59× in the
synthetic case. The small-spectrum difference is within a few percent; no small
workload speedup is claimed. All recorded masks, modulation choices, rewards and
occupancy match exactly. Raw `optimized-waste-*.json` / `reduced-waste-*.json` files
contain the settings, assertions and memory estimates.

These are environment rollout results. The earlier full PPO measurements precede
this follow-up; a proportional PPO gain has not been measured. A custom Pallas
XPM experiment is kept on the separate `perf/gn-kernel-poc` branch/worktree and is
not included in PR #62.

The expanded GN suite passes **92 tests, six skipped**, including full-network
candidate oracles at both 12 and 140 slots for uniform, corrected and non-uniform
modes, unequal modulation thresholds on reused footprints, different batched live
counts, and full dense-versus-packed SNR with empty, sparse and dense links.

## Pallas kernel backend

`--gn_kernel_backend=pallas` (see [training](training.md)) was measured against the
ordinary-JAX reference on an A100 80GB. The GPU was shared with another job at
roughly 98% utilisation, so absolute rates are conservative. The workload is
NSFNET directed, full C+L+S (1,668 slots), k=5, `all_slots` masks, load 5,000,
coherent NLI, 4 independent environments, candidate batch 64 and environment
chunk 1. Every arm starts from the same saved state (256 requests admitted per
environment by a Pallas warmup) and replays the same traffic:

```bash
python -m experimental.benchmarks.gn_scaling --gn-kernel-backend pallas --slots 1668 --envs 4 --k 5 --load 5000 --coherent --band-preference C,L,S --gn-mask-mode all_slots --gn-mask-batch-size 64 --gn-mask-env-batch-size 1 --independent-streams --kernels mask rollout --steps 4 --iterations 3 --state-input state.npz
```

| Arm | Mask, 4 envs (median) | 4-step rollout (median) | Environment transitions/s |
|---|---:|---:|---:|
| `jax` (reference) | 13.15 s | 50.70 s | 0.32 |
| `pallas` | **0.71 s** | **2.62 s** | **6.12** (19×) |
| #65 code, best flags (archived) | 0.67 s | 2.58 s | 6.20 |

All arms produce identical valid-action counts and admissions. The #65 control
used the pre-#66 XPM broadcast. Its kernels are not faster for that reason: in
this state every channel has the same launch power, so the two formulas agree.

## Mask throughput, 27 October 2026

A second pass on the Pallas `all_slots` workload: NSFNET directed, full C+L+S
(1,668 slots), k=5, load 5,000, coherent NLI, `safe_release`, candidate batch 64,
environment chunk 1, 16 environments replaying the same saved steady state
(about 6,000 admitted lightpaths each), on a dedicated A100 80GB with default XLA
flags:

```bash
python -m experimental.benchmarks.gn_scaling --gn-kernel-backend pallas --slots 1668 --envs 16 --k 5 --load 5000 --coherent --band-preference C,L,S --gn-mask-mode all_slots --gn-mask-batch-size 64 --gn-mask-env-batch-size 1 --independent-streams --state-input ss16.npz --kernels rollout --steps 16 --iterations 3 [--heuristic NAME]
```

| Workload | Before | After |
|---|---:|---:|
| Full mask, 16 environments (median) | 41.4 ms | 23.4 ms |
| Environment steps/s (all_slots mask, first valid action) | 321 | 600 |
| `ksp_ff` steps/s | 167 | 1,474 |
| `arbr_ff` steps/s | 167 | 588 |
| `kca_ff` steps/s | 167 | 592 |
| `ksp_ff` warmup from empty, 2,000 steps x 16 environments | 984 s | 23 s |

The warmup "before" was measured with the other changes already in place; the
first-fit mask alone accounts for it.

Full-mask valid-action counts and admissions are identical across the changes,
and first-fit masks select the same actions as the full mask (tested). Changes:

- **Unbatched environment chunks.** `lax.map(batch_size=1)` vmaps a batch of
  one, which turns every data-dependent `cond`/`switch` in the mask into a
  select of all branches and gives every while loop a batched predicate. Chunk 1
  is now a plain scan.
- **Delta evaluation of screened candidates.** Each candidate batch
  materialised five (batch, links, slots) network copies for the evidence kernel.
  Candidates now re-solve only their path links from explicit rows, and the
  evidence kernel applies the insertion to the shared baseline arrays on the fly
  (`delta=` in `path_snr_pallas`).
- **Tail batches.** Most passes leave a handful of candidates. Each loop
  iteration uses the smallest power-of-two fraction of the batch (down to 1/16)
  that finishes the remainder; passes with nothing available, or no screen
  survivor, are skipped.
- **Single-kernel compaction.** `jnp.nonzero` lowers to a blocked cumulative
  sum, a scatter and a broadcast; `xlron/kernels/compact.py` does it in one
  Pallas program, for mask repacks and for the three per-link compactions in the
  XPM wrapper (a full link-SNR refresh fell from 0.74 to 0.46 ms).
- **Heuristics reuse the mask.** `select_action_eval` computes the physical mask
  once; heuristics recomputed it two or three times (`heuristic_mask_from_state`).
- **First-fit masks.** `ksp_ff` reads only the first path with a valid start,
  and `ff_ksp` each path's first fit, so they get a mask
  (`gn_mask_first_fit`) that scans (path, start, format) candidates per path in
  band-preference order, jumps past each path once it settles, and for `ksp_ff`
  stops at the first feasible path. Screen blocks grow while the scan continues.
  In a nearly empty or half-filled network almost every start is spectrum-free,
  and the full mask verifies all of them; this dominated heuristic warmup, which
  RL training also uses. `ff_ksp` gains less (about 300 steps/s mid-fill), as it
  must settle every path.
- **Exact link-SNR cache under `safe_release`.** Link SNR depends only on that
  link's own channels, and every change re-solves exactly the links it touches,
  so admission re-solves only the new path's links and SNR observations skip
  their full refresh. The mask still refreshes once, as heuristics may pass a
  stale cache.

Tile sizes of the evidence and XPM kernels were swept (128 to 1,024 slots per
evidence tile; XPM row/column tiles and persistent programs); all settings were
within noise of the defaults. Environment vmapping remains slower (124 steps/s
for chunk 16 or a full vmap), as each environment's loops run to the busiest.

**Do not enable XLA's device-side loops.** With JAX 0.12,
`--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL` removes the per-iteration
host synchronisation and was up to 7.9x faster on a shared GPU, but
compute-sanitizer reports out-of-bounds reads in XLA fusions inside the
command-buffer loops (either flag alone), and RMSA-GN runs with
`departure_policy=immediate` segfaulted on A100s. It is not enabled.

## Pooled environment masks and PPO, 28 October 2026

A third pass, aimed at transformer PPO and heuristic evaluation on the full C+L+S
workload (NSFNET directed, 1,668 slots, k=5, load 5,000, coherent NLI,
`safe_release`, `all_slots`, Pallas). Masks, formats, rewards and occupancy are
identical to the previous code in every comparison below (`gn_scaling
--reference`, max abs error 0). Measured on an A100-SXM4-80GB.

| Workload, 16 environments at steady state (about 6,200 lightpaths each) | Before | After |
|---|---:|---:|
| Full mask, all environments | 27.4 ms | 5.1 ms |
| Steps/s with full masks (RL policies, `arbr_ff`, `kca_ff`, ...) | 575 | 2,872 |
| `ksp_ff` steps/s | 1,130 | 2,963 |
| `ff_ksp` steps/s | 1,115 | 2,720 |

The "after" rollout figures use `--xla_gpu_graph_min_graph_size=1`, which
`train.py` now sets by default (`gn_scaling` itself sets no XLA flags).

Transformer PPO with the GN sweep scripts' settings (32 environments, rollout 128,
8 minibatches, 4 epochs, `occupancy_snr` tokens, modulation features, 25,000
`ksp_ff` warmup steps), same seed:

| | Before | After |
|---|---:|---:|
| Training steps/s | 607 | 2,530 |
| Warmup, 25,000 steps x 32 environments | 483 s | 163 s (0.5 s from `--WARMUP_CACHE_DIR`) |
| Training compile, `--xla_gpu_force_compilation_parallelism=1` | 273 s | 291 s |
| Training compile, default XLA flags | 23 s | 27 s |
| Estimated 4.2M-step run, default flags | 2.1 h | 31 min (28 min with a cached warmup) |

The single-threaded compilation flag used for long pytest runs (see `CLAUDE.md`)
adds about four minutes to every training run; leave it out of launch scripts.
GPU PPO training is not bitwise reproducible between runs (two runs of the same
code and seed differ in their third significant figure), so training comparisons
are of throughput only; the environment comparisons above are exact.

What changed, in order of effect:

- **Pooled environment chunks.** A serial per-environment mask is latency-bound:
  about 266 kernels of a few microseconds and 19 host synchronisations per
  environment, leaving the GPU idle half the time. Masking a chunk in lockstep
  (`vmap` with a named axis whose control-flow predicates are reduced over the
  chunk) shares the loops and branches, but on its own pads every environment to
  the busiest one: the GN work grew 13x and the mask slowed to 102 ms. So the
  per-environment inputs are gathered (`all_gather`, free inside the vmap) and
  every environment's screen and full-check candidates go through one queue; the
  endpoint and evidence kernels read each lane's environment from the stacked
  arrays in place (`env=`), with no per-lane copies. Batches then cover the
  chunk, so `gn_mask_batch_size` and `gn_mask_screen_batch_size` are scaled by
  the chunk size up to 256 and 2048, where mask time plateaued. The default
  `gn_mask_env_batch_size=0` is now one chunk of all environments (it was a
  plain vmap, which executes every branch: 124 steps/s).
- **Pooled first fit.** First-fit masks keep each environment's own scan (block
  growth, settled paths, early stop); each round screens the next block of every
  unfinished environment through one queue, then checks pending survivors with
  per-environment allowances that double while checks continue.
- **Polynomial XPM re-solve.** Total power enters every XPM pair only through the
  Raman tilt `(ap - fk * P * cr)^2`, so each pair is exactly `c0 + c1 P + c2 P^2`
  (`xpm_pair_coefficients`). The kernel's `polynomial=True` mode stores each
  link's three sums once per mask; a candidate's path links then get their XPM
  at the new total power plus the inserted channel's own pairs
  (`xpm_after_insertion`) in O(slots) per link, instead of a full reduction.
- **Changed-row evidence.** Every existing lightpath whose SNR an insertion can
  change crosses one of the candidate's own links, so the evidence kernel visits
  only those rows (`changed_rows_only`, 5 instead of 44 here) with per-cell
  thresholds. Lightpaths sharing no link with the candidate path are decided once
  per mask from their baseline SNR. This also removes the predicate-packing
  kernel from these masks.
- **Unrolled transformer layers.** The encoder's `lax.scan` over layers stacked
  every layer's backward residuals into `(layers, batch, ...)` buffers: 73 copies
  and 65 GB of device copies per PPO update. Unrolled, a 512-sample minibatch's
  forward/backward pass fell from 14.3 to 11.1 ms (`policy_update_bench.py`).

- **Single-kernel CUDA graphs.** `--xla_gpu_graph_min_graph_size=1` (now in
  `train.py`'s default XLA flags) lets XLA capture even single kernels between
  host synchronisations in CUDA graphs: +8% PPO, +9% `ksp_ff`. Adding custom
  calls to command buffers was slower; device-side while loops stay off.
- **No scatter for small `nonzero`.** `release_departures_safely` picked held
  lightpaths with `jnp.nonzero(size=32)` over all 73k cells, which lowers to a
  scatter of every cell (0.8 ms per 16-environment step);
  `first_true_indices` searches the running count instead. Candidate path links
  come from `path_link_indices` rather than a per-lane `nonzero`, and
  `searchsorted` over short arrays uses `compare_all` (the default `scan` is a
  loop of launches).
- **Endpoint-only amplifier ASE.** The endpoint screen evaluated the ISRS-aware
  ASE at every slot and kept two (`_ase_with_isrs_gain_at`).
- **Additive fibre power.** A candidate fills free slots, so it adds exactly one
  channel to its links' total power; the screen adds its launch power to the
  baseline totals instead of re-reducing the spectrum (equal up to float32
  summation order).
- **Packed hop rows.** Paths are padded to the longest one's six hops (3.7 real
  on average here). The pooled full check re-solves only real (lane, hop) link
  rows, in a buffer of three quarters of the padded size with a full fallback.
- **Overlapped logging and cached warmups.** Increments run on a worker thread
  while the previous one is logged (3.5% of wall time with W&B off), and
  `--WARMUP_CACHE_DIR` reuses warmed-up states (see [training](training.md)).

Measured and not adopted: peeling the first iteration of the candidate loops
into device-side conds (no change; the step is GPU-bound now), `bfloat16`
compute for the transformer (`--compute_dtype=bfloat16`, +5% end to end, changes
numerics), and trusting the `safe_release` link-SNR cache instead of the mask's
baseline refresh (about 3%, but the cache differs from a fresh solve by float32
rounding, so masks would change). 64 environments ran 9% faster than 32 (2,769
steps/s), with a larger PPO batch.

Fixed on the way: `every_path` first-fit masks (`ff_ksp`) lost later paths'
first fits whenever an earlier path settled inside the same screen block, as the
survivor loop stopped at the first settled survivor. It now skips to the next
survivor whose path is unsettled.

Profiles after these changes. A 16-environment full-mask step is GPU-bound
(5.9 ms of kernels in about 6 ms): the mask's XLA fusions (2.9 ms, led by the
inserted rows' SNR) and its Pallas kernels (endpoint screen 1.1 ms, XPM 0.5 ms,
evidence 0.4 ms), departures and request generation 0.5 ms. In training, the
rollout is about three quarters of the time and the PPO update most of the
rest; the policy's rollout forward pass is 0.75 ms per 32-environment step. The
update's forward/backward pass is memory-bound with no dominant kernel (the
largest, the token embedding GEMM, is 8%).

```bash
python -m experimental.benchmarks.gn_scaling --gn-kernel-backend pallas --slots 1668 --envs 16 --k 5 --load 5000 --coherent --band-preference C,L,S --gn-mask-mode all_slots --gn-mask-batch-size 64 --gn-mask-env-batch-size 16 --independent-streams --state-input ss16.npz --kernels mask rollout --steps 16 --iterations 3 [--heuristic NAME] [--reference chunk1.npz]
python -m experimental.benchmarks.policy_update_bench --minibatch 512 [--forward-only]
```

## Dependencies and Gymnax

`uv.lock` was refreshed: 159 package versions changed, 20 packages were added and
14 removed. Important final versions are JAX/JAXlib 0.12.0.1, Gymnax 1.0.1, Equinox
0.13.8, Flax 0.12.10, Distrax 0.1.9, NumPy 2.5.4, SciPy 1.18.1 and Optax 0.2.8.
The [complete version changes](benchmarks/gn_20261024/dependency_updates.json) are saved.

Gymnax is retained. Its 1.0.1 release declares `jax<0.12` to bound advertised support
to combinations covered by CI, according to its [changelog](https://github.com/RobertTLange/gymnax/blob/main/CHANGELOG.md).
`pyproject.toml` uses a package/version-specific uv override for that JAX requirement.
Other Gymnax dependencies and the JAX CUDA extras remain resolved normally. The
scoped override requires uv 0.13.5 or newer; uv was updated locally and on the
benchmark host. The existing department project's Python environment was preserved.

The override is a tested XLRON configuration, not a claim that all Gymnax built-in
environments support JAX 0.12. Future Gymnax versions are deliberately outside this
version-specific override and must be evaluated normally.

The upgraded ty version makes warning-only diagnostics fail by default. The explicit
`tool.ty.terminal.error-on-warning = false` setting retains the project's existing
non-blocking warning policy while errors still fail. Warnings are still reported;
see the [official rule-level documentation](https://docs.astral.sh/ty/rules/).
The upgraded pre-commit also needs a modern Git: the local default Git 2.22 lacks
the `ls-files --deduplicate` option it uses. Final hook validation passed using the
already installed bundled Git 2.53, without changing the system Git installation.

## Verification and accuracy limits

- Full upgraded-stack suite: **2,290 passed, 488 skipped**.
- GN suite after the final batching adjustment: **84 passed, six skipped**.
- Subsequent parallel fixed-power PPO smoke tests: **two passed**; training utility
  tests after the warmup fix: **27 passed**. These extend the original suite result
  above; the whole suite was not rerun after this small warmup change.
- After rebasing onto main with ARBR: **54 GN/factory/training-utility tests passed**,
  plus **40 ARBR tests passed, ten skipped**. All pre-commit hooks and the public
  DRA guard passed; the guard's six regression tests also passed.
- Independent small-network candidate oracle checks full-network recomputation,
  stale cached SNR and successive actual steps for all three execution modes.
- Selected-link results are compared with full-network results, including repeated
  and reordered indices. Sparse-path tests cover packed paths and zero SNR.
- Recorded mask, reward and occupancy arrays match exactly in the compared cases.
  Full-precision SNR accumulation differs at ordinary float32 rounding levels
  (about 2e-6 dB in the 100-slot comparison).
- The original default GPU matrix arithmetic differs from full-precision sparse
  accumulation by up to 0.00153 dB in the recorded 420-slot case. Its masks and
  rollout outcomes still agree exactly. New benchmark defaults request `highest`
  matrix precision to make numerical comparisons meaningful.
- Type checking has warnings but no errors. The changed core files have no added
  Ruff diagnostics relative to the baseline; existing lint diagnostics remain.
- Documentation builds successfully; existing API-docstring warnings remain.
- All final pre-commit hooks pass, including type checking, with the upgraded
  environment and Git 2.53.

Equivalence tests establish preservation of the implemented model. They do not
establish agreement with an independent physical propagation model.

The [independent coefficient audit](benchmarks/gn_20261024/accuracy_audit.json) follows
the grouping and channel axes of the [cited Semrau implementation](https://github.com/dsemrau/ISRSGNmodel/blob/master/Functions/Python/ISRSGNmodel.py),
retaining XLRON's beta4 expression. It exposes two pre-existing discrepancies:

1. `_spm` and its uniform/fused inlined forms apply the leading coefficient to
   only the first of the two bracketed contributions. The reference applies it to
   their sum. The eight-channel diagnostic differs by about 1.06% in SPM efficiency.
2. The generic `_xpm` receives one-dimensional channel power/bandwidth inputs from
   the environment; their `.T` operations do not create channel-of-interest and
   interferer axes. The fused implementation uses explicit axes. The heterogeneous
   eight-channel diagnostic differs by up to 121% in XPM efficiency.

These percentages concern the diagnostic coefficients, not total SNR error or all
operating points. The production formulas were preserved so performance and physics
changes can be reviewed independently. Multislot channel-versus-slot power accounting
also deserves an independent validation case: the fibre power-budget helper deduplicates
channel blocks, while the GN tilt sums the per-slot power array.

The public DRA module is a `NotImplementedError` stub pending publication. DRA timing
and physical regression checks need the actual implementation. Also, the base model
documents a linear Raman-slope approximation with C+L applicability; a wider numerical
stress run alone cannot validate S-band or broader multiband accuracy.

## Further architecture opportunities

These are proposals inferred from the code and measurements, not implemented or
benchmarked speedup claims:

1. **Validate and unify the physical kernels first.** Add independent SPM/XPM,
   heterogeneous power, multislot and broad-spectrum references before combining
   generic/fused execution or changing channel representations.
2. **Aggregate repeated occupied centres with explicit weights.** This PR now
   skips zero interferers in the wide uniform-span path, but every occupied slot
   still contributes separately. Weighted channel aggregation could reduce pair
   calculations further. Weights must preserve the intended power/PSD accounting;
   simply dropping duplicate slots changes the current result.
3. **Cache frequency-dependent pair kernels.** The Raman term
   `(a + abar - f * total_power * Cr)^2` is quadratic in total power. On a fixed
   frequency/bandwidth grid, static pair coefficients could turn repeated
   transcendental calculations into weighted reductions/matrix products. Variable
   channel centres, bandwidths and modulation corrections require separate handling.
4. **Extend reuse across related candidates and steps.** Feasible-candidate
   compaction and identical-footprint reuse are now implemented. More reuse must
   respect power changes, departures, modulation-dependent physics and stale caches.
   Differentiable execution retains static traversal.
5. **Validate a supported custom-kernel backend.** The ordinary-JAX tiled reduction
   is implemented here; the Pallas POC is isolated in a separate worktree. Backend
   support, register use, differentiation and generic/corrected modes need their
   own validation before production adoption.
6. **Rework RSA's registry-based capacity accumulation if it becomes dominant.**
   It still evaluates registry rows across all slots and performs path ownership
   reductions. The present 420-slot measurement does not justify claiming a gain.

Exact XPM remains quadratic in the number of interacting channels for the current
direct formulation. The original masking adds candidate and network-link factors;
this change removes much redundant link/path work but does not eliminate channel
pair interactions. The limited scaling from one to 16 environments shows that
arithmetic and device data movement become important after allocation pressure is
reduced.

My assessment is that there is **no demonstrated need for a C rewrite**. JAX's
[JIT compilation](https://docs.jax.dev/en/latest/jit-compilation.html) already removes
Python interpretation from these compiled loops. Algorithmic reuse, channel
representation, fusion and compiler behavior are the immediate opportunities.
Custom native code would need to improve those same kernels and data flows; changing
the implementation language alone would not remove quadratic physical interactions
or improve the model's validity.

## Reproduction and artifacts

The benchmark is `experimental/benchmarks/gn_scaling.py`. On a GPU host:

```bash
uv self update
uv sync --locked --group gpu
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --locked python -m experimental.benchmarks.gn_scaling \
  --slots 420 --k 5 --iterations 15 --kernels link lightpath mask rollout \
  --output /tmp/gn-420.json --snapshot /tmp/gn-420.npz

CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --locked python -m experimental.benchmarks.gn_scaling \
  --slots 1376 --k 5 --warmup-steps 128 --iterations 10 \
  --kernels link lightpath mask rollout --output /tmp/gn-wide.json

uv run python -m experimental.benchmarks.gn_accuracy_audit --output /tmp/gn-audit.json
```

Other options include `--mode corrected`, `--mode nonuniform`, `--envs 16`, and
`--env-type rsa_gn_model --kernels link lightpath throughput rollout`.
`--occupancy 0.9` raises the synthetic target; the achieved occupancy is reported
and can be substantially lower because placement attempts are bounded. Use
`--reference /tmp/original.npz` for numerical/outcome assertions. Baseline and
optimized runs should use the same script, precision setting and runtime when
isolating code changes. To reproduce the original source, export commit `2256456`,
copy the benchmark script into it, and execute from that directory using the chosen
environment's absolute Python path.

Raw result JSON files are in `docs/benchmarks/gn_20261024`, including the
[original 420-slot result](benchmarks/gn_20261024/baseline-420.json) and the
[same-runtime optimized result](benchmarks/gn_20261024/batch8-420.json).
`baseline-*` uses the original source; `optimized-*` uses the restructured source;
`batch8-*`, `batch32-*`, `batch40-*` and `serial-candidates-*` record batching
experiments. Some batching experiments precede the final adaptive default; their
implementation labels explain this. The partial `baseline-latest-420.json` is marked
as timed out. Numerical comparison details are in
[validation.json](benchmarks/gn_20261024/validation.json).

The isolated department copies, Python environment, logs and binary snapshots were
left at `/tmp/xlron-gn-perf-20261024` on Turin for inspection. Benchmark processes
have exited. The existing checkout at `~/git/XLRON` was not changed.
