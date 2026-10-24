# GN simulation performance investigation — 24 October 2026

The implemented changes reduce repeated GN calculations and bound candidate-state
memory while retaining the existing admission checks. On directed NSFNET with 420
slots, code changes alone made a 16-step RMSA rollout **4.44× faster** and reduced
action-mask temporary buffers **24.48×**. A 1,376-slot run with real admitted traffic
also completed with full SNR checks.

These are performance and implementation-equivalence results. The independent
formula audit below identifies pre-existing discrepancies that need physical-model
validation before making stronger multiband accuracy claims.

## Repository and PR status

The local workspace initially contained only `.venv`. The checkout was restored
from `https://github.com/micdoh/XLRON.git`, and the performance branch was based on
remote `main`, commit `2256456` (`Bump version to 1.6.0`). A subsequent fetch confirmed
the same remote head. During final validation, PR #61 landed on main as `dce10f9`;
the performance branch was rebased onto that commit. Performance changes are on
the local `perf/gn-multiband` branch and have not been pushed.

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

### Bounded candidate batches

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
2. **Compact active channels with explicit weights.** Work currently scales with
   frequency slots even when many are empty or share a centre frequency. Static
   capacity buckets and weighted aggregation could reduce pair calculations while
   retaining JAX-compatible shapes. Weights must preserve the intended power/PSD
   accounting; simply dropping duplicate slots changes the current result.
3. **Cache frequency-dependent pair kernels.** The Raman term
   `(a + abar - f * total_power * Cr)^2` is quadratic in total power. On a fixed
   frequency/bandwidth grid, static pair coefficients could turn repeated
   transcendental calculations into weighted reductions/matrix products. Variable
   channel centres, bandwidths and modulation corrections require separate handling.
4. **Pack feasible candidates and reuse identical placements.** Formats can produce
   identical spectrum footprints; current batches still evaluate invalid lanes.
   Dynamic compaction and careful threshold handling could reduce work further.
   Differentiable execution and outer environment vmaps must remain supported.
5. **Tile and fuse the XPM reduction in JAX/Pallas.** Bound pairwise working storage
   and reduce launches without changing the physical model. Profile generated
   kernels and register use before introducing a custom kernel.
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
`--env-type rsa_gn_model --kernels link lightpath throughput rollout`. Use
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
