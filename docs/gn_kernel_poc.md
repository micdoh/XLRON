# GN kernels: performance, accuracy and reproducible comparisons

This branch continues the ordinary-JAX optimizations in PR #62 (`97373df`), which
has now merged. Custom backends remain opt-in. The supported differentiable path
and the normal environment default remain ordinary JAX. All simulations below
retain float32 physics, modulation thresholds, new-path margin, existing-path
checks and per-fibre power checks.

Throughput units: historical "training steps/s" entries count **environment
transitions collected per second through the entire PPO loop**, including
learning. They do not count gradient or optimizer updates. New reports label
environment transitions and minibatch updates separately.

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
This run inherited the CLI's **load 250**, so it is a low-load full-band throughput
test, not a measurement at the target load 5,000.
Training compile takes 15.15 s; synchronized training execution takes 22.18 s.
Neither warmup nor compilation is included in the throughput figure.

The explicit **load-5,000** rerun (`v50`), otherwise the same 16-env/two-layer
configuration, reaches **1,359.7 training steps/s**: 32,768 steps in 24.10 s.
Training compile is 14.93 s and the 3,000-request warmup executes in 23.65 s.
Longer learning runs use 25,000 warmup requests and a separate optimizer/evaluation
configuration; this short benchmark does not establish their final throughput.

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

The first W&B launch also omitted the intended load flag and inherited load 250.
Those two runs were stopped, marked `invalid_configuration` and moved to a
separate invalid-load group. The campaign now passes `--load=5000` explicitly;
preflight constructs the actual heuristic, training and evaluation environments
and checks 200 arrivals/time-unit, 1,668 slots and coherent accumulation. Invalid
runs do not count toward the blocking comparison.

The corrected campaign records are in
`/tmp/xlron-gn-kernel-v2-20261024/runs/cls-20261024-v5/{lane2,lane3}` on Turin,
with W&B group `gn-cls-20261024-v5` in `XLRON-GN-CLS`. Each lane's `manifest.json`
records commands, source hashes, run URLs and eventual exit status. Lane 2 runs
the four heuristics followed by the SNR-only transformer; lane 3 runs the
occupancy+SNR transformer. Both lanes evaluate their selected checkpoint on
held-out traffic after training.

The v5 runs finished at approximately **1,410 training steps/s** for SNR-only
and **1,354 steps/s** for occupancy+SNR (execution only). Both completed
10,485,760 steps. KSP-MSCL and MSCL-KSP failed because their capacity-loss
calculation reshaped the GN sentinel path row as a real route. The fix excludes
that single appended row, including with one path or packed path bits, and
preserves ordinary RSA capacity losses.

The initial blocking comparison is inconclusive: some replicas have unusually
high SNR blocking. SNR-only model service blocking was 30.72% / 35.66% on seeds
4242 / 4243; occupancy+SNR was 30.67% / 36.62%. KSP-FF and FF-KSP measured
35.17% / 34.62% on seed 4242, with large between-replica dispersion. Their bitrate
blocking was 45.57% / 45.81%, versus model two-seed means of 46.09% / 46.42%.
This does not establish that PPO beats either heuristic. The SNR outlier cause
has not been established and is retained in the reported metrics.

The follow-up `experimental.benchmarks.gn_cls_tuning_campaign` covers **all 23
implemented CLI heuristic choices**, each on seeds 4242/4243, plus both MSCL
heuristics with all stored interfering routes: **50 heuristic evaluations**.
The unique/non-relative most-used choices are included even where they coincide
for this environment. The unimplemented `ksp_ff_multiband` stub is excluded.
All runs retain the same full C+L+S, 1,668-slot, coherent, load-5,000 workload,
25,000-request warmup and full physical admission checks.

Each observation mode then continues its existing best checkpoint through
three 8,388,608-step trials. The first uses LR 1e-4, entropy 0.005 and 128-step
rollouts. The second uses on-policy masking and clip 0.1. The third uses
256-step rollouts, gamma 0.9995, lambda 0.995, LR 5e-5, entropy 0.002 and reward
centering. The third trial starts from the best initial/first/second checkpoint
on selection seed 4244; comparison seeds 4242/4243 are evaluated separately.
Optimizer state is restarted; actor and critic weights are retained. Source
checkpoints are preserved and new checkpoints have separate paths.

There are **76 jobs** across two queued workers: 50 heuristic evaluations,
six training trials and 20 model evaluations. Exact plans and statuses are at
`/tmp/xlron-gn-kernel-v2-20261024/runs/cls-20261025-v6/{lane2,lane3}/manifest.json`
on Turin. Workers wait for GPUs 2/3 to have at most 512 MiB allocated, take an
advisory lease against our other campaign workers, and check allocation before
every child. Both GPUs were occupied by another user when queued; no full-sized
v6 run had started at that point. Online runs use W&B group
`gn-cls-20261025-all-heuristics-tuning` in
[XLRON-GN-CLS](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS).

The opt-in host recorder saves each replica's cumulative post-warmup blocking,
a final W&B table and the first physical-state snapshot with SNR blocking over
10%. This makes the outliers inspectable without filtering them or changing
admission behavior. All 23 GN heuristic smoke checks, packed/unpacked sentinel
regressions, existing MSCL capacity-loss checks and an actual adjusted-HP PPO
checkpoint continuation pass on CPU. Native v6 CUDA validation awaits a free GPU.

The v6 campaign subsequently completed all 50 heuristic evaluations successfully.
The strongest PPO continuation uses occupancy+SNR: its best periodic service
blocking is 27.23%, and its two-seed held-out average is 31.08%. These are different
evaluation traffic sets. Its last logged valid mass averaged 2.14% (SNR-only:
0.13%), and the probability-space valid-mass loss could lose its gradient below
the `1e-8` numerical guard. A two-action example with a 100-logit disadvantage
produces zero corrective gradient with that implementation, instead of -1 on
the valid logit.

The PPO barrier now uses logsumexp differences, keeping float32 recovery
gradients for very small valid mass. Empty masks have zero barrier/gradient;
their previous constant penalty is removed without renormalizing the batch.
Normal-mass values and gradients match the probability definition. New W&B
diagnostics report current log valid mass and the fraction below the old floor
among nonempty masks. This change affects policy optimization only; the full
GN admission calculation is retained.

After the user requested the transformer paper's stabilization stack, unstarted
on-policy long-horizon trials were superseded. The active SNR-only control
reached its full 8,388,608-step budget, but controller replacement interrupted
final logging. Its 128 per-environment increment records and checkpoint are
preserved; W&B retains its crashed status with explicitly labeled recovery
counters. It is excluded from completed-run evaluation summaries. The
replacement `gn_cls_stabilized_campaign` starts all five trials from
the same selected occupancy+SNR checkpoint, with 16,777,216 additional steps per
trial and separate comparison seeds 4242/4243 (4244 for selection). Both GPUs
prioritize the observation mode that has improved most.

The [paper's Table 2](https://arxiv.org/html/2605.02075v2) gives clip 0.04, gamma
0.996, lambda 0.99, critic LR 5e-5, rollout 64 and valid-mass coefficients
0.001/0.002. All replacement trials explicitly use off-policy IAM, non-recentered
clipping, damping, two-action gating and float32. The fine-tuning lane keeps the
successful LR 1e-4, entropy 0.005 and update regime while testing VML 0.001/0.002.
A third arm lowers the damping target from 0.05 to 0.005: the GN workload has
8,340 actions but only about 4.8 valid actions per state on average in the final
logged batch, so the paper's target cannot be assumed to transfer unchanged.
The other lane uses one epoch/minibatch per 64-step rollout and compares actor
LR 1e-4/5e-4 with VML 0.002. These are paper-informed continuations, not literal
reproductions of its distance-only RMSA workloads. VML is held constant to test
its strength; the paper's large-topology runs anneal it linearly to half.

The new campaign retains directed NSFNET, five paths, 1,668 slots, full C+L+S,
load 5,000 and 25,000-request warmup. Its records are under
`/tmp/xlron-gn-kernel-v2-20261024/runs/cls-20261025-v7/` on Turin, with W&B group
`gn-cls-20261025-paper-stabilized`. Numerical-loss tests, the PPO regression
suite, actual GN checkpoint save/reload and physical-config preflight pass.

### Unrestricted slots and compact spectrum observations

The observation comparison adds `lifetime_snr`: one slot channel encodes
free/occupied/band-gap status and a smooth remaining lifetime; the other carries
smoothly scaled SNR, retaining finite values outside the former 0--30 dB clip.
Its full-band input is `44 x 3352`, compared with `44 x 3351` for occupancy+SNR.
Dense encoders at widths 128/256 are compared with a local spectrum encoder at
width 128. The local encoder retains per-slot features for scoring actions and
uses band-aware filters; ordered feature pooling does not aggregate actions.

The existing GN mask restricted starts to first/last-fit positions even with
slot aggregation one. New explicit modes offer every spectrum-feasible raw
start. `all_slots` evaluates physical feasibility before sampling;
`selected_slot` resolves modulation and checks full physical feasibility for the
chosen start, counting failed choices as blocking. Accepted tentative states
are reused instead of recomputing their GN physics. All comparison jobs pin
`aggregate_slots=1`, yielding 8,340 path/start-slot actions.

A synchronized initial mask/step microbenchmark uses the first four replicas of
the same saved full C+L+S, load-5,000 state (65.55% powered-slot occupancy), A100
80 GB PCIe, JAX 0.12.0.1 and the existing Pallas backends. It measures one step
with three warm executions and two timed executions, excluding compilation:

| Mode | Mask ms | Mask + step ms | Env steps/s |
|---|---:|---:|---:|
| First/last-fit control | 2.861 | 3.764 | 1,063 |
| Every slot, full physical mask | 109.808 | 111.001 | 36 |
| Every slot, selected-slot physical checks | 0.293 | 3.051 | 1,311 |

These are short mechanism benchmarks with different masks/selected admissions,
not equivalent-output speedups or steady-state PPO throughput. This workload
uses the benchmark's default departure observation mode and excludes the extra
post-departure SNR-cache refresh used by SNR-aware training. Both full and
selected-slot modes retain final admission checks. Full masking is about 36x
slower for this measured step. After the user required physical feasibility
before sampling, selected-slot trials were canceled before training. New
representation trials use full physical masks. Full masking streams candidates
and avoids a quadratic candidate-grouping table.

The optimized full-mask baseline uses the **actual admission GN function**;
the initial table above used the legacy fused mask GN function. This changes
neither function's equations and does not fix the deferred physical-model
discrepancies. Matched-output speedups below compare the same admission physics.

A descending-SE pass stops checking formats at a path/start once the selected
highest-efficiency feasible format is known. This alone gave little benefit.
Larger candidate batches improved utilization. A new exact-output-row screening
pass computes only the new lightpath endpoints, still summing every interferer
and retaining post-insertion Raman/ASE dependence. Candidates within 0.01 dB of
the threshold are retained conservatively; every surviving candidate receives
the original complete new/old-lightpath and power checks.

The four-state A100 follow-up, with identical saved states and masks/formats/
one-step rewards/occupancy exactly equal, measured:

| Full physical mask | Mask ms | Mask + step ms |
|---|---:|---:|
| Exhaustive reference, batch 8 | 113.63 | 114.51 |
| Descending format pruning, batch 8 | 109.76 | 110.70 |
| Descending format pruning, batch 64 | 77.14 | 78.46 |
| Endpoint screening + format pruning, batch 64 | 39.65 | 40.74 |

This is a **2.87x mask speedup** and **2.81x one-step rollout speedup**, including
all physically feasible raw starts on the five configured paths. Results use
three warm and two timed executions; larger matched rollouts and CUDA profiling
are recorded separately. Source hashes and reports are in the `masking-v3`
directory of the isolated observation stage.

The subsequent 16-state comparison uses independent traffic keys, eight rollout
steps, three warm executions and ten synchronized timed samples. Recorded
masks, selected formats, all rewards and final occupancy are exactly equal:

| Full physical mask scheduling | Mask ms | Eight-step rollout ms | Env steps/s |
|---|---:|---:|---:|
| Exhaustive reference, candidate batch 8 | 419.73 | 4661.61 | 27.46 |
| Screened, candidate batch 64, all 16 envs | 196.10 | 2158.98 | 59.29 |
| Screened, candidate batch 64, env chunks 4 | 126.18 | 1242.59 | 103.01 |
| Screened, candidate batch 64, env chunks 1 | 93.26 | 842.62 | 151.91 |

This is a **4.50x mask speedup and 5.53x rollout speedup** in the matched
independent-stream workload. Environment-only timings exclude neural inference,
PPO updates and SNR-observation refresh. The first chunk experiment streams the
complete benchmark environment operation. Production integration uses
`gn_mask_env_batch_size` to stream physical masking only, leaving policy/optimizer
batches intact. The integrated follow-up measured **420.57 to 90.13 ms** per
full mask and **4671.30 to 789.73 ms** per independent eight-step rollout:
**27.40 to 162.08 env steps/s (5.92x)**. Recorded outputs remain exactly equal.
Full-mask temporary storage falls from 363.70 to 136.47 MiB. The endpoint path
screen now retains just its two output columns through ROADM/transceiver
accumulation, avoiding a full-spectrum candidate scatter.

Reviewed reports, source hashes and CUDA statistics are under
[`docs/benchmarks/fullmask-20261025`](benchmarks/fullmask-20261025/unpruned16.json).
The [W&B comparison](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/ifqrifmb)
contains the timing table and raw evidence artifact. Nsight 2025 with CUDA graph
node tracing attributes roughly 14% of measured kernel time to SNR evidence,
9% to generic XPM, with substantial remaining reduction/scatter/packing work.
CUDA profiler timings are excluded from the synchronized performance table.
The integration reports and source hashes are in the `integrated` subdirectory.
The next workspace rewrite constructs endpoint-screening inputs only for the
changed path's link rows, retaining all interfering channels on those links.
This reduces the mask from **90.13 to 82.71 ms** and the independent eight-step
rollout from **789.73 to 731.16 ms**, reaching **175.06 environment transitions/s**.
Masks, selected formats, rewards and final occupancy still match the exhaustive
reference exactly. Reports and source hashes are in the `workspace` subdirectory.

The next prototype merges duplicate existing-path SNR predicates by path ID and
slot, keeping the strongest threshold. It computes the candidate's path SNR
across every traversed link as before; it neither reuses pre-insertion SNR nor
omits shared-link degradation. Native tests include a case where the retained
predicate is on the first link but degradation on another shared link must
reject the insertion. NaN thresholds and invalid path IDs retain their original
behavior. Packing is an experimental benchmark option and is not enabled in
the frozen training or heuristic campaigns.

A fresh paired A100 run, including packing overhead, measured:

| Full physical mask variant | Mask ms | Eight-step rollout ms | Environment transitions/s |
|---|---:|---:|---:|
| Exhaustive reference | 421.48 | 4671.61 | 27.40 |
| Path-only workspace, mask chunks 1 | 84.02 | 741.87 | 172.54 |
| Same plus duplicate-predicate packing | 80.72 | 717.39 | 178.42 |

The incremental rollout improvement is **3.41%**; the full sequence is **6.51x**
the exhaustive reference. Recorded masks, resolved modulation formats, rewards
and final occupancy match exactly. Compilation takes 4.37 seconds for the
packed mask and 6.43 seconds for its eight-step rollout. All 20 native tests
pass. These are matched environment-only measurements, not a PPO speedup claim.
The user permits idle shared GPU capacity under a 25% JAX memory budget;
GPU 3 reported no other VRAM allocation when this paired benchmark started,
but these are not guaranteed exclusive-GPU measurements.

The packed-mask trace attributes 13.3% of CUDA kernel time to path SNR evidence,
11.0% to the main generic XPM kernel and 0.3% to predicate packing. Reductions,
spectrum scans and scatters remain substantial targets. Timed-region profiling
uses Nsight 2025 with CUDA graph node tracing; profiler-instrumented timings
are excluded above. Reports, source hashes, native-test output and CUDA
statistics are in `predicates/`, with the
[W&B comparison](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/su5mbb8j).

Full PPO uses a different workload from the heuristic-action environment benchmark:
16 environments, 64 rollout steps, lifetime+SNR dense-128 transformer, a
modulation-aware actor head, strict float32 matrix multiplication, one epoch and
eight minibatches. Its synchronized throughput improves from **149.36 to 159.14
environment transitions/s** after the path workspace rewrite. These are rates
of collected transitions through rollout plus learning, not optimizer-update
rates. One rollout collects 1,024 transitions and performs eight minibatch
updates (each updating actor and critic). The latest 6.435-second rollout/update
therefore executes **1.24 minibatch updates/s**. The 162.08 and 149.36 results
must not be used to infer training overhead: their actions, rollout lengths and
observation work differ. Raw PPO reports are in `ppo` and `ppo-workspace`.

A matched decomposition now uses the same initial policy, physical states,
random keys, 16 environments and 64-step rollout for both phases. Two warm
executions precede five synchronized samples per phase:

| Phase | Seconds per 1,024 transitions | Environment transitions/s | Minibatch updates/s |
|---|---:|---:|---:|
| Policy rollout plus advantage calculation | 6.352 | 161.20 | 0 |
| Same rollout plus full PPO learning | 6.431 | 159.22 | 1.244 |

Final environment states match exactly. The median difference is 78.9 ms,
or 1.24% additional wall time, for this specific one-epoch/eight-minibatch
configuration. This difference includes retaining learning buffers, loss and
gradient calculations, optimizer updates and diagnostics; it is not an isolated
gradient-kernel timing. The rollout-only phase already includes advantages.
The benchmark repeats fixed initial inputs and excludes compilation, warmup
and W&B; it does not measure long-run learning quality. Raw evidence is in
`ppo-matched/ppo.json` and the
[matched W&B benchmark](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/hwdp5v4l).

CPU execution remains supported with the ordinary JAX backend. A bounded check
on one loaded full-C+L+S state, with an eight-core Xeon Gold 6242 CPU affinity,
compiled in 2.075 seconds and computed one full physical mask in **876.83 ms**.
This is a mask timing, not CPU environment or training throughput; the report
is in `cpu/cpu.json`.

Native mask/model validation passes 15 tests. Current CPU GN/model/optimizer
checks pass 107 tests (6 skips); PPO/behaviour-probability checks pass 23 tests,
and full-mask/chunked dense/local PPO checkpoint smoke checks pass. A subsequent
full-shape CUDA PPO preflight passed on a shared Malmö H100 NVL: 32 environments,
128-step rollouts, four epochs and eight minibatches. Its 8,192 transitions took
37.685 seconds (217.38 transitions/s, including 64 minibatch updates), with
31.762 seconds of startup. This is a short preflight, not a learning result,
and differs from the A100 throughput benchmark in hardware and PPO settings.

An optional modulation-aware residual actor head adds 57 parameters and four
features per candidate (validity, efficiency, required width and threshold),
without extending link tokens. Both dense and local encoders can use it.
Masking precedes the forward pass and PPO stores the exact acting state;
a regression checks that stored features reproduce the behaviour probability.
Modulation is still selected automatically by highest feasible efficiency;
the policy now sees its resource cost. Explicit lower-efficiency format choice
would require a separate joint/conditional action experiment, for which format
pruning after the first success would not apply.

Initial 16-replica model microbenchmarks use identical saved states and a
negative log-probability plus value loss, rather than the complete PPO loss:

| Encoder | Parameters | Forward ms | Forward/backward ms |
|---|---:|---:|---:|
| Occupancy+SNR, dense 128 | 1,650,949 | 0.896 | 2.975 |
| Lifetime+SNR, dense 128 | 1,651,077 | 1.210 | 3.104 |
| Lifetime+SNR, local 128 (initial) | 632,938 | 5.247 | 11.606 |
| Lifetime+SNR, dense 256 | 4,118,661 | 1.239 | 3.387 |

The local encoder is smaller but its initial implementation is slower. Its five
offset contractions are now fused into one masked-window contraction, with
ordinary JAX reverse-mode AD. Values/gradients match an independent offset-sum
reference within float32 tolerances. The repeat GPU benchmark measured local
forward/backward at 10.023 ms versus 11.606 ms initially (about 14% lower), while
forward increased from 5.247 to 5.611 ms. Dense-128 forward/backward remained
about 3.158 ms, so fewer parameters do not establish a speed advantage. These
encoder controls precede the optional 57-parameter candidate head. An attempted capture with the default
Nsight 2022 tool produced no usable CUDA kernel trace and exited abnormally;
those captures are excluded from performance evidence.

New comparisons initialize fresh weights and use both seeds 42/43, identical
PPO settings, full C+L+S/load 5,000 and a 25,000-request warmup. They compare all
four rows above, with 16,777,216 training steps per run plus separate model
selection/comparison seeds. Eight minibatches keep full-resolution local feature
activations bounded (256 transitions per minibatch). A separate-optimizer bug
that assigned the legacy critic transformer encoder to the actor optimizer is
fixed for all new comparisons. Existing saved policy weights remain compatible;
these trials restart their optimizer.

Native validation and short full-shape PPO preflight gate long runs. The
representation trials above remain staged; the current priority is an
8,388,608-transition continuation of the best existing occupancy+SNR dense-128
checkpoint. A zero-initialized 57-parameter modulation residual preserves its
initial policy and its existing path-sum pooling (the legacy configuration
string `transformer_actor_pooling=mean` also takes the sum branch). The
full-shape preflight passed and this continuation is
**running**, with a fresh optimizer, actor LR 1e-4, critic LR 5e-5, entropy 0.005
annealed linearly, clip 0.04, gamma 0.996, GAE lambda 0.99, off-policy IAM,
damping/gating and valid-mass coefficient 0.002. It uses 32 environments,
128-step rollouts, four update epochs, eight minibatches, candidate batches 64
and mask environment chunks 1. Every raw slot on each configured path remains
selectable when physically feasible.

Empty-state full-physics warmup was a startup bottleneck: an earlier local
encoder preflight took 345.8 seconds for 32 warmup requests and then hit its
900-second training timeout. The new experiment-only bootstrap restores 16
independently admitted physical states from a workload-validated NPZ snapshot,
cycles them to initialize 32 replicas and retains independent future traffic
keys. It refreshes physical SNR and clears measurement counters. This bypasses
the slow empty-state startup; it does not claim the bootstrap is steady state.
Final model and heuristic comparisons each add 25,000 warmup requests per
replica, then measure 524,288 aggregate transitions on separate seeds.

Frozen continuation sources are under
`/tmp/xlron-gn-malmo-fullmask-20261025/training` on Malmö GPU 2. JAX preallocation
is disabled with a 25% memory pool (about 23 GiB); the observed worker allocation
was about 17.9 GiB. The
[live primary run](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/b4e50003)
and [passed preflight](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/5b65572c)
use group `gn-cls-20261025-full-physical-best-hp`. Reviewed command plans,
source hashes and preflight timings are in `primary/`.

All 23 implemented heuristics plus two multi-route MSCL variants are planned
on seeds 4242/4243 (50 jobs). Their fresh frozen source is
`/tmp/xlron-gn-fullmask-heurs-20261025` on Turin GPU 3; ARBR-FF is running first,
followed by MSCL-KSP and KSP-FF. Historical FF/LF heuristic blocking is not a
matched baseline for unrestricted actions. Previous partial continuations and
an interrupted heuristic startup are labeled retired/excluded with their
checkpoints and logs preserved.

All new training/evaluation metrics are namespaced, with hidden step axes.
Completed historical evaluations also received `eval/` aliases: 65 runs and
33,280 history rows were copied without overwriting their original histories.
Two initial unsupported-MSCL runs contained no scalar result and were left
unchanged. Migration reports are in `logging/`; old default Charts panels are
not deleted.

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
