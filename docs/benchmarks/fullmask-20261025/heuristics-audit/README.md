# Heuristic ranking audit

The API extracts in `run-results.json` distinguish older FF/LF GN evaluations
from the still-running full-mask ARBR evaluation. None of the high-SNR-blocking
replicas was excluded. These measurements do not establish a corrected,
unrestricted-mask heuristic ranking.

## Non-GN evidence

The saved USA100 directed RMSA comparison uses 320 slots, k=70, load 620,
40 million transitions, seed 42 and 25,000-request episodes. Final logged
episode service blocking is 0.06534% for MSCL-KSP (`3e3ryzbn`) versus 0.40067%
for FF-KSP (`x3mv1yaj`), a 6.13-fold reduction. Physical/traffic settings match;
FF-KSP used four devices with 12 replicas per device, versus MSCL's one device
with 12 replicas. This confirms this comparison, not universal dominance over
all heuristics, topologies and loads.

## Historical GN evidence

Two-seed means (4242/4243), service blocking by cause, in percent:

| Heuristic | Spectrum | SNR | Total |
| --- | ---: | ---: | ---: |
| ARBR-FF | 26.017 | 0.137 | 26.154 |
| MSCL-KSP, one interfering route per pair | 24.701 | 4.797 | 29.498 |
| MSCL-KSP, all stored routes | 25.376 | 6.837 | 32.212 |

MSCL's spectrum-blocking component improves, while the SNR component reverses
the total ranking. The per-replica histories in `replica-history.json` locate
the dominant effect more precisely: three of 32 MSCL-KSP n=1 replicas account
for 96.80% of its SNR blocking. Its final mean service blocking is 29.497%,
but its median is 25.130%; the three outliers end at 54.684%, 75.098% and
89.047% blocking. ARBR's mean is 26.154%, median 26.181%, and range
25.452–26.871%. No replica is excluded from any mean. A spectrum-only objective
is a limitation, but does not by itself explain these prolonged outages.

MSCL counts the future contiguous placements lost on overlapping routes, with
uniform weights for slot widths up to the maximum demand. It does not count
future physically feasible placements, frequency-dependent noise, or modulation
downgrades caused by its current choice. ARBR's first-fit searches C,L,S band
order; MSCL's minimum-loss ties use the lowest raw slot index, which starts in
L-band for this layout. This can produce different band occupancy even with the
same feasibility mask. The `n=1` and `n=all k` scores also count different route
sets, so neither is guaranteed to dominate.

## Corrected footprint score

ARBR and MSCL previously used the nominal reach-based path efficiency to price
the allocation. RMSA-GN actually resolves modulation per path and start, so its
footprint can differ at neighboring slots. They now obtain fresh mask/formats
together and score the actual allocated width. ARBR prices its band-preferred
first-fit start; MSCL uses a width for every candidate and includes the lowest
physical modulation in its future demand cap. Ordinary RSA/RMSA behavior and
the physical feasibility mask are unchanged.

The regression tests are in `xlron/heuristics/heuristics_test.py`: stale carried
formats, band order, a concrete ARBR path-ranking reversal and independent
before/after spectrum recounts for variable-width GN placements. Existing
ARBR/MSCL and other heuristic tests also passed in split runs. A corrected
long-run blocking improvement has not yet been measured.

## Full-mask ARBR anomaly

The first 512 measured requests of replica 5 in run `625744af` accepted none.
Its saved cached SNR passes all existing-channel predicates, with a minimum
margin of +0.0040 dB. Refreshing from its current channel arrays gives 875 failed
link/slot predicates and a minimum margin of -1.1011 dB. Both ordinary JAX and
the generic Pallas backend reproduce that result. These predicates repeat
services across links and slots; they are not 875 distinct failed services.

All 16 common bootstrap states pass after a fresh physical solve, including
replica 5. Their smallest margin is only about 0.000068 dB. The anomaly is
therefore not an initially invalid saved bootstrap or solely a custom-kernel
discrepancy.

An optical counterfactual removes a single 16-slot channel on one link. Under
the current ISRS model, 263 surviving slots lose SNR, by up to 0.0710 dB. With
Raman coupling set to zero for this diagnostic only, none loses SNR. Departures
can therefore reduce other channels' SNR under the implemented model; reduced
NLI does not imply every channel improves. This isolates a per-link response,
not a network-level experiment or independent validation of the GN equations.

The full mask refreshes SNR and retains the existing global old-channel checks.
An already failing old channel can therefore reject unrelated new traffic until
the condition clears. This helps explain why a larger candidate set does not
guarantee lower long-run blocking. The width fix does not remove these checks
or resolve this separate physical-state behavior. Existing checks must not be
weakened merely to improve the reported score. See `physical-state-audit.json`
for the saved-state measurements and expiry counterfactuals.

## Captured departure transition

A controlled full-mask replay starts from valid bootstrap replica 5 with the
same future traffic key, then checks fresh, ordinary-JAX path SNR after every
step. The initial source is frozen at 4cc76cc with the endpoint compiler fix;
it deliberately retains the original heuristic scoring. No mask-valid action
was subsequently rejected by the admission checks in these replays. This
isolates the outage mechanism, rather than measuring the scoring fix's benefit.

At ARBR replay step 19,338, action 1,045 is admitted. Every active lightpath
passes immediately after insertion. A different lightpath then expires during
generation of the next arrival, and a survivor falls below its modulation
minimum. `arbr-departure-transition.json` records all three stages:

| Stage | Minimum margin over all active lightpaths | SNR of the affected survivor |
| --- | ---: | ---: |
| Before insertion | +0.010203 dB | 8.824699 dB |
| After admission, before expiry | +0.010203 dB | 8.827790 dB |
| After expiry | −0.016310 dB | 8.751421 dB |

The affected survivor is path 35 on links 2 and 24, with centre offset
−7,934.75 GHz and required SNR 8.767731 dB. Path 30, on link 2 at
+9,534.75 GHz, expires. The insertion slightly improves this survivor; the
departure causes its failure. Thus the captured event is not a new admission
that the existing-channel mask should have rejected under its current rules.

`departure-noise-decomposition.json` separates the terms on link 2:

| Quantity | Before expiry | After expiry |
| --- | ---: | ---: |
| Amplifier ASE | 9.698138 µW | 9.933185 µW |
| Nonlinear noise | 0.254198 µW | 0.251518 µW |
| Compensating amplifier gain | 65.542994 | 67.131506 |

Under the implemented ISRS-dependent gain model, removing the higher-frequency
service reduces Raman transfer into the lower-frequency survivor. Required
amplifier gain and ASE rise; the ASE increase exceeds the NLI reduction. An
independent NumPy float64 evaluation of the amplifier/ASE expression agrees
with these values. With Raman coupling disabled only for this diagnostic,
the departure slightly improves SNR instead. This verifies the code path and
arithmetic; it is not independent validation of the physical model or its
per-slot power representation.

The outage persists: at replay step 25,000, ARBR has rejected 5,289 consecutive
arrivals. `disjoint-admission-probe.json` shows a candidate on link 0 with
sufficient new-channel SNR, no collision and no power violation, but the
existing failed path on links 2/24 triggers the global SNR check. Its SNR is
unchanged by that disjoint insertion. The checks have not been relaxed.
Matched MSCL replay reaches 25,000 steps with ten shorter outage episodes and
no 100-rejection lock. These single-trajectory diagnostics are not a new
heuristic ranking.

Historical MSCL anomaly snapshots also fail a fresh ordinary-JAX solve despite
passing cached SNR (`historical-mscl-state-health.json`). The older ARBR runs
are internally consistent for their restricted-mask setting, but absence of a
catastrophic replica in those 32 samples does not establish immunity. New
full-mask ARBR results must therefore remain provisional.

An optional `existing_snr_margin` now protects a target reserve during later
admissions in both the mask and actual step. The default remains zero. When a
departure has consumed some reserve, admissions preserve the remaining
headroom without making an unchanged, still-physical disjoint path fail just
because it is below the target buffer. The original modulation minimum still
applies globally. This reduces one source of fragility but is not a guarantee
against all future departures. It has not yet been shown to improve long-run
blocking; margin comparisons must start from empty states and retain all
replicas.

Validation: the existing mask/GN regression selection passed 111 tests on CPU
(seven CUDA/unsupported cases skipped). The reserve edge-case and diagnostics
selection passed seven tests on CUDA, including predicate packing. A separate
1668-slot gate built a state from empty with one accepted existing lightpath:
ordinary JAX and the fused-endpoint/persistent-Pallas implementation both expose
8,029 valid starts, with identical modulation masks; all 32 sampled valid actions
are actually admitted. See `retained-reserve-fullcls-gate.json`. This sparse
correctness check does not measure steady-state margin performance.

The [W&B departure audit](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/5axcr2ae)
contains the transition SNR trace, per-replica summary and plot, and the evidence
artifact. The analysis uses `diagnostics/`, `replicas/` and `eval/` namespaces.
