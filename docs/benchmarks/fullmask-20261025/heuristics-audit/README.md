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

Thus MSCL's spectrum-blocking component improves, while the SNR component
reverses the total ranking. This is consistent with a spectrum-only objective
missing the cost of lost SNR headroom. It does not establish causality: the
high-SNR-blocking replicas require the separate state audit below.

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
