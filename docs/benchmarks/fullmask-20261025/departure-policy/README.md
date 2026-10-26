# Safe optical release after traffic completion

This is a design recommendation and a measured check-cost experiment, not an
implemented departure policy. The running margin/power trials remain frozen.

The timing measurements and this design are logged in
[W&B run jut5wtaf](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/jut5wtaf).

## Recommendation

Use distinct logical traffic completion and physical optical release events.
At the requested service end, finish the payload on schedule. Before turning
off its optical signal, simulate removal and check every surviving live
lightpath. Commit the removal only if their end-to-end SNR meets the chosen
threshold/reserve. Otherwise retain an idle modulated signal with the same
path, spectral shape, power and modulation statistics until removal is safe.
The GN calculation must still include this signal as an interferer and Raman
contributor. A pure unmodulated CW tone is not automatically equivalent.

| State | Payload live | Optical spectrum/power occupied | Requires payload SNR protection |
| --- | --- | --- | --- |
| Live service | Yes | Yes | Yes |
| Idle retained signal | No | Yes | No |
| Physically released | No | No | No |

The retained signal cannot silently free its slots or fibre power budget. Its
transmitter/ROADM resources also remain committed where those are modeled.
Existing modulation metadata remains necessary for physical interference
calculations, even though its own payload constraint no longer applies.

"Safe" means all surviving live paths stay above their required SNR, optionally
plus the retained 0.5 dB reserve. It does not mean zero change in SNR. Keeping
an expired signal's payload constraint would unnecessarily protect expired
signals from each other's removal and can create spurious dependencies.

If the initial state is feasible, this rule preserves feasibility inductively
under the current instantaneous steady-state model:

1. Every admitted live lightpath and every existing live lightpath is checked
   before a new optical signal is added.
2. Logical completion removes a payload constraint but changes no optical
   signal, so the other live paths' SNR is unchanged.
3. Physical release occurs only after the proposed new optical state passes
   all surviving live-path constraints.

This avoids the need to predict every future departure for controllable service
completions. It does not cover uncontrolled transmitter failures, physical
model errors or fast switching/EDFA transients absent from this simulator.
The captured failure was between two modeled steady-state loading conditions;
it was not a modeled short-lived amplifier transient.

## Event ordering and release attempts

`generate_request_rsa` currently advances to the next arrival and removes all
expired services as a batch. A final-state check can miss an outage in an
intermediate state between departures. Process each distinct service-completion
time in chronological order before the next arrival, with simultaneous
completions grouped. Do not create separate service events for every occupied
slot/hop; identify the complete lightpath consistently across its footprint.

After each logical completion, attempt safe physical releases. Retry pending
signals when the protected live set or optical loading changes: another
completion, successful admission, successful physical release or explicit
power/gain reconfiguration. Do not busy-wait or freeze the simulation clock.
Rejected arrivals without another state change need not repeat an identical
failed release test.

Release order can matter. A release that improves margins can enable another
pending release, so iterate over newly enabled candidates. If proposing a
group removal, checking only the final state is sufficient only under an
explicit atomic-switch assumption; sequential physical switches require each
intermediate optical state to be safe as well.

An exact first version can refresh the full network for a proposed release.
Later versions can refresh changed links and accumulate full end-to-end SNR for
the surviving paths that use them, retaining every interferer on those links.
A changed-link cache is valid only if it was kept current after all earlier
optical changes. With vmapped JAX conditionals, a logical no-event branch does
not by itself prove GPU work was skipped; profile the integrated event loop.

## Resource cost, learning and alternatives

Retained optical slots can raise request blocking and keep transmitters and
power committed after traffic completion. Continuing arrivals can inherit
dependence on retained Raman loading, so retention can persist or accumulate.
Track pending-signal count, hold-time distribution and oldest hold, retained
slot-time/power-time, active payload throughput and all SNR violations. Keep
these distinct from ordinary admission blocking. The RL objective must price
retained resources rather than awarding free benefit from idle optical power.

A timeout is a policy alarm or a trigger for reconfiguration, not permission
to force an unsafe switch-off while claiming zero outages. A lower modulation
threshold is not a free repair: maintaining bitrate may require additional
spectrum. A new-lightpath launch-power head also does not automatically retune
already active lightpaths after a departure.

Our recommended first comparison is margin-only versus safe optical release
with the same 0.5 dB reserve, traffic, action resolution and physical model.
Longer term, explicitly modeled power/gain/tilt reconfiguration may make more
retained signals safe to release. Dynamic amplifier reconfiguration has been
studied for GSNR management in changing C+L-band loading [1]; its assumptions
must be matched before applying it to this C+L+S model.

If retaining idle signals is not a permitted network capability, departure
checks still detect outages but cannot prevent them. A strict alternative is
admission-time checking of the known chronological departure sequence until
the candidate completes, protecting all surviving services at each event.
Given known holding times, this is a sequence of current-service departure
states, not every possible subset. With a valid departure-robust starting
state and the same check on every later admission, it can preserve the
invariant without optical keep-alives, but is substantially more work per
candidate. Checking only the next departure or only full/empty loading is not
a general guarantee. Unknown departure times require certified robust bounds
or an explicit reliability target instead.

## Measured physical-check cost

`check-cost.json` uses the recorded ARBR event from
`../heuristics-audit/arbr-departure-transition.json`: 1668 slots, 44 links,
one changed link (link 2), JAX 0.12.0.1, H100 NVL, float32, current Pallas
backends with 16 persistent row programs. Eleven synchronized repetitions,
after compilation and a warm call:

| Operation | Median |
| --- | ---: |
| Full-network SNR refresh and surviving-path check | 0.2535 ms |
| One-link refresh and surviving-path check | 0.2171 ms |
| Four-link refresh and surviving-path check | 0.2284 ms |
| Full admission mask on the healthy pre-departure state | 2.0546 ms |

Selected-link and full-network checks give exactly identical active-path SNR
and the same 12 failed link/slot predicates after removal. Those predicates
repeat one failed service across its six slots and two hops. The minimum
margin is −0.016310 dB. The healthy admission-mask reference has three valid
choices. Removing the unsafe physical release from the event keeps the
survivor at its pre-release SNR in the implemented model.

The supplied changed-link indices, valid pre-release cache and a single
captured state make this a microbenchmark. It does not measure event sorting,
live/idle bookkeeping, batching, release retries, cache maintenance, complete
rollout overhead or real amplifier transient behavior. The small selected-link
gain suggests starting with the simpler full check and profiling integration
before adding more kernels.

The measurement harness is `/tmp/xlron-departure-check-cost.py` on the local
host and Malmö, run from the frozen
`/tmp/xlron-gn-margin-power-v3-20261025` source with `PYTHONPATH=.` and
`uv run --no-sync python /tmp/xlron-departure-check-cost.py`. It borrows idle
GPU 0 compute with preallocation disabled and a 25% JAX memory budget; no
running trial sources were edited. The fixture uses the captured `before.npz`
and `after.npz` from the Turin heuristic replay and action 1045.

[1] Z. Huang et al., [Performance improvements by dynamic amplifier
reconfigurations for C+L-band optical networks in the presence of stimulated
Raman scattering](https://opg.optica.org/jocn/abstract.cfm?uri=jocn-15-6-344),
JOCN 15(6), 344–356 (2023), DOI 10.1364/JOCN.483414.
