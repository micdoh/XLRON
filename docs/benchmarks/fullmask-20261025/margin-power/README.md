# Retained SNR margin and launch-power comparison

The previous runs required 0.5 dB on a newly admitted lightpath but preserved
only its modulation minimum during later admissions. These trials explicitly
retain a target 0.5 dB reserve. They compare the cost of more conservative
admission with departure-driven outages; they do not assume that 0.5 dB is
optimal or sufficient for every future departure pattern.

| Lane | New-lightpath margin | Retained reserve | Fixed launch power |
| --- | ---: | ---: | ---: |
| Control | 0.5 dB | 0 dB | −10 dBm |
| Reserve | 0.5 dB | 0.5 dB | −10 dBm |
| Admission | 1 dB | 0.5 dB | −10 dBm |
| Power bracket | 0.5 dB | 0.5 dB | −11 and −9 dBm |

Each setting compares ARBR-FF and MSCL-KSP (one interfering route per pair),
with seeds 4242/4243 and four replicas per seed. Every replica starts empty
under its own trial's parameters and discards 25,000 warmup requests before
measuring 32,768 requests. All replicas stay in the means. The physical setup
is directed NSFNET, k=5, load 5000, 1668 slots, full C+L+S, float32, full physical
masking and no slot aggregation. The winning endpoint/persistent kernels are
enabled. Old bootstraps and running sources are not changed.

The reserve lane then queues a 4,194,304-transition transformer continuation
from the saved lower-LR/entropy checkpoint, with a fresh optimizer and fresh
network state. It uses 16 environments, rollout 128, four epochs, eight
minibatches, actor LR 1e-4, critic LR 5e-5, off-policy IAM, valid-mass loss
0.002, occupancy/SNR observations and the modulation-aware candidate head.
This run keeps fixed −10 dBm power; it is not learned-power training.

`gn_margin_campaign.py` writes exact job plans and delegates to the existing
bounded-memory worker, which records source/checkpoint fingerprints, W&B IDs,
status and commands. The experiments use at most a 25% JAX memory budget per
GPU and require idle capacity with additional free-memory headroom. The
`runs.json` snapshot identifies the hosts, lanes and current run links.

Each logging increment records per-replica blocking, the mean/median/range,
and a fresh physical SNR audit. Audit counts are link/slot predicates, which
repeat a lightpath over its slots and hops; they are not distinct failed-service
counts. These are periodic state samples, not a complete outage-event counter.
Summary keys use `diagnostics/` and `replicas/`; blocking uses `eval/` or `train/`.

The first preflight caught a command-builder bug: a separate numeric scheduling
argument removed a matching value from the existing three-value kernel tile.
The replacement helper now treats only option names as names and removes a
replaced option's old separate arguments together. A regression preserves
`--xpm-tile 16 32 4` while replacing screen sizes and adding persistent rows.
The larger sparse-network preflight was stopped and labeled as interrupted;
the bounded preflight uses eight warmup requests and 128 measured transitions.
It is a correctness check, not a blocking or throughput result. Full comparison
warmup remains 25,000 requests per replica.

The shortened preflight also exposed empty reported arrays when 32 samples were
downsampled in blocks of 64. Its corrected explicit plan uses downsampling 1;
full runs retain downsampling 64 and have sufficiently long logging increments.
All three corrected CUDA preflights complete successfully: ARBR, MSCL and a real
transformer PPO update. The latter's core losses are finite. Their recorded
commands, status and physical-health samples are in `preflight.json`; the W&B
API verification is in `wandb-verification.json`. Twenty-one focused local tests
and two legacy learned-power regressions pass; pre-commit and typing pass.

The four full lanes are running, initially in warmup:
[control](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/472de30b),
[retained 0.5 dB](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/b74123c7),
[1 dB admission / 0.5 dB retained](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/dc9e487c),
and [power bracket](https://forge.coreweave.com/wandb/micdoh/XLRON-GN-CLS/runs/e23936b2).
These links identify the first trial in each queue; `runs.json` includes all
trial IDs and the queued transformer continuation. No steady-state improvement
has been measured yet. The full lanes use the frozen v3 sources; the tiny
preflight's downsampling override was supplied in its explicit job plan.

Existing learned-power implementations use MLP/GNN architectures; the MLP
power policy delegates route/spectrum selection to a heuristic. The transformer
currently lacks a joint power policy and power-conditioned physical mask, so
that combination now fails with an explicit explanation. A joint discrete
path/slot/power mask is the intended next interface: retaining every feasible
slot while deriving modulation and width at the candidate power. The bracket
trials measure power sensitivity before that larger action-space change.
