# RL with analytic gradients from differentiable simulation (SHAC)

**Goal:** learn an RMSA policy using first-order gradients backpropagated through
XLRON's differentiable environment (`--differentiable`), instead of (or in addition
to) PPO's score-function estimator. Ultimate target: match/beat the MSCL-KSP
heuristic (~0.05% service blocking on USA100 at load 620) with a learned policy
that has totally free slot choice (`--aggregate_slots=1`).

Handoff context: `scratchpad/DIFF_HANDOFF.md` (prior session). Prior results to beat
(USA100, load 620): MSCL-KSP ~0.05%, FF-KSP ~0.34%, best pure RL (transformer,
aggregated slots) ~0.40%.

## Method

SHAC-style (Short-Horizon Actor-Critic, Xu et al. 2022) analytic policy gradient,
implemented in [`xlron/diff_sim/shac.py`](../../xlron/diff_sim/shac.py) and wired into
`train.py` behind the `--SHAC` flag:

- Policy emits a categorical over the flat (path x slot) action space
  (aggregate_slots=1 for full slot granularity).
- **Soft action, exact dynamics:** forward pass steps the env with the sampled
  integer action; backward pass replaces it with the masked-softmax *expected*
  action index (straight-through). Gradients of the (soft) reward w.r.t. the
  action scalar flow through the env's soft ops: slot-window mask (sigmoids),
  collision check (x^2/(1+x^2) violation), occupancy persistence into future steps.
- **Actor loss:** -E[ sum_{t<H} gamma^t r_t + gamma^H V(s_H) ], BPTT over
  H = ROLLOUT_LENGTH steps. Critic weights inside V(s_H) detached (SHAC).
- **Critic loss:** TD(lambda) regression on detached observations.
- Carried env state is detached between updates (truncated BPTT windows).

### Gradient pathway notes (from code reading)

- `process_path_action`: the *slot* component of the action gets a real gradient
  (`mod(action, num_slots)` -> soft slot mask); the *path* component is int-cast,
  so path choice only receives gradient signal via aliasing in the flat index.
  Expect the analytic gradient to mainly sharpen slot placement. If path learning
  stalls, the planned v2 is a hybrid PPO + SHAC loss (score function for the path
  head, analytic for slots).
- Reward in differentiable mode: `fail*check + (1-check)*success` where `check`
  is a straight-through soft violation measure -> d(reward)/d(action) is nonzero.
- `temperature` (static env param) controls the sharpness of all soft ops.
  Annealing requires recompilation, so runs use a fixed temperature (sweepable).

## Key flags

| flag | meaning |
|---|---|
| `--SHAC` | use the SHAC learner instead of PPO |
| `--differentiable` | REQUIRED: build env with soft ops |
| `--temperature` | soft-op sharpness (higher = harder ops, sharper but sparser gradients) |
| `--ROLLOUT_LENGTH` | BPTT horizon H |
| `--SHAC_VALUE_BOOTSTRAP` | include gamma^H V(s_H) in actor objective (default true) |
| `--SHAC_FORWARD` | `sample` (default) or `mode` forward action selection |
| `--SHAC_REMAT` | remat rollout steps (memory vs compute) |
| `--GAMMA`, `--GAE_LAMBDA`, `--VF_COEF`, `--ENT_COEF`, `--LR` | reused from PPO stack |

## Sanity checks completed

- 2026-07-22: `run_direct_optimization` (fixed missing flag registration) on
  nsfnet_deeprmsa_directed, 300 requests @ load 400, CPU: nonzero analytic
  gradients through the full rollout (`Mean grad. ~1e-3-5e-3`), confirming the
  env is differentiable end-to-end on this base (github/main @ v1.6.0 speedups).
  Direct *action* optimization itself degrades from the heuristic init (known
  fragility, unmasked actions) - not the target; policy learning is.

## Experiments

Logged to wandb project **DIFF_SIM_RL** (entity micdoh). See `results.md` for the
running log of runs + outcomes.

### Phase A: nsfnet validation (in progress)
`nsfnet_deeprmsa_directed`, link_resources=100, k=5, load=250, continuous
operation, warmup 3000 (heuristic), truncate_holding_time - the DeepRMSA
benchmark setting. Baseline: KSP-FF eval + PPO reference. Question: does the
analytic gradient learn at all / faster than PPO per env-step?

### Phase B: USA100 (pending Phase A)
USA100, 320 slots, load 620, transformer policy, free slot choice. Compare vs
MSCL-KSP 0.05%.

## Infrastructure

- Cluster: malmo.ee.ucl.ac.uk (H100 94GB x4; **GPUs 0 & 1 wedged, use 2 & 3 by UUID**),
  turin/geneva (A100 80GB, shared, usually busy). Direct ssh only (no ProxyJump).
- Repo on cluster: `~/git/xlron-mscl` (worktree `~/git/xlron-diffsim` for this work,
  branch `claude/rl-differentiable-simulation-77e3a3`).
- Launch scripts in this folder: `launch_shac.sh` (tmux + UUID-pinned GPU).
