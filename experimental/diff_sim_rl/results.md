# Results log — RL with differentiable-simulation gradients

wandb project: **DIFF_SIM_RL** (entity micdoh). All runs on malmo H100s unless noted.

## 2026-07-22 — Infrastructure + sanity

- Base: github/main v1.6.0 (speedups-verified), branch `claude/rl-differentiable-simulation-77e3a3`.
- Fixed `run_direct_optimization.py` (flags never registered — entry point was dead code).
- CPU sanity (direct action optimization, nsfnet, 300 req @ load 400): nonzero
  analytic gradients through the full rollout (Mean grad ~1e-3–5e-3). Gradient
  path confirmed on this base.
- SHAC learner implemented (`--SHAC`), 5/5 tests pass incl. nonzero actor
  gradient under blocking with ENT_COEF=0.

## Phase A — NSFNET (DeepRMSA setting: 100 slots, k=5, load 250, trunc. HT)

| run | surrogate | temp | LR | H | N_envs | steps | service BP | notes |
|---|---|---|---|---|---|---|---|---|
| ksp_ff baseline | - | - | - | - | 2000 | 20M | **2.60% ± 0.31%** | wandb t0hxabuu |
| random valid (SHAC t=0 start) | - | - | - | - | 256 | - | ~5.0% | untrained masked policy |
| shac_t5_lr3e4 (v1.0) | flat | 5 | 3e-4 | 32 | 256 | 10M | 4.30% ± 0.20% | wandb cnzbizzb; learned 5.0->4.3, slow |
| shac_t20_lr1e4 (v1.0) | flat | 20 | 1e-4 | 32 | 256 | ~4M (killed) | 5.7% (worse than init) | sharp temp = useless local gradients |
| shac_sc_t1_lr3e4 (v1.1) | slot_cond | 1 | 3e-4 | 32 | 256 | 10M | 5.71% ± 0.22% | learned 6.9->5.7, slow |
| shac_sc_t5_lr3e4 (v1.1) | slot_cond | 5 | 3e-4 | 32 | 256 | 10M | 7.15% ± 0.23% | flat/worse (gradient too local) |
| shac_hyb_t1 (v1.2) | slot_cond | 1 | 3e-4 | 32 | 256 | ~5M (killed) | 9.2% and rising | PG term unstable (see below) |
| shac_pg_only_t1 (v1.2) | (none) | 1 | 3e-4 | 32 | 256 | ~5M (killed) | 11.7% and rising | PG-only, same instability |
| shac_hyb_b (v1.2, 3b) | slot_cond | 1 | 1e-4 | 64 | 256 | ~7M (killed) | 8.0% and rising | gamma/H/LR/ent fix did NOT cure PG |
| shac_pg_b (v1.2, 3b) | (none) | 1 | 1e-4 | 64 | 256 | ~7M (killed) | 9.6% and rising | conclusion: don't hand-roll REINFORCE |
| shac_ppo_il (v1.3) | slot_cond | 1 | 3e-4 | 150 | 64 | 10M PPO-side | TBD | stock PPO update + analytic update interleaved |
| shac_flat_t1 (v1.3) | flat | 1 | 3e-4 | 32 | 256 | 15M | TBD | best pure-analytic recipe pushed (flat bias, wide sigmoids) |
| ppo_ref | - | - | 3e-4 | 150 | 64 | 10M | TBD | reference |

### Round-3 diagnosis (PG instability)

Both v1.2 runs (hybrid and PG-only) degraded monotonically (5% -> 9-12%) while
entropy collapsed 3.4 -> 1.5: confident learning of garbage. Cause: with
gamma=0.99 and H=32, gamma^H = 0.72 and the critic terminal value (~ -9) dwarfs
the in-window reward sum (~ -1.4), so TD-lambda targets, and hence advantages,
were dominated by the untrained critic; per-update advantage normalization then
amplified that noise to +/-1 and un-clipped REINFORCE sharpened onto it, with
value_loss rising (0.58 -> 0.79) as the critic chased the nonstationary mess.
Fix (round 3b): H=64 + gamma=0.97 (gamma^H = 0.14 -> targets grounded in real
rewards), LR 3e-4 -> 1e-4, ENT_COEF 0.001 -> 0.01.

### Round-2 interim notes

Slot-conditional surrogate fixes the ST bias (soft_gap 30-80 flat-index units ->
7-16 slot units) but the pure analytic signal is weak: t=1 grinds 6.9->5.8% by
~5M steps; t=5 flat (sigmoid support ~1/t slots -> t=5 gradient too local).
Curious round-1 observation: the flat surrogate dropped 6.9->5.0 within the
first increment -- its "lower the flat index" bias is accidentally FF-like
(low path + low slot), fast at first, saturating at 4.3%. Conclusion: the
analytic term alone under-determines path choice; round 3 pairs it with a
score-function term (v1.2 hybrid) and a PG-only ablation isolates its added value.

Baseline throughput notes: KSP-FF eval 8.0M FPS (2000 envs); SHAC trains at
~195K FPS incl. BPTT backward (N=64 smoke) on one H100. A 10M-step SHAC run
takes ~10 min wall-clock (122 increments, dominated by logging).

### Round-1 diagnosis (flat surrogate, wandb cnzbizzb)

grad_norm healthy (~1), entropy falling (3.4->2.2) but blocking nearly flat:
the policy sharpens on the PATH axis in response to SLOT-direction gradients
(the flat surrogate E[a]=S*E[p]+E[s] amplifies dL/ds into the path marginal by
S=100x; the env int-casts the path decode so there is no genuine path
gradient). soft_gap ~30-80 flat-index units confirmed a badly biased surrogate.
Fix: v1.1 slot-conditional surrogate (E[s | sampled path]); also t=20 confirmed
that sharper soft-ops degrade the gradient (sigmoid support ~1/t slots).

## Phase B — USA100 (load 620, 320 slots) — pending Phase A

Targets: MSCL-KSP ~0.05%, FF-KSP ~0.34%, best pure RL (aggregated transformer) ~0.40%.
