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
| shac_ppo_il (v1.3, H=150) | slot_cond | 1 | 3e-4 | 150 | 64 | killed pre-run | - | compile pathological (>55 min for H=150 BPTT graph) |
| shac_ppo_il_h64 (v1.3) | slot_cond | 1 | 3e-4 | 64 | 128 | 10M | TBD | interleave relaunched at H=64 |
| shac_flat_t1 (v1.3) | flat | 1 | 3e-4 | 32 | 256 | 15M | 4.42% ± 0.20% | temp barely matters for flat surrogate |
| shac_flat_t5_nostate | flat | 5 | 3e-4 | 32 | 256 | 10M | **4.31% ± 0.20%** | ABLATION: matches full-BPTT flat_t5 (4.30%) exactly |
| shac_flat_t5_seed2 | flat | 5 | 3e-4 | 32 | 256 | 10M | 4.24% ± 0.20% | replicates flat_t5 (4.30%) -- result robust |
| ppo_ref | - | - | 3e-4 | 150 | 64 | 10M | TBD | reference |

### Emerging picture (round 4, interim)

- ppo_ref (stock PPO, raw rmsa obs/action): flat at 7.3-7.4% through 4.3M steps.
  Score-function learning on the raw 500-dim action space is slow/ineffective at
  these HPs -- the DeepRMSA-paper results use the curated deeprmsa env instead.
- The analytic runs are the only ones that learn in this setup (5.0/6.9 -> 4.3/5.7%).
- shac_nostate (no cross-step gradients) tracks full BPTT (4.4% @ 7.7M vs 4.3%
  final): differentiating THROUGH the dynamics adds ~nothing here; the analytic
  value is the immediate soft-unblocking signal + the flat surrogate's
  index-lowering (FF-like) prior. The BPTT lookahead story does not hold on
  nsfnet at these settings.

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

## Phase A conclusions (2026-07-22)

1. **The analytic gradient through the differentiable env trains a policy where
   score-function methods fail outright** at matched raw obs/action settings:
   SHAC 7 -> 4.2-4.3% (robust across 2 seeds), stock PPO flat at 7.3%, vanilla
   REINFORCE diverges to 12%. This is the sample-efficiency evidence for
   differentiable simulation on this problem class.
2. **But the learning signal is NOT the BPTT lookahead.** Severing all
   cross-step gradients reproduces the result exactly (4.31% vs 4.30%). What
   remains is (a) the immediate soft-unblocking gradient and (b) the flat
   surrogate's global index-lowering prior, which is a soft first-fit bias.
   On nsfnet at these settings, differentiating *through the dynamics* adds
   nothing measurable -- the future-congestion signal the MSCL-beating thesis
   relies on is either too weak (sigmoid support ~1/T slots), drowned by ST
   bias, or genuinely absent at H=32-64.
3. **Analytic-only plateaus at ~4.3% vs KSP-FF 2.60%** -- it does not reach,
   let alone beat, the heuristic on the raw action space.
4. Hand-rolled REINFORCE (with TD-lambda baseline) is unstable at every setting
   tried; use stock PPO for any score-function component.

Implications for Phase B (USA100 vs MSCL 0.05%): pure SHAC as-is will not close
a 100x blocking gap. Candidate directions, in order of promise:
- PPO+analytic interleave (verdict pending, wandb shac_ppo_il_h64)
- richer surrogate than expected-index (e.g. distribution-level gradients on the
  soft slot mask rather than a scalar action; requires env interface change)
- temperature scheduling / much lower T with bias correction
- accepting the reframe: analytic gradient as a fast *pretrainer* (FF-prior
  learner) composed with stronger downstream RL.

## Phase B — USA100 (load 620, 320 slots) — pending Phase A

Targets: MSCL-KSP ~0.05%, FF-KSP ~0.34%, best pure RL (aggregated transformer) ~0.40%.
