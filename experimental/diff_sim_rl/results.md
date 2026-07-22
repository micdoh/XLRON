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
| shac_sc_t1_lr3e4 (v1.1) | slot_cond | 1 | 3e-4 | 32 | 256 | 10M | TBD | |
| shac_sc_t5_lr3e4 (v1.1) | slot_cond | 5 | 3e-4 | 32 | 256 | 10M | TBD | |
| ppo_ref | - | - | 3e-4 | 150 | 64 | 10M | TBD | reference |

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
