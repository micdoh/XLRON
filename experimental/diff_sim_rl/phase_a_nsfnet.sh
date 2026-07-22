#!/bin/bash
# Phase A: SHAC validation on NSFNET (DeepRMSA benchmark setting).
# Run on malmo. Usage: ./phase_a_nsfnet.sh <run: baseline|shac1|shac2|ppo>
# Env setting matches the DeepRMSA benchmark: nsfnet_deeprmsa_directed, 100 slots,
# k=5, load 250, truncated holding times, steady-state (warmup 3000).
set -euo pipefail

GPU2=GPU-18355792-796a-023f-68f9-b2952eb378ee
GPU3=GPU-fe12a3be-1f3a-122a-7229-312c32e5d08a
DIR=$(cd "$(dirname "$0")" && pwd)

COMMON="--env_type=rmsa --topology_name=nsfnet_deeprmsa_directed --link_resources=100 \
 --k=5 --load=250 --continuous_operation --truncate_holding_time \
 --ENV_WARMUP_STEPS=3000 --warmup_action_type=heuristic --path_heuristic=ksp_ff \
 --WANDB --PROJECT=DIFF_SIM_RL"

SHAC_COMMON="$COMMON --SHAC --differentiable --GAMMA=0.99 --GAE_LAMBDA=0.95 \
 --VF_COEF=0.5 --ENT_COEF=0.001 --ROLLOUT_LENGTH=32 --NUM_ENVS=256 \
 --TOTAL_TIMESTEPS=10000000 --STEPS_PER_INCREMENT=81920"

case "$1" in
  baseline)
    # KSP-FF reference at the same load (fast, non-differentiable eval)
    "$DIR/launch_shac.sh" ksp_ff_nsfnet "$GPU3" $COMMON \
      --EVAL_HEURISTIC --TOTAL_TIMESTEPS=20000000 --NUM_ENVS=2000 \
      --STEPS_PER_INCREMENT=2000000 --EXPERIMENT_NAME=ksp_ff_nsfnet_L250
    ;;
  shac1)
    "$DIR/launch_shac.sh" shac_t5_lr3e4 "$GPU2" $SHAC_COMMON \
      --temperature=5.0 --LR=3e-4 --EXPERIMENT_NAME=shac_t5_lr3e4
    ;;
  shac2)
    "$DIR/launch_shac.sh" shac_t20_lr1e4 "$GPU3" $SHAC_COMMON \
      --temperature=20.0 --LR=1e-4 --EXPERIMENT_NAME=shac_t20_lr1e4
    ;;
  # Round 2: slot-conditional surrogate (v1.1). Lower temperature widens the
  # soft-op sigmoids so the collision gradient sees free space further away.
  shac_sc_t1)
    "$DIR/launch_shac.sh" shac_sc_t1_lr3e4 "$GPU3" $SHAC_COMMON \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=1.0 --LR=3e-4 \
      --EXPERIMENT_NAME=shac_sc_t1_lr3e4
    ;;
  shac_sc_t5)
    "$DIR/launch_shac.sh" shac_sc_t5_lr3e4 "$GPU2" $SHAC_COMMON \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=5.0 --LR=3e-4 \
      --EXPERIMENT_NAME=shac_sc_t5_lr3e4
    ;;
  # Round 3: hybrid analytic + score-function actor loss (v1.2)
  hybrid_t1)
    "$DIR/launch_shac.sh" shac_hyb_t1 "$GPU2" $SHAC_COMMON \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=1.0 --LR=3e-4 \
      --SHAC_PG_COEF=1.0 --EXPERIMENT_NAME=shac_hyb_t1_lr3e4
    ;;
  # Pure score-function ablation (no analytic term): isolates what the
  # analytic gradient adds over REINFORCE-with-baseline at identical settings
  pg_only)
    "$DIR/launch_shac.sh" shac_pg_only "$GPU3" $SHAC_COMMON \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=1.0 --LR=3e-4 \
      --SHAC_PG_COEF=1.0 --SHAC_ANALYTIC_COEF=0.0 --EXPERIMENT_NAME=shac_pg_only_t1
    ;;
  # Round 3b: fix PG instability. gamma=0.99/H=32 made TD targets bootstrap-
  # dominated (gamma^H=0.72), so normalized advantages were mostly early-critic
  # noise and REINFORCE sharpened onto it (blocking 5->9-12%). H=64 + gamma=0.97
  # (gamma^H=0.14) grounds targets in real rewards; lower LR + higher entropy.
  hybrid_b)
    "$DIR/launch_shac.sh" shac_hyb_b "$GPU2" \
      --env_type=rmsa --topology_name=nsfnet_deeprmsa_directed --link_resources=100 \
      --k=5 --load=250 --continuous_operation --truncate_holding_time \
      --ENV_WARMUP_STEPS=3000 --warmup_action_type=heuristic --path_heuristic=ksp_ff \
      --WANDB --PROJECT=DIFF_SIM_RL --SHAC --differentiable \
      --GAMMA=0.97 --GAE_LAMBDA=0.95 --VF_COEF=0.5 --ENT_COEF=0.01 \
      --ROLLOUT_LENGTH=64 --NUM_ENVS=256 --TOTAL_TIMESTEPS=15000000 \
      --STEPS_PER_INCREMENT=163840 \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=1.0 --LR=1e-4 \
      --SHAC_PG_COEF=1.0 --EXPERIMENT_NAME=shac_hyb_b_g97_h64
    ;;
  # Round 4a: stock PPO + analytic interleave (settings matched to ppo_ref;
  # each update consumes 2*H*N steps, so wandb env_step undercounts 2x)
  interleave)
    "$DIR/launch_shac.sh" shac_ppo_il "$GPU2" $COMMON \
      --SHAC --differentiable --SHAC_INTERLEAVE_PPO \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=1.0 \
      --LR=3e-4 --GAMMA=0.999 --VF_COEF=0.5 \
      --ROLLOUT_LENGTH=150 --NUM_ENVS=64 \
      --TOTAL_TIMESTEPS=10000000 --STEPS_PER_INCREMENT=960000 \
      --EXPERIMENT_NAME=shac_ppo_interleave
    ;;
  # Round 4b: best pure-analytic recipe pushed further: the flat surrogate's
  # index-lowering bias (fast FF-like start, reached 4.3%) + t=1 wide sigmoids
  flat_t1)
    "$DIR/launch_shac.sh" shac_flat_t1 "$GPU3" $SHAC_COMMON \
      --SHAC_ACTION_SURROGATE=flat --temperature=1.0 --LR=3e-4 \
      --TOTAL_TIMESTEPS=15000000 --EXPERIMENT_NAME=shac_flat_t1_lr3e4
    ;;
  # Round 4c: state-gradient truncation ablation. Reference = shac_t5_lr3e4
  # (flat, T=5, full BPTT, 4.30%). If this matches it, BPTT-through-dynamics
  # contributes nothing and the analytic gradient is just immediate unblocking.
  nostate)
    "$DIR/launch_shac.sh" shac_nostate "$GPU3" $SHAC_COMMON \
      --SHAC_ACTION_SURROGATE=flat --temperature=5.0 --LR=3e-4 \
      --SHAC_TRUNCATE_STATE_GRAD --EXPERIMENT_NAME=shac_flat_t5_nostate
    ;;
  pg_only_b)
    "$DIR/launch_shac.sh" shac_pg_b "$GPU3" \
      --env_type=rmsa --topology_name=nsfnet_deeprmsa_directed --link_resources=100 \
      --k=5 --load=250 --continuous_operation --truncate_holding_time \
      --ENV_WARMUP_STEPS=3000 --warmup_action_type=heuristic --path_heuristic=ksp_ff \
      --WANDB --PROJECT=DIFF_SIM_RL --SHAC --differentiable \
      --GAMMA=0.97 --GAE_LAMBDA=0.95 --VF_COEF=0.5 --ENT_COEF=0.01 \
      --ROLLOUT_LENGTH=64 --NUM_ENVS=256 --TOTAL_TIMESTEPS=15000000 \
      --STEPS_PER_INCREMENT=163840 \
      --SHAC_ACTION_SURROGATE=slot_conditional --temperature=1.0 --LR=1e-4 \
      --SHAC_PG_COEF=1.0 --SHAC_ANALYTIC_COEF=0.0 --EXPERIMENT_NAME=shac_pg_b_g97_h64
    ;;
  ppo)
    # PPO reference with identical env + action space (full slot granularity)
    "$DIR/launch_shac.sh" ppo_ref "$GPU3" $COMMON \
      --LR=3e-4 --GAMMA=0.999 --ROLLOUT_LENGTH=150 --NUM_ENVS=64 \
      --TOTAL_TIMESTEPS=10000000 --STEPS_PER_INCREMENT=96000 \
      --EXPERIMENT_NAME=ppo_ref_nsfnet
    ;;
  *) echo "unknown run: $1"; exit 1;;
esac
