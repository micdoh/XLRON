# speedups branch — status and remaining items

Baseline (main): 16.6K SPS single-env CPU (RMSA NSFNET 100 FSU k=5 KSP-FF, load 250,
100k steps, M1 Pro, f32). After batches 1+2: **27K SPS (+62%)**, blocking bit-identical
(0.24147), 1210+632 tests pass, differentiable path verified (gradients flow).

## Done
1. [42bf19a] Boolean fast path in remove_expired_services_rsa; integer path-index
   short-circuit in get_path_index_array; integer decode in process_path_action;
   removed unusable donate_argnums on mask_slots. All non-diff-mode only; diff paths
   untouched. Bit-identical.
2. [4d7711d] Utilisation/fragmentation off the hot path -> computed per logging
   increment in log_metrics from final state. THE big win (+62%).
9. [done 2026-07-18] differentiable_compare early-out hoisted above soft-op
   construction (trace-size only; benchmark COMPILATION flat within noise:
   1.29-1.30s before, 1.14-1.30s after — the DCE'd soft ops were a small
   fraction of the compile). Bit-identical (0.24147); gradient check passes
   (grad std ~9.5e-11).
10. [done 2026-07-18] Tier hygiene: init_traffic_matrix constructed in LARGE_FLOAT
    (was SMALL_FLOAT then astype(f32) — lossy under mixed precision); rsa.py's
    duplicate LARGE_FLOAT one/zero removed, now imports env_funcs' (int32) pair so
    the RWA `path_se = one` site matches mask_slots' int path_se_array dtype
    (single required_slots specialisation). Reward base values got explicit
    LARGE_FLOAT constants to keep reward dtype float32. Bit-identical (0.24147),
    FPS unchanged (~26K, within noise), 2066 tests pass.
8. [done 2026-07-18] Dead obs build skipped in heuristic eval: shape-(1,) placeholder
   obs (train_utils.heuristic_eval_obs_placeholder) substituted into the carry at
   all three thread sites — experiment_data_setup init, get_warmup_fn loop body,
   eval_heuristic step body — gated on config.EVAL_HEURISTIC && !USE_GNN &&
   !USE_TRANSFORMER, so env.step's returned obs is unused and the ~4403-elem
   get_obs concat is DCE'd from the compiled step. EVAL_MODEL/RL/GNN paths and
   warmup_action_type='heuristic' during RL training keep real obs (gate keys on
   the outer config). SweepRewarmTest updated to mirror the placeholder carry.
   Measured same-session: 26.3-27.3K -> 28.2-28.6K FPS (~+5-7%). Bit-identical on
   benchmark (0.24147), warmup path (0.24920), NUM_ENVS=4 (0.22885), and load
   sweep + rewarm (0.17680/0.25340); rsa_gn_model heuristic eval runs (SNR checks
   untouched); gradient check passes (grad std ~9.5e-11). NOTE: DeepRMSA's
   calculate_path_stats in step_env writes into state (a live carry element), so
   the obs placeholder cannot DCE it and no existing static param can gate it —
   left as is (benchmark env is rmsa, unaffected). experimental/
   launch_power_optimization/optimize_launch_power.py builds its own full-shape
   obs carry but is already stale against current APIs (5-field Transition, old
   select_action_eval signature) — pre-existing rot, untouched.

## Remaining (verified proposals from the 4-lens audit; anchors = main @997478c)
3. **Lean info stacking** (est. 10-20%): wrappers.py:78-130 stacks ~12 scalars per
   step through scan ys; eval returns traj.info so nothing DCEs. Proposal: for
   EVAL_HEURISTIC, return only the keys the metric pipeline consumes (blocking
   comes from carry counters; check process_metrics needs: returns, lengths,
   cum_returns, accepted_*, total_bitrate, done flags) and drop the rest from the
   stacked dict. Watch: process_metrics key expectations; DOWNSAMPLE; wandb.
4. **Pre-sample the request stream** (est. 10-15%): env_funcs generate_request_rsa
   (~4 key splits + 2 jax.random.choice that re-cumsum the constant traffic matrix
   + 2 exponentials per step). Existing deterministic_requests/list_of_requests
   gather path (env_funcs.py:1239-1266) is the vehicle: pre-sample (T,) streams
   (bandwidth, src-dst, arrival, holding) before the scan, index by
   state.total_requests. Not bit-identical (key-consumption order changes) — fine
   per user. Minimal variant: precompute CDF once in params, sample via
   uniform+searchsorted.
5. **Mask-as-check step refactor** (est. 10-15%): implement_path_action writes both
   (L,S) arrays speculatively, check_no_spectrum_reuse rescans the full spectrum,
   then TWO dense undo passes run even on success (env_funcs.py:2000-2173 region).
   Proposal: validity = full_link_slot_mask[action] gather (mask_slots already ran
   on the same pre-step state); apply allocation once, multiplied by validity.
   Scope: plain RSA/RMSA/RWA check path ONLY — GN envs keep their SNR check
   (physics, not derivable from the mask); differentiable mode keeps the soft
   implement/check/undo path (gradients flow through allocation).
   This is the deepest change: step-API touch in rsa.py step_env + check_action_rsa
   callers; RL invalid actions (masking off) must still be detected -> the gather
   handles it (mask says invalid), but action_history/undo semantics for the
   diff path must be preserved.
6. **Bool action mask end-to-end** (est. 2-4%): mask born bool at
   (window_sums == 0) (env_funcs.py mask_slots) then cast f32, f32-multiplied by
   path_valid, ones-concat, and f32->f16->f32 through the RL carry. Proposal: bool
   in non-diff mode end-to-end; cast to float only at logit-masking
   (train_utils.py:1194) and model-input sites. Requires: init_link_slot_mask dtype,
   aggregate_slots bool handling, every mask write site consistent (scan carry!),
   heuristics first_fit argmax on bool (works), mod_format_mask STAYS float (-1
   sentinels). Add a bool entry to dtype_config.DTYPE_MAP.
7. **int8 occupancy tier for link_slot_array** (GPU memory win, small CPU): values
   {-1,0,1,2}; ~15 write sites need cast-per-write (mixed-precision carry rule);
   force preferred_element_type=int32 on paths @ occupied. Diff mode stays float.

## Verification recipe (used for batches 1-2)
- Speed: 3 reps of the RMSA eval command above; compare FPS.
- Correctness: service_blocking_probability must stay 0.24147 for bit-identical
  items (items 4+ change RNG or semantics — compare distributions instead).
- Tests: uv run pytest xlron/environments/... + ppo_test (full suite before merge).
- Gradients: optimize_actions 30 iters, Mean grad nonzero, actions move.

## Full audit trail
Verified findings + file:line: session scratchpad (rerun audit if stale) and the
workflow journal wf_ab0ce66e-eb5. Do NOT merge to main until the GN/RWA-LR/VONE
paths are re-tested (metrics injection touches all envs via log_metrics).
