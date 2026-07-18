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

3. [done 2026-07-18] Lean info stacking in eval: eval_heuristic._pack_info packs
   the core per-step info scalars into two vectors (int-valued counters+done
   flags in LARGE_INT, float metrics in LARGE_FLOAT) so the scan stacks 2
   buffers instead of ~10 tiny per-scalar dynamic-update-slices; _unpack_info
   restores the exact keys/dtypes after the scan, so process_metrics/
   print_metrics/CSV/wandb see an unchanged dict. Constant-zero per-step
   utilisation/fragmentation are dropped entirely for non-VONE envs;
   log_metrics' per-increment injection now *creates* those processed_data
   entries from a template metric (and reorders to the canonical metrics-list
   order) so summary rows and CSV columns are preserved — verified CSV
   byte-identical vs HEAD. GN blocked_* keys ride the int pack; throughput/
   launch_power/log_actions keys stay unpacked. LogWrapper and the RL path
   (ppo.py) untouched. Measured same-session: 3.55s/28.1K -> 3.23-3.29s/
   30.4-30.9K FPS (~+9%). Bit-identical (0.24147; NUM_ENVS=4 0.24223 matches
   HEAD); GN smoke run correct (spectrum/snr/power blocking, throughput);
   1237+25 tests pass; gradient check passes (grad std ~9.5e-11). NOTE: the
   pickled merged_out (.pkl next to EPISODE_DATA_OUTPUT_FILE) no longer
   contains per-step utilisation/fragmentation for non-VONE eval runs (they
   were constant 0 since batch 2 anyway).

4. [done 2026-07-18, variant (a) only] Fused per-step request randomness:
   generate_request_rsa/_rwalr now make ONE jax.random.uniform draw of shape
   (3+num_holding,) per step (was ~2 splits + 4-8 threefry draws) and sample by
   inverse-CDF: source-dest via searchsorted against a traffic-matrix CDF
   pre-computed in make_env (new static params field traffic_cdf, None +
   per-step-cumsum fallback when random_traffic regenerates the matrix per
   reset; traffic_array samples floor(u*n) rows); bandwidth via floor(u*n)
   (uniform) or searchsorted on the constant-folded values_bw_probs cumsum;
   arrival/holding via -log1p(-u), exactly jax.random.exponential's
   construction (helper _arrival_holding_from_uniforms; the keyed
   generate_arrival_holding_times wrapper survives for VONE and draws its
   uniforms in one fused call too, truncate_holding_time candidates included).
   NOT bit-identical (RNG consumption changed): benchmark blocking 0.24021
   (was 0.24147), NUM_ENVS=4 0.23863. Same-session FPS 29.0-30.5K -> 32.9-33.0K
   (~+10%). New GenerateRequestDistributionTest validates 100k-draw source-dest
   frequencies vs the traffic matrix (CDF and random_traffic fallback paths),
   bw uniformity, and exponential arrival/holding means; seeded-snapshot
   expectations updated in rsa_test/env_funcs_test/rwa_lightpath_reuse_test
   (new draws verified self-consistent with unchanged env logic). All env +
   ppo + heuristics tests pass (1574+618); gradient check passes (grad std
   ~9.5e-11). Variant (b) (pre-sample (T,) streams before the scan) NOT
   attempted: it needs eval_fn/learner_fn plumbing and per-env (T,)-stream
   memory that scales badly with NUM_ENVS on GPU; (a) already hit the item's
   estimated 10-15% band.

5. [done 2026-07-18] Prestate-check step refactor (the "mask-as-check" item): plain
   RSA/RMSA/RWA step no longer implements speculatively + rescans + undoes.
   New check_action_rsa_prestate decides validity on the PRE-step state: windowed
   spectrum check (jnp.take of the (L, max_slots) window at initial_slot with
   fill=0, occupied = nonzero on path links — exactly when the speculative
   allocation would produce a value > 1) + the same overflow/no-op/dummy-path
   scalar checks as check_action_rsa. Then implement_and_complete_rsa applies the
   allocation ONCE, gated on success (delta = affected_slots_mask * success), with
   the complete_step_rsa counter updates fused in. DEVIATION from the audit
   proposal: validity is computed from the spectral window, NOT gathered from
   state.full_link_slot_mask — the state's mask is STALE in heuristic eval
   (heuristics call mask_slots internally via get_action_mask but never write the
   result back to the carried state; only the RL select_action path refreshes it),
   so the gather would have accepted every ksp_ff fallback action (~24% of steps
   emit invalid p0s0 when blocked) and collapsed blocking to ~0. The windowed
   check is mathematically identical to a fresh mask gather (mask bit = window
   free && fits && real path), costs (L x max_slots) instead of O(1) — negligible
   vs the removed (L,S) passes — and needs no mask-freshness contract, so direct
   step() callers and tests keep exact old semantics for arbitrary invalid
   actions. Gate: params.__class__.__name__ == "RSAEnvParams" && !differentiable
   (exact name: DeepRMSA/multiband/RWA-LR/GN keep the old flow; GN keeps SNR
   checks; diff mode keeps the soft implement/check/undo path VERBATIM in the
   else branch). Success arithmetic bit-identical (mask*1 == mask); on fail the
   state is untouched (old flow's add-then-subtract could perturb occupied
   departure entries by float rounding — empirically no effect on the benchmark).
   Measured same-session: 3.09-3.33s/30.0-32.3K -> 2.07-2.08s/48.0-48.4K FPS
   (~+55%). Blocking bit-identical: 0.24021 all 3 reps (matches HEAD), NUM_ENVS=4
   0.23863 (matches HEAD), warmup ENV_WARMUP_STEPS=3000/50k 0.23776 (matches
   HEAD). GN heuristic-eval smoke runs. New PrestateCheckEquivalenceTest (6
   tests) asserts post-state + fail-flag equality old-vs-new flow for valid,
   occupied-slot, mid-window-overlap, overflow and no-op actions (dyadic times
   so the old undo round-trip is exact) plus full-step blocking of an invalid
   action. env_funcs+rsa 622 passed; ppo+heuristics and deeprmsa+rwalr+gn suites
   pass; gradient check passes (grad std ~9.5e-11).

## Remaining (verified proposals from the 4-lens audit; anchors = main @997478c)
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
