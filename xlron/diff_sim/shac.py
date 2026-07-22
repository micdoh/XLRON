"""SHAC-style analytic policy gradient through the differentiable XLRON environment.

Short-Horizon Actor-Critic (Xu et al. 2022, "Accelerated Policy Learning with
Parallel Differentiable Simulation"), adapted to XLRON's discrete lightpath
action space:

- The policy emits a categorical distribution over the flat (path, slot) action
  space. The environment consumes a scalar action; we form a *soft* action with
  a straight-through estimator: the forward pass uses the sampled (or argmax)
  integer action so the dynamics stay exact, while the backward pass uses the
  masked-softmax expected action index. d(reward)/d(action) therefore flows into
  the policy logits analytically, through the differentiable env ops (soft slot
  masks, soft collision checks) -- including through *future* steps, because an
  allocation persists in link_slot_array and shapes later collisions.
- Actor objective: mean over envs of [sum_{t<H} gamma^t r_t + gamma^H V(s_H)],
  backpropagated through the H-step rollout (BPTT). H = config.ROLLOUT_LENGTH.
  The critic weights inside the terminal value are detached (SHAC treats the
  critic as fixed during the actor update); the gradient still flows through
  s_H into earlier actions.
- Critic loss: TD(lambda) regression on the detached rollout (lambda =
  config.GAE_LAMBDA), predictions from detached observations so the value fit
  never backpropagates into the dynamics.

Requires --differentiable (the env must be built with soft ops) and a policy
model (MLP/GNN/Transformer). Reuses the standard optimizer/schedule stack from
train_utils (LR, MAX_GRAD_NORM, VF_COEF, ENT_COEF, GAMMA, GAE_LAMBDA).
"""

import math
from typing import Any, Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from box import Box
from gymnax.environments import environment

from xlron.environments.dataclasses import EnvParams
from xlron.environments.diff_utils import straight_through
from xlron.train.train_utils import TrainState, cast_model_for_compute

# (train_state, env_state, obs, rng_step, rng_epoch) -- matches ppo.RunnerState
RunnerState = Tuple[TrainState, Any, Any, jnp.ndarray, jnp.ndarray]

# Large negative for masking invalid action logits (matches select_action)
NEG_INF = -1e8


def _detach(tree: Any) -> Any:
    """stop_gradient over an arbitrary pytree."""
    return jax.tree.map(jax.lax.stop_gradient, tree)


def _detach_critic(model: eqx.Module) -> eqx.Module:
    """Return the model with stop_gradient applied to the critic submodule's arrays.

    Used for the terminal value in the actor objective: gradients flow through the
    *input state* but not into the critic weights.
    """
    if not hasattr(model, "critic"):
        return model
    detached_critic = jax.tree.map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x,
        model.critic,
    )
    return eqx.tree_at(lambda m: m.critic, model, detached_critic)


def _soft_action_from_logits(
    logits: jnp.ndarray,
    action_mask: jnp.ndarray,
    key: jnp.ndarray,
    config: Box,
    num_slot_actions: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Form a straight-through soft action from masked policy logits.

    Forward pass: integer action (sampled from the masked categorical, or its
    argmax when SHAC_FORWARD='mode').

    Backward pass (SHAC_ACTION_SURROGATE):
    - 'slot_conditional' (default): S * p_hard + E[s | p = p_hard], the expected
      slot index on the SAMPLED path. The env provides no analytic gradient for
      the path component (process_path_action int-casts it), and with the flat
      surrogate a slot-direction gradient dL/da would hit the path marginal with
      an S-times amplification (a = p*S + s), so the policy would respond to slot
      signals by reshuffling paths. Conditioning on the sampled path routes the
      analytic gradient exclusively into the slot distribution of that path.
    - 'flat': E[a] under the full masked softmax (the original v1.0 surrogate;
      kept for ablation).

    Returns (action, log_prob, entropy, hard_action, soft_action).
    """
    mask = jax.lax.stop_gradient(action_mask.astype(jnp.float32))
    masked_logits = logits + NEG_INF * (1.0 - mask)
    log_probs = jax.nn.log_softmax(masked_logits)
    probs = jnp.exp(log_probs)

    if config.get("SHAC_FORWARD", "sample") == "mode":
        hard_action = jnp.argmax(masked_logits, axis=-1)
    else:
        hard_action = jax.random.categorical(key, masked_logits, axis=-1)

    surrogate = config.get("SHAC_ACTION_SURROGATE", "slot_conditional")
    if surrogate == "flat":
        indices = jnp.arange(logits.shape[-1], dtype=jnp.float32)
        soft_action = jnp.sum(probs * indices, axis=-1)
    elif surrogate == "slot_conditional":
        S = num_slot_actions
        K = logits.shape[-1] // S
        probs_ks = probs.reshape(probs.shape[:-1] + (K, S))
        p_hard = hard_action // S
        # Slot distribution on the sampled path: (..., S)
        slot_probs = jnp.take_along_axis(
            probs_ks, p_hard[..., None, None].astype(jnp.int32), axis=-2
        ).squeeze(-2)
        slot_probs = slot_probs / (jnp.sum(slot_probs, axis=-1, keepdims=True) + 1e-8)
        slot_indices = jnp.arange(S, dtype=jnp.float32)
        e_slot = jnp.sum(slot_probs * slot_indices, axis=-1)
        soft_action = p_hard.astype(jnp.float32) * S + e_slot
    else:
        raise ValueError(f"Unknown SHAC_ACTION_SURROGATE: {surrogate}")

    action = straight_through(hard_action.astype(jnp.float32), soft_action)
    log_prob = jnp.take_along_axis(
        log_probs, hard_action[..., None].astype(jnp.int32), axis=-1
    ).squeeze(-1)
    entropy = -jnp.sum(probs * jnp.where(probs > 0, log_probs, 0.0), axis=-1)
    return action, log_prob, entropy, hard_action, soft_action


def _get_obs_tuple(obsv, env_state, env_params, config):
    """Mirror ppo.py obs handling: GNN/Transformer models consume (state, params)."""
    if config.USE_GNN or config.USE_TRANSFORMER:
        return (env_state.env_state, env_params)
    return tuple([obsv])


def _apply_model(model, obs, config):
    """Apply the model over the env batch dimension. obs leaves have leading NUM_ENVS."""
    if config.USE_GNN or config.USE_TRANSFORMER:
        axes = (0, None)
        if config.NUM_ENVS > 1:
            return jax.vmap(model, in_axes=axes)(*obs)
        return model(*obs)
    if config.NUM_ENVS > 1:
        return jax.vmap(model)(*obs)
    return model(*obs)


def _td_lambda_targets(
    rewards: jnp.ndarray,  # (H, N)
    values: jnp.ndarray,  # (H, N) V(s_t), detached
    bootstrap: jnp.ndarray,  # (N,) V(s_H), detached
    gamma: float,
    lam: float,
) -> jnp.ndarray:
    """Standard TD(lambda) targets over an H-step window (no terminals inside the
    window: XLRON continuous operation never terminates mid-rollout)."""

    def _scan_fn(next_target, xs):
        r_t, v_next = xs
        target = r_t + gamma * ((1.0 - lam) * v_next + lam * next_target)
        return target, target

    v_next_seq = jnp.concatenate([values[1:], bootstrap[None]], axis=0)  # (H, N)
    _, targets = jax.lax.scan(_scan_fn, bootstrap, (rewards, v_next_seq), reverse=True)
    return targets


def get_shac_learner_fn(
    env: environment.Environment,
    env_params: EnvParams,
    train_state: TrainState,
    config: Box,
) -> Callable:
    """Build the SHAC learner. Drop-in replacement for ppo.get_learner_fn:
    returns learner_fn(runner_state) -> {"runner_state", "metrics", "loss_info"}
    with metrics/loss_info shaped identically (leaves (NUM_UPDATES, H, N) /
    (NUM_UPDATES,))."""

    if not env_params.differentiable:
        raise ValueError(
            "SHAC requires --differentiable: the analytic gradient flows through "
            "the env's soft ops. Re-run with --differentiable (and a --temperature)."
        )
    if config.env_type.lower() == "vone":
        raise ValueError("SHAC does not support the VONE action space (path actions only).")

    gamma = float(config.GAMMA)
    # GAE_LAMBDA may be None (PPO anneals it); default to 0.95 for TD(lambda) targets.
    lam = float(config.GAE_LAMBDA) if config.GAE_LAMBDA is not None else 0.95
    horizon = int(config.ROLLOUT_LENGTH)
    num_envs = int(config.NUM_ENVS)
    use_bootstrap = bool(config.get("SHAC_VALUE_BOOTSTRAP", True))
    vf_coef = float(config.VF_COEF)
    # Action-space geometry for the slot-conditional surrogate (matches
    # process_path_action / RSAEnv.num_actions: A = k_paths * num_slot_actions)
    num_slot_actions = math.ceil(env_params.link_resources / env_params.aggregate_slots)

    def _mask_fn(inner_state):
        mask_result = env.action_mask(inner_state, env_params)  # ty: ignore[unresolved-attribute]
        # (action_mask, full_action_mask[, mod_format_mask]) -> use head mask
        return mask_result[0]

    def _env_step_fn(step_key, env_state, action):
        return env.step(step_key, env_state, action, env_params)

    def _rollout_step(carry, unused, model):
        env_state, obs, rng = carry
        rng, action_key, step_key = jax.random.split(rng, 3)

        pi, value = _apply_model(model, obs, config)
        logits = pi._logits

        mask = (
            jax.vmap(_mask_fn)(env_state.env_state)
            if num_envs > 1
            else _mask_fn(env_state.env_state)
        )
        # Threshold: differentiable-mode masks can be float; hard-select validity.
        mask = jax.lax.stop_gradient(mask > 0.5)

        action, log_prob, entropy, hard_action, soft_action = _soft_action_from_logits(
            logits, mask, action_key, config, num_slot_actions
        )

        step_keys = jax.random.split(step_key, num_envs) if num_envs > 1 else step_key
        step_fn = jax.vmap(_env_step_fn, in_axes=(0, 0, 0)) if num_envs > 1 else _env_step_fn
        obsv, env_state, reward, terminal, truncated, info = step_fn(step_keys, env_state, action)
        reward = reward * config.REWARD_SCALE

        next_obs = _get_obs_tuple(obsv, env_state, env_params, config)

        step_out = {
            "reward": reward,
            "value": value,
            "log_prob": log_prob,
            "entropy": entropy,
            "info": info,
            "terminal": terminal,
            "truncated": truncated,
            "action": hard_action,
            # |backward surrogate - forward action|: the straight-through bias
            "soft_gap": jnp.abs(
                jax.lax.stop_gradient(soft_action) - hard_action.astype(jnp.float32)
            ),
            # Detached obs snapshot for the critic regression. Store only the
            # state/array element -- for GNN/Transformer obs the second element is
            # env_params, which must not be stacked over the scan (static fields /
            # duplicated arrays). The tuple is rebuilt in the critic pass.
            "obs_sg": _detach(obs[0]),
        }
        return (env_state, next_obs, rng), step_out

    def _loss_fn(model_params, model_static, env_state, obs, rng, ent_coef):
        model = eqx.combine(model_params, model_static)
        model = cast_model_for_compute(model)

        rollout_step = lambda carry, unused: _rollout_step(carry, unused, model)
        if config.get("SHAC_REMAT", False):
            rollout_step = jax.checkpoint(rollout_step, prevent_cse=False)

        (final_env_state, final_obs, rng), traj = jax.lax.scan(
            rollout_step, (env_state, obs, rng), None, horizon
        )

        rewards = traj["reward"].astype(jnp.float32)  # (H, N) soft rewards
        discounts = (gamma ** jnp.arange(horizon, dtype=jnp.float32))[:, None]

        # Terminal value with critic weights detached; gradient flows through the
        # final observation into the rollout actions.
        if use_bootstrap:
            model_vsg = _detach_critic(model)
            _, terminal_value = _apply_model(model_vsg, final_obs, config)
            actor_return = jnp.sum(discounts * rewards, axis=0) + (
                gamma**horizon
            ) * terminal_value.astype(jnp.float32)
        else:
            terminal_value = jnp.zeros(rewards.shape[1:], dtype=jnp.float32)
            actor_return = jnp.sum(discounts * rewards, axis=0)

        actor_loss = -jnp.mean(actor_return)
        entropy_mean = jnp.mean(traj["entropy"])

        # ---- Critic: TD(lambda) regression on the detached rollout ----
        obs_sg = traj["obs_sg"]  # leaves (H, N, ...); first obs element only
        rebuild = (
            (lambda o: (o, env_params))
            if (config.USE_GNN or config.USE_TRANSFORMER)
            else (lambda o: tuple([o]))
        )
        values_pred = jax.vmap(lambda o: _apply_model(model, rebuild(o), config)[1])(obs_sg)
        values_pred = values_pred.astype(jnp.float32)  # (H, N)
        final_obs_sg = _detach(final_obs)
        _, bootstrap_pred = _apply_model(model, final_obs_sg, config)
        bootstrap_pred = bootstrap_pred.astype(jnp.float32)

        rewards_sg = jax.lax.stop_gradient(rewards)
        targets = _td_lambda_targets(
            rewards_sg,
            jax.lax.stop_gradient(values_pred),
            jax.lax.stop_gradient(bootstrap_pred),
            gamma,
            lam,
        )
        critic_loss = jnp.mean((values_pred - jax.lax.stop_gradient(targets)) ** 2)

        # ---- Optional score-function (policy-gradient) actor term ----
        # The analytic gradient only reaches the slot distribution (the env has no
        # path gradient); this REINFORCE-with-baseline term on the same on-policy
        # rollout provides unbiased path (and slot) learning. Advantages are the
        # TD(lambda) targets minus the value baseline, both detached.
        pg_coef = float(config.get("SHAC_PG_COEF", 0.0))
        if pg_coef > 0.0:
            adv = jax.lax.stop_gradient(targets - values_pred)
            adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
            pg_loss = -jnp.mean(traj["log_prob"] * adv)
        else:
            pg_loss = jnp.array(0.0, dtype=jnp.float32)

        analytic_coef = float(config.get("SHAC_ANALYTIC_COEF", 1.0))
        total_loss = (
            analytic_coef * actor_loss
            + pg_coef * pg_loss
            + vf_coef * critic_loss
            - ent_coef * entropy_mean
        )

        aux = {
            "final_env_state": _detach(final_env_state),
            "final_obs": _detach(final_obs),
            "traj_info": traj["info"],
            "traj_terminal": traj["terminal"],
            "traj_truncated": traj["truncated"],
            "metrics": {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "pg_loss": pg_loss,
                "entropy": entropy_mean,
                "reward_mean": jnp.mean(rewards),
                "soft_gap": jnp.mean(traj["soft_gap"]),
                "terminal_value_mean": jnp.mean(terminal_value),
            },
        }
        return total_loss, aux

    def _update_step(runner_state: RunnerState, unused: Any):
        train_state, env_state, last_obs, rng_step, rng_epoch = runner_state
        rng_step, rollout_key = jax.random.split(rng_step)

        ent_coef = train_state.ent_schedule(train_state.step)

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, aux), grads = grad_fn(
            train_state.model_params,
            train_state.model_static,
            env_state,
            last_obs,
            rollout_key,
            ent_coef,
        )

        grad_norm = optax.global_norm(grads)
        train_state = train_state.apply_gradients(grads=grads)
        train_state = eqx.tree_at(lambda s: s.step, train_state, train_state.step + 1)

        runner_state = (
            train_state,
            aux["final_env_state"],
            aux["final_obs"],
            rng_step,
            rng_epoch,
        )

        # Metrics dict mirrors ppo: traj info per step (leaves (H, N))
        metric = aux["traj_info"]

        m = aux["metrics"]
        loss_info = {
            "loss/total_loss": total_loss,
            "loss/actor_loss": m["actor_loss"],
            "loss/value_loss": m["critic_loss"],
            "loss/value_loss_scaled": m["critic_loss"] * vf_coef,
            "loss/entropy": m["entropy"],
            "loss/entropy_loss_scaled": -m["entropy"] * ent_coef,
            "loss/grad_norm": grad_norm,
            "loss/reward_mean": m["reward_mean"],
            "loss/soft_gap": m["soft_gap"],
            "loss/terminal_value_mean": m["terminal_value_mean"],
            "loss/pg_loss": m["pg_loss"],
        }
        return runner_state, (metric, loss_info)

    def learner_fn(runner_state: RunnerState) -> Dict[str, Any]:
        runner_state, (metric_info, loss_info) = jax.lax.scan(
            _update_step, runner_state, None, config.NUM_UPDATES
        )
        return {"runner_state": runner_state, "metrics": metric_info, "loss_info": loss_info}

    return learner_fn
