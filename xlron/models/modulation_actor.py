"""Score full-resolution GN candidates using their resolved modulation and width."""

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from xlron.environments.env_funcs import read_rsa_request, required_slots
from xlron.models.spectrum_transformer import ActorCriticSpectrumTransformer
from xlron.models.transformer import ActorCriticTransformer


def candidate_modulation_features(state, params):
    """Small per-action features; no extra per-link observation channels."""
    indices = state.mod_format_mask.astype(jnp.int32)
    valid = indices >= 0
    formats = params.modulations_array
    selected = formats[jnp.maximum(indices, 0)]
    _, bandwidth = read_rsa_request(state.request_array)
    widths = jax.vmap(lambda se: required_slots(bandwidth, se, params.slot_size, params.guardband))(
        selected[:, 1]
    )
    features = jnp.stack(
        (
            valid.astype(jnp.float32),
            selected[:, 1] / jnp.maximum(jnp.max(formats[:, 1]), 1.0),
            jnp.log1p(widths) / jnp.log1p(params.link_resources),
            (selected[:, 2] + params.snr_margin) / params.max_snr,
        ),
        axis=-1,
    )
    return jnp.where(valid[:, None], features, 0.0)


class ModulationAwareActorCritic(eqx.Module):
    """Residual candidate head around either transformer; critic is unchanged."""

    network: ActorCriticTransformer | ActorCriticSpectrumTransformer
    actor_candidate: eqx.nn.MLP

    def __init__(self, network, key):
        self.network = network
        head = eqx.nn.MLP(5, 1, 8, 1, activation=jax.nn.gelu, key=key)
        # Preserve the base policy at initialization/checkpoint migration.
        head = eqx.tree_at(
            lambda h: (h.layers[-1].weight, h.layers[-1].bias),
            head,
            (jnp.zeros_like(head.layers[-1].weight), jnp.zeros_like(head.layers[-1].bias)),
        )
        self.actor_candidate = head

    @property
    def actor_critic(self):
        return self.network.actor_critic

    @property
    def critic_mlp(self):
        return self.network.critic_mlp

    def __call__(self, state, params, *, enable_dropout=False, key=None):
        policy, value = self.network(state, params, enable_dropout=enable_dropout, key=key)
        features = candidate_modulation_features(state, params)
        base_logits = jnp.asarray(policy.logits)
        logits = base_logits[: features.shape[0]]
        inputs = jnp.concatenate((logits[:, None], features), axis=-1)
        correction = jax.vmap(self.actor_candidate)(inputs)[:, 0]
        logits = base_logits.at[: features.shape[0]].add(correction)
        return distrax.Categorical(logits=logits), value
