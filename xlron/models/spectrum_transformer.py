"""Local spectrum features plus compact graph context, with one action per slot."""

import math

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from xlron import dtype_config
from xlron.environments.env_funcs import get_obs_transformer, get_paths, read_rsa_request
from xlron.models.transformer import Embedder, Encoder


def band_convolution(conv, features, status):
    """Apply shared local filters without mixing across permanent band gaps."""
    slots = features.shape[0]
    radius = conv.weight.shape[-1] // 2
    segment = jnp.cumsum(status < 0)
    padded = jnp.pad(features, ((radius, radius), (0, 0)))
    ids = jnp.pad(segment, (radius, radius), constant_values=-1)
    usable = jnp.pad(status >= 0, (radius, radius))
    # Build masked local windows in one fused gather, then use one contraction
    # instead of five tiny GEMMs. Keep ordinary JAX reverse-mode differentiation.
    offsets = range(2 * radius + 1)
    patches = jnp.stack([padded[o : o + slots] for o in offsets], axis=-1)
    same = jnp.stack(
        [(ids[o : o + slots] == segment) & usable[o : o + slots] for o in offsets], axis=-1
    )
    patches = (patches * same[:, None, :]).reshape(slots, -1)
    result = patches @ conv.weight.reshape(conv.weight.shape[0], -1).T
    if conv.bias is not None:
        result = result + conv.bias[:, 0]
    return jnp.where((status >= 0)[:, None], result, 0)


def ordered_spectrum_summary(local, bins=16):
    """Ordered mean/max feature summaries; this does not aggregate slot actions."""
    width = math.ceil(local.shape[0] / bins)
    padding = bins * width - local.shape[0]
    padded = jnp.pad(local, ((0, padding), (0, 0)))
    blocks = padded.reshape(bins, width, local.shape[-1])
    valid = (jnp.arange(bins * width) < local.shape[0]).reshape(bins, width, 1)
    means = jnp.sum(blocks, axis=1) / jnp.maximum(jnp.sum(valid, axis=1), 1)
    maxima = jnp.max(jnp.where(valid, blocks, -jnp.inf), axis=1)
    maxima = jnp.where(jnp.any(valid, axis=1), maxima, 0)
    return jnp.concatenate((means, maxima), axis=-1).reshape(-1)


class SpectrumLinkEncoder(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    embedder: Embedder
    graph: Encoder
    num_wire_features: int = eqx.field(static=True)
    request_cols: int = eqx.field(static=True)

    def __init__(
        self,
        metadata_size,
        request_cols,
        embedding_size,
        intermediate_size,
        num_layers,
        num_heads,
        num_wire_features,
        dropout_rate,
        attention_dropout_rate,
        key,
        ffn_activation="relu2",
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv1d(3, 8, 5, key=k1)
        self.conv2 = eqx.nn.Conv1d(8, 8, 5, key=k2)
        self.embedder = Embedder(16 * 8 * 2 + metadata_size, embedding_size, k3)
        self.graph = Encoder(
            metadata_size,
            embedding_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_wire_features,
            dropout_rate,
            attention_dropout_rate,
            k4,
            preembedded=True,
            ffn_activation=ffn_activation,
        )
        self.num_wire_features = num_wire_features
        self.request_cols = request_cols

    def __call__(self, tokens, params, enable_dropout=False, key=None):
        start, slots = self.num_wire_features, params.link_resources
        status = tokens[:, start : start + slots]
        quality = tokens[:, start + slots : start + 2 * slots]
        frequency = params.slot_centre_freq_array.val
        frequency = frequency / jnp.maximum(jnp.max(jnp.abs(frequency)), 1.0)
        raw = jnp.stack((status, quality, jnp.broadcast_to(frequency, status.shape)), axis=-1)

        def encode_link(x, life):
            x = jax.nn.gelu(band_convolution(self.conv1, x, life))
            return jax.nn.gelu(band_convolution(self.conv2, x, life))

        local = jax.vmap(encode_link)(raw, status)
        metadata = jnp.concatenate((tokens[:, :start], tokens[:, start + 2 * slots :]), axis=-1)
        if self.request_cols:
            metadata = metadata[:, : -self.request_cols]
        summaries = jax.vmap(ordered_spectrum_summary)(local)
        embedded = jax.vmap(self.embedder)(jnp.concatenate((summaries, metadata), axis=-1))
        context = self.graph(metadata, embeddings=embedded, enable_dropout=enable_dropout, key=key)[
            "output"
        ]
        return context, local


class ActorCriticSpectrumTransformer(eqx.Module):
    actor: SpectrumLinkEncoder
    critic: SpectrumLinkEncoder
    actor_context: eqx.nn.Linear
    actor_local: eqx.nn.Linear
    actor_score: eqx.nn.Linear
    critic_mlp: eqx.nn.MLP
    value_query: jax.Array | None
    path_pool: eqx.nn.Linear | None
    num_wire_features: int = eqx.field(static=True)
    critic_pooling: str = eqx.field(static=True)
    actor_pooling: str = eqx.field(static=True)

    def __init__(
        self,
        input_size,
        embedding_size,
        intermediate_size,
        num_slot_actions,
        num_layers,
        num_heads,
        enable_dropout,
        dropout_rate,
        attention_dropout_rate,
        share_layers,
        num_wire_features,
        actor_mlp_width,
        critic_mlp_width,
        actor_mlp_depth,
        critic_mlp_depth,
        num_request_specific_cols,
        key,
        critic_pooling="mean",
        actor_pooling="sum",
        ffn_activation="relu2",
    ):
        if share_layers:
            raise ValueError("The spectrum policy uses independent actor/critic encoders")
        slots = (input_size - num_wire_features - 2 - num_request_specific_cols) // 2
        if num_slot_actions != slots:
            raise ValueError("Spectrum policy requires one action per slot (aggregate_slots=1)")
        keys = jax.random.split(key, 8)
        metadata = num_wire_features + 2 + num_request_specific_cols
        common = (
            embedding_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_wire_features,
            dropout_rate,
            attention_dropout_rate,
        )
        self.actor = SpectrumLinkEncoder(
            metadata, 0, *common, keys[0], ffn_activation=ffn_activation
        )
        self.critic = SpectrumLinkEncoder(
            metadata - num_request_specific_cols,
            num_request_specific_cols,
            *common,
            keys[1],
            ffn_activation=ffn_activation,
        )
        self.actor_context = eqx.nn.Linear(embedding_size, 8, key=keys[2])
        self.actor_local = eqx.nn.Linear(8, 8, key=keys[3])
        self.actor_score = eqx.nn.Linear(8, 1, key=keys[4])
        self.critic_mlp = eqx.nn.MLP(
            embedding_size, 1, critic_mlp_width, critic_mlp_depth, key=keys[5]
        )
        self.value_query = (
            jax.random.normal(keys[6], (embedding_size,)) * 0.02
            if critic_pooling == "attention"
            else None
        )
        self.path_pool = (
            eqx.nn.Linear(3, 1, key=keys[7]) if actor_pooling == "min_mean_max" else None
        )
        self.num_wire_features = num_wire_features
        self.critic_pooling, self.actor_pooling = critic_pooling, actor_pooling

    @property
    def actor_critic(self):
        return self.actor, self.critic

    def __call__(self, state, params, *, enable_dropout=False, key=None):
        tokens = get_obs_transformer(state, params).astype(dtype_config.COMPUTE_DTYPE)
        actor_key, critic_key = jax.random.split(key) if key is not None else (None, None)
        context, local = self.actor(tokens, params, enable_dropout, actor_key)
        value_context, _ = self.critic(tokens, params, enable_dropout, critic_key)
        context = jax.vmap(self.actor_context)(context)
        local = jax.vmap(jax.vmap(self.actor_local))(local)
        hidden = jax.nn.gelu(local + context[:, None, :])
        link_scores = jax.vmap(jax.vmap(self.actor_score))(hidden)[..., 0]
        nodes, _ = read_rsa_request(state.request_array)
        paths = get_paths(params, nodes)
        if self.path_pool is None:
            logits = paths.astype(link_scores.dtype) @ link_scores
        else:
            on_path = paths[:, :, None] > 0
            mean = (paths.astype(link_scores.dtype) @ link_scores) / jnp.maximum(
                paths.sum(axis=1, keepdims=True), 1
            )
            low = jnp.min(jnp.where(on_path, link_scores[None], jnp.inf), axis=1)
            high = jnp.max(jnp.where(on_path, link_scores[None], -jnp.inf), axis=1)
            features = jnp.nan_to_num(jnp.stack((low, mean, high), axis=-1))
            logits = jax.vmap(jax.vmap(self.path_pool))(features)[..., 0]
        logits = logits.reshape(-1)
        if params.include_no_op:
            logits = jnp.concatenate((logits, jnp.array([-1e4], logits.dtype)))
        if self.value_query is None:
            value = jnp.mean(value_context, axis=0)
        else:
            weights = jax.nn.softmax(
                value_context @ self.value_query / math.sqrt(value_context.shape[-1])
            )
            value = jnp.sum(weights[:, None] * value_context, axis=0)
        value = self.critic_mlp(value)[0]
        return distrax.Categorical(logits=logits.astype(dtype_config.PARAMS_DTYPE)), value.astype(
            dtype_config.PARAMS_DTYPE
        )
