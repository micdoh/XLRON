from typing import Callable, Tuple, Union

import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn import MultiheadAttention
from jaxtyping import (  # https://github.com/google/jaxtyping
    Array,
    Float,
    PRNGKeyArray,
)
from xlron import dtype_config
from xlron.environments.dataclasses import EnvState, EnvParams
from xlron.environments.env_funcs import (
    get_obs_transformer, 
    read_rsa_request, 
    get_path_slots,
)


class Embedder(eqx.Module):
    linear: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm

    def __init__(self, in_size: int, out_size: int, key: chex.PRNGKey):
        self.linear = eqx.nn.Linear(in_size, out_size, key=key)
        self.layernorm = eqx.nn.LayerNorm(shape=out_size)

    def __call__(self, x: Array) -> Array:
        return self.layernorm(jax.nn.gelu(self.linear(x)))


class WIRE(eqx.Module):
    """Wavelet-Induced Rotary Encodings for graphs.
    https://openreview.net/pdf?id=f7BvsdILYx

    Projects m-dimensional node features (e.g., RWSE, spectral coords)
    to rotation angles for RoPE-style positional encoding.
    """

    freq_proj: eqx.nn.Linear  # (m,) -> (embedding_size // 2,)
    embedding_size: int = eqx.field(static=True)

    def __init__(
        self,
        num_features: int,
        embedding_size: int,
        key: PRNGKeyArray,
        freq_scale: float = 0.01,
    ):
        """
        Args:
            num_features: Dimension of input position features (m)
            embedding_size: Dimension of queries/keys to rotate (must be even)
            key: PRNG key
            freq_scale: Scale for frequency initialisation
        """
        if embedding_size % 2 != 0:
            raise ValueError("embedding_size must be even")

        self.embedding_size = embedding_size

        # Project position features to angles
        # Output dim is embedding_size // 2 (one angle per 2D rotation block)
        self.freq_proj = eqx.nn.Linear(
            in_features=num_features,
            out_features=embedding_size // 2,
            use_bias=False,
            key=key,
        )

        # Optionally scale down initial frequencies for stability
        scaled_weight = self.freq_proj.weight * freq_scale
        self.freq_proj = eqx.tree_at(lambda layer: layer.weight, self.freq_proj, scaled_weight)

    def get_angles(
        self, positions: Float[Array, "num_nodes num_features"]
    ) -> Float[Array, "num_nodes half_emb"]:
        """Compute rotation angles from position features."""
        return jax.vmap(self.freq_proj)(positions)

    def rotate(
        self,
        x: Float[Array, "num_nodes embedding_size"],
        angles: Float[Array, "num_nodes half_emb"],
    ) -> Float[Array, "num_nodes embedding_size"]:
        """Apply rotary encoding to queries or keys.

        For each 2D block [x_{2i}, x_{2i+1}], rotate by angle theta_i:
            x_{2i}'   = x_{2i} * cos(theta) - x_{2i+1} * sin(theta)
            x_{2i+1}' = x_{2i} * sin(theta) + x_{2i+1} * cos(theta)
        """
        # angles: (num_nodes, embedding_size // 2)
        cos_angles = jnp.cos(angles)
        sin_angles = jnp.sin(angles)

        # Repeat for pairs: [cos_0, cos_0, cos_1, cos_1, ...]
        cos_angles = jnp.repeat(cos_angles, 2, axis=-1)
        sin_angles = jnp.repeat(sin_angles, 2, axis=-1)

        # Rotate pairs: for indices [0,1], [2,3], etc.
        # x_rotated = x * cos + rotate_pairs(x) * sin
        # where rotate_pairs swaps and negates: [x0, x1] -> [-x1, x0]
        x_pairs = x.reshape(x.shape[0], -1, 2)  # (num_nodes, num_pairs, 2)
        x_rotated_pairs = jnp.stack([-x_pairs[..., 1], x_pairs[..., 0]], axis=-1)
        x_rotated = x_rotated_pairs.reshape(x.shape)

        return x * cos_angles + x_rotated * sin_angles

    def __call__(
        self,
        queries: Float[Array, "num_nodes embedding_size"],
        keys: Float[Array, "num_nodes embedding_size"],
        positions: Float[Array, "num_nodes num_features"],
    ) -> tuple[
        Float[Array, "num_nodes embedding_size"],
        Float[Array, "num_nodes embedding_size"],
    ]:
        """Apply WIRE to queries and keys.

        Args:
            queries: Query vectors (num_nodes, embedding_size)
            keys: Key vectors (num_nodes, embedding_size)
            positions: Node position features, e.g., RWSE (num_nodes, num_features)

        Returns:
            Rotated (queries, keys)
        """
        angles = self.get_angles(positions)
        return self.rotate(queries, angles), self.rotate(keys, angles)


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: chex.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_size,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=embedding_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Array,
        enable_dropout: bool = False,
        attn_mask: Array | None = None,
        process_heads: Callable | None = None,
        key: chex.PRNGKey | None = None,
    ) -> Array:
        attention_key, dropout_key = (None, None) if key is None else jax.random.split(key)

        norm_input = jax.vmap(self.layernorm)(inputs)
        attention_output = self.attention(
            query=norm_input,
            key_=norm_input,
            value=norm_input,
            mask=attn_mask,
            inference=not enable_dropout,
            key=attention_key,
            process_heads=process_heads,
        )
        result = self.dropout(attention_output, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        return result


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    linear: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        embedding_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: chex.PRNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.linear = eqx.nn.Linear(
            in_features=embedding_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=embedding_size, key=output_key
        )
        self.layernorm = eqx.nn.LayerNorm(shape=embedding_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Array,
        enable_dropout: bool = True,
        key: chex.PRNGKey | None = None,
    ) -> Array:
        hidden = self.layernorm(inputs)

        # Feed-forward.
        hidden = self.linear(hidden)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual
        output += inputs

        return output


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        embedding_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: chex.PRNGKey,
        custom_bias: bool = False,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            embedding_size=embedding_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            embedding_size=embedding_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: Array,
        *,
        enable_dropout: bool = False,
        attn_mask: Array | None = None,
        key: chex.PRNGKey | None = None,
        process_heads: Callable | None = None,
    ) -> Array:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)

        attn_result = self.attention_block(
            inputs,
            enable_dropout=enable_dropout,
            attn_mask=attn_mask,
            key=attn_key,
            process_heads=process_heads,
        )
        # attention_block returns Array when return_attention=False (the default)
        attention_output = attn_result if isinstance(attn_result, Array) else attn_result[0]

        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output


class Encoder(eqx.Module):
    embedder_block: Embedder | eqx.nn.Identity
    layers: list[TransformerLayer]
    wire: WIRE
    num_wire_features: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_wire_features: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: chex.PRNGKey,
    ):
        embedder_key, layer_key, wire_key = jax.random.split(key, 3)
        self.num_layers = num_layers
        self.embedder_block = Embedder(
            in_size=input_size,
            out_size=embedding_size,
            key=embedder_key,
        )
        head_dim = embedding_size // num_heads
        self.wire = WIRE(num_wire_features, head_dim, wire_key)
        self.num_wire_features = num_wire_features

        layer_keys = jax.random.split(layer_key, num=num_layers)

        @eqx.filter_vmap
        def make_layer(key: chex.PRNGKey) -> TransformerLayer:
            return TransformerLayer(
                embedding_size=embedding_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                key=key,
            )

        self.layers = make_layer(layer_keys)

    def __call__(
        self,
        inputs: Array,
        *,
        enable_dropout: bool = False,
        attn_mask: Array | None = None,
        key: chex.PRNGKey | None = None,
    ) -> dict[str, Array]:
        # Index the inputs to retrieve the features used by WiRE (either spectral or random walks)
        wire_features = inputs[:, : self.num_wire_features]

        def apply_wire(q, k):
            return self.wire(q, k, wire_features)

        def process_heads(
            query_heads: Float[Array, "seq_length num_heads qk_size"],
            key_heads: Float[Array, "seq_length num_heads qk_size"],
            value_heads: Float[Array, "seq_length num_heads vo_size"],
        ) -> tuple[
            Float[Array, "seq_length num_heads qk_size"],
            Float[Array, "seq_length num_heads qk_size"],
            Float[Array, "seq_length num_heads vo_size"],
        ]:
            # vmapping over heads
            query_heads, key_heads = jax.vmap(apply_wire, in_axes=(1, 1), out_axes=(1, 1))(
                query_heads, key_heads
            )
            return query_heads, key_heads, value_heads

        embeddings = jax.vmap(self.embedder_block)(inputs)

        layer_outputs: Array | list[Array | None] = []

        # Instead of a Python for loop, use jax.lax.scan
        dynamic_layers, static_layers = eqx.partition(self.layers, eqx.is_array)

        def apply_layer(
            carry: tuple[Array, chex.PRNGKey | None], inputs: tuple[Array, Array, Array]
        ) -> tuple[tuple[Array, chex.PRNGKey | None], Array]:
            x, l_key = carry
            dynamic_layer = inputs

            # Split key for this layer
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)

            # Combine dynamic and static parts to reconstruct the layer
            layer = eqx.combine(dynamic_layer, static_layers)

            # Apply the layer
            x = layer(
                x,
                enable_dropout=enable_dropout,
                attn_mask=attn_mask,
                key=cl_key,
                process_heads=process_heads,
            )

            # Return the layer output in the scan output (not carry)
            return (x, l_key), x

        # Initial carry state (removed layer_outputs from carry)
        initial_carry = (embeddings, key)

        # Scan inputs: dynamic layers and layer-specific attention weights/biases
        scan_inputs = dynamic_layers

        (output, key), layer_outputs = jax.lax.scan(apply_layer, initial_carry, scan_inputs)

        return {"output": output, "layers": layer_outputs}


class ActorCriticTransformer(eqx.Module):
    actor_critic: eqx.nn.Shared | Tuple[eqx.Module, eqx.Module]
    actor_mlp: eqx.nn.MLP
    critic_mlp: eqx.nn.MLP
    share_layers: bool
    num_slot_actions: int

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        intermediate_size: int,
        num_slot_actions: int,
        num_layers: int,
        num_heads: int,
        enable_dropout: bool,
        dropout_rate: float,
        attention_dropout_rate: float,
        share_layers: bool,
        num_wire_features: int,
        actor_mlp_width: int,
        critic_mlp_width: int,
        actor_mlp_depth: int,
        critic_mlp_depth: int,
        key: chex.PRNGKey,
    ):
        (
            encoder_key,
            actor_key,
            critic_key,
        ) = jax.random.split(key, 3)
        self.share_layers = share_layers
        actor = Encoder(
            input_size=input_size,
            intermediate_size=intermediate_size,
            embedding_size=embedding_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_wire_features=num_wire_features,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=encoder_key,
        )
        critic = Encoder(
            input_size=input_size,
            intermediate_size=intermediate_size,
            embedding_size=embedding_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_wire_features=num_wire_features,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=encoder_key,
        )
        if self.share_layers:
            # When sharing layers, use the same encoder for both actor and critic
            self.actor_critic = (actor, actor)
        else:
            self.actor_critic = (actor, critic)
        self.actor_mlp = eqx.nn.MLP(
            in_size=embedding_size,
            width_size=actor_mlp_width,
            out_size=num_slot_actions,
            depth=actor_mlp_depth,
            key=actor_key,
        )
        self.critic_mlp = eqx.nn.MLP(
            in_size=embedding_size,
            width_size=critic_mlp_width,
            out_size=1,
            depth=critic_mlp_depth,
            key=critic_key,
        )
        self.num_slot_actions = num_slot_actions

    def __call__(
        self,
        state: EnvState,
        params: EnvParams,
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Tuple[distrax.Categorical, Array]:
        """Forward pass through the actor-critic transformer.

        Args:
            tokens: Input tokens of shape (num_links, num_wire_features + link_resources)
                   First num_wire_features columns are WiRE positional encodings
            enable_dropout: Whether to enable dropout
            key: PRNG key for dropout

        Returns:
            Tuple of (action_distribution, value)
        """
        actor, critic = self.actor_critic
        actor_key, critic_key = jax.random.split(key) if key is not None else (None, None)
        tokens = get_obs_transformer(state, params)
        
        action_tokens = actor(
            tokens,
            enable_dropout=enable_dropout,
            key=actor_key,
        )["output"]
        
        tokens_for_critic = tokens[: , :-1] # Trim tokens that are relevant to current request
        value_tokens = critic(
            tokens_for_critic,
            enable_dropout=enable_dropout,
            key=critic_key,
        )["output"]
        
        # Project action tokens back to original edge size
        action_tokens = jax.vmap(self.actor_mlp)(action_tokens)
        
        # POOLING - sum edges per path
        nodes_sd, requested_bw = read_rsa_request(state.request_array)
        init_action_array = jnp.zeros(
            params.k_paths * self.num_slot_actions, dtype=dtype_config.SMALL_FLOAT_DTYPE
        )

        def get_path_action_dist(i, action_array):
            path_features = get_path_slots(action_tokens, params, nodes_sd, i, agg_func="sum")
            action_array = jax.lax.dynamic_update_slice(
                action_array, path_features, (i * self.num_slot_actions,)
            )
            return action_array

        path_action_logits = jax.lax.fori_loop(
            0, params.k_paths, get_path_action_dist, init_action_array
        )
        action_logits = jnp.reshape(path_action_logits, (-1,))
        action_dist = distrax.Categorical(logits=action_logits)
        
        # Pool and Value
        value_tokens_mean = jnp.mean(value_tokens, axis=0)
        value = self.critic_mlp(value_tokens_mean).squeeze()

        return action_dist, value

    def sample_action(
        self,
        seed: chex.PRNGKey,
        dist: distrax.Categorical,
        log_prob: bool = False,
        deterministic: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Sample an action from the distribution"""
        action = jnp.argmax(dist.probs()) if deterministic else dist.sample(seed=seed)
        if log_prob:
            return action, dist.log_prob(action)
        return action
