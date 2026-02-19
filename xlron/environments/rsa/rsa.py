import math
from typing import Any, Optional, Tuple

import numpy as np
from gymnax.environments import environment, spaces

from xlron import dtype_config
from xlron.environments.dataclasses import (
    ActionInfo,
    EnvParams,
    EnvState,
    RSAEnvParams,
    RSAEnvState,
    RSAMultibandEnvState,
)
from xlron.environments.diff_utils import *
from xlron.environments.env_funcs import (
    calculate_path_stats,
    check_action_rmsa_gn_model,
    check_action_rsa,
    check_action_rwalr,
    complete_step_rmsa_gn_model,
    complete_step_rsa,
    complete_step_rsa_gn_model,
    complete_step_rwalr,
    generate_request_rsa,
    generate_request_rwalr,
    get_affected_slots_mask,
    get_path_and_se,
    get_path_slots,
    implement_action_rmsa_gn_model,
    implement_action_rsa,
    implement_action_rsa_gn_model,
    implement_action_rwalr,
    init_graph_tuple,
    init_link_slot_array,
    init_link_slot_departure_array,
    init_link_slot_mask,
    init_rsa_request_array,
    init_traffic_matrix,
    mask_slots,
    required_slots,
    set_band_gaps,
    update_graph_tuple,
    calculate_throughput_from_active_lightpaths
)
from xlron.environments.wrappers import *

one = jnp.array(1, dtype=dtype_config.LARGE_FLOAT_DTYPE)
zero = jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE)


class RSAEnv(environment.Environment):
    """This environment simulates the Routing Modulation and Spectrum Assignment (RMSA) problem.
    It can model RSA by setting consider_modulation_format=False in params.
    It can model RWA by setting min_bw=0, max_bw=0, and consider_modulation_format=False in params.
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        params: RSAEnvParams,
        traffic_matrix: chex.Array | None = None,
        list_of_requests: chex.Array | None = None,
        laplacian_matrix: chex.Array | None = None,
    ):
        """Initialise the environment state and set as initial state.

        Args:
            key: PRNG key
            params: Environment parameters
            traffic_matrix (optional): Traffic matrix

        Returns:
            None
        """
        super().__init__()
        state = RSAEnvState(
            current_time=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            holding_time=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            arrival_time=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_timesteps=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            total_requests=jnp.array(-1, dtype=dtype_config.LARGE_INT_DTYPE),
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(
                params, include_no_op=params.include_no_op, agg=params.aggregate_slots
            ),
            traffic_matrix=traffic_matrix
            if traffic_matrix is not None
            else init_traffic_matrix(key, params),
            list_of_requests=list_of_requests,
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            accepted_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            valid_mass=jnp.array(1.0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        )
        if params.__class__.__name__ not in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
            self.initial_state = state.replace(
                graph=init_graph_tuple(state, params, laplacian_matrix)
            )
        self._render_figure = None
        self._render_axes = None
        self._render_graph = None
        self._render_pos = None
        self._render_edge_lookup = {}
        self._render_graph_key = None

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: Optional[RSAEnvParams] = None,
    ) -> Tuple[chex.Array, RSAEnvState, float, bool, bool, dict]:
        """Performs step transitions in the environment.

        Args:
            key: PRNG key
            state: Environment state
            action: Action to take (single value array)
            params: Environment parameters

        Returns:
            obs: Observation
            state: New environment state
            reward: Reward
            terminal: True if terminal condition met
            truncated: True if max_requests reached
            info: Additional information
        """
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminal, truncated, info = self.step_env(
            key, state, action, params
        )

        def reset_fn(args):
            key_reset, state_st, params, state = args
            obs_re, state_re = self.reset(key_reset, params, state)
            return obs_re, state_re

        def continue_fn(args):
            _, state_st, _, _ = args
            return obs_st, state_st

        done = jnp.logical_or(terminal, truncated)

        obs, new_state = jax.lax.cond(
            done, reset_fn, continue_fn, (key_reset, state_st, params, state)
        )
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(new_state),
            reward,
            terminal,
            truncated,
            info,
        )

    @partial(
        jax.jit,
        static_argnums=(
            0,
            2,
        ),
    )
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[RSAEnvParams] = None,
        state: Optional[RSAEnvState] = None,
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Performs resetting of environment.

        Args:
            key: PRNG key
            params: Environment parameters

        Returns:
            obs: Observation
            state: Reset environment state
        """
        obs, state = self.reset_env(key, params, state)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: RSAEnvParams,
    ) -> Tuple[chex.Array, RSAEnvState, chex.Array, chex.Array, chex.Array, dict]:
        """Environment-specific step transition.
        1. Implement action
        2. Check if action was valid
            - If valid, calculate reward and finalise action
            - If invalid, calculate reward and undo action
        3. Generate new request, update current time, remove expired requests
        4. Update timesteps
        5. Check for terminal/truncated conditions
        6. if DeepRMSAEnv, calculate path stats
        7. if using GNN policy, update graph tuple

        Args:
            key: PRNG key
            state: Environment state
            action: Action to take (single value array)
            params: Environment parameters

        Returns:
            obs: Observation
            state: New environment state
            reward: Reward
            terminal: True if terminal condition met (e.g., end_first_blocking)
            truncated: True if max_requests reached
            info: Additional information
        """
        # Compute relevant info from action
        action_info = jit_profiler.call(params.profile, self.process_action, state, action, params)

        # Define env-type specific functions
        implement_action = implement_action_rsa
        check_action = check_action_rsa
        complete_step = complete_step_rsa
        generate_request = generate_request_rsa

        if params.__class__.__name__ == "RWALightpathReuseEnvParams":
            implement_action = implement_action_rwalr
            check_action = check_action_rwalr
            complete_step = complete_step_rwalr
            if not params.incremental_loading:
                # These are relevant to dynamic RWA-LR (upcoming)
                complete_step = complete_step_rwalr
                generate_request = generate_request_rwalr
        elif params.__class__.__name__ == "RSAGNModelEnvParams":
            implement_action = implement_action_rsa_gn_model
            check_action = check_action_rsa
            complete_step = complete_step_rsa_gn_model
        elif params.__class__.__name__ == "RMSAGNModelEnvParams":
            implement_action = implement_action_rmsa_gn_model
            check_action = check_action_rmsa_gn_model
            complete_step = complete_step_rmsa_gn_model

        # Implement action
        state = jit_profiler.call(params.profile, implement_action, state, action_info, params)

        # Check action
        check = jit_profiler.call(params.profile, check_action, state, action_info, params)

        # Calculate reward
        reward = jit_profiler.call(
            params.profile, self.calculate_reward, state, action_info, check, params
        )

        # Complete step
        state = jit_profiler.call(params.profile, complete_step, state, action_info, check, params)

        # TODO (DYNAMIC-RWALR) - calculate allocated bandwidth
        # TODO (DYNAMIC-RWALR) - generate new request if allocated DR equals requested DR, else update requested DR do not advance time do not replace source-dest
        # TODO (AFTERSTATE) - write separate functions for deterministic transition (above) and stochastic transition (below)
        state = jit_profiler.call(params.profile, generate_request, key, state, params)

        # Terminate if max_requests exceeded or, if consecutive loading,
        # then terminate if reward is failure but not before min number of timesteps before update
        terminal = self.is_terminal(state, params, action_info, reward)
        truncated = self.is_truncated(state, params)

        info = {}

        # Calculate path stats if DeepRMSAEnv
        if params.__class__.__name__ == "DeepRMSAEnvParams":
            path_stats = jit_profiler.call(
                params.profile, calculate_path_stats, state, params, state.request_array
            )
            state = state.replace(path_stats=path_stats)
        # Update graph tuple
        elif params.use_gnn:
            state = update_graph_tuple(state, params)

        # Get observation
        obs = jit_profiler.call(params.profile, self.get_obs, state, params)

        # Stash metrics so they survive the auto-reset in step().
        # The outer step() may replace state with a reset state (accepted=0,
        # empty link_slot_array, etc.), so LogWrapper reads these instead.
        info["_accepted_services"] = state.accepted_services
        info["_accepted_bitrate"] = state.accepted_bitrate
        info["_total_bitrate"] = state.total_bitrate
        info["_utilisation"] = jnp.count_nonzero(state.link_slot_array) / state.link_slot_array.size
        if hasattr(state, "throughput"):
            throughput = calculate_throughput_from_active_lightpaths(state, params)
            state = state.replace(throughput=throughput)
            info["_throughput"] = state.throughput

        return obs, state, reward, terminal, truncated, info

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        return np.asarray(jax.device_get(arr))

    def _get_render_graph(self, params: RSAEnvParams):
        import networkx as nx

        edges = self._to_numpy(params.edges.val).astype(int)
        graph_key = (
            params.num_nodes,
            params.num_links,
            bool(params.directed_graph),
            hash(edges.tobytes()),
        )
        if self._render_graph is not None and self._render_graph_key == graph_key:
            return self._render_graph, self._render_pos, self._render_edge_lookup

        lengths = self._to_numpy(params.link_length_array.val).astype(float).reshape(-1)
        g = nx.Graph()
        min_node = int(np.min(edges))
        max_node = int(np.max(edges))
        if min_node == 1 and max_node == params.num_nodes:
            g.add_nodes_from(range(1, params.num_nodes + 1))
        else:
            g.add_nodes_from(range(params.num_nodes))
        edge_lookup = {}
        for idx, (u, v) in enumerate(edges):
            u_int, v_int = int(u), int(v)
            w = float(lengths[idx]) if idx < lengths.shape[0] else 1.0
            key = tuple(sorted((u_int, v_int)))
            edge_lookup.setdefault(key, []).append(idx)
            if g.has_edge(u_int, v_int):
                if w < g[u_int][v_int]["weight"]:
                    g[u_int][v_int]["weight"] = w
                    g[u_int][v_int]["inverse_weight"] = 1.0 / max(w, 1e-6)
            else:
                g.add_edge(
                    u_int,
                    v_int,
                    weight=w,
                    inverse_weight=1.0 / max(w, 1e-6),
                )
        initial_pos = nx.circular_layout(g)
        pos = nx.spring_layout(
            g,
            k=0.12,
            iterations=100,
            pos=initial_pos,
            weight="inverse_weight",
            seed=7,
        )
        self._render_graph = g
        self._render_pos = pos
        self._render_edge_lookup = edge_lookup
        self._render_graph_key = graph_key
        return g, pos, edge_lookup

    def _get_slot_rgba(self, state: RSAEnvState, params: RSAEnvParams) -> np.ndarray:
        try:
            from matplotlib import pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for RSAEnv.render()") from exc

        slots = self._to_numpy(state.link_slot_array).astype(float)
        dep = self._to_numpy(state.link_slot_departure_array).astype(float)
        current_time = float(np.asarray(jax.device_get(state.current_time)))
        mean_holding = max(float(params.mean_service_holding_time), 1e-6)

        num_links, num_slots = slots.shape
        rgba = np.ones((num_links, num_slots, 4), dtype=float)
        rgba[..., :3] = 0.98
        rgba[..., 3] = 1.0

        gaps = slots < 0.0
        rgba[gaps, :3] = 0.15
        rgba[gaps, 3] = 0.95

        occupied = slots > 0.0
        remaining = np.where(dep > current_time, dep - current_time, dep)
        remaining = np.clip(remaining / mean_holding, 0.0, 1.0)
        palette = np.asarray(plt.get_cmap("tab20").colors)
        color_idx = np.floor(np.abs(dep) * 1000.0).astype(int) % len(palette)

        rgba[occupied, :3] = palette[color_idx[occupied]]
        # Lower alpha when closer to departure (more transparent), but not too faint.
        rgba[occupied, 3] = 0.35 + 0.60 * remaining[occupied]
        return rgba

    def render(
        self,
        state: RSAEnvState,
        params: Optional[RSAEnvParams] = None,
        mode: str = "human",
        prev_state: Optional[RSAEnvState] = None,
        action_info: Optional[ActionInfo] = None,
        check: Optional[chex.Array] = None,
        info: Optional[dict[str, Any]] = None,  # Reserved for future optional overlays.
    ):
        """Render RSA state as a link-slot heatmap plus a topology view."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError("matplotlib and networkx are required for RSAEnv.render()") from exc

        if params is None:
            raise ValueError("render() requires `params` for topology and normalization context.")

        if self._render_figure is None:
            self._render_figure = plt.figure(figsize=(16, 10))
            gs = self._render_figure.add_gridspec(
                2, 2, width_ratios=[3.0, 1.7], height_ratios=[8.0, 2.0], wspace=0.15, hspace=0.10
            )
            self._render_axes = (
                self._render_figure.add_subplot(gs[0, 0]),
                self._render_figure.add_subplot(gs[0, 1]),
                self._render_figure.add_subplot(gs[1, :]),
            )

        ax_matrix, ax_graph, ax_stats = self._render_axes
        ax_matrix.clear()
        ax_graph.clear()
        ax_stats.clear()

        rgba = self._get_slot_rgba(state, params)
        ax_matrix.imshow(rgba, aspect="auto", interpolation="nearest", origin="lower")
        ax_matrix.set_title("Link Slot State (color=request, alpha=remaining time)")
        ax_matrix.set_xlabel("Slot Index")
        ax_matrix.set_ylabel("Link Index")

        highlight_mask = None
        accepted = False
        if check is not None:
            accepted = float(np.asarray(jax.device_get(check))) < 0.5
        elif prev_state is not None:
            accepted = int(self._to_numpy(state.accepted_services)) > int(
                self._to_numpy(prev_state.accepted_services)
            )

        if accepted and action_info is not None:
            highlight_mask = self._to_numpy(action_info.affected_slots_mask).astype(float)
        elif accepted and prev_state is not None:
            prev_slots = self._to_numpy(prev_state.link_slot_array).astype(float)
            curr_slots = self._to_numpy(state.link_slot_array).astype(float)
            prev_dep = self._to_numpy(prev_state.link_slot_departure_array).astype(float)
            curr_dep = self._to_numpy(state.link_slot_departure_array).astype(float)
            highlight_mask = (
                ((curr_slots > 0.0) & (prev_slots <= 0.0)) | (curr_dep > (prev_dep + 1e-9))
            ).astype(float)
        if highlight_mask is not None:
            y_idx, x_idx = np.where(highlight_mask > 0.5)
            if x_idx.size > 0:
                ax_matrix.scatter(
                    x_idx, y_idx, marker="x", s=28, c="#111111", linewidths=0.8, alpha=0.95
                )

        g, pos, edge_lookup = self._get_render_graph(params)
        nx.draw_networkx_edges(g, pos, edge_color="#9CA3AF", width=1.6, ax=ax_graph)
        nx.draw_networkx_nodes(
            g,
            pos,
            node_size=430,
            node_color="#ffffff",
            edgecolors="#111827",
            linewidths=1.5,
            ax=ax_graph,
        )
        nx.draw_networkx_labels(g, pos, font_size=9, font_weight="bold", ax=ax_graph)

        highlight_color = "#F97316"
        if highlight_mask is not None:
            selected = np.where(highlight_mask > 0.5)
            if selected[0].size > 0:
                y0, x0 = int(selected[0][0]), int(selected[1][0])
                rgb = rgba[y0, x0, :3]
                highlight_color = tuple(float(c) for c in rgb.tolist())

        highlighted_edges = set()
        if accepted and action_info is not None:
            path = self._to_numpy(action_info.path).astype(int)
            valid_path = [int(node) for node in path if int(node) >= 0]
            for i in range(len(valid_path) - 1):
                key = tuple(sorted((valid_path[i], valid_path[i + 1])))
                if key in edge_lookup:
                    highlighted_edges.add(key)
        if not highlighted_edges and highlight_mask is not None:
            for link_idx in np.where(np.any(highlight_mask > 0.5, axis=1))[0].tolist():
                if link_idx < self._to_numpy(params.edges.val).shape[0]:
                    u, v = self._to_numpy(params.edges.val)[link_idx].astype(int).tolist()
                    highlighted_edges.add(tuple(sorted((int(u), int(v)))))

        if highlighted_edges:
            nx.draw_networkx_edges(
                g,
                pos,
                edgelist=list(highlighted_edges),
                edge_color=highlight_color,
                width=4.2,
                ax=ax_graph,
            )
        ax_graph.set_title("Topology (latest accepted request highlighted)")
        ax_graph.set_axis_off()

        req = self._to_numpy(state.request_array).astype(float).reshape(-1)
        src = int(req[0]) if req.size > 0 else -1
        dst = int(req[1]) if req.size > 1 else -1
        bit_rate = float(req[2]) if req.size > 2 else 0.0

        total_requests = max(int(np.asarray(jax.device_get(state.total_requests))), 0)
        accepted_services = int(np.asarray(jax.device_get(state.accepted_services)))
        accepted_bitrate = float(np.asarray(jax.device_get(state.accepted_bitrate)))
        total_bitrate = float(np.asarray(jax.device_get(state.total_bitrate)))
        service_bp = 1.0 - (accepted_services / max(total_requests, 1))
        bitrate_bp = 1.0 - (accepted_bitrate / max(total_bitrate, 1e-6))
        util = float(np.count_nonzero(self._to_numpy(state.link_slot_array)) / state.link_slot_array.size)

        lines = [
            f"REQUEST src={src} dst={dst} bitrate={bit_rate:.2f}",
            f"STEP {int(np.asarray(jax.device_get(state.total_timesteps)))}  TIME {float(np.asarray(jax.device_get(state.current_time))):.3f}",
            f"ACCEPTED {accepted_services} / {total_requests}",
            f"SERVICE BLOCKING {service_bp:.4f}",
            f"BITRATE BLOCKING {bitrate_bp:.4f}",
            f"UTILISATION {util:.4f}",
        ]
        if hasattr(state, "link_capacity_array"):
            cap = self._to_numpy(state.link_capacity_array)
            lines.append(f"LIGHTPATH CAPACITY (mean) {float(np.mean(cap)):.3f}")
        if check is not None:
            status = "ACCEPTED" if accepted else "BLOCKED"
            lines.append(f"LATEST ACTION {status}")

        ax_stats.set_axis_off()
        ax_stats.text(
            0.01,
            0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=12,
            color="#22C55E",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#0A0A0A", edgecolor="#22C55E", alpha=0.95),
            transform=ax_stats.transAxes,
        )
        self._render_figure.suptitle("XLRON RSA Environment Render", fontsize=14, fontweight="bold")
        self._render_figure.canvas.draw_idle()

        if mode == "human":
            plt.pause(0.001)
            return None
        if mode == "rgb_array":
            self._render_figure.canvas.draw()
            width, height = self._render_figure.canvas.get_width_height()
            rgb = np.frombuffer(self._render_figure.canvas.tostring_rgb(), dtype=np.uint8)
            return rgb.reshape((height, width, 3))
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        if self._render_figure is not None:
            from matplotlib import pyplot as plt

            plt.close(self._render_figure)
            self._render_figure = None
            self._render_axes = None

    @partial(jax.jit, static_argnums=(0, 2))
    def reset_env(
        self,
        key: chex.PRNGKey,
        params: RSAEnvParams,
        state: Optional[RSAEnvState] = None,
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Environment-specific reset.
        Generates new random traffic matrix if random_traffic is True, otherwise uses the provided traffic matrix.
        Generates new request.

        Args:
            key: PRNG key
            params: Environment parameters

        Returns:
            obs: Observation
            state: Reset environment state
        """
        # if params.multiple_topologies:
        # TODO - implement this (shuffle through topologies and use the top of the stack)
        # Question - do i need to rewrite every function to take in a params argument and index params[0]?
        # maybe in make_rsa_env function can have a params field that holds all the params (one for each topology),
        # and then cycle select from them randomly and replace the top-level params with the selected one.
        # Then need to init() the env again in order to update the state using the params
        #    raise NotImplementedError
        if params.random_traffic:
            key, key_traffic = jax.random.split(key)
            state = self.initial_state.replace(
                traffic_matrix=init_traffic_matrix(key_traffic, params)
            )
        else:
            state = self.initial_state
        state = generate_request_rsa(key, state, params)
        return self.get_obs(state, params), state

    def process_action(
        self,
        state: EnvState,
        action: Array,
        params: EnvParams,
    ) -> ActionInfo:
        """Processes action into relevant information."""
        nodes_sd, requested_datarate = jit_profiler.call(
            params.profile, read_rsa_request, state.request_array
        )
        # For GN model envs, action is [path_slot_action, launch_power].
        # Decompose into scalar path_action and power_action.
        action_1d = jnp.atleast_1d(action)
        path_action = action_1d[0]
        power_action = action_1d[1] if action_1d.shape[0] > 1 else jnp.float32(0.0)
        path_index, initial_slot_index = jit_profiler.call(
            params.profile, process_path_action, state, params, path_action
        )
        path, path_se = jit_profiler.call(
            params.profile, get_path_and_se, params, nodes_sd, path_index
        )

        # For RMSA GN model, use the SE from the selected modulation format (from masking)
        # rather than the static path_se_array value
        if params.__class__.__name__ == "RMSAGNModelEnvParams":
            path_action_int = path_action.astype(dtype_config.LARGE_INT_DTYPE)
            mod_format_index = jax.lax.dynamic_slice(
                state.mod_format_mask, (path_action_int,), (1,)
            )[0].astype(dtype_config.LARGE_INT_DTYPE)
            path_se = params.modulations_array.val[mod_format_index, 1]
        else:
            path_se = path_se if params.consider_modulation_format else one

        num_slots = jit_profiler.call(
            params.profile,
            required_slots,
            requested_datarate,
            path_se,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
        affected_slots_mask = jit_profiler.call(
            params.profile,
            get_affected_slots_mask,
            initial_slot_index,
            num_slots,
            path,
            params,
        )
        action_info = ActionInfo(
            action=path_action,
            path_index=path_index,
            initial_slot_index=initial_slot_index,
            num_slots=num_slots,
            path=path,
            se=path_se,
            requested_datarate=requested_datarate,
            nodes_sd=nodes_sd,
            affected_slots_mask=affected_slots_mask,
            power_action=power_action,
        )
        return action_info

    @partial(jax.jit, static_argnums=(0, 2))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> Array:
        """Returns mask of valid actions.
        1. Check request for source and destination nodes
        2. For each path, mask out (0) initial slots that are not valid
        See mask_slots function for more details.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            state: Environment state with action mask
        """
        action_mask = jit_profiler.call(params.profile, mask_slots, state, params)
        return action_mask

    @partial(jax.jit, static_argnums=(0, 2))
    def get_obs_unflat(
        self, state: RSAEnvState, params: RSAEnvParams
    ) -> Tuple[chex.Array, chex.Array]:
        """Retrieves observation from state.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            obs: Observation (request array, link slot array)
        """
        return (
            state.request_array,
            state.link_slot_array,
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def get_obs(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Retrieves observation from state and reshapes into single array.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            obs: Observation (flattened request array and link slot array)
        """
        return jnp.concatenate(
            (
                jnp.reshape(state.request_array, (-1,)),
                jnp.reshape(state.link_slot_array, (-1,)),
            ),
            axis=0,
        )

    def is_terminal(
        self,
        state: RSAEnvState,
        params: RSAEnvParams,
        action_info: ActionInfo,
        reward: chex.Array | None = None,
    ) -> chex.Array:
        """Check whether state transition is terminal.

        Args:
            state: Environment state
            params: Environment parameters
            reward: Reward from current step (needed for end_first_blocking check)

        Returns:
            done: Boolean termination flag
        """
        if params.end_first_blocking:
            return jnp.array(reward == self.get_reward_failure(state, action_info, params))
        elif params.terminate_on_episode_end:
            return self.is_truncated(state, params)
        else:
            return jnp.array(False)

    def is_truncated(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Check whether state transition is truncated i.e. max steps reached.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            done: Boolean termination flag
        """
        if params.continuous_operation:
            return jnp.array(False)
        else:
            return jnp.array(state.total_requests >= params.max_requests)

    @staticmethod
    def add_integer_bonus(action, scale=0.0):
        """
        Adds a small reward bonus that peaks at integer values.

        Args:
            action: The action(s) to evaluate
            scale: How strongly to encourage integer values

        Returns:
            A small reward bonus that peaks at integer values
        """
        return scale * jnp.cos(2 * jnp.pi * action)

    @staticmethod
    def penalise_non_integer_action(action, params, scale=0.0):
        """
        Penalises non-integer actions.

        Args:
            action: The action(s) to evaluate
            scale: How strongly to penalise non-integer values

        Returns:
            A penalty for non-integer actions
        """
        return -scale * jnp.abs(
            action
            - differentiable_round_simple(
                action,
                temperature=params.temperature,
                differentiable=params.differentiable,
            )
        )

    def calculate_reward(
        self, state: RSAEnvState, action_info: ActionInfo, check: Array, params: RSAEnvParams
    ) -> chex.Array:
        """Calculate reward for current state and action.

        Args:
            state: Environment state
            action_info: Processed action information
            check: Result of action validity check
            params: Environment parameters

        Returns:
            reward: Calculated reward
        """
        reward = self.get_reward_failure(state, action_info, params) * check + (
            (1 - check) * self.get_reward_success(state, action_info, params)
        )
        return reward

    def get_reward_failure(
        self,
        state: Optional[EnvState] = None,
        action_info: Optional[ActionInfo] = None,
        params: Optional[EnvParams] = None,
    ) -> chex.Array:
        """Return reward for current state.

        Args:
            state (optional): Environment state

        Returns:
            reward: Reward for failure
        """
        reward = -one
        if params.reward_type == "service":
            pass
        elif params.reward_type == "bitrate":
            reward = (
                differentiable_index(
                    state.request_array,
                    1,
                    temperature=params.temperature,
                    differentiable=params.differentiable,
                )
                * reward
                / jnp.max(params.values_bw.val)
            )
        else:
            reward = (
                reward
                * differentiable_index(
                    read_rsa_request(state.request_array),
                    1,
                    temperature=params.temperature,
                    differentiable=params.differentiable,
                )
                / jnp.max(params.values_bw.val)
                if params.maximise_throughput
                else reward
            )
        return reward

    def get_reward_success(
        self,
        state: Optional[EnvState] = None,
        action_info: Optional[ActionInfo] = None,
        params: Optional[EnvParams] = None,
    ) -> chex.Array:
        """Return reward for current state.

        Args:
            state: (optional) Environment state

        Returns:
            reward: Reward for success
        """
        reward = zero

        if params.reward_type != "service":
            reward = state.request_array[1] * reward / jnp.max(params.values_bw.val)
            if params.reward_type == "bitrate":
                pass  # No additional calculation needed
            elif params.reward_type == "snr":
                # SNR calculation...
                assert params.__class__.__name__ == "RSAGNModelEnvParams"
                path_snr = get_snr_for_path(action_info.path, state.link_snr_array, params)[
                    action_info.initial_slot_index.astype(dtype_config.LAREG_INT_DTYPE)
                ]
                # set to 0 if negative and divide by large SNR (e.g. 50. dB) to scale below 1
                # N.B. negative SNR in dB would be a fail anyway since min. required is 10dB
                path_snr_norm = jnp.where(path_snr < zero, zero, path_snr) / params.max_snr
                return reward + path_snr_norm
            elif params.reward_type == "mod_format":
                # Modulation format calculation...
                assert params.__class__.__name__ == "RSAGNModelEnvParams"
                mod_format_index = get_path_slots(
                    state.modulation_format_index_array,
                    params,
                    action_info.nodes_sd,
                    action_info.path_index,
                    agg_func="max",
                )[action_info.initial_slot_index.astype(dtype_config.LARGE_INT_DTYPE)]
                return reward + 0.05 * (one + mod_format_index)
            else:
                return reward
        else:
            reward = reward + self.penalise_non_integer_action(
                action_info.action, params
            )  # + self.add_integer_bonus(action)

        return reward

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @staticmethod
    def num_actions(params: EnvParams) -> int:
        """Number of actions possible in environment."""
        return math.ceil(params.link_resources / params.aggregate_slots) * params.k_paths

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions(params))

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(
            3  # Request array
            + params.num_links * params.link_resources  # Link slot array
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
                "current_time": spaces.Discrete(1),
                "request_array": spaces.Discrete(3),
                "link_slot_array": spaces.Discrete(params.num_links * params.link_resources),
                "link_slot_departure_array": spaces.Discrete(
                    params.num_links * params.link_resources
                ),
            }
        )


class RSAMultibandEnv(RSAEnv):
    def __init__(
        self,
        key: chex.PRNGKey,
        params: RSAEnvParams,
        traffic_matrix: chex.Array | None = None,
        list_of_requests: chex.Array | None = None,
        laplacian_matrix: chex.Array | None = None,
    ):
        super().__init__(
            key,
            params,
            traffic_matrix=traffic_matrix,
            list_of_requests=list_of_requests,
            laplacian_matrix=laplacian_matrix,
        )
        state = RSAMultibandEnvState(
            current_time=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            holding_time=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            arrival_time=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            total_timesteps=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            total_requests=jnp.array(-1, dtype=dtype_config.LARGE_INT_DTYPE),
            link_slot_array=set_band_gaps(init_link_slot_array(params), params, -1.0),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(
                params, include_no_op=params.include_no_op, agg=params.aggregate_slots
            ),
            traffic_matrix=traffic_matrix
            if traffic_matrix is not None
            else init_traffic_matrix(key, params),
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            accepted_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            list_of_requests=list_of_requests,
            valid_mass=jnp.array(1.0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))
