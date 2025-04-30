import absl
from box import Box
from xlron.environments.env_funcs import (
    init_path_link_array,
    convert_node_probs_to_traffic_matrix, make_graph, init_path_length_array, init_modulations_array,
    init_path_se_array, required_slots, init_values_bandwidth, normalise_traffic_matrix, init_link_length_array,
    init_path_capacity_array, pad_array, init_link_length_array_gn_model, generate_source_dest_pairs,
    init_list_of_requests,
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *
from xlron.environments import *
from xlron.environments.gn_model.isrs_gn_model import from_dbm


def process_config(config: Optional[Union[dict, absl.flags.FlagValues]], **kwargs) -> dict:
    """Allow configuration to be a dict, absl.flags.FlagValues, or kwargs.
    Return a Box that can be indexed like a dict or accessed like an object."""
    config = config or {}
    # Allow config to be a dict or absl.flags.FlagValues
    if isinstance(config, absl.flags.FlagValues):
        config = {k: v.value for k, v in config.__flags.items()}
    # if kwargs are passed, then include them in config
    config.update(kwargs)
    config = Box(config)  # Convert for easier access with dot or dict notation
    return config


def make(config: Optional[Union[dict, absl.flags.FlagValues]], **kwargs) -> Tuple[environment.Environment, EnvParams]:
    """Create RSA environment.

    This function is the entry point to setting up any XLRON environment.
    This function takes a dictionary of the commandline flag parameters and configures the RSA environment
    and parameters accordingly.

    See parameter_flags.py for a list of all possible parameters and their default values.

    Passing the 'log_wrapper' parameter as False will disable logging of the environment.

    Args:
        config: Configuration dictionary

    Kwargs:
        Any of the parameters in the config dictionary can be passed as keyword arguments.

    Returns:
        env: Environment
        params: Environment parameters
    """
    config = process_config(config, **kwargs)
    env_type = config.get("env_type", "").lower()
    if not env_type in [
        "rsa",
        "rmsa",
        "rwa",
        "deeprmsa",
        "rwa_lightpath_reuse",
        "rsa_gn_model",
        "rmsa_gn_model",
        "rsa_multiband",
        "vone",
    ]:
        raise ValueError(f"Invalid environment type {env_type}")

    seed = config.get("seed", 0)
    topology_name = config.get("topology_name", "conus")
    load = config.get("load", 100)
    k = config.get("k", 5)
    incremental_loading = config.get("incremental_loading", False)
    end_first_blocking = config.get("end_first_blocking", False)
    random_traffic = config.get("random_traffic", False)
    continuous_operation = config.get("continuous_operation", False)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 1e4)
    max_requests = total_timesteps if continuous_operation else config.get("max_requests", total_timesteps)
    link_resources = config.get("link_resources", 100)
    values_bw = config.get("values_bw", None)
    node_probabilities = config.get("node_probabilities", None)
    if values_bw:
        values_bw = [int(val) for val in values_bw]
    slot_size = config.get("slot_size", 12.5)
    min_bw = config.get("min_bw", 25)
    max_bw = config.get("max_bw", 100)
    step_bw = config.get("step_bw", 1)
    custom_traffic_matrix_csv_filepath = config.get("custom_traffic_matrix_csv_filepath", None)
    traffic_requests_csv_filepath = config.get("traffic_requests_csv_filepath", None)
    multiple_topologies_directory = config.get("multiple_topologies_directory", None)
    aggregate_slots = config.get("aggregate_slots", 1)
    disable_node_features = config.get("disable_node_features", False)
    disjoint_paths = config.get("disjoint_paths", False)
    log_actions = config.get("log_actions", False)
    guardband = config.get("guardband", 1)
    weight = config.get("weight", None)
    remove_array_wrappers = config.get("remove_array_wrappers", False)
    maximise_throughput = config.get("maximise_throughput", False)
    reward_type = config.get("reward_type", "bitrate")
    truncate_holding_time = config.get("truncate_holding_time", False)
    alpha = config.get("alpha", 0.2) * 1e-3
    amplifier_noise_figure = config.get("amplifier_noise_figure", 4.5)
    beta_2 = config.get("beta_2", -21.7) * 1e-27
    gamma = config.get("gamma", 1.2) * 1e-3
    span_length = config.get("span_length", 100) * 1e3
    lambda0 = config.get("lambda0", 1550) * 1e-9
    B = slot_size * link_resources  # Total modulated bandwidth

    if config.get("aggregate_slots", 1) > 1 and config.get("EVAL_HEURISTIC", False):
        raise ValueError("Cannot aggregate slots and evaluate heuristic")

    # VONE specific parameters
    node_resources = config.get("node_resources", 30)
    min_node_resources = config.get("min_node_resources", 1)
    max_node_resources = config.get("max_node_resources", 2)
    virtual_topologies = config.get("virtual_topologies", ["3_ring"])
    # Automated calculation of max edges in virtual topologies
    max_edges = 0
    for topology in virtual_topologies:
        num, shape = topology.split("_")
        max_edges = max(max_edges, int(num) - (0 if shape == "ring" else 1))

    # GN model parameters
    max_span_length = config.get("max_span_length", 100e3)
    ref_lambda = config.get("ref_lambda", 1577.5e-9)  # centre of C+L bands (1530-1625nm)
    nonlinear_coeff = config.get("nonlinear_coeff", 1.2 / 1e3)
    raman_gain_slope = config.get("raman_gain_slope", 0.028 / 1e3 / 1e12)
    attenuation = config.get("attenuation", 0.2 / 4.343 / 1e3)
    attenuation_bar = config.get("attenuation_bar", 0.2 / 4.343 / 1e3)
    dispersion_coeff = config.get("dispersion_coeff", 17 * 1e-12 / 1e-9 / 1e3)
    dispersion_slope = config.get("dispersion_slope", 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9)
    coherent = config.get("coherent", False)
    noise_figure = config.get("noise_figure", 4)
    interband_gap_width = config.get("interband_gap_width", 100)
    gap_width_slots = int(math.ceil(interband_gap_width / slot_size))
    interband_gap_start = config.get("interband_gap_start", 0)
    gap_start_slots = int(math.ceil(interband_gap_start / slot_size))
    mod_format_correction = config.get("mod_format_correction", True)
    num_roadms = config.get("num_roadms", 1)
    roadm_loss = config.get("roadm_loss", 18)
    snr_margin = config.get("snr_margin", 1)
    path_snr = True if env_type in ["rsa_gn_model", "rmsa_gn_model"] else False
    max_snr = config.get("max_snr", 50.)
    max_power = config.get("max_power", 9)
    min_power = config.get("min_power", -5)
    step_power = config.get("step_power", 1)
    default_launch_power = float(from_dbm(config.get("launch_power", 0.5)))
    optimise_launch_power = config.get("optimise_launch_power", False)
    traffic_array = config.get("traffic_array", False)
    launch_power_array = config.get("launch_power_array", None)
    pack_path_bits = config.get("pack_path_bits", False)

    # optimize_launch_power.py parameters
    num_spans = config.get("num_spans", 10)

    rng = jax.random.PRNGKey(seed)
    rng, _, _, _, _ = jax.random.split(rng, 5)
    graph = make_graph(topology_name, topology_directory=config.get("topology_directory", None))
    traffic_intensity = config.get("traffic_intensity", 0)
    mean_service_holding_time = config.get("mean_service_holding_time", 10)

    # Set traffic intensity / load
    if traffic_intensity:
        arrival_rate = traffic_intensity / mean_service_holding_time
    else:
        arrival_rate = load / mean_service_holding_time
    num_nodes = len(graph.nodes)
    num_links = len(graph.edges)
    scale_factor = config.get("scale_factor", 1.0)
    path_link_array = init_path_link_array(
        graph, k, disjoint=disjoint_paths, weight=weight, directed=graph.is_directed(),
        rwa_lr=True if env_type == "rwa_lightpath_reuse" else False, scale_factor=scale_factor, path_snr=path_snr)

    launch_power_type = config.get("launch_power_type", "fixed")
    # The launch power type determines whether to use:
    # 1. Fixed power for all channels.
    # 2. Tabulated values of power for each path.
    # 3. RL to determine power for each channel.
    # 4. Fixed power scaled by path length.
    if env_type in ["rmsa_gn_model", "rsa_gn_model"]:
        # default_launch_power_array = jnp.array([default_launch_power,])
        default_launch_power_array = jnp.full((k,), default_launch_power)
        if launch_power_type == "fixed":
            # Same power for all channels
            launch_power_array = default_launch_power_array if launch_power_array is None else launch_power_array
            launch_power_type = 1
        elif launch_power_type == "tabular":
            # The power of a channel is determined by the path it takes
            launch_power_array = jnp.zeros(path_link_array.shape[0]) \
                if launch_power_array is None else launch_power_array
            launch_power_type = 2
        elif launch_power_type == "rl":
            # RL sets power per channel
            launch_power_array = default_launch_power_array
            launch_power_type = 3
        elif launch_power_type == "scaled":
            # Power scaled by path length
            launch_power_array = default_launch_power_array if launch_power_array is None else launch_power_array
            launch_power_type = 4
        else:
            pass

    if custom_traffic_matrix_csv_filepath:
        random_traffic = False  # Set this False so that traffic matrix isn't replaced on reset
        traffic_matrix = jnp.array(np.loadtxt(custom_traffic_matrix_csv_filepath, delimiter=","))
        traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    elif node_probabilities:
        random_traffic = False  # Set this False so that traffic matrix isn't replaced on reset
        node_probabilities = [float(prob) for prob in config.get("node_probs")]
        traffic_matrix = convert_node_probs_to_traffic_matrix(node_probabilities)
    elif traffic_array:
        traffic_matrix = generate_source_dest_pairs(num_nodes, graph.is_directed())
    else:
        traffic_matrix = None

    if config.get("deterministic_requests"):
        deterministic_requests = True
        # Remove headers from array
        if traffic_requests_csv_filepath:
            list_of_requests = np.loadtxt(traffic_requests_csv_filepath, delimiter=",")[1:, :]
            list_of_requests = jnp.array(list_of_requests)
        else:
            list_of_requests = init_list_of_requests(int(max_requests))
        max_requests = len(list_of_requests)
    elif optimise_launch_power:
        deterministic_requests = True
        list_of_requests = jnp.array(config.get("list_of_requests", [0]))
    else:
        deterministic_requests = False
        list_of_requests = jnp.array([0])

    values_bw = init_values_bandwidth(min_bw, max_bw, step_bw, values_bw)

    if env_type == "rsa":
        consider_modulation_format = False
    elif env_type == "rwa":
        guardband = 0
        values_bw = jnp.array([slot_size])
        consider_modulation_format = False
    elif env_type == "rwa_lightpath_reuse":
        consider_modulation_format = False
        # Set guardband to 0 and slot size to max bandwidth to ensure that requested slots is always 1 but
        # that the bandwidth request is still considered when updating link_capacity_array
        guardband = 0
        slot_size = int(max(values_bw))
    elif env_type == "vone" and slot_size == 1:
        consider_modulation_format = False
    else:
        consider_modulation_format = True

    max_bw = max(values_bw)

    link_length_array = init_link_length_array(graph).reshape((num_links, 1))

    # Automated calculation of max slots requested
    if consider_modulation_format:
        modulations_array = init_modulations_array(config.get("modulations_csv_filepath", None))
        if weight is None:  # If paths aren't to be sorted by length alone
            path_link_array = init_path_link_array(graph, k, disjoint=disjoint_paths, directed=graph.is_directed(),
                                                   weight=weight, modulations_array=modulations_array,
                                                   path_snr=path_snr)
        path_length_array = init_path_length_array(path_link_array, graph)
        path_se_array = init_path_se_array(path_length_array, modulations_array)
        min_se = min(path_se_array)  # if consider_modulation_format
        max_slots = required_slots(max_bw, min_se, slot_size, guardband=guardband)
        max_spans = int(jnp.ceil(max(link_length_array) / max_span_length)[0])
        if env_type == "rmsa_gn_model" or env_type == "rsa_gn_model":
            link_length_array = init_link_length_array_gn_model(graph, max_span_length, max_spans)
    else:
        path_se_array = jnp.array([1])
        if env_type == "rwa_lightpath_reuse":
            path_capacity_array = init_path_capacity_array(
                link_length_array, path_link_array, min_request=min(values_bw), R_s=100e9, scale_factor=scale_factor,
                alpha=alpha, NF=amplifier_noise_figure, beta_2=beta_2, gamma=gamma, L_s=span_length, lambda0=lambda0,
                B=B*1e9,
            )
            max_requests = int(scale_factor * max_requests)
        else:
            # If considering just RSA without physical layer considerations
            link_length_array = jnp.ones((num_links, 1))
        max_slots = required_slots(max_bw, 1, slot_size, guardband=guardband)

    if env_type == "rsa_gn_model":
        consider_modulation_format = False
        path_se_array = jnp.array([1])
        max_slots = required_slots(max_bw, 1, slot_size, guardband=guardband)

    if incremental_loading:
        mean_service_holding_time = load = 1e6

    # Define edges for use with heuristics and GNNs
    edges = jnp.array(sorted(graph.edges))

    if pack_path_bits:
        path_link_array = jnp.packbits(path_link_array, axis=1)

    laplacian_matrix = jnp.array(nx.directed_laplacian_matrix(graph)) if graph.is_directed() \
        else jnp.array(nx.laplacian_matrix(graph).todense())

    params_dict = dict(
        max_requests=max_requests,
        mean_service_holding_time=mean_service_holding_time,
        k_paths=k,
        link_resources=link_resources,
        num_nodes=num_nodes,
        num_links=num_links,
        load=load,
        arrival_rate=arrival_rate,
        path_link_array=HashableArrayWrapper(path_link_array) if not remove_array_wrappers else path_link_array,
        incremental_loading=incremental_loading,
        end_first_blocking=end_first_blocking,
        edges=HashableArrayWrapper(edges) if not remove_array_wrappers else edges,
        random_traffic=random_traffic,
        path_se_array=HashableArrayWrapper(path_se_array) if not remove_array_wrappers else path_se_array,
        link_length_array=HashableArrayWrapper(link_length_array) if not remove_array_wrappers else link_length_array,
        max_slots=int(max_slots),
        consider_modulation_format=consider_modulation_format,
        slot_size=slot_size,
        continuous_operation=continuous_operation,
        aggregate_slots=aggregate_slots,
        guardband=guardband,
        deterministic_requests=deterministic_requests,
        multiple_topologies=False,
        directed_graph=graph.is_directed(),
        maximise_throughput=maximise_throughput,
        values_bw=HashableArrayWrapper(values_bw) if not remove_array_wrappers else values_bw,
        reward_type=reward_type,
        truncate_holding_time=truncate_holding_time,
        log_actions=log_actions,
        traffic_array=traffic_array,
        disable_node_features=disable_node_features,
        pack_path_bits=pack_path_bits,
    )

    if env_type == "vone":
        env_params = VONEEnvParams
        params_dict.update(
            node_resources=node_resources, min_node_resources=min_node_resources,
            max_node_resources=max_node_resources, max_edges=max_edges
        )
    elif env_type == "deeprmsa":
        env_params = DeepRMSAEnvParams
    elif env_type == "rwa_lightpath_reuse":
        env_params = RWALightpathReuseEnvParams
    elif env_type == "rsa_multiband":
        env_params = RSAMultibandEnvParams
        params_dict.update(gap_start=gap_start_slots, gap_width=gap_width_slots)
    elif "gn_model" in env_type:
        env_params = RSAGNModelEnvParams
        params_dict.update(
            ref_lambda=ref_lambda, max_spans=max_spans, max_span_length=max_span_length,
            default_launch_power=default_launch_power,
            nonlinear_coeff=nonlinear_coeff, raman_gain_slope=raman_gain_slope, attenuation=attenuation,
            attenuation_bar=attenuation_bar, dispersion_coeff=dispersion_coeff, noise_figure=noise_figure,
            dispersion_slope=dispersion_slope, coherent=coherent, gap_start=gap_start_slots, gap_width=gap_width_slots,
            roadm_loss=roadm_loss, num_roadms=num_roadms, num_spans=num_spans, launch_power_type=launch_power_type,
            snr_margin=snr_margin, last_fit=config.get("last_fit", False), max_power=max_power, min_power=min_power,
            step_power=step_power, max_snr=max_snr, mod_format_correction=mod_format_correction,
        )
        if env_type == "rmsa_gn_model":
            env_params = RMSAGNModelEnvParams
            params_dict.update(
                modulations_array=HashableArrayWrapper(
                    modulations_array) if not remove_array_wrappers else modulations_array,
            )
    else:
        env_params = RSAEnvParams

    params = env_params(**params_dict)

    # If training single model on multiple topologies, must store params for each topology within top-level params
    if multiple_topologies_directory:
        # iterate through files in directory
        params_list = []
        p = pathlib.Path(multiple_topologies_directory).glob('**/*')
        files = [x for x in p if x.is_file()]
        config.update(multiple_topologies_directory=None, remove_array_wrappers=True)
        for file in files:
            # Get filename without extension
            config.update(topology_name=file.stem, topology_directory=file.parent)
            env, params = make(config)
            params = params.replace(multiple_topologies=True)
            params_list.append(params)
        # for params in params_list, concatenate the field from each params into one array per field
        # from https://stackoverflow.com/questions/73765064/jax-vmap-over-batch-of-dataclasses
        cls = type(params_list[0])
        fields = params_list[0].__dict__.keys()
        field_dict = {}
        for k in fields:
            values = [getattr(v, k) for v in params_list]
            # values = [list(v) if isinstance(v, chex.Array) else v for v in values]
            # Pad arrays to same shape
            padded_values = HashableArrayWrapper(jnp.array(pad_array(values, fill_value=0)))
            field_dict[k] = padded_values
        params = cls(**field_dict)

    if remove_array_wrappers:
        # Only remove array wrappers if multiple_topologies=True for the inner files loop above
        env = None
    else:
        if env_type == "vone":
            env = VONEEnv(rng, params, virtual_topologies=virtual_topologies, traffic_matrix=traffic_matrix,
                          list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        elif env_type == "deeprmsa":
            env = DeepRMSAEnv(rng, params, traffic_matrix=traffic_matrix, laplacian_matrix=laplacian_matrix)
        elif env_type == "rwa_lightpath_reuse":
            env = RWALightpathReuseEnv(
                rng, params, traffic_matrix=traffic_matrix, path_capacity_array=path_capacity_array,
                list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        elif env_type == "rsa_gn_model":
            env = RSAGNModelEnv(rng, params, traffic_matrix=traffic_matrix, launch_power_array=launch_power_array,
                                list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        elif env_type == "rmsa_gn_model":
            env = RMSAGNModelEnv(rng, params, traffic_matrix=traffic_matrix, launch_power_array=launch_power_array,
                                 list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        elif env_type == "rsa_multiband":
            env = RSAMultibandEnv(rng, params, traffic_matrix=traffic_matrix, list_of_requests=list_of_requests,
                                  laplacian_matrix=laplacian_matrix)
        else:
            env = RSAEnv(rng, params, traffic_matrix=traffic_matrix, list_of_requests=list_of_requests,
                         laplacian_matrix=laplacian_matrix)

    if config.get("log_wrapper", True):
        env = LogWrapper(env)

    return env, params

