# Training with PPO

XLRON uses Proximal Policy Optimization (PPO) to train reinforcement learning agents for optical network resource allocation. The entire training loop — environment simulation, rollout collection, advantage estimation, and gradient updates — runs end-to-end in JAX, compiled as a single program that executes on GPU/TPU.

Training is accessed through the `train.py` script. The environment and traffic flags are the same as those used for [heuristic evaluation](./heuristic_evaluation.md) — this page focuses on the training-specific flags.


## Building a Training Command

A training command combines:

1. **Environment flags** — topology, resources, traffic (see [Heuristic Evaluation](./heuristic_evaluation.md#1-problem-environment-flags))
2. **Parallelism and scale** — number of environments, timesteps, rollout length
3. **Core PPO hyperparameters** — learning rate, discount factor, clipping
4. **Algorithmic features** — action masking, reward centering, prioritized sampling
5. **Schedules** — learning rate, entropy, GAE-lambda annealing
6. **Model architecture** — MLP, GNN, or Transformer
7. **Logging and output** — W&B, model saving, diagnostics

A minimal training command:

```bash
python -m xlron.train.train \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=5 \
    --load=250 \
    --continuous_operation \
    --TOTAL_TIMESTEPS=5000000 \
    --NUM_ENVS=16 \
    --ROLLOUT_LENGTH=100 \
    --LR=5e-4
```


## 1. Parallelism and Scale

### `--NUM_ENVS`

Number of parallel environment instances per learner. Each environment runs an independent simulation. The batch size for each update is `NUM_ENVS * ROLLOUT_LENGTH`. More environments improve sample diversity and statistical stability, but increase memory usage. Typical values: `16` to `2000`.

### `--ROLLOUT_LENGTH`

Number of environment steps collected per rollout before a PPO update. Longer rollouts capture more temporal context and allow better advantage estimation, but delay updates. Default: `150`. Typical values: `100` to `256`.

### `--TOTAL_TIMESTEPS`

Total environment steps across all environments. The number of PPO updates is `TOTAL_TIMESTEPS / (NUM_ENVS * ROLLOUT_LENGTH)`. Default: `1000000`.

### `--STEPS_PER_INCREMENT`

Steps per logging increment. Training is divided into `TOTAL_TIMESTEPS / STEPS_PER_INCREMENT` increments, with metrics reported after each. Default: `100000`.

### `--UPDATE_EPOCHS`

Number of passes over the rollout buffer per update step. Multiple epochs reuse the same data, improving sample efficiency at the risk of becoming too off-policy. Default: `1`. Typical values: `1` to `10`.

### `--NUM_MINIBATCHES`

Number of minibatches to split the rollout buffer into per epoch. The minibatch size is `(NUM_ENVS * ROLLOUT_LENGTH) / NUM_MINIBATCHES`. Default: `1`.

### `--NUM_LEARNERS`

Number of independent learners, each with their own neural network parameters and set of environments. Useful for running multiple seeds in parallel or for meta-learning. Default: `1`.

### `--NUM_DEVICES`

Number of accelerator devices to use. Environments and learners are distributed across devices. Default: `1`.

### `--mixed_precision`

Cuts the memory footprint of the parallel environment state so you can fit more `--NUM_ENVS` on a device. When enabled, the bulk carried env arrays are stored in narrower dtypes while everything precision-sensitive stays at 32-bit:

| Tier | Default | Mixed | Arrays |
|---|---|---|---|
| Bulk float | float32 | **float16** | spectrum occupancy (`link_slot_array`), normalised observation features |
| Time | float32 | **float16** (relative) / float32 (absolute) | `current_time`, `holding_time`, departure arrays |
| Bounded int | int32 | **int16** | per-path slot counts, bounded indices |
| Binary | int32 | **int8** | path–link incidence |
| Precision float | float32 | float32 | bitrate accumulators, physical SNR/power, importance weights |
| Counter int | int32 | int32 | `total_requests`, `total_timesteps`, `accepted_services` |
| **NN compute / params / optimizer** | float32 | **float32** | neural network weights, activations, Adam state |

Neural-network weights, activations and the optimizer stay in **float32** for training stability — observations are cast up to `--compute_dtype` (float32) at the model boundary, so the policy is numerically unchanged. Aggregate metrics (blocking probability, throughput) match the float32 baseline within statistical noise.

Time arrays use float16 only when they stay bounded (the default `--relative_arrival_times`); with absolute arrival times or `--incremental_loading` they automatically remain float32 to avoid overflow. Typical saving is ~30% of env-state memory at `--link_resources=100` (more at higher `--link_resources`), with no slowdown (a small speed-up on GPU/TPU from reduced memory bandwidth).

Each tier can be overridden individually (e.g. `--small_float_dtype=bfloat16`, `--binary_dtype=int8`, `--time_dtype=float32`); the per-tier flags take precedence over the `--mixed_precision` defaults. Default: `false`.


## 2. Core PPO Hyperparameters

### `--LR`

Initial learning rate for the Adam optimizer. This is arguably the most important hyperparameter. Default: `5e-4`. Typical range for sweeps: `1e-4` to `1e-3`.

### `--GAMMA`

Discount factor for future rewards. Values close to 1.0 make the agent more far-sighted. Default: `0.999`. Typical sweep range: `0.99` to `0.9999`.

### `--GAE_LAMBDA`

Lambda parameter for Generalized Advantage Estimation (GAE). Controls the bias-variance tradeoff in advantage estimation. Higher values reduce bias but increase variance. Default: `None` (uses automatic annealing from `INITIAL_LAMBDA` to `FINAL_LAMBDA`). Set explicitly to disable annealing, e.g. `--GAE_LAMBDA=0.95`. Typical sweep range: `0.9` to `0.999`.

### `--INITIAL_LAMBDA` / `--FINAL_LAMBDA`

When `GAE_LAMBDA` is not set, lambda is annealed from `INITIAL_LAMBDA` (default `0.9`) to `FINAL_LAMBDA` (default `0.98`) over the course of training using a hyperbolic secant schedule. This starts with lower variance (shorter horizon) and gradually increases the horizon as the value function improves. The annealing speed is controlled by `--LAMBDA_SCHEDULE_MULTIPLIER`.

### `--CLIP_EPS`

PPO clipping parameter. Limits how much the policy ratio can deviate from 1.0 per update. Smaller values are more conservative. Default: `0.2`. Typical sweep values: `0.02`, `0.04`, `0.08`, `0.16`.

### `--VF_COEF`

Value function loss coefficient. Scales the critic loss relative to the actor loss. Default: `0.5`. Typical sweep range: `0.1` to `1.0`.

### `--ENT_COEF`

Entropy bonus coefficient. Encourages exploration by penalising deterministic policies. Default: `0.0`. Typical sweep range: `0.001` to `0.1`. Can be scheduled (see [Schedules](#4-schedules)).

### `--MAX_GRAD_NORM`

Maximum gradient norm for gradient clipping. Prevents destructively large updates. Default: `0.5`.

### `--ADAM_EPS` / `--ADAM_BETA1` / `--ADAM_BETA2`

Adam optimizer parameters. Defaults: `1e-5`, `0.9`, `0.999`. Typical sweep ranges: `ADAM_BETA1` in `[0.8, 0.99]`, `ADAM_BETA2` in `[0.9, 0.999]`.

### `--WEIGHT_DECAY`

Weight decay (L2 regularization) for the AdamW optimizer. Default: `0.0`.

### `--REWARD_SCALE`

Multiply all rewards by this factor. Can help with training stability when rewards are very small or very large. Default: `1.0`. Typical sweep values: `1`, `10`.

### `--LOGR_CLIP` / `--ADV_CLIP`

Clip the log probability ratio to `[-LOGR_CLIP, +LOGR_CLIP]` and normalized advantages to `[-ADV_CLIP, +ADV_CLIP]`. Prevents numerical instabilities from extreme values. Defaults: `10.0`, `10.0`.


## 3. Algorithmic Features

### Invalid Action Masking

#### `--ACTION_MASKING`

Enable invalid action masking (default: `True`). The environment provides a mask of valid actions at each step; invalid actions receive logits of `-1e8`, ensuring they are never selected. This is critical for optical network environments where most actions are invalid at any given step.

#### `--OFF_POLICY_IAM`

Off-policy invalid action masking (default: `False`). When enabled, the importance ratio in PPO is computed as `unmasked_policy / masked_policy` rather than `masked_policy / masked_policy`. This means the ratio reflects the probability the *unmasked* (behavior) policy would have assigned to the taken action, which can be beneficial when the valid action set changes significantly between rollout and update time.

### Valid Mass and Gating

The PPO loss uses a gating mechanism to handle steps where the agent has very few or no valid actions:

- Steps with 0 valid actions are completely gated out (no gradient signal)
- Steps with only 1 valid action are gated out of the actor/entropy loss (the action is forced, so no learning signal)
- Steps with low "valid mass" (probability the unmasked policy places on valid actions) have their actor/entropy loss contribution damped

#### `--VALID_MASS_TARGET`

Threshold below which the actor/entropy loss is linearly damped based on valid mass. A step with `valid_mass = VALID_MASS_TARGET` gets full weight; `valid_mass = 0` gets zero weight. Default: `0.05`. Tune in range `0.02` to `0.1`.

#### `--VALID_MASS_LOSS_COEF`

Coefficient for an auxiliary loss that encourages the *unmasked* policy to place probability on valid actions. This helps prevent the unmasked logits from drifting away from valid actions, which would cause the valid mass to collapse and reduce learning signal. Default: `0.0` (disabled). Enable with values like `0.01` to `0.1`.

#### `--VML_SCHEDULE` / `--VML_END_FRACTION`

Schedule for the valid mass loss coefficient. Supports `constant`, `linear`, `cosine`. `VML_END_FRACTION` > 1.0 anneals the coefficient *upward* (e.g. `10.0` means the final coefficient is 10x the initial). Default: `constant`, `10.0`.

### Reward Centering

#### `--REWARD_CENTERING`

Enable reward centering (default: `False`). Subtracts a running estimate of the average reward from all rewards before computing TD errors. This can stabilize training in environments with non-zero average reward by keeping advantage estimates centered. The average reward estimate is updated using a decaying step size.

#### `--INITIAL_AVERAGE_REWARD` / `--REWARD_STEPSIZE`

Initial average reward estimate and step size for the exponential moving average update. Defaults: `0.0`, `0.001`.

### Prioritized Experience Replay

#### `--PRIO_ALPHA`

Priority exponent for prioritized sampling of the rollout buffer. `0.0` = uniform sampling (default), `1.0` = fully prioritized by absolute advantage. Samples with higher absolute advantage are replayed more often, with importance sampling corrections to maintain unbiased gradients.

#### `--PRIO_BETA0`

Initial importance sampling correction exponent. Annealed from `PRIO_BETA0` to `1.0` over training. `1.0` = full correction from the start (default). Lower values allow more biased but potentially faster early learning.

### VTrace / Puffer Advantage

#### `--RHO_CLIP` / `--C_CLIP`

Clipping parameters for VTrace-style off-policy correction in the advantage calculation. When both are set to positive values, the importance ratios in the GAE calculation are clipped:

- `RHO_CLIP`: Clips the ratio used in the TD error (`delta = rho_t * (r + gamma * V(s') - V(s))`)
- `C_CLIP`: Clips the ratio used in the GAE accumulation (`A_t = delta_t + gamma * lambda * c_t * A_{t+1}`)

When both are `<= 0` (default), standard GAE is used without clipping. When `lambda=1` and both clips are set, this reduces to the VTrace algorithm. These are useful when doing multiple epochs of updates, as the policy changes between epochs making the data off-policy.

### `--include_no_op`

Add a "no operation" action to the action space. Default: `False`.

### `--STEP_ON_GRADIENT`

When `True`, the schedule step counter increments on each gradient update (each minibatch). When `False` (default), it increments once per update loop. This affects how quickly learning rate and other schedules progress.


## 4. Schedules

XLRON supports scheduling for several hyperparameters. All schedules operate over the total number of gradient steps (`NUM_UPDATES * UPDATE_EPOCHS * NUM_MINIBATCHES`).

### Learning Rate Schedule

#### `--LR_SCHEDULE`

| Value | Description |
|-------|-------------|
| `cosine` | Cosine decay from `LR` to `LR * LR_END_FRACTION` (default) |
| `warmup_cosine` | Linear warmup to `LR * WARMUP_MULTIPLIER`, then cosine decay |
| `linear` | Linear decay from `LR` to `LR * LR_END_FRACTION` |
| `constant` | Fixed learning rate |

#### `--LR_END_FRACTION`

Final learning rate as a fraction of initial LR. Default: `0.1` (i.e. LR decays to 10% of its initial value).

#### `--WARMUP_MULTIPLIER`

Peak learning rate during warmup, as a multiple of initial LR. Default: `1.0`. Set to e.g. `2.0` to warm up to 2x the initial LR before decaying.

#### `--WARMUP_STEPS_FRACTION`

Fraction of total training steps used for warmup. Default: `0.2`.

#### `--LR_SCHEDULE_MULTIPLIER`

Multiply the schedule horizon by this factor. Values > 1.0 slow down the schedule (decay takes longer). Default: `1.0`.

### Entropy Schedule

#### `--ENT_SCHEDULE`

Schedule for the entropy coefficient. Supports `constant` (default), `linear`, `cosine`. With `linear` or `cosine`, the entropy coefficient decays from `ENT_COEF` to `ENT_COEF * ENT_END_FRACTION`.

#### `--ENT_END_FRACTION`

Final entropy coefficient as a fraction of initial. Default: `0.1`. Typical sweep values: `0.1`, `0.2`, `0.5`.

#### `--ENT_SCHEDULE_MULTIPLIER`

Multiply the entropy schedule horizon. Default: `1.0`.

### Separate Value Function Optimizer

#### `--SEPARATE_VF_OPTIMIZER`

Use a separate optimizer for the critic (value function) with its own hyperparameters. Default: `False`. When enabled, the following flags control the critic optimizer:

- `--VF_LR` — Learning rate (default: `LR / 3`)
- `--VF_LR_SCHEDULE` — Schedule type (default: same as `LR_SCHEDULE`)
- `--VF_LR_END_FRACTION` — End fraction (default: same as `LR_END_FRACTION`)
- `--VF_WARMUP_MULTIPLIER` — Warmup peak (default: same as `WARMUP_MULTIPLIER`)
- `--VF_WARMUP_STEPS_FRACTION` — Warmup fraction (default: same as `WARMUP_STEPS_FRACTION`)
- `--VF_ADAM_EPS`, `--VF_ADAM_BETA1`, `--VF_ADAM_BETA2` — Adam parameters
- `--VF_MAX_GRAD_NORM` — Gradient clipping
- `--VF_WEIGHT_DECAY` — Weight decay
- `--VF_SCHEDULE_MULTIPLIER` — Schedule horizon multiplier


## 5. Model Architecture

XLRON provides three neural network architectures. All follow an actor-critic design with shared or separate encoders.

### MLP (Default)

A simple multi-layer perceptron. Used by default when neither `--USE_GNN` nor `--USE_TRANSFORMER` is set.

| Flag | Description | Default |
|------|-------------|---------|
| `--NUM_LAYERS` | Number of hidden layers (shared by actor and critic) | `2` |
| `--NUM_UNITS` | Hidden units per layer | `64` |
| `--ACTIVATION` | Activation function (`tanh`, `relu`) | `tanh` |

The MLP takes a flat observation vector as input. Its simplicity makes it fast to train and a good baseline.

### GNN (Graph Neural Network)

A Jraph-based graph neural network that operates directly on the network topology. Enabled with `--USE_GNN`.

| Flag | Description | Default |
|------|-------------|---------|
| `--message_passing_steps` | Rounds of message passing | `3` |
| `--edge_embedding_size` | Edge embedding dimension | `128` |
| `--edge_mlp_layers` / `--edge_mlp_latent` | Edge update MLP depth/width | `2` / `128` |
| `--node_embedding_size` | Node embedding dimension | `16` |
| `--node_mlp_layers` / `--node_mlp_latent` | Node update MLP depth/width | `2` / `128` |
| `--global_embedding_size` | Global embedding dimension | `8` |
| `--global_mlp_layers` / `--global_mlp_latent` | Global update MLP depth/width | `1` / `16` |
| `--attn_mlp_layers` / `--attn_mlp_latent` | Attention MLP depth/width (0 to disable attention) | `1` / `64` |
| `--num_spectral_features` | Number of spectral features per node | `8` |
| `--gnn_layer_norm` | Layer normalization in GNN | `True` |
| `--mlp_layer_norm` | Layer normalization in MLPs | `False` |
| `--normalize_by_link_length` | Normalize features by link length | `False` |
| `--DISABLE_NODE_FEATURES` | Disable node features (use only edge features) | `False` |

The GNN has an inductive bias for graph-structured problems — it can generalize across topologies and naturally captures link adjacency. When `attn_mlp_layers > 0`, it uses GATv2-style attention on edges.

### Transformer

A transformer encoder that processes link-level tokens with graph-aware positional encodings (WIRE — Wavelet-Induced Rotary Encodings). Enabled with `--USE_TRANSFORMER`.

| Flag | Description | Default |
|------|-------------|---------|
| `--transformer_num_layers` | Number of transformer encoder layers | `1` |
| `--transformer_num_heads` | Number of attention heads | `4` |
| `--transformer_embedding_size` | Token embedding dimension | `128` |
| `--transformer_intermediate_size` | Feed-forward hidden dimension | `256` |
| `--transformer_obs_type` | Observation type: `departure` (departure times), `occupancy` (binary), or `capacity` (for RWA-LR) | `departure` |
| `--num_wire_features` | Number of spectral features for WIRE positional encoding | `8` |
| `--transformer_actor_mlp_width` / `--transformer_actor_mlp_depth` | Actor head MLP width/depth | `128` / `1` |
| `--transformer_critic_mlp_width` / `--transformer_critic_mlp_depth` | Critic head MLP width/depth | `128` / `2` |
| `--transformer_share_layers` | Share encoder layers between actor and critic | `False` |
| `--transformer_dropout_rate` / `--transformer_attention_dropout_rate` | Dropout rates | `0.1` / `0.1` |
| `--transformer_enable_dropout` | Enable dropout during training | `False` |

The transformer processes each link as a token, with per-link spectral features and request-specific information. WIRE positional encodings inject graph structure through rotary embeddings derived from the graph Laplacian.

### Slot Aggregation

#### `--aggregate_slots`

Reduce the action space by grouping frequency slots into blocks of N. The agent selects a block, and first-fit allocation selects the specific slot within the block. Default: `1` (no aggregation). Typical values for large action spaces: `10`, `20`, `100`.


## 6. Logging, Saving, and Evaluation

### Weights & Biases

#### `--WANDB`

Enable logging to W&B. See [Understanding XLRON](understanding_xlron.md#weights-biases-wandb-integration) for setup and sweep configuration.

#### `--LOG_LOSS_INFO`

Log loss metrics (actor loss, value loss, entropy, etc.) to W&B. Default: `True`.

#### `--ENHANCED_LOGGING`

Enable detailed diagnostic logging including valid fraction, clip fraction, ratio statistics, valid mass statistics, advantage statistics, and more. Useful for debugging training dynamics. Default: `False`.

### Model Saving and Loading

#### `--SAVE_MODEL`

Save the best model (by blocking probability) during training. Saved locally to `--MODEL_PATH` and uploaded to W&B if `--WANDB` is enabled. Default: `False`.

#### `--MODEL_PATH`

Path to save/load model files. If unspecified, models are saved to `models/<EXPERIMENT_NAME>.eqx`.

#### `--RETRAIN_MODEL`

Load a previously saved model and continue training. Restores model weights but not optimizer state.

#### `--EVAL_MODEL`

Load a saved model for evaluation only (no training). Uses the same evaluation infrastructure as heuristic evaluation.

#### `--KEEP_VF`

When loading a model, keep only the pre-trained value function (critic) and reinitialize the actor. Useful for transfer learning.

### Evaluation During Training

#### `--EVAL_DURING_TRAINING`

Run periodic evaluation using the current policy without exploration noise. This tracks the best-performing model across training. Default: `False`.

#### `--EVAL_TIMESTEPS`

Number of timesteps per evaluation run. Default: `10000`.

#### `--EVAL_FREQUENCY`

Run evaluation every N increments. Default: `1`.

### Other Logging Flags

#### `--DATA_OUTPUT_FILE` / `--TRAJ_DATA_OUTPUT_FILE`

Save per-episode or per-step metrics to CSV files.

#### `--DOWNSAMPLE_FACTOR`

Block-average every N consecutive data points before uploading to W&B. Default: `1`.

#### `--log_actions` / `--log_path_lengths`

Log detailed per-step action and path length data. Default: `False`.


## 7. Hyperparameter Sweeps

XLRON supports W&B hyperparameter sweeps. Define a sweep configuration YAML file and use the `wandb sweep` command to launch. See [Understanding XLRON](understanding_xlron.md#weights-biases-wandb-integration) for full details.

An example sweep config is provided at `examples/sweep.yaml`. The key hyperparameters to sweep are:

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| `LR` | `1e-4` to `1e-3` | Most impactful hyperparameter |
| `GAMMA` | `0.99` to `0.9999` | Discount factor |
| `GAE_LAMBDA` | `0.9` to `0.999` | Advantage estimation horizon |
| `CLIP_EPS` | `0.02`, `0.04`, `0.08`, `0.16` | PPO clipping |
| `VF_COEF` | `0.1` to `1.0` | Value loss weight |
| `ENT_COEF` | `0.001` to `0.1` | Exploration |
| `ENT_END_FRACTION` | `0.1`, `0.2`, `0.5` | Entropy decay |
| `REWARD_SCALE` | `1`, `10` | Reward magnitude |
| `OFF_POLICY_IAM` | `true`, `false` | Action masking style |
| `ADAM_BETA1` | `0.8` to `0.99` | Optimizer momentum |
| `ADAM_BETA2` | `0.9` to `0.999` | Optimizer second moment |
| `ROLLOUT_LENGTH` | `128`, `256` | Trajectory length |
| `aggregate_slots` | `10`, `20`, `100` | Action space reduction |

For transformer-specific sweeps, also consider: `transformer_num_heads`, `transformer_embedding_size`, `transformer_obs_type`, `transformer_actor_mlp_width`, `transformer_actor_mlp_depth`.

To launch a sweep:

```bash
wandb sweep examples/sweep.yaml
wandb agent <SWEEP_ID>
```


## 8. Flagfiles

You can save a set of flags to a text file and load them with `--flagfile=PATH`:

```
# my_flags.txt
--env_type=rmsa
--topology_name=nsfnet_deeprmsa_directed
--link_resources=100
--k=50
--load=250
--continuous_operation
--ENV_WARMUP_STEPS=3000
```

```bash
python -m xlron.train.train --flagfile=my_flags.txt --LR=1e-4 --NUM_ENVS=64
```

Command-line flags override flagfile values.


## Examples

### Example 1: DeepRMSA Reproduction

Reproduce the DeepRMSA paper training setup:

```bash
python -m xlron.train.train \
    --env_type=deeprmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=5 \
    --load=250 \
    --mean_service_holding_time=25 \
    --max_requests=1000 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --NUM_LAYERS=5 \
    --NUM_UNITS=128 \
    --NUM_ENVS=16 \
    --ROLLOUT_LENGTH=100 \
    --TOTAL_TIMESTEPS=5000000 \
    --LR=5e-5 \
    --LR_SCHEDULE=linear \
    --WARMUP_MULTIPLIER=2 \
    --UPDATE_EPOCHS=10 \
    --GAE_LAMBDA=0.9 \
    --GAMMA=0.95 \
    --ACTION_MASKING
```

### Example 2: RMSA with Transformer and Reward Centering

Train a transformer agent with reward centering and entropy scheduling:

```bash
python -m xlron.train.train \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --load=200 \
    --mean_service_holding_time=1 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --truncate_holding_time \
    --USE_TRANSFORMER \
    --transformer_num_layers=2 \
    --transformer_num_heads=4 \
    --transformer_embedding_size=128 \
    --transformer_obs_type=departure \
    --aggregate_slots=20 \
    --NUM_ENVS=64 \
    --TOTAL_TIMESTEPS=60000000 \
    --UPDATE_EPOCHS=1 \
    --LR=5e-4 \
    --REWARD_CENTERING \
    --ENT_COEF=0.01 \
    --ENT_SCHEDULE=cosine \
    --ENT_END_FRACTION=0.1 \
    --WANDB \
    --LOG_LOSS_INFO \
    --DOWNSAMPLE_FACTOR=100
```

### Example 3: GNN Agent

Train a GNN agent with graph attention:

```bash
python -m xlron.train.train \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=5 \
    --load=250 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --USE_GNN \
    --message_passing_steps=3 \
    --edge_embedding_size=128 \
    --node_embedding_size=16 \
    --attn_mlp_layers=1 \
    --num_spectral_features=8 \
    --NUM_ENVS=32 \
    --TOTAL_TIMESTEPS=10000000 \
    --LR=3e-4 \
    --GAMMA=0.999 \
    --WANDB
```

### Example 4: Training with Advanced Features

Combine multiple algorithmic features:

```bash
python -m xlron.train.train \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --truncate_holding_time \
    --NUM_ENVS=64 \
    --ROLLOUT_LENGTH=128 \
    --TOTAL_TIMESTEPS=20000000 \
    --LR=3e-4 \
    --LR_SCHEDULE=warmup_cosine \
    --WARMUP_MULTIPLIER=2 \
    --WARMUP_STEPS_FRACTION=0.1 \
    --GAMMA=0.999 \
    --CLIP_EPS=0.08 \
    --VF_COEF=0.5 \
    --ENT_COEF=0.01 \
    --ENT_SCHEDULE=cosine \
    --ENT_END_FRACTION=0.2 \
    --REWARD_CENTERING \
    --OFF_POLICY_IAM \
    --VALID_MASS_LOSS_COEF=0.01 \
    --REWARD_SCALE=10 \
    --WANDB \
    --ENHANCED_LOGGING \
    --SAVE_MODEL
```

### Example 5: Save and Evaluate a Model

Save during training:

```bash
python -m xlron.train.train \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --NUM_ENVS=64 \
    --TOTAL_TIMESTEPS=10000000 \
    --LR=3e-4 \
    --SAVE_MODEL \
    --MODEL_PATH=models/my_agent.eqx
```

Then evaluate:

```bash
python -m xlron.train.train \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --NUM_ENVS=2000 \
    --TOTAL_TIMESTEPS=20000000 \
    --EVAL_MODEL \
    --MODEL_PATH=models/my_agent.eqx
```
