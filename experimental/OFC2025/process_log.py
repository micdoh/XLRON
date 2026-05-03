def process_log(log_string):
    current_load = None
    current_values = {}

    print("load,bitrate_blocking_probability_mean,bitrate_blocking_probability_iqr_lower,bitrate_blocking_probability_iqr_upper")

    for line in log_string.split('\n'):
        # Check for new load value
        if line.startswith("Running training with load = "):
            # If we have a complete set, print it
            if current_load and len(current_values) == 3:
                print(
                    f"{current_load},{current_values.get('mean', '')},{current_values.get('lower', '')},{current_values.get('upper', '')}")
                current_values = {}

            # Get new load value
            current_load = line.split("= ")[1]

        # Check for bbp values
        elif " bitrate_blocking_probability_mean " in line:
            value = line.split()[-1]
            if value.replace('.', '').isdigit():  # Check if number
                current_values['mean'] = value
        elif " bitrate_blocking_probability_iqr_lower " in line:
            value = line.split()[-1]
            if value.replace('.', '').isdigit():
                current_values['lower'] = value
        elif " bitrate_blocking_probability_iqr_upper " in line:
            value = line.split()[-1]
            if value.replace('.', '').isdigit():
                current_values['upper'] = value

    # Print last set if complete
    if current_load and len(current_values) == 3:
        print(
            f"{current_load},{current_values.get('mean', '')},{current_values.get('lower', '')},{current_values.get('upper', '')}")


# Example usage:
test_string = """
604 uceedoh@geneva:~$ cd git/XLRON/
605 uceedoh@geneva:~/git/XLRON$ ../../run_ksplf_eval.sh 
Running training with load = 67
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load67.csv
CUDA_VISIBLE_DEVICES=2
I1022 22:17:29.615662 140403999169408 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 22:17:29.616872 140403999169408 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_221738-s3bg6jv2
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vocal-gorge-1
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/s3bg6jv2
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 67
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load67.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 67.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=23.35s
EXECUTION: Elapsed time=545.36s, FPS=3.67e+02
returns: 0.87107 ± 4.82955
lengths: 4000.00000 ± 0.00000
cum_returns: 3520.72437 ± 802.15991
accepted_services: 3822.90503 ± 13.23918
accepted_bitrate: 2786936.25000 ± 20240.04688
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.08400 ± 0.01426
service_blocking_probability: 0.04427 ± 0.00331
bitrate_blocking_probability: 0.07084 ± 0.00516
wandb: \ 0.196 MB of 0.196 MB uploaded
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▂▂▂▂▃▃▃▄▄▄▄▄▄▄▄▅▄▅▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▇▇██
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▁▁▁▁▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇███
wandb:             bitrate_blocking_probability_iqr_lower ▃▆▄▄▃▁▁▃▃▂▄▅▃▄▄▃▅█▆▄▆▆▇▄▆▆▇▆▅▄▅▇█▆▅▅▆▆▇▆
wandb:             bitrate_blocking_probability_iqr_upper ▂▃▄▂▃▃▂▁▁▂▂▃▄▃▃▂▂▃▂▃▂▄▄▄▅▄▇▇██▆▅▄▆▄▄▅▃▂▁
wandb:                  bitrate_blocking_probability_mean ▅▅▄▃▃▂▂▁▁▁▂▃▃▄▃▃▄▆▅▆▆▇▇▇███▇██▇█▇█▇▇▇██▇
wandb:                   bitrate_blocking_probability_std ██▇▇▆▆▆▆▆▆▇▆▆▆▆▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▃▃▃▂▂▁▁▁▁▁
wandb:                              cum_returns_iqr_lower ▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇███
wandb:                              cum_returns_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇███
wandb:                                   cum_returns_mean ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                                    cum_returns_std ▁▁▁▁▂▂▂▂▂▃▃▃▃▄▄▅▅▅▅▅▅▆▆▆▇▇▇▇▇▇▇█████████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▂▃▃▃▃▅▅▆▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇▇██
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▄▅▇▇▇▆▇▆██▇▆▆▅▇▄▅▅▆▅▆▇▇██▇▆▆▇▆▆▆▅▅▅▆▆▆
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▄▃▃▂▄▄▄▄▄▅▅▅▅▆▅▅▆█▇▇▆▆▅▅▅▅▄▄▄▄▅▅▄▄▃▂▁▂▂▁
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▃▂▃▃▂▂▅▅▄▃▄▃▅▇▆▅▅▅▇▆▆▅▇▆▅▆█▇▆▅▅▄▃▂▂▄▂▁▁▄
wandb:      episode_end_bitrate_blocking_probability_mean ▄▅▆▇▆▇▇██▇▆▆▆█▇▇▇███▇▇▆▅▅▄▂▄▃▂▃▂▂▃▂▂▂▁▂▂
wandb:       episode_end_bitrate_blocking_probability_std ▅▅▇▇███▇▇▇▇▇▆▆▆▅▆▄▄▄▅▄▄▄▄▄▅▄▃▃▃▂▂▂▂▁▁▂▁▁
wandb:                  episode_end_cum_returns_iqr_lower ▁▁▁▁▁▁▁▁▂▂▂▃▃▃▃▃▄▄▄▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇██
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▂▂▁▁▁▁▂▂▂▃▃▃▃▃▄▄▄▄▃▃▃▃▃▃▄▄▄▅▆▆▆▆▆▇▇▇▇█
wandb:                       episode_end_cum_returns_mean ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        episode_end_cum_returns_std ▁▂▃▂▂▂▁▂▁▁▂▂▂▂▂▂▃▃▂▂▁▂▂▂▃▃▄▄▅▅▆▆▆▆▆▇▆▇▇█
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▃▁▃▁█▃▃▃▅▆▆▂▃▃▆▃▃▃▂▁▂▃▄▃▄▅▆▁▃▄▄▃▂▄▄▃▄▃▂▆
wandb:                            episode_end_returns_std ▅▃▆▂▇▄▃▄▆▅▅▄▃█▇▃▃▆▃▄▁▅▁▂▆▅▆▄▂▅▆▃▃█▄▅▄▅▆▅
wandb: episode_end_service_blocking_probability_iqr_lower █▇██▇▆▆▅▄▄▃▃█▇▇▆▅▅▄▄▃▂▂▆▆▆▅▅▄▃▃▂▁▁▅▄▅▄▄▃
wandb: episode_end_service_blocking_probability_iqr_upper ▃▃▂▃▂▂▁▄▃▃▃▃▅▅▆▅█▇▇▆▇▆▆▅██▇▇▆▆▅▅▄▄▃▃▃▃▅▅
wandb:      episode_end_service_blocking_probability_mean ▅▆▇▇▇▇███▇▆▆▆█▇▇▇███▇▇▆▅▅▄▃▄▃▂▃▂▂▃▂▂▂▁▂▂
wandb:       episode_end_service_blocking_probability_std ▆▆▇▇██▇▇▇▇▇▇▆▆▆▅▆▄▄▄▄▄▄▄▄▄▄▄▃▃▃▂▂▂▂▂▁▂▂▁
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▃▃▆▇██▅▄▃▁▅▅▆█▆▅▆▄▅▃▃▃▃▃▄▆▇▄▅▂▃▆█▆▆▇▅▅▃▁
wandb:                  episode_end_utilisation_iqr_upper ▆▅▅▃▂▁▃▆▆▃▇▄▅▅█▇▆▇▂▂▄▃▃▃▃▂▆█▇▆▅█▅▁▄▃▆▇▃▃
wandb:                       episode_end_utilisation_mean ▅▅▄▃▅▅▂▄▃▄▇▆▇▆▇▇▇▅▄▅▃▆▆▇▆▅█▆▅▆▆▆█▅▅▄▅▅▂▁
wandb:                        episode_end_utilisation_std ██▇▅▆▄▅▆▆▆▆▅▄▄▄▄▃▃▁▃▄▄▃▂▃▂▃▅▆▆▇▇▆▅▅▆▇▇▇█
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▂▅▅▅▄▅▆▆▇██▆▄█▆▅▄▅▇▆▅▂▃▂▂▇▆▄▄▃▆▇▅▃▁▅▂▁▂▅
wandb:                                        returns_std ▁▇▁▃▆▆▄▅▅▄▇▇▄▆▅▄▄▇▄▇▆▂▄▂▇▇▇▂▃▂█▇█▃▆▇▁▂▂▅
wandb:             service_blocking_probability_iqr_lower ▄▄▁▁▁▁▃▃▃▃▃▃▂▄▅▂▂▇▇▄▄▆▇▆▆▆▆█▇▅█▅▇▅▇▄▆▆▆▆
wandb:             service_blocking_probability_iqr_upper ▃▅▆▄▅▃▁▁▂▄▄▃▅▄▄▅▃▂▅█▇▇▄▅▅▇▇▆██▇▇▃▆▄▄▇▆▃▂
wandb:                  service_blocking_probability_mean ▅▅▅▃▃▂▂▁▁▁▁▃▄▄▃▃▄▆▅▆▆▇▇▇███▇▇▇▇█▇▇▇▆▆▇▇▇
wandb:                   service_blocking_probability_std ██▇▇▆▆▆▆▆▆▇▆▆▆▆▅▅▄▄▄▅▄▄▄▄▄▄▄▄▄▃▃▂▂▂▁▁▁▁▁
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▆▆▄▆▄▃▇▃▄▄▇▅▂▃▃▂▅▅▇▆▃▆▃▅▄▃▃▄▄▃▅▄▂▃▅▅▁▇█▅
wandb:                              utilisation_iqr_upper ▆▇▆██▆▄▄▅▃▅▆▆▅▆▅▅▇▆▆▆▆▅▄▆▆▃▃▁▃▃▄▃▂▇▇▅▇▆█
wandb:                                   utilisation_mean ▅▆▆▇▅▅▄▃▃▃▅▅▄▅▄▄▄▇▇▆▅▄▃▄▅▄▁▂▂▂▃▃▁▂▅▅▄▇█▇
wandb:                                    utilisation_std ▆▆▅▇▇▆▆▆▆▆▆▅█▇▅▄▅▄▅▆▇▅▄▄▆▅▄▄▄▄▄▅▅▁▅▅▆▃▄▆
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2771800.0
wandb:                         accepted_bitrate_iqr_upper 2803000.0
wandb:                              accepted_bitrate_mean 2786936.25
wandb:                               accepted_bitrate_std 20240.04688
wandb:                        accepted_services_iqr_lower 3813.0
wandb:                        accepted_services_iqr_upper 3832.0
wandb:                             accepted_services_mean 3822.90503
wandb:                              accepted_services_std 13.23918
wandb:             bitrate_blocking_probability_iqr_lower 0.06732
wandb:             bitrate_blocking_probability_iqr_upper 0.07461
wandb:                  bitrate_blocking_probability_mean 0.07084
wandb:                   bitrate_blocking_probability_std 0.00516
wandb:                              cum_returns_iqr_lower 2922.99084
wandb:                              cum_returns_iqr_upper 4081.3927
wandb:                                   cum_returns_mean 3520.72437
wandb:                                    cum_returns_std 802.15991
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 2145950.0
wandb:             episode_end_accepted_bitrate_iqr_upper 2174400.0
wandb:                  episode_end_accepted_bitrate_mean 2160553.0
wandb:                   episode_end_accepted_bitrate_std 17929.14844
wandb:            episode_end_accepted_services_iqr_lower 2954.0
wandb:            episode_end_accepted_services_iqr_upper 2970.0
wandb:                 episode_end_accepted_services_mean 2962.04492
wandb:                  episode_end_accepted_services_std 11.25047
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.06681
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.07463
wandb:      episode_end_bitrate_blocking_probability_mean 0.07068
wandb:       episode_end_bitrate_blocking_probability_std 0.00563
wandb:                  episode_end_cum_returns_iqr_lower 2170.61096
wandb:                  episode_end_cum_returns_iqr_upper 3191.7887
wandb:                       episode_end_cum_returns_mean 2711.35986
wandb:                        episode_end_cum_returns_std 693.61035
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower 0.0
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean 1.55336
wandb:                            episode_end_returns_std 5.2612
wandb: episode_end_service_blocking_probability_iqr_lower 0.04163
wandb: episode_end_service_blocking_probability_iqr_upper 0.04679
wandb:      episode_end_service_blocking_probability_mean 0.04419
wandb:       episode_end_service_blocking_probability_std 0.00363
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.07213
wandb:                  episode_end_utilisation_iqr_upper 0.09175
wandb:                       episode_end_utilisation_mean 0.08249
wandb:                        episode_end_utilisation_std 0.01447
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower 0.0
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean 0.87107
wandb:                                        returns_std 4.82955
wandb:             service_blocking_probability_iqr_lower 0.042
wandb:             service_blocking_probability_iqr_upper 0.04675
wandb:                  service_blocking_probability_mean 0.04427
wandb:                   service_blocking_probability_std 0.00331
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 544.81054
wandb:                              utilisation_iqr_lower 0.07312
wandb:                              utilisation_iqr_upper 0.09125
wandb:                                   utilisation_mean 0.084
wandb:                                    utilisation_std 0.01426
wandb: 
wandb: 🚀 View run vocal-gorge-1 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/s3bg6jv2
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 3 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_221738-s3bg6jv2/logs
Completed training for load = 67
----------------------------------------
Running training with load = 135
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load135.csv
CUDA_VISIBLE_DEVICES=2
I1022 22:27:52.783435 139882764381056 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 22:27:52.784409 139882764381056 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_222800-w80gf86v
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run golden-elevator-3
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/w80gf86v
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 135
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load135.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 135.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=23.32s
EXECUTION: Elapsed time=543.93s, FPS=3.68e+02
returns: 1.00575 ± 4.99362
lengths: 4000.00000 ± 0.00000
cum_returns: 4810.92920 ± 983.77563
accepted_services: 3822.68994 ± 13.21605
accepted_bitrate: 2786708.75000 ± 20182.25781
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.16802 ± 0.01861
service_blocking_probability: 0.04433 ± 0.00330
bitrate_blocking_probability: 0.07092 ± 0.00516
wandb: 
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▂▂▂▂▃▃▃▄▄▄▄▄▄▄▄▅▄▅▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▇▇██
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▁▁▁▁▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇██
wandb:             bitrate_blocking_probability_iqr_lower ▃▅▄▄▂▂▁▄▁▁▄▄▁▃▃▃▄█▅▄▆▆▆▅▇▆▇▆▅▅▇▇▇▆▆▅▆▅▆▅
wandb:             bitrate_blocking_probability_iqr_upper ▃▄▄▂▃▄▂▂▃▂▂▄▄▃▄▃▃▄▂▄▄▄▄▅▆▇▇▇██▇▅▄▆▄▄▅▃▂▁
wandb:                  bitrate_blocking_probability_mean ▅▅▄▃▃▂▂▁▁▁▂▃▃▄▃▃▄▆▅▆▆▇▇▇███▇██▇█▇▇▇▇▇▇█▇
wandb:                   bitrate_blocking_probability_std ██▇▆▆▆▆▆▆▆▇▆▆▆▆▅▄▄▄▄▄▄▄▄▄▄▄▄▄▄▃▃▂▂▂▁▁▁▁▁
wandb:                              cum_returns_iqr_lower ▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇████
wandb:                              cum_returns_iqr_upper ▁▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇██
wandb:                                   cum_returns_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                    cum_returns_std ▁▁▁▁▁▂▂▂▂▃▃▃▃▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇████████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▂▃▃▃▃▅▅▆▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇▇██
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▄▆█▇▇▆▇▇██▇▆▆▅▇▄▅▅▅▅▆▆▇▇█▇▆▆▆▅▆▆▅▅▅▆▆▅
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▅▅▅▄▄▄▄▅▄▅▅▆▆▆▅▆▇█▇▆▅▆▅▆▆▅▄▅▄▃▅▆▅▄▃▂▁▁▁▁
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▇▆▇▆▅▅▅▅▅▄▄▃▇█▇▆▅▅▆▆▅▅▆▆▅▆█▇▆▆▅▄▃▂▃▄▃▂▁▅
wandb:      episode_end_bitrate_blocking_probability_mean ▄▅▆▇▆▇▇██▇▆▆▆█▇▇▇███▇▇▆▅▅▄▂▄▃▂▃▂▂▃▂▂▂▁▂▂
wandb:       episode_end_bitrate_blocking_probability_std ▅▅▇▇██▇▇▇▇▇▇▆▆▆▅▅▄▄▄▄▄▄▄▄▄▄▄▃▃▃▂▂▂▂▁▁▂▁▁
wandb:                  episode_end_cum_returns_iqr_lower ▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▄▄▄▅▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇▇████
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▃▃▂▂▃▃▃▄▃▃▃▄▄▅▅▅▅▆▆▆▆▇▇██
wandb:                       episode_end_cum_returns_mean ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                        episode_end_cum_returns_std ▁▂▂▂▂▃▃▂▂▂▂▃▃▃▄▅▅▄▅▅▄▅▅▅▅▆▆▆▆▆▆▇▇▇██▇▇▇█
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▄▄▅▅▆▆▂▅▄▅▅▂▅▅▅▂▆▂▄▁█▅█▅▅▄█▄▄▄▂▆▅▄▂▇▆▂▁▇
wandb:                            episode_end_returns_std ▅▅▆▅▅▆▄▅▅▅▄▅▆█▆▁▆▆▆▄▅▆▅▅▆▅▇▇▃▄▄▆▆▇▃▆▆▄▅▆
wandb: episode_end_service_blocking_probability_iqr_lower █▇▆▆▅▅▄▃▃█▇▇▆▆▅▄▄▃▂▂▆▅▆▅▅▄▃▃▂▂▁▅▆▅▄▄▃▃▂▁
wandb: episode_end_service_blocking_probability_iqr_upper ▂▂▁▅▄▄▃▃▃▃▂▂▅▅██▇▇▆▆▆▆▅▅▇▇▆▆▅▅▄▄▃▃▂▃▅▅▄▄
wandb:      episode_end_service_blocking_probability_mean ▅▆▇▇▇▇███▇▇▆▆█▇▇▇███▇▇▆▅▅▄▃▄▃▂▃▂▂▃▂▂▂▁▂▂
wandb:       episode_end_service_blocking_probability_std ▆▆▇▇██▇▇▇▇▇▇▆▆▅▅▅▄▄▄▄▄▄▄▄▄▄▄▃▃▃▂▂▂▂▂▁▂▂▁
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▂▂▁▂▂▂▂▂▁▁▂▃▃▄▄▄▄▄▄▄▃▄▅▅▅▅▇▆▆▇█▇▇▇▇▆▆▅▅▅
wandb:                  episode_end_utilisation_iqr_upper ▃▃▃▁▂▂▂▃▂▃▄▅▅▅▄▄▄▄▃▄▄▄▆▅▅▃▅▅▄▅▆▆▆▆▇▆▆▇▇█
wandb:                       episode_end_utilisation_mean ▂▁▁▁▂▃▂▂▁▁▂▂▃▄▄▄▅▅▅▅▄▅▅▅▅▄▅▅▅▆▆▆███▇█▇▆▇
wandb:                        episode_end_utilisation_std ▇██▇▇▆▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▅▃▂▁▁▁▁▁▃▃▃▃▃▄▄▅▆▇
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▄▅▇▄▂▄▃▆▄█▄▇▄▇▄▄▅▃▅▅▃▁▅▃▁▅▆▃▄▃▂▆▂▅▂▄▃▂▂▃
wandb:                                        returns_std ▄█▄▂▄▄▂▅▂▅▃▆▅▆▃▄▅▅▃▅▄▂▆▅▆▅█▁▃▁▃▅▅▆▇▆▂▄▂▂
wandb:             service_blocking_probability_iqr_lower ▃▃▃▃▃▂▂▂▂▂▂▂▁▄▄▃▃▆▆▃▃▆▆▅▅▅██▇▆▇▇▇▄▆▃▆▆▆▆
wandb:             service_blocking_probability_iqr_upper ▆▅▆▄▄▇▂▁▄▃▃▂▄▄▃▇▂▄▄█▇▇▆█▅▇▆▇█▇▇▆▂▆▄▃▆▆▂▁
wandb:                  service_blocking_probability_mean ▅▅▅▃▃▂▂▁▁▁▂▃▄▄▃▃▄▆▅▆▆▇▇▇███▇▇▇▇█▇▇▇▆▆▇▇▆
wandb:                   service_blocking_probability_std ██▇▆▆▆▆▆▆▆▆▆▅▅▆▅▅▄▄▄▅▄▄▄▄▄▄▄▄▄▃▂▂▂▁▁▁▁▁▁
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▃▄▆█▅▃▅▅▄▆▅▅▄▄▄▆▆▆▅▇▅▆▇▅▆▆▄▄▃▂▄▃▂▂▁▃▂▄▆▆
wandb:                              utilisation_iqr_upper ▃▆▇▆██▅▅▃▂▂▅▃▄▄▃▁▂▄▆▄▃▃▄▃▄▃▅▂▂▂▂▄▂▂▂▂▃▂▄
wandb:                                   utilisation_mean ▄▆▇██▆▅▄▅▄▅▆▅▅▄▄▄▅▅█▆▅▅▄▆▆▃▄▃▂▁▁▂▁▁▂▂▅▆▇
wandb:                                    utilisation_std ▇▆▆▅▆██▆▅▅▅▃▄▃▄▄▁▁▃▅▂▃▁▃▄▅▄▅▃▃▃▄▄▁▄▅▄▂▃▂
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2771800.0
wandb:                         accepted_bitrate_iqr_upper 2803000.0
wandb:                              accepted_bitrate_mean 2786708.75
wandb:                               accepted_bitrate_std 20182.25781
wandb:                        accepted_services_iqr_lower 3813.0
wandb:                        accepted_services_iqr_upper 3832.0
wandb:                             accepted_services_mean 3822.68994
wandb:                              accepted_services_std 13.21605
wandb:             bitrate_blocking_probability_iqr_lower 0.06732
wandb:             bitrate_blocking_probability_iqr_upper 0.07464
wandb:                  bitrate_blocking_probability_mean 0.07092
wandb:                   bitrate_blocking_probability_std 0.00516
wandb:                              cum_returns_iqr_lower 4120.48328
wandb:                              cum_returns_iqr_upper 5468.96606
wandb:                                   cum_returns_mean 4810.9292
wandb:                                    cum_returns_std 983.77563
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 2145950.0
wandb:             episode_end_accepted_bitrate_iqr_upper 2173800.0
wandb:                  episode_end_accepted_bitrate_mean 2160353.75
wandb:                   episode_end_accepted_bitrate_std 17856.33398
wandb:            episode_end_accepted_services_iqr_lower 2954.0
wandb:            episode_end_accepted_services_iqr_upper 2970.0
wandb:                 episode_end_accepted_services_mean 2961.85986
wandb:                  episode_end_accepted_services_std 11.19689
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.06694
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.07471
wandb:      episode_end_bitrate_blocking_probability_mean 0.07076
wandb:       episode_end_bitrate_blocking_probability_std 0.00561
wandb:                  episode_end_cum_returns_iqr_lower 3105.73523
wandb:                  episode_end_cum_returns_iqr_upper 4164.72302
wandb:                       episode_end_cum_returns_mean 3678.67993
wandb:                        episode_end_cum_returns_std 828.55914
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower 0.0
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean 1.66396
wandb:                            episode_end_returns_std 5.62508
wandb: episode_end_service_blocking_probability_iqr_lower 0.04163
wandb: episode_end_service_blocking_probability_iqr_upper 0.04679
wandb:      episode_end_service_blocking_probability_mean 0.04425
wandb:       episode_end_service_blocking_probability_std 0.00361
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.15395
wandb:                  episode_end_utilisation_iqr_upper 0.18538
wandb:                       episode_end_utilisation_mean 0.16812
wandb:                        episode_end_utilisation_std 0.02089
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower 0.0
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean 1.00575
wandb:                                        returns_std 4.99362
wandb:             service_blocking_probability_iqr_lower 0.042
wandb:             service_blocking_probability_iqr_upper 0.04675
wandb:                  service_blocking_probability_mean 0.04433
wandb:                   service_blocking_probability_std 0.0033
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 543.39018
wandb:                              utilisation_iqr_lower 0.15356
wandb:                              utilisation_iqr_upper 0.1794
wandb:                                   utilisation_mean 0.16802
wandb:                                    utilisation_std 0.01861
wandb: 
wandb: 🚀 View run golden-elevator-3 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/w80gf86v
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_222800-w80gf86v/logs
Completed training for load = 135
----------------------------------------
Running training with load = 202
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load202.csv
CUDA_VISIBLE_DEVICES=2
I1022 22:37:53.134530 139707968564096 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 22:37:53.135262 139707968564096 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_223800-xn6471xo
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run fallen-mountain-4
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/xn6471xo
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 202
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load202.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 202.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=22.64s
EXECUTION: Elapsed time=545.12s, FPS=3.67e+02
returns: 0.45953 ± 4.66645
lengths: 4000.00000 ± 0.00000
cum_returns: 5341.04248 ± 1130.38135
accepted_services: 3799.15991 ± 45.10947
accepted_bitrate: 2764473.00000 ± 38021.14844
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.24537 ± 0.02193
service_blocking_probability: 0.05021 ± 0.01128
bitrate_blocking_probability: 0.07833 ± 0.01174
wandb: 
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▂▂▃▃▄▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇██
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▂▂▃▃▄▄▅▅▅▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇██
wandb:             bitrate_blocking_probability_iqr_lower ▂▃▁▂▁▂▁▃▃▃▄▄▅▅▅▅▅▅▅▅▅▅▅▅▄▅▅▄▄▅▆▇█▇▇████▆
wandb:             bitrate_blocking_probability_iqr_upper █▄▇▅▅▁▃▁▃▂▃▃▃▄▂▆▅▅▃▅▄▆▅█▇▅▆▆▆▅▄▄▄▄▃▂▂▃▂▄
wandb:                  bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▂▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇███
wandb:                   bitrate_blocking_probability_std ▁▂▃▃▄▅▆▇▇██████▇▇▇▇▇▇▆▆▆▆▆▆▆▆▅▅▅▅▅▆▆▆▆▇▇
wandb:                              cum_returns_iqr_lower ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              cum_returns_iqr_upper ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                   cum_returns_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                                    cum_returns_std ▁▁▁▂▂▂▂▃▃▃▃▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇▇█████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▃▇▆▅▅▄▄▃▅▅▄▇▇█▇▇▆▅▄▅▄▄▃▂▃▃▂▂▁▅▄▄▃▄▅▅▅▄▄▃
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▅▅█▇▇▆▆▅▄▃▃▂▂▅▄▄▄▃▃▂▆▆▅▅▄▄▃▃▃▂▂▄▃▃▂▂▁▃▄▄
wandb:      episode_end_bitrate_blocking_probability_mean ▁▂▃▃▃▄▄▅▅▅▄▄▄▆▅▅▆▇▇▇▇▇▇▆▇▆▅▆▆▆▇▇▇███▇▇██
wandb:       episode_end_bitrate_blocking_probability_std ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇███
wandb:                  episode_end_cum_returns_iqr_lower ▁▂▂▂▂▂▂▂▂▂▂▃▂▂▂▃▃▃▃▃▄▄▅▅▅▅▆▆▆▆▇▆▆▇▇▇▇███
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▂▂▂▂▂▂▂▂▃▃▂▃▃▃▄▄▄▄▄▄▄▄▄▅▅▅▅▅▆▇▇▇▇▇▇▇██
wandb:                       episode_end_cum_returns_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        episode_end_cum_returns_std ▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▄▃▂▃█▅▃▅▅▄▆▃▃▃▆▃▃▁▅▂▄▂▄▂▄▆▅▂▄▃▃▅▂▂▅▃▄▄▂▃
wandb:                            episode_end_returns_std ▅▆▄▃█▄▄▇▆▃▅▄▃▆▆▂▄▃▄▄▃▅▁▁▆█▃▄▃▄▆▄▂▇▆▅▄▅▆▂
wandb: episode_end_service_blocking_probability_iqr_lower ▄▃▃▂▁▁▄▃▄▄▃▃▇▆▆▅██▇▆▆▅▅▄▅▄▄▃▃▂▂▁▄▄▃▄▃▃▂▁
wandb: episode_end_service_blocking_probability_iqr_upper ▆▅▅▄▇▆▆▅▆█▇▇▆▆▆▅▅▅▅▄▄▃▅▅▄▄▄▃▃▃▃▅▄▄▃▃▂▂▂▁
wandb:      episode_end_service_blocking_probability_mean ▁▂▃▃▃▄▄▄▄▄▄▄▄▅▅▅▆▆▆▇▆▆▆▆▆▆▅▆▆▆▇▆▇▇▇▇▇▇██
wandb:       episode_end_service_blocking_probability_std ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▄▃▁▂▃▄▄▃█▅▄▆▆▇▅█▇▅▄▅▄▄▂▆▇█▆▇▆▅▇▃▅▅▄▆▄▅▂▁
wandb:                  episode_end_utilisation_iqr_upper ▅▅▆▆▅▄▄▅▃▄▆▃▄▂▁▂▁▂▂▃▅▅▆▆▆▅▄▄▆█▇▆█▆▆▅▃▆▄▇
wandb:                       episode_end_utilisation_mean █▇▇▇▇▆▆▇▆▆▆▆▇▇▆▇▆▅▆▆▆▇▇▇▇▇▇▆▆▆▅▄▆▄▃▂▂▂▁▂
wandb:                        episode_end_utilisation_std ▆██▇▇▅▅▇▅▆▅▄▅▄▄▄▅▅▆▆▇██▆▄▂▁▂▁▁▂▂▃▂▃▃▅▅▇▇
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▅▅▆▅▅▄▆▄▆█▅▅▅▇█▅▆▄▇▇▇▅▅▇▃▅▅▆▆▄▃▄▄▅▁▆▅▂▆▅
wandb:                                        returns_std ▃▆▁▂▆▂▅▅▃▃▃▅▆▄▆▁▆▄▄▆█▆▅▆▆▄▆▂▇▁▅▂▅▄▄█▄▂▆▅
wandb:             service_blocking_probability_iqr_lower ▁▃▂▂▂▂▃▂▂▂▂▃▄▄▄▄▄▅▄▄▄▄▄▅▆▅▄▄▅▄▆▄▅▅▆▇█▇▇▇
wandb:             service_blocking_probability_iqr_upper ▆█▇▇▄▅▆▃▅▄▃▆▄▃▅▅▄▃▂▄▁▃▂▃▅▄▄▄▄▃▃▂▂▁▂▂▁▃▂▅
wandb:                  service_blocking_probability_mean ▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                   service_blocking_probability_std ▁▂▂▃▄▅▆▇▇██████▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇█
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▄▅▄▅▅▃▄▃▃▃▆▅▄▅▄▇▆▅▄▅▆▆▇▇█▆▆▄▄▃▃▂▂▃▃▁▂▃▄▄
wandb:                              utilisation_iqr_upper ▆▅▆▇▆██▅▄▁▄▆▄▆▄▅▄▃▃▄▆▃▃▅▃▄▂▃▃▃▂▃▃▁▂▂▁▂▃▅
wandb:                                   utilisation_mean ███▇▇▇▆▄▄▄▅▆▆▆▅▆▆▆▆▆▇▆▇▇▇▅▃▃▃▃▄▂▁▂▁▁▁▃▄▅
wandb:                                    utilisation_std ▇▆▇▅▇██▇█▇▇▇▇▄▄▄▄▃▄▅▃▂▁▂▃▄▄▄▄▅▅▆▅▅▆▇▅▅▅▅
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2753750.0
wandb:                         accepted_bitrate_iqr_upper 2782750.0
wandb:                              accepted_bitrate_mean 2764473.0
wandb:                               accepted_bitrate_std 38021.14844
wandb:                        accepted_services_iqr_lower 3793.75
wandb:                        accepted_services_iqr_upper 3814.0
wandb:                             accepted_services_mean 3799.15991
wandb:                              accepted_services_std 45.10947
wandb:             bitrate_blocking_probability_iqr_lower 0.07348
wandb:             bitrate_blocking_probability_iqr_upper 0.08061
wandb:                  bitrate_blocking_probability_mean 0.07833
wandb:                   bitrate_blocking_probability_std 0.01174
wandb:                              cum_returns_iqr_lower 4691.11987
wandb:                              cum_returns_iqr_upper 6099.15491
wandb:                                   cum_returns_mean 5341.04248
wandb:                                    cum_returns_std 1130.38135
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 2133400.0
wandb:             episode_end_accepted_bitrate_iqr_upper 2160450.0
wandb:                  episode_end_accepted_bitrate_mean 2144640.0
wandb:                   episode_end_accepted_bitrate_std 28584.5625
wandb:            episode_end_accepted_services_iqr_lower 2940.0
wandb:            episode_end_accepted_services_iqr_upper 2958.0
wandb:                 episode_end_accepted_services_mean 2945.58984
wandb:                  episode_end_accepted_services_std 29.42502
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.07218
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.08092
wandb:      episode_end_bitrate_blocking_probability_mean 0.07753
wandb:       episode_end_bitrate_blocking_probability_std 0.01053
wandb:                  episode_end_cum_returns_iqr_lower 3490.62305
wandb:                  episode_end_cum_returns_iqr_upper 4619.16797
wandb:                       episode_end_cum_returns_mean 4075.30859
wandb:                        episode_end_cum_returns_std 926.948
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower 0.0
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean 1.3978
wandb:                            episode_end_returns_std 5.1522
wandb: episode_end_service_blocking_probability_iqr_lower 0.0455
wandb: episode_end_service_blocking_probability_iqr_upper 0.05131
wandb:      episode_end_service_blocking_probability_mean 0.0495
wandb:       episode_end_service_blocking_probability_std 0.0095
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.22831
wandb:                  episode_end_utilisation_iqr_upper 0.26235
wandb:                       episode_end_utilisation_mean 0.24534
wandb:                        episode_end_utilisation_std 0.02388
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower 0.0
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean 0.45953
wandb:                                        returns_std 4.66645
wandb:             service_blocking_probability_iqr_lower 0.0465
wandb:             service_blocking_probability_iqr_upper 0.05156
wandb:                  service_blocking_probability_mean 0.05021
wandb:                   service_blocking_probability_std 0.01128
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 544.57493
wandb:                              utilisation_iqr_lower 0.229
wandb:                              utilisation_iqr_upper 0.26181
wandb:                                   utilisation_mean 0.24537
wandb:                                    utilisation_std 0.02193
wandb: 
wandb: 🚀 View run fallen-mountain-4 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/xn6471xo
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_223800-xn6471xo/logs
Completed training for load = 202
----------------------------------------
Running training with load = 270
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load270.csv
CUDA_VISIBLE_DEVICES=2
I1022 22:47:51.844153 140367385254784 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 22:47:51.844875 140367385254784 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_224759-gb6o8sf7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run brisk-dawn-5
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/gb6o8sf7
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 270
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load270.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 270.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=23.03s
EXECUTION: Elapsed time=543.71s, FPS=3.68e+02
returns: 1.22989 ± 6.28552
lengths: 4000.00000 ± 0.00000
cum_returns: 4802.04688 ± 1452.82227
accepted_services: 3686.70996 ± 108.54675
accepted_bitrate: 2662568.00000 ± 78785.08594
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.30565 ± 0.02229
service_blocking_probability: 0.07832 ± 0.02714
bitrate_blocking_probability: 0.11229 ± 0.02650
wandb: 
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▆▇▇▇██████
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▆▆▆▆▆▆▆▆▆▇▇▇▇██████
wandb:             bitrate_blocking_probability_iqr_lower ▁▁▂▂▃▃▂▂▂▂▃▂▂▃▂▃▄▄▃▄▄▄▄▅▆▆▇▆▆▇▇▇▆▆▇█▇▆▇▇
wandb:             bitrate_blocking_probability_iqr_upper ▅▅▅█▅▅▅▅▅▆▆▆▆▄▆▅▃▂▁▂▁▄▄▂▅▅▇▆▅▅▅▅▄▅▅▆▆▇▆█
wandb:                  bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇██████
wandb:                   bitrate_blocking_probability_std ▂▂▁▁▁▁▁▂▂▃▃▄▄▅▅▅▅▅▆▇▇████▇▇▆▆▅▆▆▇▇███▇▇▆
wandb:                              cum_returns_iqr_lower ▁▁▁▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              cum_returns_iqr_upper ▁▁▁▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇███
wandb:                                   cum_returns_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                    cum_returns_std ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇██████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▂▂▂▂▂▂▃▃▃▄▄▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇███
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▄▃▂▂▁▁▃▄▃▂▂▃▃▄▃▅▅▅▇█▇▇▇▇▆▅▆▆▅▅▅▅▄▄▅▅▆▆██
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▃▄▃▂▅▅▅▇▆▅▄▃▃▂▁▃▃▃▂▂▅▄▄▅█▇▆▆▅▅▄▄█▇▇▆▅▅▄▃
wandb:      episode_end_bitrate_blocking_probability_mean ▁▁▂▂▃▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▆▅▅▆▆▆▆▆▆▇▇▇▇▇██
wandb:       episode_end_bitrate_blocking_probability_std ████▇▇▇█▇▇▇▇▇▇▇▆▆▅▅▄▄▄▄▄▄▄▃▃▃▂▃▃▃▃▃▃▂▂▁▁
wandb:                  episode_end_cum_returns_iqr_lower ▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▇▇▇▇▇█▇███████
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▁▂▂▂▂▃▂▂▃▃▃▃▃▃▃▄▃▃▄▄▄▄▅▅▆▆▆▆▆▆▇▇▇▇████
wandb:                       episode_end_cum_returns_mean ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        episode_end_cum_returns_std ▁▁▁▁▂▂▂▂▂▂▂▂▂▃▃▃▄▄▄▄▄▄▄▄▄▄▄▅▅▆▆▆▆▆▇▇▇▇██
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▅▂▂▃█▅▂▃▃▄▄▄▅▁▆▅▆▄▄▂▅▃▅▄▄▆▆▄▄▄▄▅▅▂▂▃▅▅▄▅
wandb:                            episode_end_returns_std █▂▄▁▇▆▁▂▅▂▃▅▅▃▆▃▇█▃▃▅▆▂▃▇▇▆▇▃▄▆▅▅▇▂▄▅▅▇▅
wandb: episode_end_service_blocking_probability_iqr_lower ▂▂▁▄▃▃▆▆▅▅▄▃▆▆▅▅▇▇▆▅▆▅▄▄▃▂▆▅▄▄▃▂▅▄▃▄▃▂▆█
wandb: episode_end_service_blocking_probability_iqr_upper ▁▅▇▆▅▆▅▄▆▆▆▅▄▃▂▂▅▄▇▆▅▅██▇▆▆▅▇▆▅▅▄▃▂▃▅▄▃▂
wandb:      episode_end_service_blocking_probability_mean ▁▁▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▅▆▆▅▅▆▆▆▆▆▆▇▇▇▇▇██
wandb:       episode_end_service_blocking_probability_std ███▇▇▆▆▇▆▆▆▅▅▅▅▄▄▃▃▂▃▂▃▂▂▂▂▁▂▁▂▂▂▂▂▂▂▂▁▁
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▄▅▃▂▂▃▁▂▂▃▄▁▄▆▆▅▇▅▄▂▂▁▆▁▅▄█▅▄▅▆▅▅▆▄▆▅▆▆▅
wandb:                  episode_end_utilisation_iqr_upper ▃▆▃▃▄▃▁▁▂▄▅▆▅▅▅▅▅▅▆▆▇▇█▄▅▅▅▅▄▆▆▆██▇▆▆▄▅▅
wandb:                       episode_end_utilisation_mean ▃▃▂▁▂▂▁▁▁▂▄▄▅▅▆▆▆▅▅▅▄▅▅▆▆▆▇▆▆▆▆▆██▇▆▇▆▅▅
wandb:                        episode_end_utilisation_std ▆▇▆▇▇▇▆█▇▆█▆▄▄▅▄▅▃▅▆▆▆▆▄▂▁▁▂▅▅▅▅▅▅▄▅▅▅▅▇
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▃▇▆▄█▆▅▆▆█▇▆▅▆▅█▆▅▇▄▄▅▄▆▁▃▄▅▆▄▃▆▃▄▂▄▅▄▃▆
wandb:                                        returns_std ▂▆▃▂█▅▄▅▅▃▅▅▅▃▄▆▅▆▅▆▄▅▄▆▃▂▃▃▅▃▃▅▄▃▄▅▄▄▁▅
wandb:             service_blocking_probability_iqr_lower ▂▃▃▂▄▄▄▂▁▂▁▂▃▃▃▃▅▄▅▅▅▆▆▇▆▇▇▇▇█▆▇▇▇▇█▆▆▇▆
wandb:             service_blocking_probability_iqr_upper ▄▃▄▃▂▂▃▇▆▆▆▆▆▆▆▅▅▃▂▂▁▂▁▃▂▄▅▄▅▄▆▇▇▆▆██▆██
wandb:                  service_blocking_probability_mean ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇██████
wandb:                   service_blocking_probability_std ▁▁▁▁▁▁▂▂▃▃▃▄▄▄▅▅▅▅▆▇▇████▇▇▆▆▅▆▆▆▇▇▇▇▇▆▆
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▅▇▇█▅▆▄▄▂▃▄▆▂▅▅▅▃▆▅▅█▇▇▅█▇▅▆▅▆▅▄▃▅▅▃▂▁▄▅
wandb:                              utilisation_iqr_upper ▅▆██▆▅▆▅▄▄▅▄▄▄▃▁▁▃▃▄▃▂▄▃▃▃▄▃▃▂▂▄▂▂▃▂▃▂▄▄
wandb:                                   utilisation_mean ▅▇▇█▇▅▅▅▄▃▅▅▃▄▃▁▁▄▅▄▄▅▆▅▆▆▅▅▅▅▄▃▃▂▂▂▁▃▅▆
wandb:                                    utilisation_std ▄▃▄▄▄▅▇▆▆▇██▇▆▄▃▂▂▃▅▆▄▄▅▄▅▅▃▂▁▂▃▃▁▅▅▅▃▁▂
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2660700.0
wandb:                         accepted_bitrate_iqr_upper 2701750.0
wandb:                              accepted_bitrate_mean 2662568.0
wandb:                               accepted_bitrate_std 78785.08594
wandb:                        accepted_services_iqr_lower 3694.25
wandb:                        accepted_services_iqr_upper 3733.0
wandb:                             accepted_services_mean 3686.70996
wandb:                              accepted_services_std 108.54675
wandb:             bitrate_blocking_probability_iqr_lower 0.10019
wandb:             bitrate_blocking_probability_iqr_upper 0.11319
wandb:                  bitrate_blocking_probability_mean 0.11229
wandb:                   bitrate_blocking_probability_std 0.0265
wandb:                              cum_returns_iqr_lower 4089.74908
wandb:                              cum_returns_iqr_upper 5567.5752
wandb:                                   cum_returns_mean 4802.04688
wandb:                                    cum_returns_std 1452.82227
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 2062800.0
wandb:             episode_end_accepted_bitrate_iqr_upper 2098700.0
wandb:                  episode_end_accepted_bitrate_mean 2070328.0
wandb:                   episode_end_accepted_bitrate_std 58505.46094
wandb:            episode_end_accepted_services_iqr_lower 2865.0
wandb:            episode_end_accepted_services_iqr_upper 2894.25
wandb:                 episode_end_accepted_services_mean 2864.04492
wandb:                  episode_end_accepted_services_std 79.56879
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.09881
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.11224
wandb:      episode_end_bitrate_blocking_probability_mean 0.10947
wandb:       episode_end_bitrate_blocking_probability_std 0.02512
wandb:                  episode_end_cum_returns_iqr_lower 3026.24048
wandb:                  episode_end_cum_returns_iqr_upper 4349.99805
wandb:                       episode_end_cum_returns_mean 3659.28857
wandb:                        episode_end_cum_returns_std 1097.32556
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower 0.0
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean 1.50873
wandb:                            episode_end_returns_std 6.16012
wandb: episode_end_service_blocking_probability_iqr_lower 0.06607
wandb: episode_end_service_blocking_probability_iqr_upper 0.07551
wandb:      episode_end_service_blocking_probability_mean 0.07582
wandb:       episode_end_service_blocking_probability_std 0.02568
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.29289
wandb:                  episode_end_utilisation_iqr_upper 0.32347
wandb:                       episode_end_utilisation_mean 0.30667
wandb:                        episode_end_utilisation_std 0.0239
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower 0.0
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean 1.22989
wandb:                                        returns_std 6.28552
wandb:             service_blocking_probability_iqr_lower 0.06675
wandb:             service_blocking_probability_iqr_upper 0.07644
wandb:                  service_blocking_probability_mean 0.07832
wandb:                   service_blocking_probability_std 0.02714
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 543.16779
wandb:                              utilisation_iqr_lower 0.29101
wandb:                              utilisation_iqr_upper 0.31966
wandb:                                   utilisation_mean 0.30565
wandb:                                    utilisation_std 0.02229
wandb: 
wandb: 🚀 View run brisk-dawn-5 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/gb6o8sf7
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_224759-gb6o8sf7/logs
Completed training for load = 270
----------------------------------------
Running training with load = 337
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load337.csv
CUDA_VISIBLE_DEVICES=2
I1022 22:57:49.967713 139649314286464 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 22:57:49.968596 139649314286464 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_225757-x4fvg3z7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run misty-lion-6
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/x4fvg3z7
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 337
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load337.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 337.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=23.34s
EXECUTION: Elapsed time=545.27s, FPS=3.67e+02
returns: 0.88244 ± 6.56695
lengths: 4000.00000 ± 0.00000
cum_returns: 3445.92310 ± 2240.03491
accepted_services: 3496.96484 ± 206.61922
accepted_bitrate: 2501577.00000 ± 144725.50000
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.34310 ± 0.02998
service_blocking_probability: 0.12576 ± 0.05165
bitrate_blocking_probability: 0.16591 ± 0.04911
wandb: / 0.196 MB of 0.196 MB uploaded
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇████
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:             bitrate_blocking_probability_iqr_lower ▁▂▁▁▁▂▂▂▂▂▃▃▃▃▃▃▂▃▃▃▄▃▄▄▅▅▅▅▆▆▆▆▇▇▇▇████
wandb:             bitrate_blocking_probability_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃▃▃▄▅▅▆▆▇▇▇██
wandb:                  bitrate_blocking_probability_mean ▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇███
wandb:                   bitrate_blocking_probability_std ▆▆▇▇▇█████▇▇▆▆▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▂▂▂▂▁▁▁▁▁▁▁
wandb:                              cum_returns_iqr_lower ▁▁▁▁▁▁▂▂▂▁▂▂▂▂▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇██▇▇▇▇███
wandb:                              cum_returns_iqr_upper ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇████
wandb:                                   cum_returns_mean ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇████
wandb:                                    cum_returns_std ▁▁▁▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇██
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▁▅▅▄▆▆▅█▇██▇▅▅▇▆▆▅▆▆▅▄▄▄▄▃▂▂▃▄▃▃▂▁▁▃▅▅▅▄
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▅▇▇█▇▇▆▅▅▄▄▆▅▄▃▃▅▄▄▃▂▁▃▃▄▄▃▄▃▅▇▇▆▅▅▇▇▆▆▆
wandb:      episode_end_bitrate_blocking_probability_mean ▁▁▂▂▂▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▅▆▆▆▇▇▇▇▇██
wandb:       episode_end_bitrate_blocking_probability_std ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇██
wandb:                  episode_end_cum_returns_iqr_lower ▁▃▁▁▃▃▄▃▃▃▃▃▄▄████▆▇▆▅▅▅▃▂▆▄▃▁▁▁▁▁▂▂▅▃▃▄
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▂▂▁▁▂▂▂▂▃▄▄▄▄▄▄▄▅▅▆▆▆▆▆▇▆▆▆▆▆▇▇▇▇▇▇▇██
wandb:                       episode_end_cum_returns_mean ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:                        episode_end_cum_returns_std ▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇█
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▃▆▂▂▃▇▅▅▇▅▃▂▃▃█▃▆▄▃▄▅▄▅▄▃▄▆▄▆▆▂▅▂▄▅▃▄▂▁▅
wandb:                            episode_end_returns_std ▄▅▃▂▄▅▄▄▇▄▁▄▃▆▇▂▆█▂▄▅▅▄▄▅▅▅▅▅▅▄▅▂▇▄▄▅▂▅▅
wandb: episode_end_service_blocking_probability_iqr_lower ▂▁▃▃▂▁▄▃▂▅▄▇▆█▇▆▅▅▄▄▆▆▅▆▆▇▆▆▆▆█▇▇▆▅▇▇▆█▇
wandb: episode_end_service_blocking_probability_iqr_upper ▇▇▆▆▅▄▄▄▃▂▁▃▅▄▃▂▁▃▃▅▄▃▂▂▃▃▅▅▆▆▅▄▅▇▆▇█▇▆▅
wandb:      episode_end_service_blocking_probability_mean ▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:       episode_end_service_blocking_probability_std ▁▁▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▄▄▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇▇██
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▂▄▃▂▂▃▃▃▂▁▃▄▄▆▂▂▃▁▂▁▃▃▄▅▆▆▆█▆▅▇▅▃▆▆▆▄▅▅▅
wandb:                  episode_end_utilisation_iqr_upper ▅▅▆▃▅▄▃▄▁▁▄▂▅▃▄▇▆▆▁▃▃▄▄▄▅▅▄▅██▄▄█▇▅▅█▅▄▆
wandb:                       episode_end_utilisation_mean ▂▂▂▁▂▂▁▁▂▂▁▂▂▄▄▄▄▃▃▃▃▃▄▅▅▅▆▆▆▇█▇██▇▆▆▆▅▆
wandb:                        episode_end_utilisation_std ▆▆▆▅▆▆▄▆▆▆▇▆▇█▇▇█▆▇▆▇▆▄▄▃▂▁▂▄▃▃▃▄▃▆▇█▇▇▇
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▃▆▅▅▅▄▆▆▅▇▆█▇▅▇▄▆▃▇▅▇▆▆▆▁▄▄▄▃▂▂▅▅▂▄▆▆▄▃▆
wandb:                                        returns_std ▃▅▄▃▇▂▆▅▃▄▅▇▆▃▆▅▅▄▆▇▇▆▆█▅▄▅▃▄▁▄▆▇▂▆▇▅▅▃▆
wandb:             service_blocking_probability_iqr_lower ▁▁▁▂▂▃▃▃▃▃▃▄▄▄▃▄▄▄▄▄▅▅▄▆▇▇▇▇▇▇▇█▇▇▇▇███▇
wandb:             service_blocking_probability_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃▄▅▅▆▆▇▇██▇█
wandb:                  service_blocking_probability_mean ▁▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇███
wandb:                   service_blocking_probability_std ▆▆▇▇▇█████▇▇▆▆▅▅▅▅▅▄▄▄▄▄▄▄▃▃▃▂▂▂▂▁▁▁▁▁▁▁
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▁▄▃▃▃▄▃▂▃▃▄▅▅▄▅▄▄▆▅▅▂▇█▆▁▄▂▄▅▅▆▃▁▂▃▁▂▅▃▄
wandb:                              utilisation_iqr_upper ▆▅▆██▆▇▇▇▃▃▅▁▃▃▄▅▂▃▃▁▄▂▄▄▄▂▂▃▂▅▄▃▄▁▃▃▄▅▃
wandb:                                   utilisation_mean ▃▄▅█▇██▅▆▄▅█▆▇▆▆▅▆▄▅▅▅▆▆▃▄▂▃▄▃▅▃▁▃▂▃▁▂▃▃
wandb:                                    utilisation_std ▆▇▆▆▇▇▆▅▄▅▄▂▂▁▂▃▃▂▃▃▃▄▄▆▆▅▆▄▄▃▄▄▅▅▆▇▇███
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2468900.0
wandb:                         accepted_bitrate_iqr_upper 2583800.0
wandb:                              accepted_bitrate_mean 2501577.0
wandb:                               accepted_bitrate_std 144725.5
wandb:                        accepted_services_iqr_lower 3439.75
wandb:                        accepted_services_iqr_upper 3615.0
wandb:                             accepted_services_mean 3496.96484
wandb:                              accepted_services_std 206.61922
wandb:             bitrate_blocking_probability_iqr_lower 0.13761
wandb:             bitrate_blocking_probability_iqr_upper 0.18002
wandb:                  bitrate_blocking_probability_mean 0.16591
wandb:                   bitrate_blocking_probability_std 0.04911
wandb:                              cum_returns_iqr_lower 2540.88287
wandb:                              cum_returns_iqr_upper 4878.53174
wandb:                                   cum_returns_mean 3445.9231
wandb:                                    cum_returns_std 2240.03491
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 1961400.0
wandb:             episode_end_accepted_bitrate_iqr_upper 2014650.0
wandb:                  episode_end_accepted_bitrate_mean 1954386.875
wandb:                   episode_end_accepted_bitrate_std 118676.1875
wandb:            episode_end_accepted_services_iqr_lower 2753.5
wandb:            episode_end_accepted_services_iqr_upper 2811.25
wandb:                 episode_end_accepted_services_mean 2728.89502
wandb:                  episode_end_accepted_services_std 168.73499
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.13332
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.15498
wandb:      episode_end_bitrate_blocking_probability_mean 0.15929
wandb:       episode_end_bitrate_blocking_probability_std 0.05173
wandb:                  episode_end_cum_returns_iqr_lower 2059.79321
wandb:                  episode_end_cum_returns_iqr_upper 3816.7962
wandb:                       episode_end_cum_returns_mean 2699.25
wandb:                        episode_end_cum_returns_std 1796.00476
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower 0.0
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean 1.10091
wandb:                            episode_end_returns_std 6.49532
wandb: episode_end_service_blocking_probability_iqr_lower 0.09285
wandb: episode_end_service_blocking_probability_iqr_upper 0.11149
wandb:      episode_end_service_blocking_probability_mean 0.11943
wandb:       episode_end_service_blocking_probability_std 0.05445
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.33182
wandb:                  episode_end_utilisation_iqr_upper 0.36344
wandb:                       episode_end_utilisation_mean 0.34465
wandb:                        episode_end_utilisation_std 0.02883
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower 0.0
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean 0.88244
wandb:                                        returns_std 6.56695
wandb:             service_blocking_probability_iqr_lower 0.09625
wandb:             service_blocking_probability_iqr_upper 0.14006
wandb:                  service_blocking_probability_mean 0.12576
wandb:                   service_blocking_probability_std 0.05165
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 544.72722
wandb:                              utilisation_iqr_lower 0.33108
wandb:                              utilisation_iqr_upper 0.35993
wandb:                                   utilisation_mean 0.3431
wandb:                                    utilisation_std 0.02998
wandb: 
wandb: 🚀 View run misty-lion-6 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/x4fvg3z7
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_225757-x4fvg3z7/logs
Completed training for load = 337
----------------------------------------
Running training with load = 405
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load405.csv
CUDA_VISIBLE_DEVICES=2
I1022 23:07:50.222089 140536184302464 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 23:07:50.222857 140536184302464 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_230758-gqz03efp
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run laced-frog-7
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/gqz03efp
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 405
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load405.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 405.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=23.62s
EXECUTION: Elapsed time=545.22s, FPS=3.67e+02
returns: -0.31262 ± 6.21658
lengths: 4000.00000 ± 0.00000
cum_returns: 1794.42236 ± 2936.89917
accepted_services: 3294.03003 ± 289.70532
accepted_bitrate: 2336182.00000 ± 202185.59375
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.37106 ± 0.03485
service_blocking_probability: 0.17649 ± 0.07243
bitrate_blocking_probability: 0.22111 ± 0.06739
wandb: / 0.196 MB of 0.196 MB uploaded
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇███
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                               accepted_bitrate_std ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇███
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                              accepted_services_std ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇███
wandb:             bitrate_blocking_probability_iqr_lower ▁▁▁▁▁▁▁▃▃▃▃▃▄▄▃▄▄▄▄▅▆▅▅▆▆▅▆▆▆▆▆▆▇▇▇▇████
wandb:             bitrate_blocking_probability_iqr_upper ▂▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▁▂▃▄▄▄▄▄▃▄▆▆▆▇▇▇▇███▇▇▇▇
wandb:                  bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇███
wandb:                   bitrate_blocking_probability_std ▇▇▇▇▇███▇▇▇▆▆▅▄▄▃▃▂▁▁▁▁▁▁▁▂▂▂▂▃▃▃▃▄▅▅▅▆▆
wandb:                              cum_returns_iqr_lower ▇▆███▇▇▄▃▃▃▃▃▃▁▂▃▅▅▁▃▅▆▄▅▄▄▅▆▅▅▅▄▃▁▂▂▂▁▃
wandb:                              cum_returns_iqr_upper ▁▁▂▂▂▃▃▃▃▃▄▃▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▇▇▇█▇███▇█
wandb:                                   cum_returns_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███████
wandb:                                    cum_returns_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇███
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇██
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▁▃▁▃▃▄▃▄▅▄▄▅▅▅▅▅▅▆▆▅▄▃▃▆█▇▇▅▄▃▂▃▂▄▃▃▃▅▃▃
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▆▇▆█▇▇▆▆▅▄▃▂▂▂▃▃▅▅▄▅▄▄▃▃▂▂▃▄▄▃▂▂▃▄▃▂▁▁▂▃
wandb:      episode_end_bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:       episode_end_bitrate_blocking_probability_std ▃▃▂▂▂▂▁▁▁▁▁▁▂▂▂▂▂▂▂▃▄▅▅▆▆▇▇▇▇▇▇▇██████▇█
wandb:                  episode_end_cum_returns_iqr_lower ▄▄▃▃▂▃▂▂▂▂▁▁▂▁▂▃▂▄▄▅▆▆▆▇▆▆▆▆▆▅▆▆▇█▇▆▇█▇▇
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▆▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇██
wandb:                       episode_end_cum_returns_mean ▁▁▁▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▇▇▇▇▇▇▇▇███
wandb:                        episode_end_cum_returns_std ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower █████████████████▁███▁██████████████████
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▆▅▄▃▇▅▁▄▄▇▃▁▅▂█▄▅▁▇▄▆▃▅▃▅▆▅▆▅▇▅▅▁▃▄▅▃▄▅▃
wandb:                            episode_end_returns_std ▇▅▇▃█▇▂▅▆▅▃▂▄▄█▁▇▅▆▄▆▇▃▅▄▇▇█▅▇▇▄▂█▃▅▅▅▇▅
wandb: episode_end_service_blocking_probability_iqr_lower ▂▂▁▁▂▂▁▂▂▃▂▃▄▄▅▅▅▇▆▆▇▇▇█▇▇▇▇█████▇▇▆▇▇▇▇
wandb: episode_end_service_blocking_probability_iqr_upper ▆▇▇█▇▇▇▆▆▅▅▄▄▄▄▄▅▅▄▅▄▄▃▃▃▃▃▄▃▃▃▂▃▃▂▂▁▁▂▂
wandb:      episode_end_service_blocking_probability_mean ▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:       episode_end_service_blocking_probability_std ▃▃▂▂▂▂▁▁▁▁▁▁▂▁▁▂▂▂▂▂▄▄▄▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇█
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▆▃▃▁▁▄▇▆▄▅▂▂▅▃▆▅▅▄▂▂▆▄▆▇█▆▅▅▃▃▅▅▆▅▄▂▁▅▁▅
wandb:                  episode_end_utilisation_iqr_upper ▅▅▆▆▅▄▆▇▆█▅▅▅▅▃▄▅▃▄▄▁▂▂▁▃▁▄▆▄▂▄▄▆▅▅▄▂▄▇▄
wandb:                       episode_end_utilisation_mean ██████▆▅▅▅▅▅▅▅▅▆▆▄▃▄▃▃▄▄▄▃▃▂▂▂▃▃▄▃▃▁▁▁▁▂
wandb:                        episode_end_utilisation_std ▂▂▂▂▂▁▁▁▂▂▂▂▁▁▁▁▂▁▂▂▃▃▃▂▃▂▂▃▄▄▄▄▅▅▆▇▇▇██
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ████████████████████████▂▂████████▁█████
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▃▇▅▂▅▅▅▄▄▇▅▇▆▆█▆▇▅▇▄▇▄▅▅▂▂▄▃▆▄▄▆▄▄▁▃▄▃▄▂
wandb:                                        returns_std ▃█▃▁█▅▂▄▅▅▄▆▅▆▇▇▄▄▇▅█▄▅▅▆▅▃▂█▅▅▇▅▆▅▅▄▂▃▂
wandb:             service_blocking_probability_iqr_lower ▁▂▃▃▂▂▃▃▃▃▃▄▄▄▄▄▄▄▄▅▅▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:             service_blocking_probability_iqr_upper ▂▁▁▁▁▂▃▃▃▃▃▃▃▃▃▃▃▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆███▇▇▇▇▇
wandb:                  service_blocking_probability_mean ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇███
wandb:                   service_blocking_probability_std ▇▇▇▇▇███▇▇▇▆▅▅▄▄▃▃▂▁▁▁▁▁▁▁▂▂▂▂▃▃▃▃▄▄▅▅▅▆
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▅▇▇█▇▅▃▆▄▅▆▄▃▄▅▄▄▆▄▅▆██▅▅▅▅▆▆▄▄▃▅▅▅▁▅▄▆▃
wandb:                              utilisation_iqr_upper ▆▆▃▇▆▇▃▅▄█▇▇▄▃▄▄▅█▇▃▃▅▂▄▃▄▅▄▅▄▃▃▁▂▅▅▅▆▄▇
wandb:                                   utilisation_mean █▇▆▆▆▅▅▆▅▆▆▇▅▆▆▅▅█▇▇▇▇▇▆▄▄▃▃▄▃▃▂▂▂▂▂▁▁▃▂
wandb:                                    utilisation_std ▁▁▁▂▃▅▅▅▄▃▃▂▃▂▂▃▄▄▄▄▄▃▂▄▅▆▆▆▅▅▆▆▇▆███▇▇▇
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2282400.0
wandb:                         accepted_bitrate_iqr_upper 2465400.0
wandb:                              accepted_bitrate_mean 2336182.0
wandb:                               accepted_bitrate_std 202185.59375
wandb:                        accepted_services_iqr_lower 3213.0
wandb:                        accepted_services_iqr_upper 3482.25
wandb:                             accepted_services_mean 3294.03003
wandb:                              accepted_services_std 289.70532
wandb:             bitrate_blocking_probability_iqr_lower 0.17848
wandb:             bitrate_blocking_probability_iqr_upper 0.24107
wandb:                  bitrate_blocking_probability_mean 0.22111
wandb:                   bitrate_blocking_probability_std 0.06739
wandb:                              cum_returns_iqr_lower 604.58664
wandb:                              cum_returns_iqr_upper 3642.14484
wandb:                                   cum_returns_mean 1794.42236
wandb:                                    cum_returns_std 2936.89917
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 1801850.0
wandb:             episode_end_accepted_bitrate_iqr_upper 1927050.0
wandb:                  episode_end_accepted_bitrate_mean 1833197.125
wandb:                   episode_end_accepted_bitrate_std 157195.03125
wandb:            episode_end_accepted_services_iqr_lower 2533.25
wandb:            episode_end_accepted_services_iqr_upper 2715.75
wandb:                 episode_end_accepted_services_mean 2580.40503
wandb:                  episode_end_accepted_services_std 226.23892
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.16993
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.23019
wandb:      episode_end_bitrate_blocking_probability_mean 0.21144
wandb:       episode_end_bitrate_blocking_probability_std 0.06791
wandb:                  episode_end_cum_returns_iqr_lower 767.44327
wandb:                  episode_end_cum_returns_iqr_upper 3002.27942
wandb:                       episode_end_cum_returns_mean 1469.53711
wandb:                        episode_end_cum_returns_std 2251.42773
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower 0.0
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean -0.0286
wandb:                            episode_end_returns_std 6.50876
wandb: episode_end_service_blocking_probability_iqr_lower 0.12367
wandb: episode_end_service_blocking_probability_iqr_upper 0.18256
wandb:      episode_end_service_blocking_probability_mean 0.16734
wandb:       episode_end_service_blocking_probability_std 0.073
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.36369
wandb:                  episode_end_utilisation_iqr_upper 0.3919
wandb:                       episode_end_utilisation_mean 0.37285
wandb:                        episode_end_utilisation_std 0.03228
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower -0.875
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean -0.31262
wandb:                                        returns_std 6.21658
wandb:             service_blocking_probability_iqr_lower 0.12944
wandb:             service_blocking_probability_iqr_upper 0.19675
wandb:                  service_blocking_probability_mean 0.17649
wandb:                   service_blocking_probability_std 0.07243
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 544.67376
wandb:                              utilisation_iqr_lower 0.36122
wandb:                              utilisation_iqr_upper 0.39289
wandb:                                   utilisation_mean 0.37106
wandb:                                    utilisation_std 0.03485
wandb: 
wandb: 🚀 View run laced-frog-7 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/gqz03efp
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_230758-gqz03efp/logs
Completed training for load = 405
----------------------------------------
Running training with load = 472
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load472.csv
CUDA_VISIBLE_DEVICES=2
I1022 23:17:50.077556 140184287083392 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 23:17:50.078945 140184287083392 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_231801-30shcs25
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run floral-breeze-8
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/30shcs25
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 472
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load472.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 472.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=24.18s
EXECUTION: Elapsed time=545.36s, FPS=3.67e+02
returns: 0.32895 ± 6.94570
lengths: 4000.00000 ± 0.00000
cum_returns: 436.88217 ± 3619.31128
accepted_services: 3124.38501 ± 360.33511
accepted_bitrate: 2199838.00000 ± 249531.35938
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.39678 ± 0.03653
service_blocking_probability: 0.21890 ± 0.09008
bitrate_blocking_probability: 0.26656 ± 0.08328
wandb: 
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇████
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇████
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇████
wandb:             bitrate_blocking_probability_iqr_lower ▁▁▂▂▂▂▂▃▃▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇███
wandb:             bitrate_blocking_probability_iqr_upper ▁▁▁▁▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▆▆▆▆▆▆▆▅▆▇▇▇▇▇▇▇█████
wandb:                  bitrate_blocking_probability_mean ▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇██████
wandb:                   bitrate_blocking_probability_std ▇███▇▇██▇▇▇▆▆▇▇▇▇▆▆▆▆▆▆▆▅▅▅▅▄▄▄▃▃▃▂▂▂▂▁▁
wandb:                              cum_returns_iqr_lower █▇▆▆▆▆▇▆▆▆▆▅▆▄▄▅▅▅▅▅▅▄▃▃▃▂▃▃▂▂▂▂▁▂▂▁▁▁▁▁
wandb:                              cum_returns_iqr_upper ▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▅▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇█████
wandb:                                   cum_returns_mean █▇▇▇████▇▇▇▇▇▆▅▅▄▄▅▄▄▃▂▁▁▁▁▁▁▁▁▁▂▂▂▃▃▃▄▄
wandb:                                    cum_returns_std ▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▆▇▇▇▇██
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇████
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▁▂▁▁▂▂▁▂▃▃▃▃▂▄▅▄▄▄▅▅▇▇██▇▇▆▅▅▄▄▄▄▄▄▅▅▅▅▇
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▁▂▂▂▂▁▁▂▂▂▂▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▂▂▃▃▄▄▅▅▆▇▇█
wandb:      episode_end_bitrate_blocking_probability_mean ▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇███████
wandb:       episode_end_bitrate_blocking_probability_std ▁▁▂▂▃▃▃▄▄▅▅▅▆▆▆▇▇▇▇▇▇███████▇▇▆▆▆▆▆▆▅▅▅▅
wandb:                  episode_end_cum_returns_iqr_lower █████████▇▇▇▆▆▅▅▅▄▄▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁
wandb:                  episode_end_cum_returns_iqr_upper ▁▁▂▂▂▁▁▁▂▂▁▁▂▂▂▂▃▃▄▄▄▄▅▄▅▄▅▅▄▄▅▅▅▆▆▆▇▇██
wandb:                       episode_end_cum_returns_mean ██▇█▇▅▅▅▆▆▅▅▅▄▅▄▃▃▃▃▃▂▂▁▂▂▂▂▃▃▄▄▄▄▄▅▆▇▆▇
wandb:                        episode_end_cum_returns_std ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▃▃▃▃▇▁▃▇▃▃▃▃▃▁█▃▃▁█▃▃▃██▇▃███████▃▇███▃▃
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▄▄▂▅▄▂▂▆▆▄▅▅▅▃▇▄▃▃▅▃▆▂▄▃▅▅▄▅▇▄█▆▆▇▃▆▅▆▁▆
wandb:                            episode_end_returns_std ▅▆▃▆▄▃▂▆▇▄▆▆▃▆▇▃▄▆▅▄▆▃▃▃▆▆▄▆▇▄▇▇▆█▃▄▆▅▁▆
wandb: episode_end_service_blocking_probability_iqr_lower ▂▂▁▂▂▃▂▃▄▃▃▂▃▄▅▅▅▇▆▆▆▆▇▇▆▅▆▆▇█▇██▇█▇▆█▇▇
wandb: episode_end_service_blocking_probability_iqr_upper ▇███▇▇▇████▇▇▇▇▇▆▆▆▆▅▆▅▅▄▄▄▄▄▃▃▃▂▂▂▁▁▁▂▃
wandb:      episode_end_service_blocking_probability_mean ▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇███████
wandb:       episode_end_service_blocking_probability_std ▁▁▂▂▃▃▃▄▄▅▅▅▆▆▇▇▇▇▇▇████████▇▇▇▇▆▆▆▆▅▅▅▅
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▅▅▄█▇▅▃▆█▇▆▅▄▆▄▆▄▃▃▁▄▃▃▃▅▄▄▄▂▁▆▄▄▄▄▄▄▆▆▆
wandb:                  episode_end_utilisation_iqr_upper ▄▂▄▄▂▁▂▁▁▁▃▃▁▂▂▇▃▆▅▆▃▄▄▄▄▅▆▇▇██▇▆▇▆▇▆█▇▇
wandb:                       episode_end_utilisation_mean ▆▆▄▄▄▁▁▂▂▂▃▃▅▅▅▃▃▂▁▂▃▄▃▃▆▅▆▃▃▄█▅▆▄▅▅▅▆▆▆
wandb:                        episode_end_utilisation_std ▂▁▂▂▃▂▃▄▅▆▆▆▆▆▆▅▆▇▇▇██▇▇▆▅▅▆▅▅▄▄▆▆▅▄▅▅▄▅
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▃▁██████████▁▃▃▃█▃▃▃▃▁▃█▃▃█▃██████▁█████
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▅▃▄▄▄▅▄▆▄▅▄▇▂▂▄▄█▅▆▁▄▂▃█▄▃▃▄▆▄▅▆▆▄▁▆▃▅▄▄
wandb:                                        returns_std ▆▅▂▃▄▃▄▆▄▁▂▆▅▄▄▆▅▆▆▂▄▃▂█▆▄▁▂▆▃▃▅▇▁▂▆▃▅▃▃
wandb:             service_blocking_probability_iqr_lower ▁▁▂▃▂▂▂▃▄▃▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇█▇▇▇▇▇█▇▇▇██▇█
wandb:             service_blocking_probability_iqr_upper ▂▂▁▁▁▁▂▃▃▃▃▃▄▅▄▅▅▅▆▆▆▆▆▆▅▅▅▅▅▅▅▆▆▆▇██▇▇█
wandb:                  service_blocking_probability_mean ▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇██████
wandb:                   service_blocking_probability_std ▇███▇████▇▇▇▇▇▇▇▇▆▆▆▆▆▆▆▆▅▅▅▅▄▄▄▃▃▃▂▂▂▁▁
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▇▆▅▅▆▆▆▆▅▅▄▄▄▄▁▄▄▃▄▄▃▄▃▅▆▅▄▆▅▄▆▅▅▅▄▄▃▅▇█
wandb:                              utilisation_iqr_upper ▆▅▆▇██▇▃▅▄▃▄▅▄▄▃▅▇▃▄▃▄▃▂▃▄▃▁▂▃▅▅▄▅▃▄▂▄▄▆
wandb:                                   utilisation_mean ▄▄▄▅▅▇▆▄▅▄▄▄▂▂▂▃▅▅▄▃▂▁▂▂▃▃▂▃▃▃▄▄▃▄▅▅▄▅▇█
wandb:                                    utilisation_std ▆██▇▆▇▇▆▅▅▃▁▁▁▃▂▃▃▃▃▃▃▃▄▃▃▂▃▂▃▃▃▂▃▂▃▃▂▃▂
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 2106000.0
wandb:                         accepted_bitrate_iqr_upper 2377750.0
wandb:                              accepted_bitrate_mean 2199838.0
wandb:                               accepted_bitrate_std 249531.35938
wandb:                        accepted_services_iqr_lower 2971.75
wandb:                        accepted_services_iqr_upper 3391.0
wandb:                             accepted_services_mean 3124.38501
wandb:                              accepted_services_std 360.33511
wandb:             bitrate_blocking_probability_iqr_lower 0.20715
wandb:             bitrate_blocking_probability_iqr_upper 0.30051
wandb:                  bitrate_blocking_probability_mean 0.26656
wandb:                   bitrate_blocking_probability_std 0.08328
wandb:                              cum_returns_iqr_lower -1452.289
wandb:                              cum_returns_iqr_upper 2991.11688
wandb:                                   cum_returns_mean 436.88217
wandb:                                    cum_returns_std 3619.31128
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 1674050.0
wandb:             episode_end_accepted_bitrate_iqr_upper 1859200.0
wandb:                  episode_end_accepted_bitrate_mean 1729691.0
wandb:                   episode_end_accepted_bitrate_std 200111.9375
wandb:            episode_end_accepted_services_iqr_lower 2374.75
wandb:            episode_end_accepted_services_iqr_upper 2640.0
wandb:                 episode_end_accepted_services_mean 2452.42993
wandb:                  episode_end_accepted_services_std 288.24149
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.1999
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.28077
wandb:      episode_end_bitrate_blocking_probability_mean 0.25596
wandb:       episode_end_bitrate_blocking_probability_std 0.0863
wandb:                  episode_end_cum_returns_iqr_lower -820.8203
wandb:                  episode_end_cum_returns_iqr_upper 2310.74487
wandb:                       episode_end_cum_returns_mean 474.36203
wandb:                        episode_end_cum_returns_std 2812.83838
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower -3.5
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean 0.36709
wandb:                            episode_end_returns_std 7.10646
wandb: episode_end_service_blocking_probability_iqr_lower 0.14811
wandb: episode_end_service_blocking_probability_iqr_upper 0.2337
wandb:      episode_end_service_blocking_probability_mean 0.20864
wandb:       episode_end_service_blocking_probability_std 0.09301
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.3874
wandb:                  episode_end_utilisation_iqr_upper 0.42055
wandb:                       episode_end_utilisation_mean 0.39488
wandb:                        episode_end_utilisation_std 0.04123
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower 0.0
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean 0.32895
wandb:                                        returns_std 6.9457
wandb:             service_blocking_probability_iqr_lower 0.15225
wandb:             service_blocking_probability_iqr_upper 0.25706
wandb:                  service_blocking_probability_mean 0.2189
wandb:                   service_blocking_probability_std 0.09008
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 544.81234
wandb:                              utilisation_iqr_lower 0.39125
wandb:                              utilisation_iqr_upper 0.41917
wandb:                                   utilisation_mean 0.39678
wandb:                                    utilisation_std 0.03653
wandb: 
wandb: 🚀 View run floral-breeze-8 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/30shcs25
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_231801-30shcs25/logs
Completed training for load = 472
----------------------------------------
Running training with load = 540
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load540.csv
CUDA_VISIBLE_DEVICES=2
I1022 23:27:55.500290 140481313155968 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 23:27:55.501566 140481313155968 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_232803-as7vpeik
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run proud-dawn-9
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/as7vpeik
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 540
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load540.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 540.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=24.80s
EXECUTION: Elapsed time=541.70s, FPS=3.69e+02
returns: -0.67137 ± 7.05963
lengths: 4000.00000 ± 0.00000
cum_returns: -1421.89966 ± 4544.36914
accepted_services: 2907.74487 ± 454.16309
accepted_bitrate: 2034995.00000 ± 309303.56250
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.40562 ± 0.04601
service_blocking_probability: 0.27306 ± 0.11354
bitrate_blocking_probability: 0.32148 ± 0.10346
wandb: 
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇███
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇████
wandb:                        accepted_services_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▁▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇███
wandb:             bitrate_blocking_probability_iqr_lower ▁▁▁▁▁▁▂▂▂▂▂▃▃▄▄▄▄▄▄▄▅▅▅▅▆▆▆▆▇▆▇▇▇▇▇█████
wandb:             bitrate_blocking_probability_iqr_upper ▁▁▂▃▂▂▂▂▃▃▄▄▅▄▅▆▆▆▅▅▆▆▆▆▆▆▆▇▇▇█▇▇▇▇▇▇▇██
wandb:                  bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇████
wandb:                   bitrate_blocking_probability_std ▆▆▆▇▇████▇▇▇▆▆▅▆▅▅▅▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁▁▂▂▃▃▃
wandb:                              cum_returns_iqr_lower ███▇▇▇▇▇▇▇▇▆▆▅▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▁
wandb:                              cum_returns_iqr_upper ▁▁▂▁▂▂▃▂▂▂▃▃▃▃▄▄▃▄▄▄▅▆▅▅▅▅▅▅▆▆▆▅▅▅▅▆▆▇▇█
wandb:                                   cum_returns_mean ████▇▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                                    cum_returns_std ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇███
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▂▂▂▂▃▃▃▃▃▃▃▃▃▃▃▃▄▄▅▅▅▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇█
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇███
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▄▅▅▅▆▆▆▇▇▇▇▇▇████
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▁▁▁▃▅▇▅▇▇▇▇▆▇▆▇▆▅▄▃▃▂▂▃▃▃▂▃▅▄▅▅▄▃▆▇█▇▆▆▇
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▃▃▃▄▄▄▅▅▅▆▆▆▆▇▇▇██████▇▇
wandb:      episode_end_bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:       episode_end_bitrate_blocking_probability_std ▁▁▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇██
wandb:                  episode_end_cum_returns_iqr_lower █▇▇▇█▇▇▇▇███▇▇▇▇▆▆▅▅▄▄▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▂▁
wandb:                  episode_end_cum_returns_iqr_upper ▃▃▂▂▁▂▂▂▂▂▃▃▂▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▄▃▄▄▄▃▄▄▆▇█
wandb:                       episode_end_cum_returns_mean ███▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▄▄▄▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                        episode_end_cum_returns_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ▅▅▁▅▅▅▅▅▅▅▅▁▅▁▅▅▅▅▅▄▅▁█▅▅▅▅▅▅▅▅▅▅▁▅▅▅▅▁▅
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▅▄▁▄▆▄▄▁▃▃▆▄▅▂▅▁▃▂▄▁▂▃▇▄▂▄▆▄█▆▃▄▃▄▄▃▃▄▃▄
wandb:                            episode_end_returns_std ▇▆▅▄▇▆▄▃▄▃▆█▅▆▇▁▅▇▅▄▃▆▄▆▄▆▇▆▇▅▆▄▄█▅▅▅▅▇▇
wandb: episode_end_service_blocking_probability_iqr_lower ▁▃▃▃▅▆▅▄▃▄▅▅▆▅██▆▆▇▆▇▇▆▅▅▅▅▄▃▂▃▃▅▄▃▄▃▂▃▂
wandb: episode_end_service_blocking_probability_iqr_upper ▁▁▂▂▃▃▄▄▅▅▅▅▅▅▅▅▅▅▅▅▆▆▇▇▇▇▇▆▆▆▆▆▆▆▆▇▇▇▇█
wandb:      episode_end_service_blocking_probability_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:       episode_end_service_blocking_probability_std ▁▁▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇██
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▆▆██▅▆▇▇▃▄▄▃▁▁▃▂▃▃▅▄▃▄▄▄▄▄▂▃▂▂▄▃▃▃▃▄▃▄▆▅
wandb:                  episode_end_utilisation_iqr_upper ▆▅▇▆██▆▆▆▇█▇▇▇█▇▇▆▅▄▅▃▄▄▁▁▁▂▃▃▃▄▅▅▆▅▇▅▅▃
wandb:                       episode_end_utilisation_mean ██▇▆▅▅▅▅▄▆▅▄▅▅▅▆▅▃▄▄▃▃▃▅▅▂▃▂▂▂▂▂▅▃▂▁▂▂▃▃
wandb:                        episode_end_utilisation_std ▅▄▅▄▂▁▁▂▂▂▃▄▄▅▄▄▃▃▃▃▄▄▅▆▆▆▆▆▆▆▆▆▆▆▆▅▇▇██
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ▅▅█▅▅▁▅▅▅▅▁▅▅▅▅▅▅▅▅▅▅▅▅▅▁▁▅▅█▅▅▅█▅▁▄▅▅▄▅
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▄▆▇▄▄▃▄▆▅▅▃▅▄▃▄▄▅▄▇▃▅▄▃▂▁▁▃▅█▃▃▆▅▄▂▃▅▅▄▄
wandb:                                        returns_std ▃▆▄▄▅▄▂█▄▂▄▅▅▄▃▅▅▄▇▃▄▄▄▁▂▁▃▃█▁▂▆▃▄▅▃▅▄▆▄
wandb:             service_blocking_probability_iqr_lower ▁▁▁▁▁▂▂▂▂▃▃▃▄▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇███
wandb:             service_blocking_probability_iqr_upper ▁▁▂▂▂▂▂▂▃▃▄▄▅▅▅▅▆▅▅▅▆▅▅▅▆▆▇▇███▇▇███▇███
wandb:                  service_blocking_probability_mean ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇████
wandb:                   service_blocking_probability_std ▅▆▆▇▇▇███▇▇▇▆▆▅▅▅▅▅▄▄▃▃▃▃▂▂▂▂▂▁▁▁▁▁▂▂▃▃▃
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ▇▄▅▄▆▆▇▆▅▄▆▇▅▃▄▃▁▂▂▁▁▂▂▃▄▄▃▄▄▁▂▁▂▂▆▅▅▆█▆
wandb:                              utilisation_iqr_upper ▇█▆▇▇█▆▅▄▄▄▅▃▃▁▁▂▃▄▃▄▃▄▃▄▄▁▄▃▃▅▃▃▃▃▄▄▅▅▅
wandb:                                   utilisation_mean ████▇▇▇▆▅▅▅▅▃▂▂▂▂▃▃▂▂▂▃▃▃▂▁▂▂▂▃▂▂▂▂▂▂▄▄▄
wandb:                                    utilisation_std ▆▆▆▇█▇▇▆▅▅▅▄▃▂▁▂▄▃▄▃▄▄▃▃▄▄▅▅▅▅▅▄▅▅▅▆▆▆▆▆
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 1898000.0
wandb:                         accepted_bitrate_iqr_upper 2287350.0
wandb:                              accepted_bitrate_mean 2034995.0
wandb:                               accepted_bitrate_std 309303.5625
wandb:                        accepted_services_iqr_lower 2703.0
wandb:                        accepted_services_iqr_upper 3286.25
wandb:                             accepted_services_mean 2907.74487
wandb:                              accepted_services_std 454.16309
wandb:             bitrate_blocking_probability_iqr_lower 0.23726
wandb:             bitrate_blocking_probability_iqr_upper 0.36877
wandb:                  bitrate_blocking_probability_mean 0.32148
wandb:                   bitrate_blocking_probability_std 0.10346
wandb:                              cum_returns_iqr_lower -3854.68433
wandb:                              cum_returns_iqr_upper 2069.69379
wandb:                                   cum_returns_mean -1421.89966
wandb:                                    cum_returns_std 4544.36914
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 1545100.0
wandb:             episode_end_accepted_bitrate_iqr_upper 1796200.0
wandb:                  episode_end_accepted_bitrate_mean 1629558.0
wandb:                   episode_end_accepted_bitrate_std 243217.625
wandb:            episode_end_accepted_services_iqr_lower 2207.0
wandb:            episode_end_accepted_services_iqr_upper 2579.0
wandb:                 episode_end_accepted_services_mean 2324.1499
wandb:                  episode_end_accepted_services_std 356.20889
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.22657
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.33497
wandb:      episode_end_bitrate_blocking_probability_mean 0.29899
wandb:       episode_end_bitrate_blocking_probability_std 0.10503
wandb:                  episode_end_cum_returns_iqr_lower -1868.54242
wandb:                  episode_end_cum_returns_iqr_upper 1717.64417
wandb:                       episode_end_cum_returns_mean -631.57733
wandb:                        episode_end_cum_returns_std 3493.01074
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower -5.25
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean -0.64961
wandb:                            episode_end_returns_std 7.43917
wandb: episode_end_service_blocking_probability_iqr_lower 0.1678
wandb: episode_end_service_blocking_probability_iqr_upper 0.28783
wandb:      episode_end_service_blocking_probability_mean 0.25003
wandb:       episode_end_service_blocking_probability_std 0.11494
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.38923
wandb:                  episode_end_utilisation_iqr_upper 0.43878
wandb:                       episode_end_utilisation_mean 0.40757
wandb:                        episode_end_utilisation_std 0.04678
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower -5.25
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean -0.67137
wandb:                                        returns_std 7.05963
wandb:             service_blocking_probability_iqr_lower 0.17844
wandb:             service_blocking_probability_iqr_upper 0.32425
wandb:                  service_blocking_probability_mean 0.27306
wandb:                   service_blocking_probability_std 0.11354
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 541.15819
wandb:                              utilisation_iqr_lower 0.39298
wandb:                              utilisation_iqr_upper 0.43701
wandb:                                   utilisation_mean 0.40562
wandb:                                    utilisation_std 0.04601
wandb: 
wandb: 🚀 View run proud-dawn-9 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/as7vpeik
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_232803-as7vpeik/logs
Completed training for load = 540
----------------------------------------
Running training with load = 607
Output file: /home/uceedoh/git/XLRON/data/launch_power_train_out_load607.csv
CUDA_VISIBLE_DEVICES=2
I1022 23:37:53.419354 140615815650176 xla_bridge.py:889] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
I1022 23:37:53.421069 140615815650176 xla_bridge.py:889] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Available devices: [cuda(id=0)]
Local devices: [cuda(id=0)]
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
XLA_PYTHON_CLIENT_PREALLOCATE=true
wandb: Currently logged in as: micdoh. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.1
wandb: Run data is saved locally in /home/uceedoh/git/XLRON/wandb/run-20241022_233801-fgmrt0qn
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run confused-fog-10
wandb: ⭐️ View project at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL
wandb: 🚀 View run at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/fgmrt0qn
logtostderr False
alsologtostderr False
log_dir 
v 0
verbosity 0
logger_levels {}
stderrthreshold fatal
showprefixforinfo True
run_with_pdb False
pdb_post_mortem False
pdb False
run_with_profiling False
profile_file None
use_cprofile_for_profiling True
only_check_args False
SEED 42
NUM_LEARNERS 1
NUM_DEVICES 1
NUM_ENVS 200
ROLLOUT_LENGTH 150
NUM_UPDATES 1
MINIBATCH_SIZE 1
TOTAL_TIMESTEPS 200000.0
UPDATE_EPOCHS 10
NUM_MINIBATCHES 1
LR 0.0005
GAMMA 0.999
GAE_LAMBDA 0.95
CLIP_EPS 0.2
ENT_COEF 0.0
VF_COEF 0.5
ADAM_EPS 1e-05
ADAM_BETA1 0.9
ADAM_BETA2 0.999
LAYER_NORM False
MAX_GRAD_NORM 0.5
ACTIVATION tanh
LR_SCHEDULE warmup_cosine
SCHEDULE_MULTIPLIER 1.0
WARMUP_PEAK_MULTIPLIER 1.0
WARMUP_STEPS_FRACTION 0.2
WARMUP_END_FRACTION 0.1
NUM_LAYERS 2
NUM_UNITS 64
VISIBLE_DEVICES 2
PREALLOCATE_MEM True
PREALLOCATE_MEM_FRACTION 0.95
PRINT_MEMORY_USE False
WANDB True
SAVE_MODEL False
DEBUG False
DEBUG_NANS False
NO_TRUNCATE False
ORDERED True
NO_PRINT_FLAGS False
MODEL_PATH None
PROJECT LAUNCH_POWER_EVAL
EXPERIMENT_NAME 607
DOWNSAMPLE_FACTOR 1
DISABLE_JIT False
ENABLE_X64 False
ACTION_MASKING False
LOAD_MODEL False
DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/launch_power_train_out_load607.csv
PLOTTING True
EMULATED_DEVICES None
log_actions False
PROFILE False
env_type rsa_gn_model
load 607.0
mean_service_holding_time 25.0
k 5
topology_name nsfnet_deeprmsa_directed
link_resources 115
max_requests 10.0
max_timesteps 10.0
min_bw 25
max_bw 100
step_bw 1
values_bw ['400', '600', '800', '1200']
slot_size 100.0
incremental_loading False
end_first_blocking False
continuous_operation True
aggregate_slots 1
disjoint_paths False
guardband 0
symbol_rate 100
scale_factor 1.0
weight weight
modulations_csv_filepath ./examples/modulations.csv
traffic_requests_csv_filepath None
topology_directory None
multiple_topologies_directory None
traffic_intensity 0.0
maximise_throughout False
use_gn_model False
include_isrs False
reward_type bitrate
truncate_holding_time False
ENV_WARMUP_STEPS 3000
random_traffic False
custom_traffic_matrix_csv_filepath None
alpha 0.2
amplifier_noise_figure 4.5
beta_2 -21.7
gamma 0.0012
span_length 100.0
lambda0 1550.0
node_resources 4
virtual_topologies ['3_ring']
min_node_resources 1
max_node_resources 1
node_probs None
EVAL_HEURISTIC True
path_heuristic ksp_lf
node_heuristic random
USE_GNN False
gnn_latent 64
message_passing_steps 3
output_edges_size 64
output_nodes_size 64
output_globals_size 64
gnn_mlp_layers 2
normalize_by_link_length False
EVAL_MODEL False
model None
min_traffic 0.0
max_traffic 1.0
step_traffic 0.1
deterministic False
ref_lambda 1.5775e-06
launch_power 0.5
launch_power_type fixed
nonlinear_coefficient 0.0012
raman_gain_slope 2.8e-17
attenuation 4.605111673958094e-05
attenuation_bar 4.605111673958094e-05
dispersion_coeff 1.6999999999999996e-05
dispersion_slope 67.0
noise_figure 4.0
num_roadms 1.0
roadm_loss 18.0
coherent True
mod_format_correction False
interband_gap 100.0
gap_start 44
snr_margin 0.01
max_power 9.0
min_power -5.0
first_fit False
optimise_launch_power False
EVAL_STEPS 100
OPTIMIZATION_ITERATIONS 5
traffic_array False
list_of_requests None
? False
help False
helpshort False
helpfull False
helpxml False
chex_n_cpu_devices 1
chex_assert_multiple_cpu_devices False
test_srcdir 
test_tmpdir /tmp/absl_testing
test_random_seed 301
test_randomize_ordering_seed 
xml_output_file 
chex_skip_pmap_variant_if_single_device True
op_conversion_fallback_to_while_loop True
delta_threshold 0.5
tt_check_filter False
tt_single_core_summaries False
runtime_oom_exit True
hbm_oom_exit True

---BEGINNING COMPILATION---
Independent learners: 1
Environments per learner: 200
Number of devices: 1
Learners per device: 1
Timesteps per learner: 200000.0
Timesteps per environment: 1000.0
Total timesteps: 200000.0
Total updates: 6.0
Batch size: 30000
Minibatch size: 30000

/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.batching.BatchTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
/home/uceedoh/xlron_env/lib64/python3.11/site-packages/jax/_src/core.py:678: FutureWarning: unhashable type: <class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>. Attempting to hash a tracer will lead to an error in a future JAX release.
  warnings.warn(
COMPILATION: Elapsed time=23.82s
EXECUTION: Elapsed time=541.68s, FPS=3.69e+02
returns: -1.58887 ± 5.96898
lengths: 4000.00000 ± 0.00000
cum_returns: -2480.74609 ± 4831.71826
accepted_services: 2781.65991 ± 506.14600
accepted_bitrate: 1935639.00000 ± 342178.40625
total_bitrate: 2999449.75000 ± 18955.25977
utilisation: 0.41966 ± 0.04800
service_blocking_probability: 0.30458 ± 0.12654
bitrate_blocking_probability: 0.35458 ± 0.11450
wandb: 
wandb: Run history:
wandb:                         accepted_bitrate_iqr_lower ▁▁▁▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇▇██
wandb:                         accepted_bitrate_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                              accepted_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                               accepted_bitrate_std ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                        accepted_services_iqr_lower ▁▁▁▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇▇███
wandb:                        accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                             accepted_services_mean ▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              accepted_services_std ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:             bitrate_blocking_probability_iqr_lower ▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇█▇█▇██
wandb:             bitrate_blocking_probability_iqr_upper ▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▃▃▄▄▄▄▅▅▆▇▇███████████▇█
wandb:                  bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇█████
wandb:                   bitrate_blocking_probability_std ▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▃▃▃▄▄▄▅▅▅▆▆▆▇▇▇▇▇███████
wandb:                              cum_returns_iqr_lower ██████▇▇▇▇▆▆▆▆▆▆▅▅▅▄▄▄▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:                              cum_returns_iqr_upper ▁▂▂▃▂▄▄▃▄▅▄▂▆▆▅▄▄▃▄▂▁▁▂▃▄▆▇▇▇▆▇▇█▆█▇█▅▅▅
wandb:                                   cum_returns_mean █████▇▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                                    cum_returns_std ▁▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                                           env_step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                      episode_count ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_lower ▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇███
wandb:             episode_end_accepted_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:                  episode_end_accepted_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                   episode_end_accepted_bitrate_std ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:            episode_end_accepted_services_iqr_lower ▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇██
wandb:            episode_end_accepted_services_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                 episode_end_accepted_services_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                  episode_end_accepted_services_std ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇███
wandb: episode_end_bitrate_blocking_probability_iqr_lower ▃▂▂▂▂▂▂▂▁▂▃▃▃▃▃▂▅▄▄▃▃▄▅▅▆▆███▇█▇█▇▇▆▆▆▇▆
wandb: episode_end_bitrate_blocking_probability_iqr_upper ▇▇▆▆▇██▇▇▇▆▆▆▇▇▇█▇▇▆▅▆▄▃▄▅▄▄▃▃▃▂▁▁▁▁▂▂▂▂
wandb:      episode_end_bitrate_blocking_probability_mean ▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:       episode_end_bitrate_blocking_probability_std ▄▃▄▃▃▄▄▅▆▄▄▃▅▅▅▅▅▄▅▄▅▄▅▆█▇▆▅▄▄▂▃▃▃▂▂▁▂▃▄
wandb:                  episode_end_cum_returns_iqr_lower ▂▂▂▂▁▁▂▂▂▃▃▄▃▂▂▃▃▂▂▂▃▁▄▄▄▄▅▅▃▃▃▅▄▄▆▇▇▇▇█
wandb:                  episode_end_cum_returns_iqr_upper ▁▂▃▃▅▅▁▄▄▄▄▅▄▅▄▇▆▅██▆▆▄▇▅██▆▇▅▆▅▇▆▆▇█▇▇▆
wandb:                       episode_end_cum_returns_mean ███▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▃▂▂▂▂▂▁▁▁▁
wandb:                        episode_end_cum_returns_std ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇██
wandb:                      episode_end_lengths_iqr_lower ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                      episode_end_lengths_iqr_upper ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                           episode_end_lengths_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                            episode_end_lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      episode_end_returns_iqr_lower ██▁█▆█▁██▆▁██▁███▁████████▁███▁██▁██▁█▆█
wandb:                      episode_end_returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                           episode_end_returns_mean ▆▇▃▅▅▆▃▅▇▃▄▄▅▃▇▆▄▂▅▇█▄▇▄▃▅▄▂▅▂▆▇▆▁▄█▃▆▅▆
wandb:                            episode_end_returns_std ▅▇▃▄▅▅▃▄▇▂▄▃▃▅▅▂▄▆▂▅▇▃▄▅▃▆▅▂▅▁█▆▅▂▂█▅▃█▇
wandb: episode_end_service_blocking_probability_iqr_lower ▂▁▁▂▂▃▃▃▂▃▂▃▄▄▃▃▄▃▃▄▅▆▆▅▅▅▅▆▇▇█▇██▇▇▇▆▇▆
wandb: episode_end_service_blocking_probability_iqr_upper ▁▁▂▃▃▄▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇███▇▇▇▇███▇▇▇▇▆▆▆▆
wandb:      episode_end_service_blocking_probability_mean ▁▁▁▂▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:       episode_end_service_blocking_probability_std ▁▁▂▁▂▂▂▃▄▂▃▂▃▃▃▄▄▃▄▃▅▄▅▆█▇▇▆▅▅▄▄▄▃▃▄▃▄▅▆
wandb:                episode_end_total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                episode_end_total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                     episode_end_total_bitrate_mean ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:                      episode_end_total_bitrate_std ▁▁▄▃▂▂▄▄▄▃▂▃▄▄▅▄▅▃▅▄▅▅▄▅▅▆▄▃▄▄▅▅▆▅▇▇█▆▇▅
wandb:                  episode_end_utilisation_iqr_lower ▃▄▃▄▅▇▄▅▄▃▂▄▄▅▆▂▄▃▂▁▃▄▃▅▆▆█▅▅▆▆▅▃▄▆▆▇▆▃▂
wandb:                  episode_end_utilisation_iqr_upper ██▃▃▅▅▅▃▅▅▂▄▄▄▃▃▃▃▂▂▄▃▄▃▆▄▂▄▅▃▃▄▄▂▂▁▁▂▃▃
wandb:                       episode_end_utilisation_mean ██▆▆▇▇▇▇▇▆▆▆▇████▆▄▅▅▆▅▅▅▄▄▃▃▂▂▃▄▃▃▃▂▃▂▁
wandb:                        episode_end_utilisation_std ██▆▅▇▇▆▆▄▄▃▃▄▄▅▄▄▃▃▃▃▃▂▂▁▂▁▁▁▂▂▃▄▃▃▃▃▃▄▅
wandb:                                  lengths_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  lengths_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                       lengths_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                        lengths_std ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                  returns_iqr_lower ████▁▆█▁██▆██▁█▁█▁█▁█▁▁▆▁▁▆██▁█▁██▁▁██▁█
wandb:                                  returns_iqr_upper ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                                       returns_mean ▇███▄▄▇▅▅█▅▆▄▄▆▄▇▃▇▃▆▅▃▅▁▂▃▅▃▄▅▄▅▆▃▆▆▄▅▅
wandb:                                        returns_std ▅▇▅▆▄▃▆▅▂▃▅▆▂▅▆▅▅▄█▂▇▆▂▅▃▃▂▅▃▃▅▃▄▅▄█▃▁▅▄
wandb:             service_blocking_probability_iqr_lower ▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▆▅▆▆▆▆▆▇▇▇▇▇▇██
wandb:             service_blocking_probability_iqr_upper ▁▁▁▁▁▁▁▂▂▂▂▂▂▃▂▂▃▃▄▄▄▅▅▆▆▆▇▇████████▇▇▇▇
wandb:                  service_blocking_probability_mean ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇█████
wandb:                   service_blocking_probability_std ▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▃▃▃▄▄▄▅▅▅▅▆▆▇▇▇▇▇███████
wandb:                            total_bitrate_iqr_lower ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████
wandb:                            total_bitrate_iqr_upper ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                 total_bitrate_mean ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                                  total_bitrate_std ▁▁▁▁▁▂▃▃▄▅▅▆▅▅▅▆▆▆▆▇▇▇▇▇▆▆▆▅▄▅▄▄▅▅▅▆▆▇██
wandb:                                      training_time ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:                              utilisation_iqr_lower ██████▇▇▇▇▇▇▅▅▄▄▃▄▂▃▄▃▄▃▂▂▂▂▁▁▁▁▂▂▃▄▄▂▃▃
wandb:                              utilisation_iqr_upper ▇▇██▆▅▆▅▃▄▂▅▃▂▂▃▁▃▂▂▁▂▂▂▃▃▃▄▅▃▁▁▂▂▂▁▃▁▃▄
wandb:                                   utilisation_mean ████▇▇▇▇▇▆▆▆▅▄▃▃▃▃▃▃▃▃▃▂▂▂▂▂▁▁▁▁▁▂▁▁▂▂▂▃
wandb:                                    utilisation_std ▃▂▂▂▂▃▃▂▁▁▂▃▃▄▄▄▄▄▄▄▄▅▆▆▆███▇▇▇▇▇▇▇▇▆▆▅▅
wandb: 
wandb: Run summary:
wandb:                         accepted_bitrate_iqr_lower 1740600.0
wandb:                         accepted_bitrate_iqr_upper 2205600.0
wandb:                              accepted_bitrate_mean 1935639.0
wandb:                               accepted_bitrate_std 342178.40625
wandb:                        accepted_services_iqr_lower 2488.75
wandb:                        accepted_services_iqr_upper 3193.0
wandb:                             accepted_services_mean 2781.65991
wandb:                              accepted_services_std 506.146
wandb:             bitrate_blocking_probability_iqr_lower 0.26333
wandb:             bitrate_blocking_probability_iqr_upper 0.42262
wandb:                  bitrate_blocking_probability_mean 0.35458
wandb:                   bitrate_blocking_probability_std 0.1145
wandb:                              cum_returns_iqr_lower -5114.73291
wandb:                              cum_returns_iqr_upper 1093.24292
wandb:                                   cum_returns_mean -2480.74609
wandb:                                    cum_returns_std 4831.71826
wandb:                                           env_step 999
wandb:                                      episode_count 98
wandb:             episode_end_accepted_bitrate_iqr_lower 1442850.0
wandb:             episode_end_accepted_bitrate_iqr_upper 1739500.0
wandb:                  episode_end_accepted_bitrate_mean 1565582.0
wandb:                   episode_end_accepted_bitrate_std 244116.85938
wandb:            episode_end_accepted_services_iqr_lower 2075.0
wandb:            episode_end_accepted_services_iqr_upper 2506.0
wandb:                 episode_end_accepted_services_mean 2245.02002
wandb:                  episode_end_accepted_services_std 360.44986
wandb: episode_end_bitrate_blocking_probability_iqr_lower 0.25079
wandb: episode_end_bitrate_blocking_probability_iqr_upper 0.37677
wandb:      episode_end_bitrate_blocking_probability_mean 0.32655
wandb:       episode_end_bitrate_blocking_probability_std 0.10521
wandb:                  episode_end_cum_returns_iqr_lower -2752.58502
wandb:                  episode_end_cum_returns_iqr_upper 1055.70502
wandb:                       episode_end_cum_returns_mean -1307.90247
wandb:                        episode_end_cum_returns_std 3442.27441
wandb:                      episode_end_lengths_iqr_lower 3099.0
wandb:                      episode_end_lengths_iqr_upper 3099.0
wandb:                           episode_end_lengths_mean 3099.0
wandb:                            episode_end_lengths_std 0.0
wandb:                      episode_end_returns_iqr_lower -5.25
wandb:                      episode_end_returns_iqr_upper 0.0
wandb:                           episode_end_returns_mean -0.62659
wandb:                            episode_end_returns_std 7.80615
wandb: episode_end_service_blocking_probability_iqr_lower 0.19135
wandb: episode_end_service_blocking_probability_iqr_upper 0.33043
wandb:      episode_end_service_blocking_probability_mean 0.27557
wandb:       episode_end_service_blocking_probability_std 0.11631
wandb:                episode_end_total_bitrate_iqr_lower 2312900.0
wandb:                episode_end_total_bitrate_iqr_upper 2336250.0
wandb:                     episode_end_total_bitrate_mean 2324898.75
wandb:                      episode_end_total_bitrate_std 17564.25781
wandb:                  episode_end_utilisation_iqr_lower 0.4129
wandb:                  episode_end_utilisation_iqr_upper 0.45949
wandb:                       episode_end_utilisation_mean 0.42674
wandb:                        episode_end_utilisation_std 0.04631
wandb:                                  lengths_iqr_lower 4000.0
wandb:                                  lengths_iqr_upper 4000.0
wandb:                                       lengths_mean 4000.0
wandb:                                        lengths_std 0.0
wandb:                                  returns_iqr_lower -5.25
wandb:                                  returns_iqr_upper 0.0
wandb:                                       returns_mean -1.58887
wandb:                                        returns_std 5.96898
wandb:             service_blocking_probability_iqr_lower 0.20175
wandb:             service_blocking_probability_iqr_upper 0.37781
wandb:                  service_blocking_probability_mean 0.30458
wandb:                   service_blocking_probability_std 0.12654
wandb:                            total_bitrate_iqr_lower 2984450.0
wandb:                            total_bitrate_iqr_upper 3011850.0
wandb:                                 total_bitrate_mean 2999449.75
wandb:                                  total_bitrate_std 18955.25977
wandb:                                      training_time 541.13813
wandb:                              utilisation_iqr_lower 0.39116
wandb:                              utilisation_iqr_upper 0.45474
wandb:                                   utilisation_mean 0.41966
wandb:                                    utilisation_std 0.048
wandb: 
wandb: 🚀 View run confused-fog-10 at: https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/runs/fgmrt0qn
wandb: ️⚡ View job at https://wandb.ai/micdoh/LAUNCH_POWER_EVAL/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjQ4MDI2NjE2Mg==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241022_233801-fgmrt0qn/logs
Completed training for load = 607
----------------------------------------
606 uceedoh@geneva:~/git/XLRON$ Read from remote host geneva.ee.ucl.ac.uk: Operation timed out
Connection to geneva.ee.ucl.ac.uk closed.
client_loop: send disconnect: Broken pipe
MDs-MacBook-Pro-2:~ michaeldoherty$ 
"""

if __name__ == "__main__":
    process_log(test_string)