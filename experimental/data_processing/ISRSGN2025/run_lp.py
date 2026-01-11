import subprocess
import numpy as np
import os
import time

if __name__ == "__main__":
    # Set the base directory from which to run all commands
    base_directory = "/home/uceedoh/git/XLRON"

    # Create a unique screen name (using timestamp to avoid conflicts)
    screen_name = f"power_runs_{int(time.time())}"

    #env_type = "rsa"
    env_type = "rsa_gn_model"

    num_envs = 2000 if env_type == "rsa" else 1

    base_command = (f"/home/uceedoh/xlron_env/bin/python3.11 /home/uceedoh/git/XLRON/xlron/train/train.py "
                    f" --env_type={env_type} --DOWNSAMPLE_FACTOR 100 --guardband=0 --incremental_loading "
                    f" --end_first_blocking --NUM_ENVS {num_envs} "
                    f" --EVAL_HEURISTIC --SMALL_FLOAT_DTYPE float32 --LARGE_FLOAT_DTYPE float32 "
                    f" --MED_INT_DTYPE int32 --monitor_active_lightpaths --coherent --WANDB ")

    topology = "nsfnet_deeprmsa_directed"
    path_heuristics = ["ksp_ff", "ff_ksp"] if env_type == "rsa" else ["ksp_ff"] # ["ff_ksp"]
    slot_sizes = [100, 50, 25, 12.5, 6.25, 3.125]
    powers = np.arange(10, 31, 1) if env_type != "rsa" else [0]
    k_values = np.arange(1, 10, 1) if env_type == "rsa" else [4] # [3]
    project_name = f"NetSim_{env_type}"

    # Create screen session
    print(f"Creating new screen session named: {screen_name}")
    subprocess.run(f"screen -dmS {screen_name}", shell=True)

    # Small delay to ensure screen is created
    time.sleep(1)

    # Create a script file with all commands to run in the screen
    script_path = os.path.join(base_directory, "run_power_script.sh")

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {base_directory}\n")

        for path_heuristic in path_heuristics:
            for k in k_values:
                for slot_size in slot_sizes:
                    values_bw = "100" if env_type != "rsa" else f"{slot_size}"
                    for power in powers:
                        link_resources = int(15000 / slot_size)
                        # Todo - adjust number of steps based on total network capacity
                        total_timesteps = 20000 # 10 * num_envs * link_resources * 44 / 3 / (float(values_bw) / slot_size)
                        # Todo - consider doing values_bw from 12.5 up with different sizes
                        experiment_name = f"{path_heuristic}_k{k}"
                        experiment_name = experiment_name if env_type == "rsa" else experiment_name + f"_slot{slot_size}_power{power:.1f}"
                        command = base_command + (
                         f" --max_power_per_fibre {power:.2f} "
                         f" --topology_name {topology} "
                         f" --TOTAL_TIMESTEPS {total_timesteps} "
                         f" --slot_size {slot_size} "
                         f" --link_resources {link_resources} "
                         f" --path_heuristic {path_heuristic} "
                         f" --PROJECT {project_name} "
                         f" --EXPERIMENT_NAME {experiment_name} "
                         f" --values_bw {values_bw} "
                         f" --k {k} "
                        )
                        f.write(f"echo 'Running with path_heuristic={path_heuristic}, k={k}, slot_size={slot_size}, power={power:.1f}'\n")
                        f.write(f"{command}\n")
                        f.write("echo '========================================='\n")

        # Add command to keep screen alive after all commands complete
        f.write("echo 'All runs completed!'\n")
        f.write("echo 'Press Ctrl+C to exit'\n")
        f.write("while true; do sleep 10; done\n")

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Run the script in the screen
    screen_cmd = f"screen -r {screen_name} -X stuff 'bash {script_path}\n'"
    subprocess.run(screen_cmd, shell=True)

    print(f"All commands have been sent to screen session '{screen_name}'")
    print(f"To view the progress, use: screen -r {screen_name}")
    print("To detach from the screen (once attached), press Ctrl+A followed by D")