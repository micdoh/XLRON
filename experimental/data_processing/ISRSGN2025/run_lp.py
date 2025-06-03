import subprocess
import numpy as np
import os
import time

if __name__ == "__main__":
    # Set the base directory from which to run all commands
    base_directory = "/home/uceedoh/git/XLRON"

    # Create a unique screen name (using timestamp to avoid conflicts)
    screen_name = f"power_runs_{int(time.time())}"

    env_type = "rsa"
    env_type = "rsa_gn_model"

    num_envs = 2000 if env_type == "rsa" else 1

    base_command = (f"/home/uceedoh/xlron_env/bin/python3.11 /home/uceedoh/git/XLRON/xlron/train/train.py "
                    f"--env_type={env_type} --DOWNSAMPLE_FACTOR 100 --guardband=0 --incremental_loading "
                    f"--end_first_blocking --values_bw 100 --NUM_ENVS {num_envs} --path_heuristic=ksp_ff "
                    f"--EVAL_HEURISTIC --SMALL_FLOAT_DTYPE float32 --LARGE_FLOAT_DTYPE float32"
                    f" --MED_INT_DTYPE int32 --monitor_active_lightpaths --coherent --WANDB")

    topology = "nsfnet_deeprmsa_directed"
    total_timesteps = 10000000 if env_type == "rsa" else 25000
    path_heuristics = ["ksp_ff", "ff_ksp"] if env_type == "rsa" else ["ff_ksp"]
    slot_sizes = [100] if env_type == "rsa" else [150, 125, 100, 75, 50, 25, 12.5]
    power_dict = {
        150: np.arange(-8, 1, 1),
        125: np.arange(-9, 0, 1),
        100: np.arange(-10, -1, 1),
        75: np.arange(-12, -2, 1),
        50: np.arange(-15, -5, 1),
        25: np.arange(-17, -5, 1),
        12.5: np.arange(-20, -5, 1),
    }
    guardband_dict = {
        150: [0, 1],
        125: [0, 1],
        100: [0, 1],
        75: [0, 1],
        50: [0, 1, 2],
        25: [0, 1, 2, 4],
        12.5: [0, 1, 2, 4, 8],
    }
    k_values = np.arange(1, 21, 1) if env_type == "rsa" else [3]
    project_name = f"LP_{env_type}_{topology}"

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
                    powers = np.arange(0, 1, 1) if env_type == "rsa" else power_dict[slot_size]
                    for guardband in guardband_dict[slot_size]:
                        for power in powers:
                            link_resources = int(15000 / slot_size)
                            experiment_name = f"{path_heuristic}_k{k}"
                            experiment_name = experiment_name if env_type == "rsa" else experiment_name + f"_slot{slot_size}_power{power:.1f}"
                            command = base_command + (
                             f" --launch_power {power:.2f} "
                             f" --topology_name {topology} "
                             f" --TOTAL_TIMESTEPS {total_timesteps} "
                             f" --slot_size {slot_size} "
                             f" --link_resources {link_resources} "
                             f" --path_heuristic {path_heuristic} "
                             f" --PROJECT {project_name} "
                             f" --EXPERIMENT_NAME {experiment_name} "
                             f" --k {k} "
                             f" --guardband {guardband} "
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