import json
import os
import shutil
import collections
import random

from nutfuser import config
from nutfuser import utils

Job = collections.namedtuple("Job", ["id", "town", "port", "tm_port"])

if __name__ == "__main__":

    # GET NUTFUSER FOLDER
    NUTFUSER = os.path.dirname(os.path.realpath(__file__))

    # ASK where to put the data_jobs FOLDER
    done = False
    while not done:
        try:
            possible_path = input("Where to save the 'evaluation_jobs' FOLDER? : ")
            if not os.path.isdir(possible_path):
                print(utils.color_error_string(f"Cannot find '{possible_path}'!"))
                continue
            script_and_jobs_path = os.path.join(possible_path, "evaluation_jobs")
            if os.path.isdir(script_and_jobs_path):
                print(utils.color_error_string(f"'evaluation_jobs' folder already present in '{possible_path}'!"))
                overwrite = None
                while overwrite not in ["yes", "no"]:
                    overwrite = input("Do you want to overwrite it? ['yes' or 'no'] : ")
                if overwrite == "yes":
                    shutil.rmtree(script_and_jobs_path)
                else:
                    raise KeyboardInterrupt
            done = True
        except KeyboardInterrupt:
            print(utils.color_info_string("\nBye!\n"))
            exit()
        except:
            continue
    os.mkdir(script_and_jobs_path)
    folder_in_train_logs = os.listdir(os.path.join(NUTFUSER, "train_logs"))
    if "nvidia_log" in folder_in_train_logs:
        folder_in_train_logs.remove("nvidia_log")
    if "output_log" in folder_in_train_logs:
        folder_in_train_logs.remove("output_log")
    networks_weights = []
    for folder in folder_in_train_logs:
        files_in_network_folder = os.listdir(os.path.join(NUTFUSER, "train_logs", folder))
        for file in files_in_network_folder:
            if file[:5] == "model":
                networks_weights.append(os.path.join(NUTFUSER, "train_logs", folder, file))
    with open(os.path.join(script_and_jobs_path, 'weight_list.json'), 'w') as f:
        json.dump(networks_weights, f, sort_keys=False, indent=4)
    print(f"Found out {len(networks_weights)} networks to evaluate!")

    # ASKING NETWORKS TO EVALUATE
    done = False
    while not done:
        try:
            are_u_done = input(f"I have created the {os.path.join(script_and_jobs_path, 'weight_list.json')}"
                               f" file with all the stuff that I think you want to evaluate."
                               f" Please open it and modify it if you want after that write 'done'!")
            if are_u_done == "done":
                done = True
            else:
                done = False
        except KeyboardInterrupt:
            print(utils.color_info_string("Bye!"))
            exit()

    with open(os.path.join(script_and_jobs_path, 'weight_list.json'), 'r') as f:
        networks_weights = json.load(f)
    print(f"After your modification found out {len(networks_weights)} networks to evaluate!")

    # ASK CARLA FOLDER
    done = False
    while not done:
        try:
            CARLA_PATH = input("Where is the Carla folder? [absolute path] : ")
            if os.path.isdir(CARLA_PATH):
                done = True
        except:
            done = False
    print(utils.color_info_string(f"Found Carla in {CARLA_PATH}!"))

    # ASK SLURM ACCOUNT
    done = False
    while not done:
        try:
            account_name = str(input("What is you slurm account name? [string] : "))
            done = True
        except KeyboardInterrupt:
            print(utils.color_info_string("Bye!"))
            exit()
        except:
            done = False

    # GET EVALUATION SCENARIOS
    folder_in_evaluation_routes = os.listdir(os.path.join(NUTFUSER, "evaluation_routes"))
    all_scenarios = []
    for evaluation_folder in folder_in_evaluation_routes:
        scenarios = os.listdir(os.path.join(NUTFUSER, "evaluation_routes", evaluation_folder))
        for scenario in scenarios:
            if scenario == "evaluation.xml":
                continue
            all_scenarios.append(os.path.join(NUTFUSER, "evaluation_routes", evaluation_folder, scenario))

    # GENERATE EVALUATION FOLDER
    os.mkdir(os.path.join(script_and_jobs_path, "jobs"))
    os.mkdir(os.path.join(script_and_jobs_path, "logs"))
    os.mkdir(os.path.join(script_and_jobs_path, "completed"))
    os.mkdir(os.path.join(script_and_jobs_path, "results"))
    for weight_path in networks_weights:
        net_name = os.path.dirname(weight_path).split("/")[-1]
        if "_" in net_name:
            net_name = net_name.split("_")[0]
        os.mkdir(os.path.join(script_and_jobs_path, "results", net_name))
        for scenario_town in folder_in_evaluation_routes:
            if "_" in scenario_town:
                scenario_town = scenario_town.split("_")[0]
            os.mkdir(os.path.join(script_and_jobs_path, "results", net_name, scenario_town))

    # GENERATE JOBS
    jobs = []
    print("Parsing Folders and creating a job list...")
    for weight_path in networks_weights:
        for scenario_path in all_scenarios:
            net_name = os.path.dirname(weight_path).split("/")[-1]
            scenario_town = os.path.dirname(scenario_path).split("/")[-1]
            if "_" in net_name:
                net_name = net_name.split("_")[0]
            if "_" in scenario_town:
                scenario_town = scenario_town.split("_")[0]
            jobs.append(
                {
                    "net_name": net_name,
                    "scenario_town": scenario_town,
                    "scenario": scenario_path.split("/")[-1].split("_")[0],
                    "scenario_path": scenario_path,
                    "weight_path": weight_path,
                    "save_path": os.path.join(script_and_jobs_path, "results", net_name, scenario_town),
                    "completed_path": os.path.join(script_and_jobs_path, "completed")
                }
            )
    print(f"Created a list of {len(jobs)} jobs!")

    # CREATE JOB FILE
    for i, job in enumerate(jobs):

        with open(os.path.join(script_and_jobs_path, "jobs", f"{i}.sbatch"), "w") as file:
            file.write(
                f"""#!/bin/sh
#SBATCH --job-name=j_{i}
#SBATCH --partition=boost_usr_prod
#SBATCH -o {os.path.join(script_and_jobs_path, "logs", f"{i}.log")}
#SBATCH -e {os.path.join(script_and_jobs_path, "logs", f"{i}.log")}
#SBATCH --mail-type=FAIL,BEGIN
#SBATCH --mail-user=enrico.mannocci3@unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --account={account_name}

# print info about current job

PORT={2000+i}
TM_PORT={5000+i}
JOB_ID={i}

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Evaluation JOB STARTED: $dt"

cd {NUTFUSER}
source bin/activate
python single_evaluation.py --evaluation_route {job['scenario_path']} --where_to_save {job['save_path']} --carla_path {CARLA_PATH} --weight_path {job['weight_path']} --id {i} --completed_path {job['completed_path']} --rpc_port $PORT --tm_port $TM_PORT
""")

    with open(os.path.join(script_and_jobs_path, "launch_all_jobs.sh"), "w") as file:
        file.write(
            f"""#!/bin/sh

cd {os.path.join(script_and_jobs_path, "jobs")}

""")
        for i, job in enumerate(jobs):
            file.write(f"sbatch {i}.sbatch\n")
        file.write("cd ..")
