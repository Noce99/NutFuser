import sys
import os
import stat
import shutil
import collections

from nutfuser import config
from nutfuser import utils

Job = collections.namedtuple("Job", ["id", "town", "port", "tm_port"])

if __name__ == "__main__":

    # ASK where to put the data_jobs FOLDER
    done = False
    while not done:
        try:
            possible_path = input("Where to save the data_jobs FOLDER? : ")
            if not os.path.isdir(possible_path):
                print(utils.color_error_string(f"Cannot find '{possible_path}'!"))
                continue
            script_and_jobs_path = os.path.join(possible_path, "data_jobs")
            if os.path.isdir(script_and_jobs_path):
                print(utils.color_error_string(f"'data_jobs' folder already present in '{possible_path}'!"))
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

    # ASKING NUMBER OF JOBS TO CREATE
    done = False
    while not done:
        try:
            num_of_jobs = int(input("How many jobs do you want to run? [int number] : "))
            done = True
        except KeyboardInterrupt:
            print(utils.color_info_string("Bye!"))
            exit()
        except:
            done = False
    print(utils.color_info_string(f"I will create {num_of_jobs} job files!"))
    JOBS = [Job(id=i,
                town=list(config.TOWN_DICT.keys())[i%len(config.TOWN_DICT.keys())],
                port=2001+i*5,
                tm_port=(2001+i*5)*2)
            for i in range(num_of_jobs)]

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

    # ASK NUMBER OF FRAMES
    done = False
    while not done:
        try:
            num_of_frames = int(input("How many frames per directory? (5000 is in [1; 2] GB per folder) [int number] : "))
            done = True
        except KeyboardInterrupt:
            print(utils.color_info_string("Bye!"))
            exit()
        except:
            done = False
    
    # ASK NUMBER OF OTHER NPC
    done = False
    while not done:
        try:
            num_of_npc = int(input("How many npc? (both cars and pedestrian) [int number] : "))
            done = True
        except KeyboardInterrupt:
            print(utils.color_info_string("Bye!"))
            exit()
        except:
            done = False

    # GET NUTFUSER FOLDER
    NUTFUSER = os.path.dirname(os.path.realpath(__file__))

    # POPULATING data_jobs FOLDER
    """
    FROM CARLA DOCUMENTATIONS:
    terminal 1: ./CarlaUE4.sh -carla-rpc-port=4000 # simulation A
    terminal 2: ./CarlaUE4.sh -carla-rpc-port=5000 # simulation B
    terminal 3: python3 spawn_npc.py --port 4000 --tm-port 4050 # TM-Server A connected to simulation A
    terminal 4: python3 spawn_npc.py --port 5000 --tm-port 5050 # TM-Server B connected to simulation B
    """
    os.mkdir(os.path.join(script_and_jobs_path, "jobs"))
    os.mkdir(os.path.join(script_and_jobs_path, "logs"))
    for job in JOBS:
        with open(os.path.join(script_and_jobs_path, "jobs", f"job_{job.id}.sh"), "w") as file:
            file.write(
f"""#!/bin/sh
#SBATCH --job-name=j_{job.id}
#SBATCH --partition=boost_usr_prod
#SBATCH -o {os.path.join(script_and_jobs_path, "logs", f"{job.id}.log")}
#SBATCH -e {os.path.join(script_and_jobs_path, "logs", f"{job.id}.log")}
#SBATCH --mail-type=FAIL,BEGIN
#SBATCH --mail-user=enrico.mannocci3@unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# print info about current job

TOWN={job.town}
PORT={job.port}
TM_PORT={job.tm_port}
JOB_ID={job.id}

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job on Town $TOWN started: $dt"

cd {NUTFUSER}
source bin/activate
python generate_data.py --carla_path {CARLA_PATH} --town $TOWN --rpc_port $PORT --tm_port $TM_PORT --job_id $JOB_ID --dataset_path {config.DATASET_PATH} --num_of_frames {num_of_frames} --num_of_vehicle {num_of_npc} --num_of_walkers {num_of_npc}
"""         )

    with open(os.path.join(script_and_jobs_path, "launch_all_jobs.sh"), "w") as file:
        file.write(
f"""#!/bin/sh

cd {os.path.join(script_and_jobs_path, "jobs")}

"""     )
        for job in JOBS:
            file.write(f"sbatch job_{job.id}.sh\n")
        file.write("cd ..")
