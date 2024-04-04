import sys
import os
import stat
import shutil
import collections

from nutfuser import config
from nutfuser import utils

Job = collections.namedtuple("Job", ["id", "town", "port", "tm_port"])


if __name__ == "__main__":

    # CREATING scripts_and_jobs FOLDER
    if len(sys.argv) == 1:
        print(utils.color_error_string("I need an argument with the dataset path!"))
        exit()
    config.DATASET_PATH = sys.argv[1]
    if not os.path.isdir(config.DATASET_PATH):
        try:
            os.mkdir(config.DATASET_PATH)
        except:
            print(utils.color_error_string(f"Something bad with this path: '{config.DATASET_PATH}'"))
            exit()
    if os.path.isdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs")):
        print(utils.color_error_string(f"'scripts_and_jobs' folder already present in '{os.path.join(config.DATASET_PATH, 'scripts_and_jobs')}'!"))
        overwrite = None
        while overwrite not in ["yes", "no"]:
            overwrite = input("Do you want to overwrite it? ['yes' or 'no'] : ")
        if overwrite == "yes":
            shutil.rmtree(os.path.join(config.DATASET_PATH, "scripts_and_jobs"))
        else:
            print("Bye!")
            exit()
    os.mkdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs"))

    # ASKING NUMBER OF JOBS TO CREATE
    done = False
    while not done:
        try:
            num_of_jobs = int(input("How many jobs do you want to run? [int number] : "))
            done = True
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

    # GET NUTFUSER FOLDER
    NUTFUSER = os.path.dirname(os.path.realpath(__file__))

    # POPULATING scripts_and_jobs FOLDER
    """
    FROM CARLA DOCUMENTATIONS:
    terminal 1: ./CarlaUE4.sh -carla-rpc-port=4000 # simulation A
    terminal 2: ./CarlaUE4.sh -carla-rpc-port=5000 # simulation B
    terminal 3: python3 spawn_npc.py --port 4000 --tm-port 4050 # TM-Server A connected to simulation A
    terminal 4: python3 spawn_npc.py --port 5000 --tm-port 5050 # TM-Server B connected to simulation B
    """
    os.mkdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "jobs"))
    os.mkdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs"))
    for job in JOBS:
        with open(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "jobs", f"job_{job.id}.sh"), "w") as file:
            file.write(
f"""#!/bin/sh
#SBATCH --job-name=j_{job.id}
#SBATCH --partition=boost_usr_prod
#SBATCH -o {os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs", f"{job.id}.log")}
#SBATCH -e {os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs", f"{job.id}.log")}
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
python generate_data.py --carla_path {CARLA_PATH} --town $TOWN --rpc_port $PORT --tm_port $TM_PORT --job_id $JOB_ID
"""         )

    with open(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "launch_all_jobs.sh"), "w") as file:
        file.write(
f"""#!/bin/sh

cd {os.path.join(config.DATASET_PATH, "scripts_and_jobs", "jobs")}

"""     )
        for job in JOBS:
            file.write(f"sbatch job_{job.id}.sh\n")
        file.write("cd ..")
