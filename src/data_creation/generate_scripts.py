import sys
import os
import stat
import shutil
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
import utils

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
                port=3000+i*2,
                tm_port=(3000+i*2)*2)
            for i in range(num_of_jobs)]

    # ASK CARLA FOLDER
    done = False
    while not done:
        try:
            carla_path = input("Where is the Carla folder? [absolute path] : ")
            CARLAUE4 = os.path.join(carla_path, "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping")
            if os.path.isfile(CARLAUE4):
                done = True
        except:
            done = False
    print(utils.color_info_string(f"Found Carla in {CARLAUE4}!"))

    # GET NUTFUSER FOLDER
    NUTFUSER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    # POPULATING scripts_and_jobs FOLDER
    """
    FROM CARLA DOCUMENTATIONS:
    terminal 1: ./CarlaUE4.sh -carla-rpc-port=4000 # simulation A 
    terminal 2: ./CarlaUE4.sh -carla-rpc-port=5000 # simulation B
    terminal 3: python3 spawn_npc.py --port 4000 --tm-port 4050 # TM-Server A connected to simulation A
    terminal 4: python3 spawn_npc.py --port 5000 --tm-port 5050 # TM-Server B connected to simulation B
    """
    with open(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "launch_data_creation.sh"), "w") as file:
        file.write(
f"""#!/bin/sh
PATH=$PATH:/leonardo/home/userexternal/emannocc/xdg-user-dirs-0.18/

TOWN=$1
PORT=$2
TM_PORT=$3
JOB_ID=$4
echo "Selected Town = $TOWN"
echo "Selected Port = $PORT"

while true
do
echo "Starting Carla!"
{CARLAUE4} -RenderOffScreen -nosound -carla-rpc-port=$PORT &
PID_CARLA=$!
echo "Waiting 60 s!"
sleep 60
cd {NUTFUSER}
source bin/activate
cd src/data_creation
echo "Setting up the world!"
python setup_world.py $TOWN $PORT
echo "Waiting 10 s!"
sleep 10
echo "Starting generating the traffic!"
python generate_traffic.py --hero -n 30 -w 30 --respawn --hybrid --port=$PORT --tm-port=$TM_PORT &
PID_TRAFFIC=$!
echo "Waiting 60 s!"
sleep 60
echo "STARTING GETTING DATA!"
python take_data_without_records.py $TOWN $PORT $JOB_ID
echo "Killing everything!"
kill -9 $PID_TRAFFIC
sleep 10
kill -9 $PID_CARLA
sleep 10
echo "Killed everything!"
done
"""
        )
    st = os.stat(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "launch_data_creation.sh"))
    os.chmod(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "launch_data_creation.sh"), st.st_mode | stat.S_IEXEC) # chmod +x

    os.mkdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "jobs"))
    os.mkdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs"))
    for job in JOBS:
        os.mkdir(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs", f"{job.id}"))
        with open(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "jobs", f"job_{job.id}.sh"), "w") as file:
            file.write(
f"""#!/bin/sh
#SBATCH --job-name=creation_job_0
#SBATCH --partition=boost_usr_prod
#SBATCH -o {os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs", f"{job.id}", "out.log")}
#SBATCH -e {os.path.join(config.DATASET_PATH, "scripts_and_jobs", "logs", f"{job.id}", "err.log")}
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

{os.path.join(config.DATASET_PATH, "scripts_and_jobs", "launch_data_creation.sh")} $TOWN $PORT $TM_PORT $JOB_ID
"""         )
            
    with open(os.path.join(config.DATASET_PATH, "scripts_and_jobs", "launch_all_jobs.sh"), "w") as file:
        file.write(
f"""#!/bin/sh

cd {os.path.join(config.DATASET_PATH, "scripts_and_jobs", "jobs")}

"""     )
        for job in JOBS:
            file.write(f"sbatch job_{job.id}.sh\n")