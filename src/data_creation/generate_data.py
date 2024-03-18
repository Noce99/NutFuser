import argparse
import subprocess
import multiprocessing
import os
import sys
import time
import random
import signal

import take_data_without_records
from generate_traffic_new import generate_traffic
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import utils
import config


def get_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--carla_ip',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)'
    )
    argparser.add_argument(
        '--rpc_port',
        default=2000,
        help='Carla RPC port (deafult: 2000)',
        type=int
    )
    argparser.add_argument(
        '--tm_port',
        default=8000,
        help='Traffic Manager port (default: 8000)',
        type=int
    )
    argparser.add_argument(
        '--town',
        default=15,
        help='Witch town to select (default: 15)',
        type=int
    )
    argparser.add_argument(
        '--job_id',
        default=0,
        help='Job ID to use (default: 0)',
        type=int
    )
    argparser.add_argument(
        '--backbone_data',
        help='If you want to generate backbone data',
        action='store_true'
    )
    argparser.add_argument(
        '--driving_data',
        help='If you want to generate backbone data',
        action='store_true'
    )
    argparser.add_argument(
        '--end_of_egg_file',
        help='How the end file should end to be valid! (default: py3.7-linux-x86_64.egg)',
        default="py3.7-linux-x86_64.egg",
        type=str
    )
    argparser.add_argument(
        '--carla_path',
        help='Path to the Carla Installation!',
        required=True,
        type=str
    )
    args = argparser.parse_args()
    if args.town not in config.TOWN_DICT:
        error = f"Invalid Town Index! [{args.town}]\n" + \
                 "Possible Town Index:\n"
        for key in config.TOWN_DICT:
            error += f"{key} -> {config.TOWN_DICT[key]}\n"
        raise Exception(utils.color_error_string(error))
    return args

def check_integrity_of_carla_path(args):
    """
    Check Integrity of the Carla Path
    """
    # (1) Check that the Carla's Path really exists
    if not os.path.isdir(args.carla_path):
        raise Exception(utils.color_error_string(f"The given Carla Path doesn't exist! [{args.carla_path}]"))
    # (2) Check that the egg file is really present and it works: being able to import carla!
    carla_pythonapi_dist_path = os.path.join(args.carla_path, "PythonAPI/carla/dist")
    if not os.path.isdir(carla_pythonapi_dist_path):
        raise Exception(utils.color_error_string(f"The given Carla doen't contains a PythonAPI! [{carla_pythonapi_dist_path}]"))
    egg_files = [file for file in os.listdir(carla_pythonapi_dist_path) if file[-len(args.end_of_egg_file):] == args.end_of_egg_file]
    if len(egg_files) == 0:
        raise Exception(utils.color_error_string(f"The given Carla doen't contains a \"*{args.end_of_egg_file}\" file in \"{carla_pythonapi_dist_path}\""))
    if len(egg_files) > 1:
        raise Exception(utils.color_error_string(f"The given Carla contains to many \"*{args.end_of_egg_file}\" files in \"{carla_pythonapi_dist_path}\"\n" +
                                                  "Set a more restrict search with the \"--end_of_egg_file\" arguments!"))
    egg_file_path = os.path.join(carla_pythonapi_dist_path, egg_files[0])
    # Now that we have a unique egg file we add it to the python path!
    sys.path.append(egg_file_path)
    # (3) Check that the CarlaUE4 executable is present
    carlaUE4_folder = os.path.join(args.carla_path, "CarlaUE4/Binaries/Linux/")
    if not os.path.isdir(carlaUE4_folder):
        raise Exception(utils.color_error_string(f"The folder in wicth I was expecting \"CarlaUE4-Linux-Shipping\" doesn't exists! [{carlaUE4_folder}]"))
    files = os.listdir(carlaUE4_folder)
    if "CarlaUE4-Linux-Shipping" not in files:
        raise Exception(utils.color_error_string(f"I cannot find \"CarlaUE4-Linux-Shipping\" executable in \"{carlaUE4_folder}\"!"))
    carlaUE4_path = os.path.join(carlaUE4_folder, "CarlaUE4-Linux-Shipping")
    return egg_file_path, carlaUE4_path

def start_up_carla_server(args, carla_server_is_up, process_to_kill):
    carla_process = subprocess.Popen(
        ["/usr/bin/stdbuf", "-o0", carlaUE4_path, "-RenderOffScreen", "-nosound", f"-carla-rpc-port={args.rpc_port}", "-opengl"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    process_to_kill.put(carla_process.pid)
    # We will wait Carla to start up!
    while True:
        carla_output = carla_process.stdout.readline()
        if "Disabling core dumps." in carla_output:
            break
        return_code = carla_process.poll()
        if return_code is not None:
            # The Carla process died before starting up!
            raise Exception(utils.color_error_string(f"The Carla process died while starting up!"))
    # The Carla process is up, we will wait 10 second just to be sure!
    time.sleep(10)
    carla_server_is_up.set()
    # I will run this function to take care when carla server die!
    while True:
        return_code = carla_process.poll()
        if return_code is not None:
            break

def setup_world(args):
    print("Setting up the world...")
    client = carla.Client(args.carla_ip, args.rpc_port)
    client.set_timeout(20.0)
    client.load_world(config.TOWN_DICT[args.town])
    print("World setted up!")

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    egg_file_path, carlaUE4_path = check_integrity_of_carla_path(args)
    try:
        import carla
    except:
        raise Exception(utils.color_error_string(f"Not able to import Carla from [{egg_file_path}]"))
    

    process_to_kill = multiprocessing.Queue()
    # (1) LAUNCH CARLA SERVER
    carla_server_is_up = multiprocessing.Event()
    check_carla_process = multiprocessing.Process(target=start_up_carla_server, args=(args, carla_server_is_up, process_to_kill))
    check_carla_process.start()
    # Let's wait till Carla Server is Up!
    while True:
        if not check_carla_process.is_alive():
            check_carla_process.join()
            while not process_to_kill.empty():
                os.kill(process_to_kill.get(), signal.SIGKILL)
            raise Exception(utils.color_error_string(f"Carla crashed while starting!"))
        if carla_server_is_up.is_set():
            break

    # (3) SET UP THE WORLD
    set_up_world_process = multiprocessing.Process(target=setup_world, args=(args, ))
    set_up_world_process.start()

    while True:
        if not check_carla_process.is_alive():
            check_carla_process.join()
            set_up_world_process.kill()
            raise Exception(utils.color_error_string(f"Carla crashed while setting up the world!"))
        if not set_up_world_process.is_alive():
            set_up_world_process.join()
            break
    
    # (4) SET UP TRAFFIC MANAGER
    you_can_tick = multiprocessing.Event()
    traffic_manager_is_up = multiprocessing.Event()
    set_up_traffic_manager_process = multiprocessing.Process(target=generate_traffic, args=(egg_file_path, args.carla_ip, args.rpc_port, args.tm_port, 30, 30, traffic_manager_is_up, you_can_tick))
    set_up_traffic_manager_process.start()

    while True:
        if not check_carla_process.is_alive():
            check_carla_process.join()
            set_up_traffic_manager_process.kill()
            while not process_to_kill.empty():
                os.kill(process_to_kill.get(), signal.SIGKILL)
            raise Exception(utils.color_error_string(f"Carla crashed!"))
        if not set_up_traffic_manager_process.is_alive():
            set_up_traffic_manager_process.join()
            check_carla_process.kill()
            while not process_to_kill.empty():
                os.kill(process_to_kill.get(), signal.SIGKILL)
            raise Exception(utils.color_error_string(f"Traffic Manager crashed!"))
        if traffic_manager_is_up.is_set():
            break
    print("All Setted Up properly!")

    # (5) LAUNCH DATA CREATION PROCESS
    ego_vehicle_found_event = multiprocessing.Event()
    starting_data_loop_event = multiprocessing.Event()
    finished_taking_data_event = multiprocessing.Event()
    data_creation_process = multiprocessing.Process(target=take_data_without_records.take_data_backbone,
                                                    args=(egg_file_path, args.town, args.rpc_port, args.job_id,
                                                          ego_vehicle_found_event, starting_data_loop_event,
                                                          finished_taking_data_event, you_can_tick))
    data_creation_process.start()

    start_time = time.time()

    already_printed = False
    while True:
        if not check_carla_process.is_alive():
            try:
                check_carla_process.join()
                set_up_traffic_manager_process.kill()
                data_creation_process.kill()
                while not process_to_kill.empty():
                    os.kill(process_to_kill.get(), signal.SIGKILL)
            except:
                pass
            raise Exception(utils.color_error_string(f"Carla crashed!"))
        if not set_up_traffic_manager_process.is_alive():
            try:
                set_up_traffic_manager_process.join()
                check_carla_process.kill()
                data_creation_process.kill()
                while not process_to_kill.empty():
                    os.kill(process_to_kill.get(), signal.SIGKILL)
            except:
                pass
            raise Exception(utils.color_error_string(f"Traffic Manager crashed!"))
        if not data_creation_process.is_alive():
            try:
                data_creation_process.join()
                check_carla_process.kill()
                set_up_traffic_manager_process.kill()
                while not process_to_kill.empty():
                    os.kill(process_to_kill.get(), signal.SIGKILL)
            except:
                pass
            raise Exception(utils.color_error_string(f"Data Creation crashed!"))
        if not ego_vehicle_found_event.is_set() and time.time()-start_time > 10:
            try:
                data_creation_process.kill()
                check_carla_process.kill()
                set_up_traffic_manager_process.kill()
                while not process_to_kill.empty():
                    os.kill(process_to_kill.get(), signal.SIGKILL)
            except:
                pass
            raise Exception(utils.color_error_string(f"Data Creation is not able to find out the Ego Vehicle!"))
        if starting_data_loop_event.is_set() and not already_printed:
            print(utils.color_info_string("Starting Taking Data!"))
            already_printed = True
        if finished_taking_data_event.is_set():
            print(utils.color_info_string("Finishing Taking Data!"))
            break

    # (6) CLEANING EVERYTHING
    set_up_traffic_manager_process.kill()
    check_carla_process.kill()
    while not process_to_kill.empty():
        os.kill(process_to_kill.get(), signal.SIGKILL)

    data_creation_process.join()


