import argparse
import multiprocessing
import os
import pathlib
import time
import signal
import psutil
from ctypes import c_int
from tabulate import tabulate

from nutfuser.data_creation import take_data_without_records
from nutfuser import utils
from nutfuser import config
from nutfuser.carla_interface.run_carla import  check_integrity_of_carla_path, \
                                                launch_carla_server_saifly_and_wait_till_its_up, \
                                                set_up_world_saifly_and_wait_till_its_setted_up, \
                                                set_up_traffic_manager_saifly_and_wait_till_its_up

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
        help='How the egg file should end to be valid! (default: py3.7-linux-x86_64.egg)',
        default="py3.7-linux-x86_64.egg",
        type=str
    )
    argparser.add_argument(
        '--carla_path',
        help='Path to the Carla Installation!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--num_of_vehicle',
        help='Number of Vehicle to spawn!',
        required=False,
        default=30,
        type=int
    )
    argparser.add_argument(
        '--num_of_walkers',
        help='Number of Walkers to spawn!',
        required=False,
        default=30,
        type=int
    )
    argparser.add_argument(
        '--num_of_frames',
        help='Number of Frames to take!',
        required=False,
        default=10,
        type=int
    )
    args = argparser.parse_args()
    if args.town not in config.TOWN_DICT:
        error = f"Invalid Town Index! [{args.town}]\n" + \
                 "Possible Town Index:\n"
        for key in config.TOWN_DICT:
            error += f"{key} -> {config.TOWN_DICT[key]}\n"
        raise Exception(utils.color_error_string(error))
    return args

def kill_all(carla_server_pid, traffic_manager_pid, data_creation_pid):
    try:
        os.kill(carla_server_pid, signal.SIGKILL)
    except:
        pass
    try:
        os.kill(traffic_manager_pid, signal.SIGKILL)
    except:
        pass
    try:
        os.kill(data_creation_pid, signal.SIGKILL)
    except:
        pass

if __name__ == "__main__":
    args = get_arguments()
    egg_file_path, carlaUE4_path = check_integrity_of_carla_path(args)
    try:
        import carla
    except:
        raise Exception(utils.color_error_string(f"Not able to import Carla from [{egg_file_path}]"))

    print(utils.get_a_title("STARTING LAUNCHING ALL THE PROCESS", color="blue"))
    print(utils.color_info_success(f"Find out a valid carla in {egg_file_path}!"))

    # (0) SET UP LOGS AND DATASETFOLDER
    nutfuser_folder_path = pathlib.Path(__file__).parent.resolve()
    carla_log_path = os.path.join(nutfuser_folder_path, "logs", f"carla_server_logs_{args.job_id}.log")
    traffic_manager_log_path = os.path.join(nutfuser_folder_path, "logs", f"tarffic_manager_logs_{args.job_id}.log")
    datasets_path = os.path.join(nutfuser_folder_path, "datasets")

    if not os.path.isdir(datasets_path):
        try:
            os.mkdir(datasets_path)
        except:
            Exception(utils.color_error_string(f"Unable to create [{datasets_path}] dir!"))

    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        a_table.append([arg, getattr(args, arg)])
    a_table.append(["Dataset Path", datasets_path])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    # (1) LAUNCH CARLA SERVER
    print("Launching Carla Server...")
    os.environ["PATH"] = f"{os.environ['PATH']}:/leonardo/home/userexternal/emannocc/xdg-user-dirs-0.18/"
    carla_server_pid = multiprocessing.Value(c_int)
    carla_was_correctly_started_up = launch_carla_server_saifly_and_wait_till_its_up(
            rpc_port=args.rpc_port,
            carla_server_pid=carla_server_pid,
            carlaUE4_path=carlaUE4_path,
            logs_path=carla_log_path,
        )

    if not carla_was_correctly_started_up:
        raise Exception(utils.color_error_string(f"Carla crashed while starting!"))

    print(utils.color_info_string("(1/3)\tCarla Server is UP!"))

    # (3) SET UP THE WORLD
    world_was_correctly_setted_up = set_up_world_saifly_and_wait_till_its_setted_up(
            carla_ip=args.carla_ip,
            rpc_port=args.rpc_port,
            town_number=args.town,
            carla_server_pid=carla_server_pid
        )
    
    if not world_was_correctly_setted_up:
        raise Exception(utils.color_error_string(f"Failed to set up world!"))

    print(utils.color_info_string("(2/3)\tWorld was correctly setted up!"))

    # (4) SET UP TRAFFIC MANAGER
    traffic_manager_pid = multiprocessing.Value(c_int)
    carla_is_ok,\
    traffic_manager_is_ok,\
    you_can_tick,\
    traffic_manager_is_up = set_up_traffic_manager_saifly_and_wait_till_its_up(
            carla_ip=args.carla_ip,
            rpc_port=args.rpc_port,
            tm_port=args.tm_port,
            number_of_vehicles=args.num_of_vehicle,
            number_of_walkers=args.num_of_walkers,
            carla_server_pid=carla_server_pid,
            traffic_manager_pid=traffic_manager_pid,
            logs_path=traffic_manager_log_path
        )

    if not carla_is_ok:
        raise Exception(utils.color_error_string(f"Carla crashed while setting up Traffic Manager!"))
    if not traffic_manager_is_ok:
        raise Exception(utils.color_error_string(f"Traffic Manager Crashed!"))

    print(utils.color_info_string("(3/3)\tTraffic Manager Setted Up properly!"))

    # (5) LAUNCH DATA CREATION PROCESS
    data_creation_pid = multiprocessing.Value(c_int)
    ego_vehicle_found_event = multiprocessing.Event()
    finished_taking_data_event = multiprocessing.Event()
    data_creation_process = multiprocessing.Process(target=take_data_without_records.take_data_backbone,
                                                    args=(egg_file_path, args.town, args.rpc_port, args.job_id,
                                                          ego_vehicle_found_event,finished_taking_data_event,
                                                          you_can_tick, args.num_of_frames, datasets_path))
    data_creation_process.start()
    data_creation_pid.value = data_creation_process.pid

    print(utils.get_a_title(f"STARTING TO TAKE DATA [{args.num_of_frames}]", color="green"))
    start_time = time.time()
    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            kill_all(carla_server_pid, traffic_manager_pid, data_creation_pid)
            raise Exception(utils.color_error_string(f"Carla crashed!"))
        if not psutil.pid_exists(traffic_manager_pid.value):
            kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise Exception(utils.color_error_string(f"Traffic Manager crashed!"))
        if not psutil.pid_exists(data_creation_pid.value):
            kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise Exception(utils.color_error_string(f"Data Creation crashed!"))
        if not ego_vehicle_found_event.is_set() and time.time()-start_time > 10:
            kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise Exception(utils.color_error_string(f"Data Creation is not able to find out the Ego Vehicle!"))
        if finished_taking_data_event.is_set():
            break
    
    print(utils.get_a_title(f"FINISHED TO TAKE DATA [{args.num_of_frames}]", color="green"))
    
    # (6) CLEANING EVERYTHING
    kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
