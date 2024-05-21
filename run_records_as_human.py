import argparse
import multiprocessing
import os
import pathlib
import time
import signal
import psutil
from ctypes import c_int
from tabulate import tabulate

from nutfuser.evaluation import recorder
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
        '--wait_carla_for',
        help='How many seconds wait for Carla! (default: 100)',
        required=False,
        default=100,
        type=int
    )
    argparser.add_argument(
        '--record_path',
        help=f'Witch record file to run!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--show_carla_window',
        help=f'If the carla window is displayed or not! (default: True)',
        required=False,
        default=True,
        type=bool
    )
    args = argparser.parse_args()
    if args.record_path[:2] == "./":
        args.record_path = args.record_path[2:]
    if not str(pathlib.Path(__file__).parent.resolve()) in args.record_path:
        args.record_path = os.path.join(pathlib.Path(__file__).parent.resolve(), args.record_path)
    if not os.path.isfile(args.record_path):
        raise Exception(utils.color_error_string(f"Cannot find the {args.record_path} file!"))
    return args


pids_to_be_killed = []

def run_all(args):
    # (1) LAUNCH CARLA SERVER
    print("Launching Carla Server...")
    os.environ["PATH"] = f"{os.environ['PATH']}:/leonardo/home/userexternal/emannocc/xdg-user-dirs-0.18/"
    carla_server_pid = multiprocessing.Value(c_int)
    carla_was_correctly_started_up = launch_carla_server_saifly_and_wait_till_its_up(
            rpc_port=args.rpc_port,
            carla_server_pid=carla_server_pid,
            carlaUE4_path=carlaUE4_path,
            logs_path=carla_log_path,
            how_many_seconds_to_wait=args.wait_carla_for,
            show_carla_window=args.show_carla_window
        )

    pids_to_be_killed.append(carla_server_pid.value)

    if not carla_was_correctly_started_up:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while starting!"))
    
    print(utils.color_info_string("(1/3)\tCarla Server is UP!"))

    """
    # (3) SET UP THE WORLD
    world_was_correctly_setted_up = set_up_world_saifly_and_wait_till_its_setted_up(
            carla_ip=args.carla_ip,
            rpc_port=args.rpc_port,
            town_number=args.town,
            carla_server_pid=carla_server_pid
        )
    
    if not world_was_correctly_setted_up:
        raise utils.NutException(utils.color_error_string(f"Failed to set up world!"))

    print(utils.color_info_string("(2/3)\tWorld was correctly setted up!"))

    # (4) SET UP TRAFFIC MANAGER
    traffic_manager_pid = multiprocessing.Value(c_int)
    carla_is_ok,\
    traffic_manager_is_ok,\
    you_can_tick,\
    traffic_manager_is_up,\
    set_up_traffic_manager_process = set_up_traffic_manager_saifly_and_wait_till_its_up(
            carla_ip=args.carla_ip,
            rpc_port=args.rpc_port,
            tm_port=args.tm_port,
            number_of_vehicles=args.num_of_vehicle,
            number_of_walkers=args.num_of_walkers,
            carla_server_pid=carla_server_pid,
            traffic_manager_pid=traffic_manager_pid,
            logs_path=traffic_manager_log_path
        )

    pids_to_be_killed.append(traffic_manager_pid.value)

    if not carla_is_ok:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while setting up Traffic Manager!"))
    if not traffic_manager_is_ok:
        raise utils.NutException(utils.color_error_string(f"Traffic Manager Crashed!"))

    print(utils.color_info_string("(3/3)\tTraffic Manager Setted Up properly!"))
    """

    # (5) LAUNCH RECORDER
    player_pid = multiprocessing.Value(c_int)
    finished_playing_event = multiprocessing.Event()
    player_process = multiprocessing.Process(target=recorder.play_record,
                                                args=(egg_file_path,
                                                      args.record_path,
                                                      args.rpc_port,
                                                      finished_playing_event))
    player_process.start()
    player_pid.value = player_process.pid
    pids_to_be_killed.append(player_pid.value)

    print(utils.get_a_title(f"STARTING TO PLAY RECORD", color="green"))
    start_time = time.time()
    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            #kill_all(carla_server_pid, traffic_manager_pid, data_creation_pid)
            raise utils.NutException(utils.color_error_string(f"Carla crashed!"))
        if not player_process.is_alive():
            #kill_all(carla_server_pid.value, traffic_manager_pid.value, data_creation_pid.value)
            raise utils.NutException(utils.color_error_string(f"Record Player crashed!"))
        if finished_playing_event.is_set():
            break
    
    print(utils.get_a_title(f"FINISHED TO PLAY RECORD", color="green"))

    # (6) CLEANING EVERYTHING
    try:
        os.kill(carla_server_pid.value, signal.SIGKILL)
    except:
        pass
    return True

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
    carla_log_path = os.path.join(nutfuser_folder_path, "logs", f"carla_server_logs_human.log")
    traffic_manager_log_path = os.path.join(nutfuser_folder_path, "logs", f"traffic_manager_logs_human.log")

    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        a_table.append([arg, getattr(args, arg)])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    # LET'S RUN ALL
    try:
        run_all(args)
    except utils.NutException as e:
        print(e.message)
        args.rpc_port += 1
        for pid in pids_to_be_killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except:
                pass
        pids_to_be_killed = []
    except KeyboardInterrupt:
        for pid in pids_to_be_killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except:
                pass
        exit()
