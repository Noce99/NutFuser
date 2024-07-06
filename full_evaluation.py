import argparse
import multiprocessing
import os
import pathlib
import time
import signal
import psutil
from ctypes import c_int
from tabulate import tabulate
import sys

from nutfuser.data_creation import take_data_without_records, take_data_just_position_for_evaluation
from nutfuser import utils
from nutfuser import config
from datetime import datetime

sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), "scenario_runner"))
from nutfuser.carla_interface.run_carla import  check_integrity_of_carla_path, \
                                                launch_carla_server_saifly_and_wait_till_its_up, \
                                                launch_scenario_runner_saifly_and_wait_till_its_up

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
        '--evaluation_routes',
        help=f'Path to the evaluation_routes folder (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), "evaluation_routes")})',
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "evaluation_routes"),
        required=False,
        type=str
    )
    argparser.add_argument(
        '--where_to_save',
        help=f'Path to the evaluation folder (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), "evaluation")})',
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "evaluation"),
        required=False,
        type=str
    )
    argparser.add_argument(
        '--wait_carla_for',
        help='How many seconds wait for Carla! (default: 10)',
        required=False,
        default=10,
        type=int
    )
    argparser.add_argument(
        '--show_images',
        help=f'If you want to see the images of the car moving',
        action='store_true'
    )
    args = argparser.parse_args()
    return args

def kill_all(carla_server_pid, evaluation_pid):
    try:
        os.kill(carla_server_pid, signal.SIGKILL)
    except:
        pass
    try:
        os.kill(evaluation_pid, signal.SIGKILL)
    except:
        pass

pids_to_be_killed = []

def run_all(args, route_path, weights_path, output_dir_path):
    # (1) LAUNCH CARLA SERVER
    print("Launching Carla Server...")
    os.environ["PATH"] = f"{os.environ['PATH']}:/leonardo/home/userexternal/emannocc/xdg-user-dirs-0.18/"
    carla_server_pid = multiprocessing.Value(c_int)
    carla_was_correctly_started_up = launch_carla_server_saifly_and_wait_till_its_up(
            rpc_port=args.rpc_port,
            carla_server_pid=carla_server_pid,
            carlaUE4_path=carlaUE4_path,
            logs_path=carla_log_path,
            how_many_seconds_to_wait=args.wait_carla_for
        )

    pids_to_be_killed.append(carla_server_pid.value)

    if not carla_was_correctly_started_up:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while starting!"))
    
    print(utils.color_info_string("(1/3)\tCarla Server is UP!"))

    # (2) LAUNCH SCENARIO RUNNER
    print("Launching Scenario Runner...")
    agent_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "scenario_runner", "srunner", "autoagents", "nutfuser_autonomous_agent.py")
    args_list = ["--route",         route_path,
                 "--agent",         agent_path,
                 "--timeout",       "30",
                 "--agentConfig",   weights_path,
                 "--json",
                 "--outputDir",     output_dir_path]
    if args.show_images:
        args_list.append("--show_images")
    scenario_runner_pid = multiprocessing.Value(c_int)
    scenario_runner_is_done = multiprocessing.Event()
    carla_is_ok, scenario_runner_is_ok, launch_scenario_runner_process = launch_scenario_runner_saifly_and_wait_till_its_up(
            scenario_runner_pid=scenario_runner_pid,
            carla_server_pid=carla_server_pid,
            args_list=args_list,
            scenario_runner_is_done=scenario_runner_is_done
        )

    pids_to_be_killed.append(scenario_runner_pid.value)

    if not carla_is_ok:
        raise utils.NutException(utils.color_error_string(f"Carla crashed while setting up Scenario Runner!"))
    if not scenario_runner_is_ok:
        raise utils.NutException(utils.color_error_string(f"Scenario Runner Crashed!"))
    
    print(utils.color_info_string("(2/3)\tScenario Runner is UP!"))
    
    while True:
        if not psutil.pid_exists(carla_server_pid.value):
            raise utils.NutException(utils.color_error_string(f"Carla crashed!"))
        if not launch_scenario_runner_process.is_alive():
            break
        if scenario_runner_is_done.is_set():
            break

    # (6) CLEANING EVERYTHING
    print(utils.color_info_success(f"Scenario Runner Finished!"))
    kill_all(carla_server_pid.value, scenario_runner_pid.value)
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
    carla_log_path = os.path.join(nutfuser_folder_path, "logs", f"carla_server_evaluation_logs.log")
    scenario_runner_log_path = os.path.join(nutfuser_folder_path, "logs", f"scenario_runner.log")

    # (1) SHOW A TABLE WITH ALL THE ARGUMENTS
    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        a_table.append([arg, getattr(args, arg)])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    # (2-3) LET'S CREATE A LIST OF EVALUATIONS TO DO & LET'S CREATE THE OUTPUT FOLDERS
    evaluation_worklist = []
    evaluation_routes_folders = os.listdir(args.evaluation_routes)
    a_table_head = ["Evaluation Route", "Flow?"]
    a_table = []
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    os.mkdir(os.path.join(args.where_to_save, f"EVAL_{current_time}"))
    for element in evaluation_routes_folders:
        os.mkdir(os.path.join(args.where_to_save, f"EVAL_{current_time}", element))
        evaluation_worklist.append({"route_path":os.path.join(args.evaluation_routes, element, "evaluation.xml"),
                                    "weights_path":"/home/enrico/Projects/Carla/NutFuser/train_logs/full_net_NO_flow/model_0030.pth",
                                    "where_to_save":os.path.join(args.where_to_save, f"EVAL_{current_time}", element)})
        evaluation_worklist.append({"route_path":os.path.join(args.evaluation_routes, element, "evaluation.xml"),
                                    "weights_path":"/home/enrico/Projects/Carla/NutFuser/train_logs/full_net_flow/model_0030.pth",
                                    "where_to_save":os.path.join(args.where_to_save, f"EVAL_{current_time}", element)})
        a_table.append([element, True])
        a_table.append([element, False])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    # (4) LET'S RUN THE EVALUATION
    for k, element in enumerate(evaluation_worklist):
        to_print = f"# Evaluation n. {k+1}/{len(evaluation_worklist)} #"
        print(utils.color_info_string("#"*len(to_print)))
        print(utils.color_info_string(to_print))
        print(utils.color_info_string("#"*len(to_print)))
        # LET'S RUN ALL
        for i in range(config.MAX_NUM_OF_ATTEMPTS):
            try:
                print(utils.get_a_title(f"ATTEMPT [{i+1}/{config.MAX_NUM_OF_ATTEMPTS}]", "blue"))
                if run_all(args,
                           route_path=element["route_path"],
                           weights_path=element["weights_path"],
                           output_dir_path=element["where_to_save"]):
                    break
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

    
