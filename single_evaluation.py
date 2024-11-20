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

sys.path.append("/leonardo_scratch/fast/IscrC_ADC/CARLA_0_9_15/PythonAPI/carla/")


from nutfuser import config

sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), "scenario_runner"))


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
        '--evaluation_route',
        help=f'Path to the evaluation_route JSON file!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--where_to_save',
        help=f'Path to the evaluation JSON file!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--wait_carla_for',
        help='How many seconds wait for Carla! (default: 150)',
        required=False,
        default=150,
        type=int
    )
    argparser.add_argument(
        '--show_images',
        help=f'If you want to see the images of the car moving',
        action='store_true'
    )
    argparser.add_argument(
        '--weight_path',
        help=f'Path to the weights',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--id',
        help=f'ID of the evaluation!',
        required=False,
        default="-1",
        type=str
    )
    argparser.add_argument(
        '--completed_path',
        help=f'I will save a file there when the job will be completed!',
        required=False,
        default=None,
        type=str
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


def run_all(args, route_path, weights_path, output_dir_path, id):
    from nutfuser import utils
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
    agent_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "scenario_runner", "srunner", "autoagents",
                              "nutfuser_autonomous_agent.py")
    args_list = ["--route", route_path,
                 "--agent", agent_path,
                 "--timeout", "300",
                 "--agentConfig", weights_path,
                 "--json",
                 "--outputDir", output_dir_path,
                 "--port", str(args.rpc_port),
                 "--trafficManagerPort", str(args.tm_port)]
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
    if args.completed_path is not None:
        with open(os.path.join(args.completed_path, id), "w") as file:
            file.write(f"DONE [{id}]!\n")
            file.write(f"route_path = '{route_path}'\n")
            file.write(f"weights_path = '{weights_path}'\n")
            file.write(f"output_dir_path = '{output_dir_path}'\n")
    return True


if __name__ == "__main__":
    args = get_arguments()
    from nutfuser.carla_interface.add_carla_library import add_carla_library_to_path
    egg_file_path, carlaUE4_path = add_carla_library_to_path(args.carla_path, args.end_of_egg_file)
    from nutfuser import utils
    from nutfuser.carla_interface.run_carla import launch_carla_server_saifly_and_wait_till_its_up, \
        launch_scenario_runner_saifly_and_wait_till_its_up
    try:
        import carla
    except:
        raise Exception(utils.color_error_string(f"Not able to import Carla from [{egg_file_path}]"))


    print(utils.get_a_title("STARTING LAUNCHING ALL THE PROCESS", color="blue"))
    print(utils.color_info_success(f"Find out a valid carla in {egg_file_path}!"))

    # (0) SET UP LOGS AND DATASETFOLDER
    nutfuser_folder_path = pathlib.Path(__file__).parent.resolve()
    carla_log_path = os.path.join(nutfuser_folder_path, "logs", f"carla_server_evaluation_logs_{args.id}.log")
    scenario_runner_log_path = os.path.join(nutfuser_folder_path, "logs", f"scenario_runner.log")

    # (1) SHOW A TABLE WITH ALL THE ARGUMENTS
    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        a_table.append([arg, getattr(args, arg)])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    if not os.path.isfile(args.evaluation_route):
        raise utils.NutException(f"{args.evaluation_route} does not exists!")
    if not os.path.isfile(args.weight_path):
        raise utils.NutException(f"{args.weight_path} does not exists!")
    folder_where_to_save = os.path.dirname(args.where_to_save)
    print(folder_where_to_save)
    if not os.path.isdir(folder_where_to_save):
        raise utils.NutException(f"{folder_where_to_save} does not exists!")

    for i in range(config.MAX_NUM_OF_ATTEMPTS):
        try:
            print(utils.get_a_title(f"ATTEMPT [{i + 1}/{config.MAX_NUM_OF_ATTEMPTS}]", "blue"))
            if run_all(args,
                       route_path=args.evaluation_route,
                       weights_path=args.weight_path,
                       output_dir_path=args.where_to_save,
                       id=args.id):
                break
        except utils.NutException as e:
            print(e.message)
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
