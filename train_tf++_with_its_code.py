import argparse
import os
import pathlib
from tabulate import tabulate
import subprocess
import datetime
import psutil
import signal
import time

from nutfuser import utils

def get_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--dataset_train",
        help="Where to take the data for training!",
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets"),
        type=str
    )
    argparser.add_argument(
        "--dataset_validation",
        help="Where to take the data for validation!",
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets"),
        type=str
    )
    argparser.add_argument(
        "--weight",
        help="Where to take the pretrained weights!",
        required=False,
        default=None,
        type=str
    )
    argparser.add_argument(
        '--just_backbone',
        help='Set if you want to train just the backbone!',
        action='store_true'
    )
    argparser.add_argument(
        '--batch_size',
        help='Batch size of the training!',
        required=False,
        default=10,
        type=int
    )
    argparser.add_argument(
        '--train_flow',
        help='If set we train also Optical Flow!',
        action='store_true'
    )
    args = argparser.parse_args()
    # THERE I CHECK THE ARGUMENTS
    if args.dataset_train == args.dataset_validation:
        print(utils.color_info_string(
            "WARNING:"
        )+
        utils.color_error_string(
            "The training and validation set are the same!"
        ))
    if args.just_backbone is False and args.weight is None:
        raise  utils.NutException(utils.color_error_string(
            "You cannot train on the all network (not just the backbone) without giving some weights.\n"+
            "The whole trainign process is composed firstly by a backbone training of 30 epochs and secondly"+
            " by a full network training with the previusly computed weights!"))
    if not os.path.isdir(args.dataset_train):
        raise  utils.NutException(utils.color_error_string(
            f"The folder '{args.dataset_train}' does not exist!"))
    if not os.path.isdir(args.dataset_validation):
        raise  utils.NutException(utils.color_error_string(
            f"The folder '{args.dataset_validation}' does not exist!"))
    if args.weight is not None and not os.path.isfile(args.weight):
        raise  utils.NutException(utils.color_error_string(
            f"The file '{args.weight}' does not exist!"))
    # THERE I PROPERLY CHECK THAT THE DATASETFOLDERS ARE GOOD BUILTED
    for folder in os.listdir(args.dataset_train):
        folder_path = os.path.join(args.dataset_train, folder)
        if os.path.isdir(folder_path):
            utils.check_dataset_folder(folder_path)
    for folder in os.listdir(args.dataset_validation):
        folder_path = os.path.join(args.dataset_validation, folder)
        if os.path.isdir(folder_path):
            utils.check_dataset_folder(folder_path)
    if args.train_flow:
        args.train_flow = 1
    else:
        args.train_flow = 0
    return args

if __name__=="__main__":
    args = get_arguments()

    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        a_table.append([arg, getattr(args, arg)])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    nutfuser_path = pathlib.Path(__file__).parent.resolve()
    shell_train_path = os.path.join(nutfuser_path, "nutfuser", "neural_networks", "tf++", "shell_train.sh")
    train_script_path = os.path.join(nutfuser_path, "nutfuser", "neural_networks", "tf++", "train.py")
    train_logs_folder = os.path.join(nutfuser_path, "train_logs")
    venv_to_source_path = os.path.join(nutfuser_path, "bin", "activate")

    if not os.path.isdir(train_logs_folder):
        os.mkdir(train_logs_folder)

    now = datetime.datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    if not os.path.isdir(os.path.join(train_logs_folder, "nvidia_log")):
        os.mkdir(os.path.join(train_logs_folder, "nvidia_log"))
    nvidia_log = os.path.join(train_logs_folder, "nvidia_log", f"{current_time}")
    
    with open(nvidia_log, 'w') as logs_file:
        nvidia_process = subprocess.Popen(
                    ["nvidia-smi", "--loop=20"],
                    universal_newlines=True,
                    stdout=logs_file,
                    stderr=logs_file,
                )
        
    if not os.path.isdir(os.path.join(train_logs_folder, "output_log")):
        os.mkdir(os.path.join(train_logs_folder, "output_log"))
    output_log = os.path.join(train_logs_folder, "output_log", f"{current_time}")

    with open(output_log, 'w') as logs_file:
        train_process = subprocess.Popen(
                    [shell_train_path, f"{venv_to_source_path}", f"{train_script_path}", f"{args.dataset_train}", f"{args.dataset_validation}", f"{train_logs_folder}", f"{args.batch_size}", f"{args.train_flow}"],
                    universal_newlines=True,
                    stdout=logs_file,
                    stderr=logs_file,
                )
    
    train_pid = train_process.pid
    nvidia_pid = nvidia_process.pid

    print("Waiting 10 seconds that the training process starts!")
    for i in range(10):
        print(f"{i}, ", end="", flush=True)
        time.sleep(1)
    print()

    torch_run_pid = None
    children_pids = []

    for proc in psutil.process_iter():
        if proc.name() == "torchrun":
            torch_run_pid = proc.pid
            child = proc.children(recursive=True)
            for el in child:
                children_pids.append(el.pid)

    print(f"Ctrl-C For closing the Training!")

    try:
        while True:
            if not psutil.pid_exists(train_pid):
                break
    except KeyboardInterrupt:
        pass

    for pid in children_pids:
        os.kill(pid, signal.SIGKILL)
    os.kill(torch_run_pid, signal.SIGKILL)
    os.kill(train_pid, signal.SIGKILL)
    os.kill(nvidia_pid, signal.SIGKILL)
    print()
    print("Killed Everything!")
