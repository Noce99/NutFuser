import sys
import pathlib
import argparse

import nutfuser.config as config
import nutfuser.utils as utils
from nutfuser.neural_networks.data_loader import backbone_dataset
from nutfuser.neural_networks.model import LidarCenterNet
from nutfuser.neural_networks.trainer import Trainer
from nutfuser.neural_networks.tfpp_config import GlobalConfig

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch import optim
import json
import pickle
import os
import datetime
import numpy as np
import random
import time
from tabulate import tabulate
from tqdm import tqdm
import multiprocessing

def get_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--dataset_train",
        help=f"Where to take the data for training! (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), 'datasets', 'train_dataset')})",
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets", "train_dataset"),
        type=str
    )
    argparser.add_argument(
        "--dataset_validation",
        help="Where to take the data for validation! (default: None)",
        required=False,
        default=None,
        type=str
    )
    argparser.add_argument(
        '--just_backbone',
        help='Set if you want to train just the backbone! (default: False)',
        action='store_true'
    )
    argparser.add_argument(
        '--batch_size',
        help='Batch size of the training! (default: 10)',
        required=False,
        default=10,
        type=int
    )
    argparser.add_argument(
        '--train_flow',
        help='If set we train also Optical Flow! (default: False)',
        action='store_true'
    )
    argparser.add_argument(
        "--resume_path",
        help="Path to an old training log folder that we want to resume! (default: None)",
        default=None,
        required=False,
        type=str
    )
    argparser.add_argument(
        '--epochs',
        help='How many epochs do you want to train? (default: 30)',
        required=False,
        default=30,
        type=int
    )
    args = argparser.parse_args()
    # THERE I CHECK THE ARGUMENTS
    if args.dataset_validation is None:
        args.dataset_validation = args.dataset_train
    if args.dataset_train == args.dataset_validation:
        print(utils.color_info_string(
            "WARNING:"
        )+
        utils.color_error_string(
            "The training and validation set are the same!"
        ))
    if args.just_backbone is False and args.weights_path is None:
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
    if args.resume_path is not None and not os.path.isdir(args.resume_path):
        raise  utils.NutException(utils.color_error_string(
            f"The folder '{args.resume_path}' that you asked to resume from does not exist!"))
    # THERE I PROPERLY CHECK THAT THE DATASETFOLDERS ARE WELL BUILTED
    for folder in tqdm(os.listdir(args.dataset_train)):
        folder_path = os.path.join(args.dataset_train, folder)
        if os.path.isdir(folder_path):
            utils.check_dataset_folder(folder_path)
    
    if args.dataset_validation != args.dataset_train:
        for folder in tqdm(os.listdir(args.dataset_validation)):
            folder_path = os.path.join(args.dataset_validation, folder)
            if os.path.isdir(folder_path):
                utils.check_dataset_folder(folder_path)
    return args

def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(rank: int, world_size: int, cpu_number: int, train_dataset_path: str, validation_dataset_path: str, train_just_backbone: bool, train_flow: bool, weight_path: str, batch_size: int, epochs: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "19991"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    my_train_dataset = backbone_dataset(rank=rank, dataset_path=train_dataset_path)
    my_validation_dataset = backbone_dataset(rank=rank, dataset_path=validation_dataset_path)

    tfpp_config_file = GlobalConfig()
    if train_just_backbone:
        tfpp_config_file.use_controller_input_prediction = False
    else:
        tfpp_config_file.use_controller_input_prediction = True
    if train_flow:
        tfpp_config_file.use_flow = True
    else:
        tfpp_config_file.use_flow = False
    my_model = LidarCenterNet(tfpp_config_file)
    optimizer = ZeroRedundancyOptimizer(
        my_model.parameters(),
        optimizer_class=optim.AdamW,
        lr=config.LEARNING_RATE,
        amsgrad=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        my_train_dataset,
        # drop_last=True,
        shuffle=True)
    dataloader_train = torch.utils.data.DataLoader(
        my_train_dataset,
        sampler=sampler,
        batch_size=batch_size,
        # worker_init_fn=seed_worker,
        # generator=torch.Generator(device='cpu').manual_seed(torch.initial_seed()),
        num_workers=int(cpu_number/world_size),
        pin_memory=False,
        #drop_last=True,
        shuffle=False)
    print(f"int(cpu_number/world_size) = {int(cpu_number/world_size)}")
    milestones = [config.REDUCE_LR_FIRST_TIME, config.REDUCE_LR_SECOND_TIME]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=config.LEARNING_RATE_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    my_model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[rank])

    now = datetime.datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    log_dir = os.path.join(os.getcwd(), "train_logs", f"logs_{current_time}")
    if rank == 0:
        if not os.path.isdir(os.path.join(os.getcwd(), "train_logs")):
            os.makedirs(os.path.join(os.getcwd(), "train_logs"))
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'args.txt'), 'w', encoding='utf-8') as f:
            json.dump("I should write all the args there! Since now I waas to lazy!", f, indent=2)
        with open(os.path.join(log_dir, 'config.pickle'), 'wb') as f2:
            pickle.dump("I should write all the args there! Since now I waas to lazy!", f2, protocol=4)
    else:
        time.sleep(1)
        
    trainer = Trainer(
                model,
                optimizer,
                dataloader_train,
                my_train_dataset,
                my_validation_dataset,
                scheduler,
                scaler,
                rank,
                log_dir,
                train_just_backbone,
                train_flow,
                )
    trainer.train_for_epochs(epochs)

if __name__ == "__main__":
    args = get_arguments()
    gpu_number = torch.cuda.device_count()
    cpu_number = multiprocessing.cpu_count()

    # Let's show all the training arguments with a table
    a_table_head = ["Argument", "Value"]
    a_table = []
    for arg in vars(args):
        if arg == "resume_path" and getattr(args, arg) is None:
            a_table.append([arg, "!NEW TRAINING!"])
            continue
        a_table.append([arg, getattr(args, arg)])
    a_table.append(["GPUs:", gpu_number])
    a_table.append(["CPUs:", cpu_number])
    print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    # There we start the multiprocessing training
    torch.multiprocessing.spawn(main, args=(    gpu_number,
                                                cpu_number,
                                                args.dataset_train,
                                                args.dataset_validation,
                                                args.just_backbone,
                                                args.train_flow,
                                                args.resume_path,
                                                args.batch_size,
                                                args.epochs), nprocs=gpu_number)
