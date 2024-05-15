import sys
import pathlib

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

def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(rank: int, world_size: int, train_dataset_path: str, validation_dataset_path: str, train_just_backbone: bool, train_flow: bool):
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
        batch_size=config.BATCH_SIZE,
        # worker_init_fn=seed_worker,
        # generator=torch.Generator(device='cpu').manual_seed(torch.initial_seed()),
        # num_workers=4*world_size, # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813
        pin_memory=False,
        #drop_last=True,
        shuffle=False)
    milestones = [config.REDUCE_LR_FIRST_TIME, config.REDUCE_LR_SECOND_TIME]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=config.LEARNING_RATE_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    my_model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[rank])

    if rank == 0:
        now = datetime.datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
        if not os.path.isdir(os.path.join(os.getcwd(), "train_logs")):
            os.makedirs(os.path.join(os.getcwd(), "train_logs"))
        log_dir = os.path.join(os.getcwd(), "train_logs", f"logs_{current_time}")
        os.makedirs(log_dir)

        with open(os.path.join(log_dir, 'args.txt'), 'w', encoding='utf-8') as f:
            json.dump("I should write all the args there! Since now I waas to lazy!", f, indent=2)
        with open(os.path.join(log_dir, 'config.pickle'), 'wb') as f2:
            pickle.dump("I should write all the args there! Since now I waas to lazy!", f2, protocol=4)
    else:
        writer = None

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
    trainer.train_for_epochs(2)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    train_dataset_path = "/home/enrico/Projects/Carla/NutFuser/datasets/train_dataset"
    validation_dataset_path = "/home/enrico/Projects/Carla/NutFuser/datasets/train_dataset"
    train_just_backbone = True
    train_flow = False
    # world_size = 2
    torch.multiprocessing.spawn(main, args=(world_size, train_dataset_path, validation_dataset_path, train_just_backbone, train_flow), nprocs=world_size)
