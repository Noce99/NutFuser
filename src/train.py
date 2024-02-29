import config
import utils
from data_loader import backbone_dataset
from model import simple_model
from trainer import Trainer

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import json
import pickle
import os
import datetime
import numpy as np
import random
import copy

def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_training(a_dataset):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    number_of_gpus = torch.cuda.device_count()
    device = torch.device(f'cuda:{rank}')

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://', # Default
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=15))
    
    a_model = simple_model().to(device)
    start_epoch = config.START_EPOCH
    if config.PRETRAINED_WEIGHT_PATH is not None:
        a_model.load_state_dict(torch.load(config.PRETRAINED_WEIGHT_PATH, map_location=device), strict=False)

    model = torch.nn.parallel.DistributedDataParallel(
        a_model,
        device_ids=None,
        output_device=None,
        broadcast_buffers=False)
    
    params = model.parameters()
    optimizer = ZeroRedundancyOptimizer(params, optimizer_class=optim.AdamW, lr=config.LEARNING_RATE, amsgrad=True)
    if config.PRETRAINED_OPTIMIZER_PATH is not None:
        optimizer.load_state_dict(torch.load(config.PRETRAINED_OPTIMIZER_PATH, map_location=device))
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum(np.prod(p.size()) for p in model_parameters)
    if rank == 0:
        print('Total trainable parameters: ', num_params)
    
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        a_dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank,
        drop_last=True)
    
    dataloader_train = torch.utils.data.DataLoader(
        a_dataset,
        sampler=sampler_train,
        batch_size=config.BATCH_SIZE,
        worker_init_fn=seed_worker,
        generator=torch.Generator(device='cpu').manual_seed(torch.initial_seed()),
        num_workers=4*number_of_gpus, # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813
        pin_memory=False,
        drop_last=True)
    
    if rank == 0:
        now = datetime.datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
        if not os.path.isdir(os.path.join(os.getcwd(), "train_logs")):
            os.makedirs(os.path.join(os.getcwd(), "train_logs"))
        log_dir = os.path.join(os.getcwd(), "train_logs", f"logs_{current_time}")
        os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        with open(os.path.join(log_dir, 'args.txt'), 'w', encoding='utf-8') as f:
            json.dump("I should write all the args there! Since now I waas to lazy!", f, indent=2)
        with open(os.path.join(log_dir, 'config.pickle'), 'wb') as f2:
            pickle.dump("I should write all the args there! Since now I waas to lazy!", f2, protocol=4)
    else:
        writer = None
    
    milestones = [config.REDUCE_LR_FIRST_TIME, config.REDUCE_LR_SECOND_TIME]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=config.LEARNING_RATE_DECAY,
        verbose=False)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    if config.PRETRAINED_SCHEDULER_PATH is not None:
        scheduler.load_state_dict(torch.load(config.PRETRAINED_SCHEDULER_PATH, map_location=device))
    if config.PRETRAINED_SCALER_PATH is not None:
        scaler.load_state_dict(torch.load(config.PRETRAINED_SCALER_PATH, map_location=device))

    print(utils.color_error_string("I'M LAZY SO FOR NOW VAL = TRAIN!"))
    dataloader_val = dataloader_train

    return model, optimizer, dataloader_train, dataloader_val, writer, device, scheduler, scaler, rank, world_size, start_epoch
    

if __name__ == "__main__":
    my_dataset = backbone_dataset()
    training_stuff = setup_training(my_dataset)
    trainer = Trainer(*training_stuff)
    my_dataset.close()
