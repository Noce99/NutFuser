import config

from tqdm import tqdm
import torch
import random
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import cv2


class Trainer:
    def __init__(self,
                model,
                optimizer,
                dataloader_train,
                dataset_train,
                scheduler,
                scaler,
                rank,
                log_dir):
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataset_train = dataset_train
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scheduler = scheduler
        self.scaler = scaler
        self.rank = rank

        self.epoch = 0
        self.qualitative_validation_index = 0
        self.tensor_board_index = 0
        visual_validation_path = os.path.join(self.log_dir, "visual_validation")
        os.mkdir(visual_validation_path)
    
    def train(self):
        self.model.train()

        num_batches = 0
        loss_total = 0
        losses_total = {key: 0.0 for key in ["loss_semantic",
                                             "loss_bev_semantic",
                                             "loss_depth"]}

        self.optimizer.zero_grad()
        for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):
            
            losses = self.load_data_use_model_compute_loss(data, validation=False)
            loss = torch.zeros(1, dtype=torch.float32, device=self.rank)

            for key, value in losses.items():

                loss += value * 0.33
                losses_total[key] += float(value.item()) * 0.33

            loss_total += float(loss.item())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            num_batches += 1

            if i%20 == 0:
                self.log_losses_tensorboard(total_loss=loss_total, detailed_loss=losses_total, num_batches=num_batches, prefix='')

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def train_for_epochs(self, epochs):
        self.trainng_start_time = time.time()
        for epoch in range(epochs):
            if self.rank == 0:
                print(f"EPOCH {epoch}")
            self.epoch = epoch
            if self.rank == 0:
                self.qualitative_validation()
                torch.cuda.empty_cache()
            self.train()
            torch.cuda.empty_cache()

            self.scheduler.step()
            if self.rank == 0:
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)


    def load_data_use_model_compute_loss(self, data, validation=False):
        # Elements inside the data dict are already tensors! Magic!
        for data_folder in config.DATASET_FOLDER_STRUCT:
            data_name, _ = data_folder
            data[data_name].to(self.rank)
        
        target_point = None
        command = None
        ego_vel = None

        rgb = data["rgb_A_0"].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        semantic_label = F.one_hot(data["semantic_0"][:, :, :, 0].type(torch.LongTensor), 8).permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        bev_semantic_label = F.one_hot(torch.rot90(data["bev_semantic"], 3, [1, 2])[:, :, :, 0].type(torch.LongTensor), 6).permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        depth_label = (data["depth_0"][:, :, :, 0]/255).contiguous().to(self.rank, dtype=torch.float32)
        lidar = data["bev_lidar"][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)

        pred_wp,\
        pred_target_speed,\
        pred_checkpoint,\
        pred_semantic, \
        pred_bev_semantic, \
        pred_depth, \
        pred_bounding_box, _, \
        pred_wp_1, \
        selected_path = self.model(rgb=rgb,
                            lidar_bev=lidar,
                            target_point=target_point,
                            ego_vel=ego_vel,
                            command=command)
        
        compute_loss = self.model.module.compute_loss
        losses = compute_loss(  pred_semantic=pred_semantic,
                                pred_bev_semantic=pred_bev_semantic,
                                pred_depth=pred_depth,
                                semantic_label=semantic_label,
                                bev_semantic_label=bev_semantic_label,
                                depth_label=depth_label,
                            )

        return losses


    def log_losses_tensorboard(self, total_loss, detailed_loss, num_batches, prefix=''):
        if self.rank == 0:
            self.writer.add_scalar(prefix + 'loss_total', total_loss / num_batches, self.tensor_board_index)
        
            for key, value in detailed_loss.items():
                self.writer.add_scalar(prefix + key, value / num_batches, self.tensor_board_index)
                self.tensor_board_index += 1

    @torch.inference_mode()
    def qualitative_validation(self):
        self.model.eval()
        print("Creating Visual Evaluation Data!")

        some_random_idxs = random.sample(range(1, len(self.dataset_train)), 10)
        
        folder_path = os.path.join(self.log_dir, "visual_validation", str(self.qualitative_validation_index))
        self.qualitative_validation_index += 1
        os.mkdir(folder_path)

        k = 0
        for idx in some_random_idxs:
            data = self.dataset_train[idx]
            target_point = None
            command = None
            ego_vel = None

            rgb = torch.from_numpy(data["rgb_A_0"]).permute(2, 0, 1).contiguous()[None, :].to(self.rank, dtype=torch.float32)
            lidar = torch.from_numpy(data["bev_lidar"])[None, :][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)

            pred_wp,\
            pred_target_speed,\
            pred_checkpoint,\
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            pred_bounding_box, _, \
            pred_wp_1, \
            selected_path = self.model(rgb=rgb,
                                lidar_bev=lidar,
                                target_point=target_point,
                                ego_vel=ego_vel,
                                command=command)
            
            pred_depth = (pred_depth[0, :, :].detach().cpu().numpy()*255).astype(np.uint8)
            pred_semantic = torch.argmax(pred_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
            pred_bev_semantic = torch.argmax(pred_bev_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
    
            depth_comparison = np.zeros((pred_depth.shape[0]*2, pred_depth.shape[1]), dtype=np.uint8)
            depth_comparison[0:pred_depth.shape[0], :] = pred_depth
            depth_comparison[pred_depth.shape[0]:, :] = data["depth_0"][:, :, 0]

            semantic_comparison = np.zeros((pred_semantic.shape[0]*2, pred_semantic.shape[1]), dtype=np.uint8)
            semantic_comparison[0:pred_semantic.shape[0], :] = pred_semantic
            semantic_comparison[pred_depth.shape[0]:, :] = data["semantic_0"][:, :, 0]

            bev_semantic_comparison = np.zeros((pred_bev_semantic.shape[0]*2, pred_bev_semantic.shape[1]), dtype=np.uint8)
            bev_semantic_comparison[0:pred_bev_semantic.shape[0], :] = pred_bev_semantic
            bev_semantic_comparison[pred_bev_semantic.shape[0]:, :] = np.rot90(data["bev_semantic"][:, :, 0], 3)
            
            cv2.imwrite(os.path.join(folder_path, f"depth_{k}.png"), depth_comparison)
            cv2.imwrite(os.path.join(folder_path, f"semantic_{k}.png"), semantic_comparison*30)
            cv2.imwrite(os.path.join(folder_path, f"bev_semantic_{k}.png"), bev_semantic_comparison*30)
            k += 1
        print("Finished to create Visual Evaluation Data!")
