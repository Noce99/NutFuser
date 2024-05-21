import nutfuser.config as config
import nutfuser.utils as utils

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
                dataset_validation,
                scheduler,
                scaler,
                rank,
                log_dir,
                train_just_backbone,
                train_flow):
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.scheduler = scheduler
        self.scaler = scaler
        self.rank = rank
        self.log_dir = log_dir
        self.train_just_backbone = train_just_backbone
        self.train_flow = train_flow

        self.epoch = 0
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.tensor_board_i = self.epoch * config.NUM_OF_TENSORBOARD_LOGS_PER_EPOCH
        self.tensor_board_val_i = self.epoch
        if not self.train_just_backbone:
            self.tensor_board_i += 30 * config.NUM_OF_TENSORBOARD_LOGS_PER_EPOCH
            self.tensor_board_val_i += 30
        self.nut_validation_i = 0
        steps_after_log_train = (len(self.dataloader_train) - 1) / config.NUM_OF_TENSORBOARD_LOGS_PER_EPOCH
        self.when_to_log_train = [0] + [round(i*steps_after_log_train) for i in range(1, config.NUM_OF_TENSORBOARD_LOGS_PER_EPOCH)]

        visual_validation_path = os.path.join(self.log_dir, "visual_validation")
        if self.rank == 0:
            os.mkdir(visual_validation_path)

    def train(self):
        self.model.train()

        num_batches = 0
        loss_total = 0
        losses_total = {key: 0.0 for key in ["loss_semantic",
                                             "loss_bev_semantic",
                                             "loss_depth"]}
        if self.train_flow:
            losses_total["loss_flow"] = 0.0
        if not self.train_just_backbone:
            losses_total["loss_target_speed"] = 0.0
            losses_total["loss_checkpoint"] = 0.0

        self.optimizer.zero_grad()
        for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):

            losses = self.load_data_use_model_compute_loss(data, validation=False)
            loss = torch.zeros(1, dtype=torch.float32, device=self.rank)

            for key, value in losses.items():
                if key == "loss_flow":
                    loss += value * config.FLOW_LOSS_MULTIPLIER
                    losses_total[key] += float(value.item()) * config.FLOW_LOSS_MULTIPLIER
                else:
                    loss += value
                    losses_total[key] += float(value.item())

            loss_total += float(loss.item())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            num_batches += 1

            if i in self.when_to_log_train:
                self.log_losses_tensorboard(total_loss=loss_total, detailed_total_losses=losses_total, num_batches=num_batches, prefix='')

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    def train_for_epochs(self, epochs):
        self.trainng_start_time = time.time()
        for epoch in range(epochs):
            if self.rank == 0:
                print(f"EPOCH {epoch}")
            self.epoch = epoch
            self.train()
            torch.cuda.empty_cache()
            if self.rank == 0:
                self.qualitative_validation()
                torch.cuda.empty_cache()


            self.scheduler.step()
            if self.rank == 0:
                self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)


    def load_data_use_model_compute_loss(self, data, validation=False):
        # Validation = True will compute additional metrics not used for optimization
        # Load data used in both methods

        bb_center_heatmap = None
        bb_wh = None
        bb_yaw_class = None
        bb_yaw_res = None
        bb_offset = None
        bb_velocity = None
        bb_brake_target = None
        bb_pixel_weight = None
        bb_avg_factor = None
        ego_waypoint = None
        target_point = None
        command = None
        ego_vel = None
        target_speed = None
        checkpoint_label = None

        rgb_a = data["rgb_A_0"].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        rgb_b = data["rgb_B_0"].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        rgb = torch.concatenate([rgb_a, rgb_b], dim=1)
        semantic_label = F.one_hot(data["semantic_0"][:, :, :, 0].type(torch.LongTensor), 8).permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        bev_semantic_label = F.one_hot(torch.rot90(data["bev_semantic"], 3, [1, 2])[:, :, :, 0].type(torch.LongTensor), 6).permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        depth_label = (data["depth_0"][:, :, :, 0]/255).contiguous().to(self.rank, dtype=torch.float32)
        lidar = data["bev_lidar"][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)
        flow_label = (data["optical_flow_0"][:, :, :, :2] / 2**15 - 1).permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)

        if not self.train_just_backbone:
            target_point = data["targetpoint"].to(self.rank, dtype=torch.float32)
            if target_point.shape[1] == 3:
                target_point = target_point[:, :-1]
            command = None
            ego_vel = data["input_speed"].to(self.rank, dtype=torch.float32)
            ego_vel = ego_vel[:, None]
            target_speed = data["target_speed"].to(self.rank, dtype=torch.float32)
            checkpoint_label = data["waypoints"].to(self.rank, dtype=torch.float32)
            if checkpoint_label.shape[2] == 3:
                checkpoint_label = checkpoint_label[:, :, :-1]
        else:
            target_point = None
            command = None
            ego_vel = None
        
        pred_wp,\
        pred_target_speed,\
        pred_checkpoint,\
        pred_semantic, \
        pred_bev_semantic, \
        pred_depth, \
        pred_bounding_box, _, \
        pred_wp_1, \
        selected_path, \
        pred_flow = self.model( rgb=rgb,
                                lidar_bev=lidar,
                                target_point=target_point,
                                ego_vel=ego_vel,
                                command=command)
        
        compute_loss = self.model.module.compute_loss
        losses = compute_loss(pred_wp=pred_wp,
                              pred_target_speed=pred_target_speed,
                              pred_checkpoint=pred_checkpoint,
                              pred_semantic=pred_semantic,
                              pred_bev_semantic=pred_bev_semantic,
                              pred_depth=pred_depth,
                              pred_flow=pred_flow,
                              pred_bounding_box=pred_bounding_box,
                              waypoint_label=ego_waypoint,
                              target_speed_label=target_speed,
                              checkpoint_label=checkpoint_label,
                              semantic_label=semantic_label,
                              bev_semantic_label=bev_semantic_label,
                              depth_label=depth_label,
                              flow_label=flow_label,
                              center_heatmap_label=bb_center_heatmap,
                              wh_label=bb_wh,
                              yaw_class_label=bb_yaw_class,
                              yaw_res_label=bb_yaw_res,
                              offset_label=bb_offset,
                              velocity_label=bb_velocity,
                              brake_target_label=bb_brake_target,
                              pixel_weight_label=bb_pixel_weight,
                              avg_factor_label=bb_avg_factor,
                              pred_wp_1=pred_wp_1,
                              selected_path=selected_path)
        return losses


    def log_losses_tensorboard(self, total_loss, detailed_total_losses, num_batches, prefix=''):
        # Collecting the losses from all GPUs has led to issues.
        # I simply log the loss from GPU 0 for now they should be similar.
        if self.rank == 0:

            if prefix == '':
                index = self.tensor_board_i
            else:
                index = self.tensor_board_val_i

            self.writer.add_scalar(prefix + 'loss_total', total_loss / num_batches, index)

            for key, value in detailed_total_losses.items():
                if value != 0.0:
                    self.writer.add_scalar(prefix + key, value / num_batches, index)

            if prefix == '':
                self.tensor_board_i += 1
            else:
                self.tensor_board_val_i += 1

    @torch.inference_mode()
    def qualitative_validation(self):
        self.model.eval()
        print("Creating Visual Evaluation Data!")

        some_random_idxs = random.sample(range(1, len(self.dataset_validation)), 10)

        folder_path = os.path.join(self.log_dir, "visual_validation", str(self.nut_validation_i))
        self.nut_validation_i += 1
        os.mkdir(folder_path)

        k = 0
        for idx in some_random_idxs:
            data = self.dataset_validation[idx]
            if not self.train_just_backbone:
                target_point = torch.from_numpy(data["targetpoint"]).to(self.rank, dtype=torch.float32)
                if target_point.shape[0] == 3:
                    target_point = target_point[:-1]
                target_point = target_point[None, :]
                command = None
                ego_vel = torch.tensor([data["input_speed"]]).to(self.rank, dtype=torch.float32)
                ego_vel = ego_vel[:, None]
                target_speed = torch.from_numpy(data["target_speed"]).to(self.rank, dtype=torch.float32)
                checkpoint_label = torch.from_numpy(data["waypoints"]).to(self.rank, dtype=torch.float32)
                if checkpoint_label.shape[1] == 3:
                    checkpoint_label = checkpoint_label[:, :-1]
            else:
                target_point = None
                command = None
                ego_vel = None

            rgb_a = torch.from_numpy(data["rgb_A_0"]).permute(2, 0, 1).contiguous()[None, :].to(self.rank, dtype=torch.float32)
            rgb_b = torch.from_numpy(data["rgb_A_0"]).permute(2, 0, 1).contiguous()[None, :].to(self.rank, dtype=torch.float32)
            rgb = torch.concatenate([rgb_a, rgb_b], dim=1)
            lidar = torch.from_numpy(data["bev_lidar"])[None, :][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(self.rank, dtype=torch.float32)

            pred_wp,\
            pred_target_speed,\
            pred_checkpoint,\
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            pred_bounding_box, _, \
            pred_wp_1, \
            selected_path, \
            pred_flow = self.model( rgb=rgb,
                                    lidar_bev=lidar,
                                    target_point=target_point,
                                    ego_vel=ego_vel,
                                    command=command)

            pred_depth = (pred_depth[0, :, :].detach().cpu().numpy()*255).astype(np.uint8)
            pred_semantic = torch.argmax(pred_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
            pred_bev_semantic = torch.argmax(pred_bev_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
            if self.train_flow:
                pred_flow = ((pred_flow + 1)*(2**15)).permute(0, 2, 3, 1)[0, :, :, :].contiguous().detach().cpu().numpy()

            depth_comparison = np.zeros((pred_depth.shape[0]*2, pred_depth.shape[1]), dtype=np.uint8)
            depth_comparison[0:pred_depth.shape[0], :] = pred_depth
            depth_comparison[pred_depth.shape[0]:, :] = data["depth_0"][:, :, 0]

            semantic_comparison = np.zeros((pred_semantic.shape[0]*2, pred_semantic.shape[1]), dtype=np.uint8)
            semantic_comparison[0:pred_semantic.shape[0], :] = pred_semantic
            semantic_comparison[pred_depth.shape[0]:, :] = data["semantic_0"][:, :, 0]

            bev_semantic_comparison = np.zeros((pred_bev_semantic.shape[0]*2, pred_bev_semantic.shape[1], 3), dtype=np.uint8)
            pred_bev_semantic = np.rot90(pred_bev_semantic, 1)
            bev_semantic_comparison[0:pred_bev_semantic.shape[0], :, 0] = pred_bev_semantic
            bev_semantic_comparison[0:pred_bev_semantic.shape[0], :, 1] = pred_bev_semantic
            bev_semantic_comparison[0:pred_bev_semantic.shape[0], :, 2] = pred_bev_semantic

            rgb_ground_truth = np.zeros((data["bev_semantic"].shape[0], data["bev_semantic"].shape[1], 3))
            rgb_ground_truth[:, :] = data["bev_semantic"]
            if not self.train_just_backbone:
                for i in range(checkpoint_label.shape[0]):
                    rgb_ground_truth = cv2.circle(rgb_ground_truth, (int(128-checkpoint_label[i, 0]*256/config.BEV_SQUARE_SIDE_IN_M),
                                                                    int(128-checkpoint_label[i, 1]*256/config.BEV_SQUARE_SIDE_IN_M)),
                                                                    2, (0, 255, 0), -1)
                    rgb_ground_truth = cv2.circle(rgb_ground_truth, (int(128-pred_checkpoint[0, i, 0]*256/config.BEV_SQUARE_SIDE_IN_M),
                                                                    int(128-pred_checkpoint[0, i, 1]*256/config.BEV_SQUARE_SIDE_IN_M)),
                                                                    3, (0, 0, 255), -1)
                list_target_speed = [float(el) for el in target_speed]
                list_predicted_speed = [float(el) for el in torch.nn.functional.softmax(pred_target_speed[0], dim=0)]
                cv2.putText(rgb_ground_truth, f"{list_target_speed[0]:.2f}, {list_target_speed[1]:.2f}, {list_target_speed[2]:.2f}, {list_target_speed[3]:.2f}", (0, 128+60), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.6, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(rgb_ground_truth, f"{list_predicted_speed[0]:.2f}, {list_predicted_speed[1]:.2f}, {list_predicted_speed[2]:.2f}, {list_predicted_speed[3]:.2f}", (0, 128+90), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.6, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            bev_semantic_comparison[pred_bev_semantic.shape[0]:, :, :] = rgb_ground_truth # np.rot90(data["bev_semantic"][:, :, 0], 3)

            if self.train_flow:
                flow_comparison = np.zeros((pred_flow.shape[0]*2, pred_flow.shape[1], 3), dtype=np.uint8)
                # print(f"PRED\t0 : [{np.min(pred_flow[:, :, 0])}; {np.max(pred_flow[:, :, 0])}] 1 : [{np.min(pred_flow[:, :, 1])}; {np.max(pred_flow[:, :, 1])}]")
                # print(f"LABEL\t0 : [{np.min(data['optical_flow_0'][:, :, 0])}; {np.max(data['optical_flow_0'][:, :, 0])}] 1 : [{np.min(data['optical_flow_0'][:, :, 1])}; {np.max(data['optical_flow_0'][:, :, 1])}]")
                flow_comparison[0:pred_flow.shape[0], :, :] = utils.optical_flow_to_human(pred_flow)
                flow_comparison[pred_flow.shape[0]:, :, :] = utils.optical_flow_to_human(data["optical_flow_0"][:, :, :2])

            cv2.imwrite(os.path.join(folder_path, f"depth_{k}.png"), depth_comparison)
            cv2.imwrite(os.path.join(folder_path, f"semantic_{k}.png"), semantic_comparison*30)
            cv2.imwrite(os.path.join(folder_path, f"bev_semantic_{k}.png"), bev_semantic_comparison*30)
            if self.train_flow:
                cv2.imwrite(os.path.join(folder_path, f"flow_{k}.png"), flow_comparison)
            k += 1
        print("Finished to create Visual Evaluation Data!")