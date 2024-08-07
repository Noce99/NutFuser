"""
The main model structure
"""
from nutfuser.neural_networks.transfuser import TransfuserBackbone, TransformerDecoderLayerWithAttention, TransformerDecoderWithAttention
import nutfuser.neural_networks.transfuser_utils as t_u

import numpy as np
from pathlib import Path
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from copy import deepcopy
import math
import os
import cv2
import sys
import numpy as np

import pdb
import datetime

SHOW_STUFF = False
M_to_PX = 256/64
OUTPUT_VIDEO = None

class LidarCenterNet(nn.Module):
    """
    The main model class. It can run all model configurations.
    """

    def __init__(self, config):
        super().__init__()

        # SHOWING IMAGES by Noce
        self.SAVED_FRAMES = 0
        self.frame_skipped = 0
        self.skip_each = 10
        global OUTPUT_VIDEO
        if SHOW_STUFF:
            cv2.namedWindow('Output', cv2.WINDOW_AUTOSIZE)
        else:
            if OUTPUT_VIDEO is not None:
                OUTPUT_VIDEO.release()
            now = datetime.datetime.now()
            file_name = f"{now.day}_{now.month}_{now.year}-{now.hour}-{now.minute}-{now.second}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            OUTPUT_VIDEO = cv2.VideoWriter(f"/home/enrico/Projects/Carla/carla_garage/output_video_evaluation/{file_name}", fourcc, 15, (1576, 828))
        self.config = config

        self.speed_histogram = []
        self.make_histogram = int(os.environ.get('HISTOGRAM', 0))

        if self.config.backbone == 'transFuser': # <--------
            self.backbone = TransfuserBackbone(config)
        else:
            raise ValueError('The chosen vision backbone does not exist. '
                             'The options are: transFuser, aim, bev_encoder')

        if self.config.use_tp:
            target_point_size = 2
        else:
            target_point_size = 0

        self.extra_sensors = self.config.use_velocity or self.config.use_discrete_command
        extra_sensor_channels = 0
        if self.extra_sensors: # <------------ True
            extra_sensor_channels = self.config.extra_sensor_channels
            if self.config.transformer_decoder_join: # <------------------------ True
                extra_sensor_channels = self.config.gru_input_size

        # prediction heads
        if self.config.use_semantic:
            self.semantic_decoder = t_u.PerspectiveDecoder(
                in_channels=self.backbone.num_image_features,
                out_channels=self.config.num_semantic_classes,
                inter_channel_0=self.config.deconv_channel_num_0,
                inter_channel_1=self.config.deconv_channel_num_1,
                inter_channel_2=self.config.deconv_channel_num_2,
                scale_factor_0=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_0,
                scale_factor_1=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_1)

        if self.config.use_bev_semantic:
            self.bev_semantic_decoder = nn.Sequential(
                nn.Conv2d(self.config.bev_features_chanels,
                          self.config.bev_features_chanels,
                          kernel_size=(3, 3),
                          stride=1,
                          padding=(1, 1),
                          bias=True), nn.ReLU(inplace=True),
                nn.Conv2d(self.config.bev_features_chanels,
                          self.config.num_bev_semantic_classes,
                          kernel_size=(1, 1),
                          stride=1,
                          padding=0,
                          bias=True),
                nn.Upsample(size=(self.config.lidar_resolution_height, self.config.lidar_resolution_width),
                            mode='bilinear',
                            align_corners=False))

            # Computes which pixels are visible in the camera. We mask the others.
            _, valid_voxels = t_u.create_projection_grid(self.config)
            valid_bev_pixels = torch.max(valid_voxels, dim=3, keepdim=False)[0].unsqueeze(1)
            # Conversion from CARLA coordinates x depth, y width to image coordinates x width, y depth.
            # Analogous to transpose after the LiDAR histogram
            valid_bev_pixels = torch.transpose(valid_bev_pixels, 2, 3).contiguous()
            valid_bev_pixels_inv = 1.0 - valid_bev_pixels
            # Register as parameter so that it will automatically be moved to the correct GPU with the rest of the network
            self.valid_bev_pixels = nn.Parameter(valid_bev_pixels, requires_grad=False)
            self.valid_bev_pixels_inv = nn.Parameter(valid_bev_pixels_inv, requires_grad=False)

        if self.config.use_depth:
            self.depth_decoder = t_u.PerspectiveDecoder(
                in_channels=self.backbone.num_image_features,
                out_channels=1,
                inter_channel_0=self.config.deconv_channel_num_0,
                inter_channel_1=self.config.deconv_channel_num_1,
                inter_channel_2=self.config.deconv_channel_num_2,
                scale_factor_0=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_0,
                scale_factor_1=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_1)

        if self.config.use_flow:
            self.flow_decoder = t_u.PerspectiveDecoder(
                  in_channels=self.backbone.num_image_features,
                  out_channels=2,
                  inter_channel_0=self.config.deconv_channel_num_0,
                  inter_channel_1=self.config.deconv_channel_num_1,
                  inter_channel_2=self.config.deconv_channel_num_2,
                  scale_factor_0=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_0,
                  scale_factor_1=self.backbone.perspective_upsample_factor // self.config.deconv_scale_factor_1)

        if self.config.use_controller_input_prediction:# <------------ True
            if self.config.transformer_decoder_join: # <------------------------ True
                ts_input_channel = self.config.gru_input_size
            else:
                ts_input_channel = self.config.gru_hidden_size
            self.target_speed_network = nn.Sequential(nn.Linear(ts_input_channel, ts_input_channel), nn.ReLU(inplace=True),
                                                      nn.Linear(ts_input_channel, len(config.target_speeds)))

        if self.config.use_controller_input_prediction or self.config.use_wp_gru: # <------------ True or False
            if self.config.transformer_decoder_join: # <------------------------ True
                decoder_norm = nn.LayerNorm(self.config.gru_input_size)
                if self.config.tp_attention: # NOT ENTERING
                    self.tp_encoder = nn.Sequential(nn.Linear(2, 128), nn.ReLU(inplace=True),
                                                    nn.Linear(128, self.config.gru_input_size))
                    self.tp_pos_embed = nn.Parameter(torch.zeros(1, self.config.gru_input_size))

                    # Pytorch does not support attention visualization, so we need a custom implementation.
                    decoder_layer = TransformerDecoderLayerWithAttention(self.config.gru_input_size,
                                                                         self.config.num_decoder_heads,
                                                                         activation=nn.GELU())
                    self.join = TransformerDecoderWithAttention(decoder_layer,
                                                                num_layers=self.config.num_transformer_decoder_layers,
                                                                norm=decoder_norm)
                else: # <-------------------------
                    decoder_layer = nn.TransformerDecoderLayer(self.config.gru_input_size,
                                                               self.config.num_decoder_heads,
                                                               activation=nn.GELU(),
                                                               batch_first=True)
                    self.join = torch.nn.TransformerDecoder(decoder_layer,
                                                            num_layers=self.config.num_transformer_decoder_layers,
                                                            norm=decoder_norm)
                # We don't have an encoder, so we directly use it on the features
                self.encoder_pos_encoding = PositionEmbeddingSine(self.config.gru_input_size // 2, normalize=True)
                self.extra_sensor_pos_embed = nn.Parameter(torch.zeros(1, self.config.gru_input_size))

                self.change_channel = nn.Conv2d(self.backbone.num_features, self.config.gru_input_size, kernel_size=1)

                if self.config.use_wp_gru:# NOT ENTERING
                    if self.config.multi_wp_output:
                        self.wp_query = nn.Parameter(
                            torch.zeros(1, 2 * (config.pred_len // self.config.wp_dilation) + 1, self.config.gru_input_size))

                        self.wp_decoder = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                                          hidden_size=self.config.gru_hidden_size,
                                                                          waypoints=(config.pred_len // self.config.wp_dilation),
                                                                          target_point_size=target_point_size)
                        self.wp_decoder_1 = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                                            hidden_size=self.config.gru_hidden_size,
                                                                            waypoints=(config.pred_len // self.config.wp_dilation),
                                                                            target_point_size=target_point_size)
                        self.select_wps = nn.Linear(self.config.gru_input_size, 1)
                    else:
                        self.wp_query = nn.Parameter(
                            torch.zeros(1, (config.pred_len // self.config.wp_dilation), self.config.gru_input_size))

                        self.wp_decoder = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                                          hidden_size=self.config.gru_hidden_size,
                                                                          waypoints=(config.pred_len // self.config.wp_dilation),
                                                                          target_point_size=target_point_size)

                if self.config.use_controller_input_prediction: # <------------ True
                    # + 1 for the target speed token
                    self.checkpoint_query = nn.Parameter(
                        torch.zeros(1, self.config.predict_checkpoint_len + 1, self.config.gru_input_size))

                    self.checkpoint_decoder = GRUWaypointsPredictorInterFuser(input_dim=self.config.gru_input_size,
                                                                              hidden_size=self.config.gru_hidden_size,
                                                                              waypoints=self.config.predict_checkpoint_len,
                                                                              target_point_size=target_point_size)

                self.reset_parameters()

            else:
                if self.config.learn_origin:
                    join_output_features = self.config.gru_hidden_size + 2
                else:
                    join_output_features = self.config.gru_hidden_size
                # waypoints prediction
                self.join = nn.Sequential(
                    nn.Linear(self.backbone.num_features + extra_sensor_channels, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, join_output_features),
                    nn.ReLU(inplace=True),
                )

                if self.config.use_wp_gru:# NOT ENTERING
                    self.wp_decoder = GRUWaypointsPredictorTransFuser(self.config,
                                                                      pred_len=(config.pred_len // self.config.wp_dilation),
                                                                      hidden_size=self.config.gru_hidden_size,
                                                                      target_point_size=target_point_size)

                if self.config.use_controller_input_prediction: # <------------ True
                    self.checkpoint_decoder = GRUWaypointsPredictorTransFuser(self.config,
                                                                              pred_len=self.config.predict_checkpoint_len,
                                                                              hidden_size=self.config.gru_hidden_size,
                                                                              target_point_size=target_point_size)

        if self.config.use_wp_gru or self.config.use_controller_input_prediction: # <------------ False or True
            if self.extra_sensors: # <------------ True
                extra_size = 0
                if self.config.use_velocity: # <------------ True
                    # Lazy version of normalizing the input over the dataset statistics.
                    self.velocity_normalization = nn.BatchNorm1d(1, affine=False)
                    extra_size += 1
                if self.config.use_discrete_command: # <------------ True
                    extra_size += 6
                self.extra_sensor_encoder = nn.Sequential(nn.Linear(extra_size, 128), nn.ReLU(inplace=True),
                                                          nn.Linear(128, extra_sensor_channels), nn.ReLU(inplace=True))

        # pid controllers for waypoints
        self.turn_controller = t_u.PIDController(k_p=config.turn_kp,
                                                 k_i=config.turn_ki,
                                                 k_d=config.turn_kd,
                                                 n=config.turn_n)
        self.speed_controller = t_u.PIDController(k_p=config.speed_kp,
                                                  k_i=config.speed_ki,
                                                  k_d=config.speed_kd,
                                                  n=config.speed_n)

        # PID controller for directly predicted input
        self.turn_controller_direct = t_u.PIDController(k_p=self.config.turn_kp,
                                                        k_i=self.config.turn_ki,
                                                        k_d=self.config.turn_kd,
                                                        n=self.config.turn_n)

        self.speed_controller_direct = t_u.PIDController(k_p=self.config.speed_kp,
                                                         k_i=self.config.speed_ki,
                                                         k_d=self.config.speed_kd,
                                                         n=self.config.speed_n)
        if self.config.use_speed_weights:
            self.speed_weights = torch.tensor(self.config.target_speed_weights)
        else:
            self.speed_weights = torch.ones_like(torch.tensor(self.config.target_speed_weights))

        self.semantic_weights = torch.tensor(self.config.semantic_weights)
        self.bev_semantic_weights = torch.tensor(self.config.bev_semantic_weights)

        if self.config.use_label_smoothing:
            label_smoothing = self.config.label_smoothing_alpha
        else:
            label_smoothing = 0.0

        self.loss_speed = nn.CrossEntropyLoss(weight=self.speed_weights, label_smoothing=label_smoothing)

        self.loss_semantic = nn.CrossEntropyLoss(weight=self.semantic_weights, label_smoothing=label_smoothing)
        self.loss_bev_semantic = nn.CrossEntropyLoss(weight=self.bev_semantic_weights,
                                                     label_smoothing=label_smoothing,
                                                     ignore_index=-1)
        if self.config.multi_wp_output:
            self.selection_loss = nn.BCEWithLogitsLoss()

    def reset_parameters(self):
        if self.config.use_wp_gru:# NOT ENTERING
            nn.init.uniform_(self.wp_query)
        if self.config.use_controller_input_prediction: # <------------ True
            nn.init.uniform_(self.checkpoint_query)
        if self.extra_sensors: # <------------ True
            nn.init.uniform_(self.extra_sensor_pos_embed)
        if self.config.tp_attention: # NOT ENTERING
            nn.init.uniform_(self.tp_pos_embed)

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command):

        # Noce
        # print(f"RGB: {rgb.shape} [{rgb.min()}; {rgb.max()}]")
        # print(f"Lidar BEV: {lidar_bev.shape} [{lidar_bev.min()}; {lidar_bev.max()}]")
        # print(f"Target Point: {target_point.shape} [{target_point.min()}; {target_point.max()}]")
        # print(f"Ego Velocity: {ego_vel.shape} [{ego_vel.min()}; {ego_vel.max()}]")
        # print(f"Command: {command.shape} [{command.min()}; {command.max()}]")
        # ----


        bs = rgb.shape[0]
        # print(self.config.backbone) # transFuser

        if self.config.backbone == 'transFuser': # <--------
            bev_feature_grid, fused_features, image_feature_grid = self.backbone(rgb, lidar_bev)
        elif self.config.backbone == 'aim':
            fused_features, image_feature_grid = self.backbone(rgb)
        elif self.config.backbone == 'bev_encoder':
            bev_feature_grid, fused_features, image_feature_grid = self.backbone(rgb, lidar_bev)
        else:
            raise ValueError('The chosen vision backbone does not exist. '
                             'The options are: transFuser, aim, bev_encoder')

        # AFTER BACKBONE
        """
        bev_feature_grid = torch.Size([1, 64, 64, 64])
        fused_features = torch.Size([1, 1512, 8, 8])
        image_feature_grid = torch.Size([1, 1512, 8, 32])
        """

        pred_wp = None
        pred_target_speed = None
        pred_checkpoint = None
        attention_weights = None
        pred_wp_1 = None
        selected_path = None

        if self.config.use_wp_gru or self.config.use_controller_input_prediction: # <------- False or True
            if self.config.transformer_decoder_join: # <------------------------ True
                fused_features = self.change_channel(fused_features)
                fused_features = fused_features + self.encoder_pos_encoding(fused_features)
                fused_features = torch.flatten(fused_features, start_dim=2)
                if self.config.tp_attention: # NOT ENTERING
                    num_pixel_tokens = fused_features.shape[2]

            # Concatenate extra sensor information
            if self.extra_sensors: # <------------ True
                extra_sensors = []
                if self.config.use_velocity: # <------------ True
                    extra_sensors.append(self.velocity_normalization(ego_vel))
                if self.config.use_discrete_command: # <------------ True
                    extra_sensors.append(command)
                extra_sensors = torch.cat(extra_sensors, axis=1)
                extra_sensors = self.extra_sensor_encoder(extra_sensors)

                if self.config.transformer_decoder_join: # <------------------------ True
                    extra_sensors = extra_sensors + self.extra_sensor_pos_embed.repeat(bs, 1)
                    fused_features = torch.cat((fused_features, extra_sensors.unsqueeze(2)), axis=2)
                else: # NOT ENTERING
                    fused_features = torch.cat((fused_features, extra_sensors), axis=1)

            if self.config.transformer_decoder_join: # <------------------------ True
                fused_features = torch.permute(fused_features, (0, 2, 1))
                if self.config.use_wp_gru: # NOT ENTERING
                    if self.config.multi_wp_output:
                        joined_wp_features = self.join(self.wp_query.repeat(bs, 1, 1), fused_features)
                        num_wp = (self.config.pred_len // self.config.wp_dilation)
                        pred_wp = self.wp_decoder(joined_wp_features[:, :num_wp], target_point)
                        pred_wp_1 = self.wp_decoder_1(joined_wp_features[:, num_wp:2 * num_wp], target_point)
                        selected_path = self.select_wps(joined_wp_features[:, 2 * num_wp])
                    else:
                        joined_wp_features = self.join(self.wp_query.repeat(bs, 1, 1), fused_features)
                        pred_wp = self.wp_decoder(joined_wp_features, target_point)
                if self.config.use_controller_input_prediction:# <------------ True
                    if self.config.tp_attention: # NOT ENTERING
                        tp_token = self.tp_encoder(target_point)
                        tp_token = tp_token + self.tp_pos_embed
                        fused_features = torch.cat((fused_features, tp_token.unsqueeze(1)), axis=1)
                        joined_checkpoint_features, attention = self.join(self.checkpoint_query.repeat(bs, 1, 1), fused_features)
                        gru_attention = attention[:, :self.config.predict_checkpoint_len]
                        # Average attention for the WP tokens
                        gru_attention = torch.mean(gru_attention, dim=1)[0]
                        vision_attention = torch.sum(gru_attention[:num_pixel_tokens])
                        add = 0
                        if self.extra_sensors: # <------------ True
                            add = 1
                            speed_attention = gru_attention[num_pixel_tokens:num_pixel_tokens + add]
                        tp_attention = gru_attention[num_pixel_tokens + add:]
                        attention_weights = [vision_attention.item(), speed_attention.item(), tp_attention.item()]
                    else: # <-------------------
                        # print(f"checkpoint_query = [{self.checkpoint_query.shape}] [{type(self.checkpoint_query)}]")
                        # print(f"checkpoint_query.repeat = [{self.checkpoint_query.repeat(bs, 1, 1).shape}] [{type(self.checkpoint_query.repeat(bs, 1, 1))}]")
                        # print(f"bs = [{bs}]")

                        joined_checkpoint_features = self.join(self.checkpoint_query.repeat(bs, 1, 1), fused_features)

                    gru_features = joined_checkpoint_features[:, :self.config.predict_checkpoint_len]
                    target_speed_features = joined_checkpoint_features[:, self.config.predict_checkpoint_len]
                    pred_checkpoint = self.checkpoint_decoder(gru_features, target_point)
                    # print(f"pred_checkpoint = {pred_checkpoint.shape}")
                    pred_target_speed = self.target_speed_network(target_speed_features)
                    # print(f"pred_target_speed = {pred_target_speed.shape}")

            else: # NOT ENTERING
                joined_features = self.join(fused_features)
                gru_features = joined_features
                target_speed_features = joined_features[:, :self.config.gru_hidden_size]

                if self.config.use_wp_gru:# NOT ENTERING
                    pred_wp = self.wp_decoder(gru_features, target_point)
                if self.config.use_controller_input_prediction:# <------------ True
                    pred_checkpoint = self.checkpoint_decoder(gru_features, target_point)
                    pred_target_speed = self.target_speed_network(target_speed_features)

        # Auxiliary tasks
        pred_semantic = None
        if self.config.use_semantic:
            pred_semantic = self.semantic_decoder(image_feature_grid)

        pred_depth = None
        if self.config.use_depth:
            pred_depth = self.depth_decoder(image_feature_grid) # [1, 1, 256, 1024]
            pred_depth = torch.sigmoid(pred_depth).squeeze(1) # [1, 256, 1024]

        pred_flow = None
        if self.config.use_flow:
            pred_flow = self.flow_decoder(image_feature_grid)

        pred_bev_semantic = None
        if self.config.use_bev_semantic:
            pred_bev_semantic = self.bev_semantic_decoder(bev_feature_grid)
            # Mask invisible pixels. They will be ignored in the loss
            pred_bev_semantic = pred_bev_semantic * self.valid_bev_pixels

        pred_bounding_box = None
        if self.config.detect_boxes:
            pred_bounding_box = self.head(bev_feature_grid) # tuple of 7 elements

        # NOCE

        # SHOWING IMAGES
        """

        my_imag_in_np = rgb[0].cpu().numpy().astype(np.uint8)
        my_imag_in_np = np.moveaxis(my_imag_in_np, 0, -1)
        my_imag_in_np = cv2.cvtColor(my_imag_in_np, cv2.COLOR_RGB2BGR)

        my_lidar_in_np = (lidar_bev[0]*255).cpu().numpy().astype(np.uint8)
        my_lidar_in_np = np.moveaxis(my_lidar_in_np, 0, -1)
        my_lidar_in_np = cv2.cvtColor(my_lidar_in_np, cv2.COLOR_GRAY2RGB)
        my_lidar_in_np[123:133, 120:137, 0] = 255
        my_lidar_in_np[123:133, 120:137, 1] = 0
        my_lidar_in_np[123:133, 120:137, 2] = 0

        my_pred_depth_in_np = (pred_depth*255).cpu().detach().numpy().astype(np.uint8) # (1, 256, 1024)
        my_pred_depth_in_np = np.moveaxis(my_pred_depth_in_np, 0, -1)

        my_pred_bev_semantic = torch.nn.functional.softmax(pred_bev_semantic).cpu() # (1, 11, 256, 256)
        BEV_SEMNATIC_TRASHOLD = 0.9
        bev_semantic_segmentation_colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255),
                                            (255, 0, 255), (125, 0, 255), (255, 125, 0), (0, 125, 255), (0, 255, 125), (255, 0, 125)]
        bev_semantic = torch.zeros((pred_bev_semantic.shape[-2], pred_bev_semantic.shape[-1], 3))
        for i in range(7):
          bev_semantic[my_pred_bev_semantic[0, i] > BEV_SEMNATIC_TRASHOLD] = torch.tensor(bev_semantic_segmentation_colors[i], dtype=torch.float)
        bev_semantic = bev_semantic.numpy()

        my_pred_semantic = torch.nn.functional.softmax(pred_semantic).cpu() # (1, 7, 256, 1024)
        SEMNATIC_TRASHOLD = 0.9
        semantic_segmentation_colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255), (125, 0, 255)]
        semantic = torch.zeros((pred_semantic.shape[-2], pred_semantic.shape[-1], 3))
        for i in range(7):
          semantic[my_pred_semantic[0, i] > SEMNATIC_TRASHOLD] = torch.tensor(semantic_segmentation_colors[i], dtype=torch.float)
        semantic = semantic.numpy()

        output_image = np.ones((256*3+20*3, 1024+20+256+20+256, 3), dtype=my_imag_in_np.dtype)*255
        output_image[0:256, 0:1024] = my_imag_in_np
        output_image[256+20:256+20+256, 0:1024, :] = my_pred_depth_in_np
        output_image[256+20+256+20:256+20+256+20+256, 0:1024] = semantic

        my_lidar_in_np = cv2.resize(my_lidar_in_np, (256*2, 256*2), interpolation = cv2.INTER_AREA)

        # BOUNDING BOXES

        carla_bb = self.convert_features_to_bb_metric(pred_bounding_box)
        for element in carla_bb:
          c_x = element[0]
          c_y = element[1]
          delta_x = element[2]
          delta_y = element[3]
          corners = [[- delta_x, - delta_y], [- delta_x, delta_y],
                     [delta_x, delta_y], [delta_x, - delta_y]]
          cos = np.cos(element[4])
          sin = np.sin(element[4])
          for i, corner in enumerate(corners):
            new_x = int((corner[0]*cos - corner[1]*sin + c_x)*8+256)
            new_y = int((corner[0]*sin + corner[1]*cos + c_y)*8+256)
            corners[i][0] = new_x
            corners[i][1] = new_y
          corners = np.array(corners, np.int32)
          corners = corners.reshape((-1, 1, 2))
          isClosed = True
          color = (255, 0, 0)
          thickness = 2
          my_lidar_in_np = cv2.polylines(my_lidar_in_np, [corners], isClosed, color, thickness)

        # CHECKPOINTS

        check_points_in_px = pred_checkpoint*M_to_PX
        for checkpoint in check_points_in_px[0]:
          cv2.circle(my_lidar_in_np, (int(checkpoint[0]*2)+128*2, int(checkpoint[1]*2)+128*2), 2, (0, 0, 255))
        way_point = (int(target_point[0, 0]*2*M_to_PX)+128*2, int(target_point[0, 1]*2*M_to_PX)+128*2)
        if way_point[0] > 255*2:
          way_point = (255*2, way_point[1])
        elif way_point[0] < 0:
          way_point = (0, way_point[1])
        if way_point[1] > 255*2:
          way_point = (way_point[0], 255*2)
        elif way_point[1] < 0:
          way_point = (way_point[0], 0)
        cv2.circle(my_lidar_in_np, way_point, 5, (0, 255, 0))

        output_image[0:256*2, 1024+20:1024+20+256*2] = np.rot90(my_lidar_in_np)
        output_image[256+20+256+20:256+20+256+20+256, 1024+20+256+20:1024+20+256+20+256] = np.rot90(bev_semantic)

        # SPEED

        prob_of_choosing_a_speed = F.softmax(pred_target_speed[0], dim=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        color = (0, 0, 0)
        thickness = 2
        isClosed = True
        bar_color_1 = (255, 0, 0)
        bar_color_2 = (0, 128, 255)
        bar_thickness = 2
        speeds = [0, 7, 18, 29]
        for i in range(1, 5):
          output_image = cv2.putText(output_image, f" {speeds[i-1]} km/h", (1024+20, 256+20+256+40*i), font, fontScale, color, thickness, cv2.LINE_AA)
          corners = [[1024+140, 256+20+256+40*i-20], [1024+140, 256+20+256+40*i+5], [1024+140+100, 256+20+256+40*i+5], [1024+140+100, 256+20+256+40*i-20]]
          bar = [[1024+140, 256+20+256+40*i-20], [1024+140, 256+20+256+40*i+5],
                 [1024+140+int(100*prob_of_choosing_a_speed[i-1]), 256+20+256+40*i+5],
                 [1024+140+int(100*prob_of_choosing_a_speed[i-1]), 256+20+256+40*i-20]]
          corners = np.array(corners, np.int32)
          bar = np.array(bar, np.int32)
          corners = corners.reshape((-1, 1, 2))
          bar = bar.reshape((-1, 1, 2))
          output_image = cv2.fillPoly(output_image, [bar], bar_color_2)
          output_image = cv2.polylines(output_image, [corners], isClosed, bar_color_1, bar_thickness)



        global OUTPUT_VIDEO

        if pred_depth.shape[0] == 1:
          if SHOW_STUFF:
            cv2.imshow('Output', output_image)
            cv2.waitKey(1)
          else:
            self.frame_skipped += 1
            if self.frame_skipped > self.skip_each:
              self.frame_skipped = 0
              OUTPUT_VIDEO.write(output_image)
        """
        # ----

        return pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic, pred_depth, \
          pred_bounding_box, attention_weights, pred_wp_1, selected_path, pred_flow

# FORWARD FINISH THERE!!!!!!!!!!!!!!!!!!!!

    def compute_loss(self, pred_wp, pred_target_speed, pred_checkpoint, pred_semantic, pred_bev_semantic, pred_depth, pred_flow,
                     pred_bounding_box, pred_wp_1, selected_path, waypoint_label, target_speed_label, checkpoint_label,
                     semantic_label, bev_semantic_label, depth_label, flow_label, center_heatmap_label, wh_label, yaw_class_label,
                     yaw_res_label, offset_label, velocity_label, brake_target_label, pixel_weight_label,
                     avg_factor_label):
        loss = {}

        if self.config.use_controller_input_prediction:# <------------ True
            loss_target_speed = self.loss_speed(pred_target_speed, target_speed_label)
            loss.update({'loss_target_speed': loss_target_speed})

            loss_wp = torch.mean(torch.abs(pred_checkpoint - checkpoint_label))
            loss.update({'loss_checkpoint': loss_wp})

        if self.config.use_semantic:
            loss_semantic = self.loss_semantic(pred_semantic, semantic_label)
            loss.update({'loss_semantic': loss_semantic})

        if self.config.use_bev_semantic:
            visible_bev_semantic_label = self.valid_bev_pixels.squeeze(1).int() * bev_semantic_label
            # Set 0 class to ignore index -1
            visible_bev_semantic_label = (self.valid_bev_pixels.squeeze(1).int() - 1) + visible_bev_semantic_label
            loss_bev_semantic = self.loss_bev_semantic(pred_bev_semantic, visible_bev_semantic_label)
            loss.update({'loss_bev_semantic': loss_bev_semantic})

        if self.config.use_depth:
            loss_depth = F.l1_loss(pred_depth, depth_label)
            loss.update({'loss_depth': loss_depth})

        if self.config.use_flow:
            loss_flow = nn.MSELoss(reduction="mean")(pred_flow, flow_label) # maybe reduction should be sum?
            # print(f"pred_flow : [{torch.min(pred_flow)}; {torch.max(pred_flow)}]")
            # print(f"flow_label : [{torch.min(flow_label)}; {torch.max(flow_label)}]")
            loss.update({'loss_flow': loss_flow})

        return loss

    def convert_features_to_bb_metric(self, bb_predictions):
        bboxes = self.head.get_bboxes(bb_predictions[0], bb_predictions[1], bb_predictions[2], bb_predictions[3],
                                      bb_predictions[4], bb_predictions[5], bb_predictions[6])[0]

        # filter bbox based on the confidence of the prediction
        bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]

        carla_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = t_u.bb_image_to_vehicle_system(bbox, self.config.pixels_per_meter, self.config.min_x, self.config.min_y)
            carla_bboxes.append(bbox)

        return carla_bboxes

    def control_pid_direct(self, pred_target_speed, pred_angle, speed):
        """
        PID controller used for direct predictions
        """
        if self.make_histogram:
            self.speed_histogram.append(pred_target_speed * 3.6)

        # Convert to numpy
        speed = speed[0].data.cpu().numpy()

        # Target speed of 0 means brake
        brake = pred_target_speed < 0.01

        # We can't steer while the car is stopped
        if speed < 0.01:
            pred_angle = 0.0

        steer = self.turn_controller_direct.step(pred_angle)

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(float(steer), 3)

        if not brake:
            if (speed / pred_target_speed) > self.config.brake_ratio:
                brake = True

        if brake:
            target_speed = 0.0
        else:
            target_speed = pred_target_speed

        delta = np.clip(target_speed - speed, 0.0, self.config.clip_delta)

        throttle = self.speed_controller_direct.step(delta)

        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)

        if brake:
            throttle = 0.0

        return steer, throttle, brake

    def control_pid(self, waypoints, velocity):
        """
        Predicts vehicle control with a PID controller.
        Used for waypoint predictions
        """
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()

        speed = velocity[0].data.cpu().numpy()

        # m / s required to drive between waypoint 0.5 and 1.0 second in the future
        one_second = int(self.config.carla_fps // (self.config.wp_dilation * self.config.data_save_freq))
        half_second = one_second // 2
        desired_speed = np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0

        if self.make_histogram:
            self.speed_histogram.append(desired_speed * 3.6)

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0

        # To replicate the slow TransFuser behaviour we have a different distance
        # inside and outside of intersections (detected by desired_speed)
        if desired_speed < self.config.aim_distance_threshold:
            aim_distance = self.config.aim_distance_slow
        else:
            aim_distance = self.config.aim_distance_fast

        # We follow the waypoint that is at least a certain distance away
        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            # When we don't move we don't want the angle error to accumulate in the integral
            angle = 0.0
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

        return steer, throttle, brake

    def create_optimizer_groups(self, weight_decay):
        """
            This long function is unfortunately doing something very simple and is
            being very defensive:
            We are separating out all parameters of the model into two buckets:
            those that will experience
            weight decay for regularization and those that won't
            (biases, and layernorm/embedding weights).
            We are then returning the optimizer groups.
            """

        # separate out all parameters to those that will and won't experience
        # regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and 'conv.' in pn:  # Add decay for convolutional layers.
                    decay.add(fpn)
                elif pn.endswith('weight') and '.bn' in pn:  # No decay for batch norms.
                    no_decay.add(fpn)
                elif pn.endswith('weight') and '.ln' in pn:  # No decay for layer norms.
                    no_decay.add(fpn)
                elif pn.endswith('weight') and 'downsample.0.weight' in pn:  # Conv2D layer with stride 2
                    decay.add(fpn)
                elif pn.endswith('weight') and 'downsample.1.weight' in pn:  # BN layer
                    no_decay.add(fpn)
                elif pn.endswith('weight') and '.attn' in pn:  # Attention linear layers
                    decay.add(fpn)
                elif pn.endswith('weight') and 'channel_to_' in pn:  # Convolutional layers for channel change
                    decay.add(fpn)
                elif pn.endswith('weight') and '.mlp' in pn:  # MLP linear layers
                    decay.add(fpn)
                elif pn.endswith('weight') and 'target_speed_network' in pn:  # MLP linear layers
                    decay.add(fpn)
                elif pn.endswith('weight') and 'join.' in pn and not '.norm' in pn:  # MLP layers
                    decay.add(fpn)
                elif pn.endswith('weight') and 'join.' in pn and '.norm' in pn:  # Norm layers
                    no_decay.add(fpn)
                elif pn.endswith('_ih') or pn.endswith('_hh'):
                    # all recurrent weights will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('_emb') or '_token' in pn:
                    no_decay.add(fpn)
                elif pn.endswith('_embed'):
                    no_decay.add(fpn)
                elif 'bias_ih_l0' in pn or 'bias_hh_l0' in pn:
                    no_decay.add(fpn)
                elif 'weight_ih_l0' in pn or 'weight_hh_l0' in pn:
                    decay.add(fpn)
                elif '_query' in pn or 'weight_hh_l0' in pn:
                    no_decay.add(fpn)
                elif 'valid_bev_pixels' in pn:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (len(inter_params) == 0), f'parameters {str(inter_params)} made it into both decay/no_decay sets!'
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f'parameters {str(param_dict.keys() - union_params)} were not ' \
           f'separated into either decay/no_decay set!'

        # create the pytorch optimizer object
        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(list(decay))],
                'weight_decay': weight_decay,
            },
            {
                'params': [param_dict[pn] for pn in sorted(list(no_decay))],
                'weight_decay': 0.0,
            },
        ]
        return optim_groups

    def init_visualization(self):
        # Privileged map access for visualization
        pass

    @torch.no_grad()
    def visualize_model(  # pylint: disable=locally-disabled, unused-argument
        self,
        save_path,
        step,
        rgb,
        lidar_bev,
        target_point,
        pred_wp,
        pred_semantic=None,
        pred_bev_semantic=None,
        pred_depth=None,
        pred_checkpoint=None,
        pred_speed=None,
        pred_bb=None,
        gt_wp=None,
        gt_bbs=None,
        gt_speed=None,
        gt_bev_semantic=None,
        wp_selected=None):
        # 0 Car, 1 Pedestrian, 2 Red light, 3 Stop sign
        color_classes = [np.array([255, 165, 0]), np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([250, 160, 160])]

        size_width = int((self.config.max_y - self.config.min_y) * self.config.pixels_per_meter)
        size_height = int((self.config.max_x - self.config.min_x) * self.config.pixels_per_meter)

        scale_factor = 4
        origin = ((size_width * scale_factor) // 2, (size_height * scale_factor) // 2)
        loc_pixels_per_meter = self.config.pixels_per_meter * scale_factor

        ## add rgb image and lidar
        if self.config.use_ground_plane:
            images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)
        else:
            images_lidar = np.concatenate(list(lidar_bev.detach().cpu().numpy()[0][:1]), axis=1)

        images_lidar = 255 - (images_lidar * 255).astype(np.uint8)
        images_lidar = np.stack([images_lidar, images_lidar, images_lidar], axis=-1)

        images_lidar = cv2.resize(images_lidar,
                                  dsize=(images_lidar.shape[1] * scale_factor, images_lidar.shape[0] * scale_factor),
                                  interpolation=cv2.INTER_NEAREST)
        # # Render road over image
        # road = self.ss_bev_manager.get_road()
        # # Alpha blending the road over the LiDAR
        # images_lidar = road[:, :, 3:4] * road[:, :, :3] + (1 - road[:, :, 3:4]) * images_lidar

        if pred_bev_semantic is not None:
            bev_semantic_indices = np.argmax(pred_bev_semantic[0].detach().cpu().numpy(), axis=0)
            converter = np.array(self.config.bev_classes_list)
            converter[1][0:3] = 40
            bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
            alpha = np.ones_like(bev_semantic_indices) * 0.33
            alpha = alpha.astype(np.float)
            alpha[bev_semantic_indices == 0] = 0.0
            alpha[bev_semantic_indices == 1] = 0.1

            alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            alpha = np.expand_dims(alpha, 2)
            bev_semantic_image = cv2.resize(bev_semantic_image,
                                            dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                            interpolation=cv2.INTER_NEAREST)

            images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

        if gt_bev_semantic is not None:
            bev_semantic_indices = gt_bev_semantic[0].detach().cpu().numpy()
            converter = np.array(self.config.bev_classes_list)
            converter[1][0:3] = 40
            bev_semantic_image = converter[bev_semantic_indices, ...].astype('uint8')
            alpha = np.ones_like(bev_semantic_indices) * 0.33
            alpha = alpha.astype(np.float)
            alpha[bev_semantic_indices == 0] = 0.0
            alpha[bev_semantic_indices == 1] = 0.1

            alpha = cv2.resize(alpha, dsize=(alpha.shape[1] * 4, alpha.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            alpha = np.expand_dims(alpha, 2)
            bev_semantic_image = cv2.resize(bev_semantic_image,
                                            dsize=(bev_semantic_image.shape[1] * 4, bev_semantic_image.shape[0] * 4),
                                            interpolation=cv2.INTER_NEAREST)
            images_lidar = bev_semantic_image * alpha + (1 - alpha) * images_lidar

            images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)

        # Draw wps
        # Red ground truth
        if gt_wp is not None:
            gt_wp_color = (255, 255, 0)
            for wp in gt_wp.detach().cpu().numpy()[0]:
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)

        # Green predicted checkpoint
        if pred_checkpoint is not None:
            for wp in pred_checkpoint.detach().cpu().numpy()[0]:
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                           radius=8,
                           lineType=cv2.LINE_AA,
                           color=(0, 128, 255),
                           thickness=-1)

        # Blue predicted wp
        if pred_wp is not None:
            pred_wps = pred_wp.detach().cpu().numpy()[0]
            num_wp = len(pred_wps)
            for idx, wp in enumerate(pred_wps):
                color_weight = 0.5 + 0.5 * float(idx) / num_wp
                wp_x = wp[0] * loc_pixels_per_meter + origin[0]
                wp_y = wp[1] * loc_pixels_per_meter + origin[1]
                cv2.circle(images_lidar, (int(wp_x), int(wp_y)),
                           radius=8,
                           lineType=cv2.LINE_AA,
                           color=(0, 0, int(color_weight * 255)),
                           thickness=-1)

        # Draw target points
        if self.config.use_tp:
            x_tp = target_point[0][0] * loc_pixels_per_meter + origin[0]
            y_tp = target_point[0][1] * loc_pixels_per_meter + origin[1]
            cv2.circle(images_lidar, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

        # Visualize Ego vehicle
        sample_box = np.array([
            int(images_lidar.shape[0] / 2),
            int(images_lidar.shape[1] / 2), self.config.ego_extent_x * loc_pixels_per_meter,
            self.config.ego_extent_y * loc_pixels_per_meter,
            np.deg2rad(90.0), 0.0
        ])
        images_lidar = t_u.draw_box(images_lidar, sample_box, color=(0, 200, 0), pixel_per_meter=16, thickness=4)

        if pred_bb is not None:
            for box in pred_bb:
                inv_brake = 1.0 - box[6]
                color_box = deepcopy(color_classes[int(box[7])])
                color_box[1] = color_box[1] * inv_brake
                box = t_u.bb_vehicle_to_image_system(box, loc_pixels_per_meter, self.config.min_x, self.config.min_y)
                images_lidar = t_u.draw_box(images_lidar, box, color=color_box, pixel_per_meter=loc_pixels_per_meter)

        if gt_bbs is not None:
            gt_bbs = gt_bbs.detach().cpu().numpy()[0]
            real_boxes = gt_bbs.sum(axis=-1) != 0.
            gt_bbs = gt_bbs[real_boxes]
            for box in gt_bbs:
                box[:4] = box[:4] * scale_factor
                images_lidar = t_u.draw_box(images_lidar, box, color=(0, 255, 255), pixel_per_meter=loc_pixels_per_meter)

        images_lidar = np.rot90(images_lidar, k=1)

        rgb_image = rgb[0].permute(1, 2, 0).detach().cpu().numpy()

        if wp_selected is not None:
            colors_name = ['blue', 'yellow']
            colors_idx = [(0, 0, 255), (255, 255, 0)]
            images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
            cv2.putText(images_lidar, 'Selected: ', (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(images_lidar, f'{colors_name[wp_selected]}', (850, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        colors_idx[wp_selected], 2, cv2.LINE_AA)

        if pred_speed is not None:
            pred_speed = pred_speed.detach().cpu().numpy()[0]
            images_lidar = np.ascontiguousarray(images_lidar, dtype=np.uint8)
            t_u.draw_probability_boxes(images_lidar, pred_speed, self.config.target_speeds)

        all_images = np.concatenate((rgb_image, images_lidar), axis=0)
        all_images = Image.fromarray(all_images.astype(np.uint8))

        store_path = str(str(save_path) + (f'/{step:04}.png'))
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        all_images.save(store_path)


class GRUWaypointsPredictorInterFuser(nn.Module):
    """
    A version of the waypoint GRU used in InterFuser.
    It embeds the target point and inputs it as hidden dimension instead of input.
    The scene state is described by waypoints x input_dim features which are added as input instead of initializing the
    hidden state.
    """

    def __init__(self, input_dim, waypoints, hidden_size, target_point_size):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        if target_point_size > 0:
            self.encoder = nn.Linear(target_point_size, hidden_size)
        self.target_point_size = target_point_size # 2
        self.hidden_size = hidden_size # 64
        self.decoder = nn.Linear(hidden_size, 2)
        self.waypoints = waypoints # 10

    def forward(self, x, target_point):
        # x.shape = [BS, 10, 256]
        # target_point.shape = [BS, 2]
        bs = x.shape[0]
        if self.target_point_size > 0:
            z = self.encoder(target_point).unsqueeze(0) # [1, BS, 64]
        else:
            z = torch.zeros((1, bs, self.hidden_size), device=x.device) # [1, BS, 64]
        output, _ = self.gru(x, z) # [BS, 10, 64]
        output = output.reshape(bs * self.waypoints, -1) # [BS*10, 64]
        output = self.decoder(output).reshape(bs, self.waypoints, 2) # [BS*10, 2] -> [BS, 10, 2]
        output = torch.cumsum(output, 1) # [BS, 10, 2]
        return output


class GRUWaypointsPredictorTransFuser(nn.Module):
    """
    The waypoint GRU used in TransFuser.
    It enters the target point as input.
    The hidden state is initialized with the scene features.
    The input is autoregressive and starts either at 0 or learned.
    """

    def __init__(self, config, pred_len, hidden_size, target_point_size):
        super().__init__()
        self.wp_decoder = nn.GRUCell(input_size=2 + target_point_size, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, 2)
        self.config = config
        self.prediction_len = pred_len

    def forward(self, z, target_point):
        output_wp = []

        # initial input variable to GRU
        if self.config.learn_origin:
            x = z[:, self.config.gru_hidden_size:(self.config.gru_hidden_size + 2)]  # Origin of the waypoints
            z = z[:, :self.config.gru_hidden_size]
        else:
            x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()

        # autoregressive generation of output waypoints
        for _ in range(self.prediction_len):
            if self.config.use_tp:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x

            z = self.wp_decoder(x_in, z)
            dx = self.output(z)

            x = dx + x

            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp


class PositionEmbeddingSine(nn.Module):
    """
    Taken from InterFuser
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize: # TRUE
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
