"""
Center Net Head implementation adapted from MM Detection
"""
import pathlib
import sys
sys.path.append(str(pathlib.Path(
    __file__).parent.resolve().parent.resolve().parent.resolve().parent.resolve()))
import nutfuser.neural_networks.tfpp.transfuser_utils as t_u
import numpy as np
import torch
from torch import nn


class LidarCenterNetHead(nn.Module):
    """
    Objects as Points Head. CenterHead use center_point to indicate object's position.
    Paper link <https://arxiv.org/abs/1904.07850>
    Args:
        config: Gobal TransFuser config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.heatmap_head = self._build_head(config.bb_input_channel, config.num_bb_classes)
        self.wh_head = self._build_head(config.bb_input_channel, 2)
        self.offset_head = self._build_head(config.bb_input_channel, 2)
        self.yaw_class_head = self._build_head(config.bb_input_channel, config.num_dir_bins)
        self.yaw_res_head = self._build_head(config.bb_input_channel, 1)

        # We use none reduction because we weight each pixel according to the number of bounding boxes.
        self.loss_center_heatmap = t_u.gaussian_focal_loss
        self.loss_wh = nn.L1Loss(reduction='none')
        self.loss_offset = nn.L1Loss(reduction='none')
        self.loss_dir_class = nn.CrossEntropyLoss(reduction='none')
        self.loss_dir_res = nn.SmoothL1Loss(reduction='none')

    @staticmethod
    def _build_head(in_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                              nn.Conv2d(in_channel, out_channel, kernel_size=1))
        return layer

    def forward(self, feat):
        """
        Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()  # [1, 4, 64, 64]
        wh_pred = self.wh_head(feat)                             # [1, 2, 64, 64]
        offset_pred = self.offset_head(feat)                     # [1, 2, 64, 64]
        yaw_class_pred = self.yaw_class_head(feat)               # [1, 12, 64, 64]
        yaw_res_pred = self.yaw_res_head(feat)                   # [1, 1, 64, 64]

        return center_heatmap_pred, wh_pred, offset_pred, yaw_class_pred, yaw_res_pred

    def loss(self, center_heatmap_pred, wh_pred, offset_pred, yaw_class_pred, yaw_res_pred,
                   center_heatmap_target, wh_target, yaw_class_target, yaw_res_target, offset_target,
                   pixel_weight, avg_factor):
        """
        Compute losses of the head.

        Args:
            center_heatmap_preds (Tensor): center predict heatmaps for all levels with shape (B, num_classes, H, W).
            wh_preds (Tensor): wh predicts for all levels with shape (B, 2, H, W).
            offset_preds (Tensor): offset predicts for all levels with shape (B, 2, H, W).

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        # The number of valid bounding boxes can vary.
        # The avg factor represents the amount of valid bounding boxes in the batch.
        # We don't want the empty bounding boxes to have an impact therefore we use reduction sum and divide by the
        # actual number of bounding boxes instead of using standard mean reduction.
        # The weight sets all pixels without a bounding box to 0.
        # Add small epsilon to have numerical stability in the case where there are no boxes in the batch.
        avg_factor = avg_factor.sum()
        avg_factor = avg_factor + torch.finfo(torch.float32).eps
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target,
                                                       reduction='sum') / avg_factor
        # The avg factor is multiplied by the number of channels to yield a proper mean.
        # For the other predictions this value is 1, so it is omitted.
        loss_wh = (self.loss_wh(wh_pred, wh_target) * pixel_weight).sum() / (avg_factor * wh_pred.shape[1])
        loss_offset = ((self.loss_offset(offset_pred, offset_target) * pixel_weight).sum() /
                       (avg_factor * wh_pred.shape[1]))
        loss_yaw_class = (self.loss_dir_class(yaw_class_pred, yaw_class_target) * pixel_weight[:, 0]).sum() / avg_factor
        loss_yaw_res = (self.loss_dir_res(yaw_res_pred, yaw_res_target) * pixel_weight[:, 0:1]).sum() / avg_factor

        losses = dict(loss_center_heatmap=loss_center_heatmap,
                      loss_wh=loss_wh,
                      loss_offset=loss_offset,
                      loss_yaw_class=loss_yaw_class,
                      loss_yaw_res=loss_yaw_res)

        return losses

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """
        Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.config.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

    def angle2class(self, angle):
        """
        Convert continuous angle to a discrete class and a small regression number from class center angle to current angle.
        Args:
            angle (float): Angle is from 0-2pi (or -pi~pi),
              class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).
        Returns:
            tuple: Encoded discrete class and residual.
            """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.config.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
        return int(angle_cls), angle_res
