from termcolor import colored
import numpy as np
import math
import cv2
import torch

import json

import xml.etree.ElementTree as ET

from tqdm import tqdm
from datetime import datetime
from pynvml import *
from types import ModuleType

from nutfuser.raft_flow_colormap import flow_to_image
import nutfuser.config as config
from nutfuser.neural_networks.tfpp_config import GlobalConfig
from nutfuser.neural_networks.model import LidarCenterNet


def color_error_string(string):
    return colored(string, "red", attrs=["bold"])  # , "blink"


def color_info_string(string):
    return colored(string, "yellow", attrs=["bold"])


def color_info_success(string):
    return colored(string, "green", attrs=["bold"])


def get_a_title(string, color):
    line = "#" * (len(string) + 2)
    final_string = line + "\n#" + string + "#\n" + line
    return colored(final_string, color, attrs=["bold"])


def convert_gps_to_carla(gps_array):
    position_carla_coord = gps_array * np.array([111324.60662786, 111319.490945, 1])
    position_carla_coord = np.concatenate([np.expand_dims(position_carla_coord[:, 1], 1),
                                           np.expand_dims(-position_carla_coord[:, 0], 1),
                                           np.expand_dims(position_carla_coord[:, 2], 1)],
                                          1)
    return position_carla_coord


def lat_lon_to_normalize_carla_cords(gps_array, origin=None, den_x=None, den_y=None, min_x=None, min_y=None):
    points = convert_gps_to_carla(gps_array)
    if origin is None:
        origin = points[0].copy()
    points -= origin
    if den_x is None or den_y is None or min_x is None or min_y is None:
        max_x = np.max(points[:, 0])
        min_x = np.min(points[:, 0])
        max_y = np.max(points[:, 1])
        min_y = np.min(points[:, 1])
        den_x = max_x - min_x
        den_y = max_y - min_y
    denominator = max(den_x, den_y)
    if denominator == 0:
        points[:, 0] = 0
        points[:, 1] = 0
        return points, origin, den_x, den_y, min_x, min_y
    if den_x > den_y:
        points[:, 0] = (points[:, 0] - min_x) / denominator
        points[:, 1] = (points[:, 1] - min_y + (den_x - den_y) / 2) / denominator
    else:
        points[:, 0] = (points[:, 0] - min_x + (den_y - den_x) / 2) / denominator
        points[:, 1] = (points[:, 1] - min_y) / denominator
    return points, origin, den_x, den_y, min_x, min_y


def calculate_point_each_meter(points, den_x, den_y):
    denominator = max(den_x, den_y)
    points_in_m = points * denominator
    points_with_equal_distance = []
    subframe_position = []
    tmp_total_distance = 0
    for i in range(len(points_in_m) - 1):
        distance = math.sqrt(
            (points_in_m[i, 0] - points_in_m[i + 1, 0]) ** 2 + (points_in_m[i, 1] - points_in_m[i + 1, 1]) ** 2)
        tmp_total_distance += distance
        if tmp_total_distance > 1.0:
            points_with_equal_distance.append(points[i])
            subframe_position.append(i)
            tmp_total_distance = 0
    return np.array(points_with_equal_distance), subframe_position


def optical_flow_to_human(flow):
    flow = (flow - 2 ** 15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]

    output = flow_to_image(flow, clip_flow=60)
    output *= 255
    return output


def optical_flow_to_human_with_path(optical_flow_path):
    flow = cv2.imread(optical_flow_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow.astype(np.float32)
    flow = flow[:, :, :2]

    return optical_flow_to_human(flow)


def optical_flow_to_human_slow(optical_flow_path):
    """
    Not used anymore! Keeping just for history!
    """
    flow = cv2.imread(optical_flow_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow.astype(np.float32)
    flow = flow[:, :, :2]
    flow = (flow - 2 ** 15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]
    flow[:, :, 0] /= H * 0.5
    flow[:, :, 1] /= W * 0.5
    output = np.zeros((H, W, 3), dtype=np.uint8)
    rad2ang = 180. / math.pi
    for i in range(H):
        for j in range(W):
            vx = flow[i, j, 0]
            vy = flow[i, j, 1]
            angle = 180. + math.atan2(vy, vx) * rad2ang
            if angle < 0:
                angle = 360. + angle
                pass
            angle = math.fmod(angle, 360.)
            norm = math.sqrt(vx * vx + vy * vy)
            shift = 0.999
            a = 1 / math.log(0.1 + shift)
            raw_intensity = a * math.log(norm + shift)
            if raw_intensity < 0.:
                intensity = 0.
            elif raw_intensity > 1.:
                intensity = 1.
            else:
                intensity = raw_intensity
            S = 1.
            V = intensity
            H_60 = angle * 1. / 60.
            C = V * S
            X = C * (1. - abs(math.fmod(H_60, 2.) - 1.))
            m = V - C
            r = 0.
            g = 0.
            b = 0.
            angle_case = int(H_60)
            if angle_case == 0:
                r = C
                g = X
                b = 0
            elif angle_case == 1:
                r = X
                g = C
                b = 0
            elif angle_case == 2:
                r = 0
                g = C
                b = X
            elif angle_case == 3:
                r = 0
                g = X
                b = C
            elif angle_case == 4:
                r = X
                g = 0
                b = C
            elif angle_case == 5:
                r = C
                g = 0
                b = X
            else:
                r = 1
                g = 1
                b = 1
            R = int((r + m) * 255)
            G = int((g + m) * 255)
            B = int((b + m) * 255)
            output[i, j, 0] = R
            output[i, j, 1] = G
            output[i, j, 2] = B
    return output


class NutException(Exception):
    """
    Exception Raised during Carla's StartUp!
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def check_dataset_folder(dataset_path):
    all_files = os.listdir(dataset_path)
    camera_indexes = [int(el[len("rgb_A_"):]) for el in all_files if "rgb_A_" in el]

    folder_that_should_be_there = [(f"rgb_A_{i}", ".jpg") for i in camera_indexes] + \
                                  [(f"rgb_B_{i}", ".jpg") for i in camera_indexes] + \
                                  [(f"depth_{i}", ".png") for i in camera_indexes] + \
                                  [(f"optical_flow_{i}", ".png") for i in camera_indexes] + \
                                  [(f"semantic_{i}", ".png") for i in camera_indexes] + \
                                  [("bev_semantic", ".png"), ("bev_lidar", ".png")]
    max_index = None
    for folder, extention in folder_that_should_be_there:
        if folder not in all_files:
            raise Exception(color_error_string(f"Cannot find out {folder} in '{dataset_path}'"))
        all_frames = os.listdir(os.path.join(dataset_path, folder))
        try:
            all_index = [int(frame[:-len(extention)]) for frame in all_frames]
            if max_index is None:
                max_index = max(all_index)
            for i in range(0, max_index):
                if i not in all_index:
                    raise Exception(color_error_string(
                        f"Missing frame {i} in '{os.path.join(dataset_path, folder)}'\n[{all_frames}]"))
        except:
            raise Exception(color_error_string(
                f"Some strange frame name inside '{os.path.join(dataset_path, folder)}'\n[{all_frames}]"))
    return camera_indexes, max_index


import time


def create_validation_video(folders_path):
    print(color_info_string(f"Creating Validation Video in {folders_path}..."))
    for folder in os.listdir(folders_path):
        full_folder_path = os.path.join(folders_path, folder)
        if not os.path.isdir(full_folder_path):
            continue
        all_frames = os.listdir(full_folder_path)
        int_all_frames = [int(element[:-4]) for element in all_frames]
        int_all_frames.sort()
        example_of_frame = cv2.imread(os.path.join(full_folder_path, all_frames[0]))
        exctention = all_frames[0][-4:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(folders_path, f"{folder}.mp4"), fourcc, 15,
                                (example_of_frame.shape[1], example_of_frame.shape[0]))
        for frame in tqdm(int_all_frames, desc=folder):
            img = cv2.imread(os.path.join(full_folder_path, f"{frame}{exctention}"))
            video.write(img)
        video.release()
    print(color_info_success(f"Created Validation Video in {folders_path}!"))


def create_compariso_validation_video(folders_path_A, folders_path_B, where_to_save):
    print(color_info_string(f"Creating Comparison Video in {where_to_save}..."))
    folders_in_A = os.listdir(folders_path_A)
    folders_in_B = os.listdir(folders_path_B)
    folders = [element for element in folders_in_A if element in folders_in_B]
    for folder in folders:
        full_folder_path_A = os.path.join(folders_path_A, folder)
        full_folder_path_B = os.path.join(folders_path_B, folder)
        if not os.path.isdir(full_folder_path_A) or not os.path.isdir(full_folder_path_B):
            continue
        all_frames_A = os.listdir(full_folder_path_A)
        all_frames_B = os.listdir(full_folder_path_B)
        int_all_frames_A = [int(element[:-4]) for element in all_frames_A]
        int_all_frames_B = [int(element[:-4]) for element in all_frames_B]
        if int_all_frames_A != int_all_frames_B:
            print(color_error_string("Frames in 'full_folder_path_A' and 'full_folder_path_B' do not match!"))
            exit()
        int_all_frames_A.sort()
        int_all_frames = int_all_frames_A
        del int_all_frames_B
        example_of_frame = cv2.imread(os.path.join(full_folder_path_A, all_frames_A[0]))
        exctention = all_frames_A[0][-4:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(where_to_save, f"comparison_{folder}.mp4"), fourcc, 15,
                                (example_of_frame.shape[1], int((example_of_frame.shape[0] / 2) * 3)))
        for frame in tqdm(int_all_frames, desc=f"comparison_{folder}"):
            img_A = cv2.imread(os.path.join(full_folder_path_A, f"{frame}{exctention}"))
            img_B = cv2.imread(os.path.join(full_folder_path_B, f"{frame}{exctention}"))
            img = np.zeros(shape=(int((img_A.shape[0] / 2) * 3), img_A.shape[1], img_A.shape[2]), dtype=img_A.dtype)
            img[:img_A.shape[0], :] = img_A
            img[img_A.shape[0] // 2:, :] = img_B
            video.write(img)
        video.release()
    print(color_info_success(f"Created Comparison Video in {where_to_save}!"))


def print_nvidia_gpu_status_on_log_file(log_file, delay_in_seconds):
    with open(log_file, 'w') as log:
        nvmlInit()
        log.write(f"Driver Version: {nvmlSystemGetDriverVersion()}\n")
        deviceCount = nvmlDeviceGetCount()
        log.write(f"I found out {deviceCount} GPUs:\n")
        gpus = []
        for i in range(deviceCount):
            gpus.append(nvmlDeviceGetHandleByIndex(i))
            log.write(f"\tDevice {i} : {nvmlDeviceGetName(gpus[-1])}\n")
        used_total = {i: 0 for i in range(len(gpus))}
        total_total = {i: 0 for i in range(len(gpus))}
        free_total = {i: 0 for i in range(len(gpus))}
        while True:
            for i in range(100):
                for i, gpu in enumerate(gpus):
                    info = nvmlDeviceGetMemoryInfo(gpu)
                    used_total[i] += info.used / 10 ** 9
                    total_total[i] += info.total / 10 ** 9
                    free_total[i] += info.free / 10 ** 9
                time.sleep(delay_in_seconds / 100)
            for i, _ in enumerate(gpus):
                used_total[i] /= 100
                total_total[i] /= 100
                free_total[i] /= 100
            now = datetime.now()
            current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
            log.write(f"{current_time}:\n")
            for i, _ in enumerate(gpus):
                log.write(f"\t{used_total[i]:.2f} GB / {total_total[i]:.2f} GB [free: {free_total[i]:.2f} GB]\n")
                log.flush()


def lidar_to_histogram_features_tfpp_original(lidar):
    MAX_HIST_POINTS = 5

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(-32, 32, 256 + 1)
        ybins = np.linspace(-32, 32, 256 + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > 5] = 5
        overhead_splat = hist / 5
        # The transpose here is an efficient axis swap.
        # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
        # (x height channel, y width channel)
        return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < -2.5 + 100]
    lidar = lidar[lidar[..., 2] > -2.5 + 0.2]
    features = splat_points(lidar)
    features = np.stack([features], axis=-1)
    features = np.transpose(features, (2, 0, 1))
    features *= 255
    features = features.astype(np.uint8)
    return features


def process_array_and_tensor(an_array_or_tensor, denormalize=False, data_dims=2, channels=1, dtype=None, argmax=False,
                             softmax=False):
    if channels == 1:
        expected_dims = data_dims
    else:
        expected_dims = data_dims + 1

    actual_dims = len(an_array_or_tensor.shape)
    if actual_dims < expected_dims:
        raise NutException(f"The expected dimension [{expected_dims}] is bigger than the actual one {actual_dims}!")

    # Denormalization
    if denormalize:
        maximum = an_array_or_tensor.max()
        if maximum <= 1.:
            an_array_or_tensor *= 255
        else:
            raise NutException(
                f"Asked to denormalize but seems that the tensor/array is not normalized! [max = {maximum:.3f} > 1.0]")
    # Check if Batch Size is Present
    if an_array_or_tensor.shape[0] == 1 and actual_dims > 1:
        an_array_or_tensor = an_array_or_tensor[0]

    # Argmax if requested
    if argmax:
        if isinstance(an_array_or_tensor, torch.Tensor):
            an_array_or_tensor = torch.argmax(an_array_or_tensor, dim=0)
        else:
            raise NutException(f"Asked to apply argmax but it's not a torch.Tensor! [{type(an_array_or_tensor)}]")

    actual_dims = len(an_array_or_tensor.shape)
    # Check correct ammount of channels
    if channels == 1 and actual_dims > expected_dims:
        an_array_or_tensor = an_array_or_tensor[:, :, 0]
    elif channels != 1:
        if an_array_or_tensor.shape[-1] != channels:
            raise NutException(
                f"Wrong number of channels! Expected {channels} but found {an_array_or_tensor.shape[-1]}!")

    actual_dims = len(an_array_or_tensor.shape)
    if actual_dims < expected_dims:
        raise NutException(
            f"The expected dimension [{expected_dims}] is bigger than the actual one {actual_dims} after some change!")

    if isinstance(an_array_or_tensor, torch.Tensor):
        if softmax:
            an_array_or_tensor = an_array_or_tensor.contiguous().detach().cpu()
            an_array_or_tensor = torch.nn.functional.softmax(an_array_or_tensor, dim=0)
            return (an_array_or_tensor.numpy()).astype(dtype)
        if dtype is not None:
            return (an_array_or_tensor.contiguous().detach().cpu().numpy()).astype(dtype)
        else:
            return an_array_or_tensor.contiguous().detach().cpu().numpy()


def create_depth_comparison(predicted_depth, label_depth=None):
    predicted_depth = process_array_and_tensor(predicted_depth, denormalize=True, data_dims=2, channels=1,
                                               dtype=np.uint8, argmax=False)
    if label_depth is not None:
        label_depth = process_array_and_tensor(label_depth, denormalize=False, data_dims=2, channels=1, dtype=np.uint8,
                                               argmax=False)

        depth_comparison = np.zeros((predicted_depth.shape[0] * 2, predicted_depth.shape[1]), dtype=np.uint8)
        depth_comparison[0:predicted_depth.shape[0], :] = predicted_depth
        depth_comparison[label_depth.shape[0]:, :] = label_depth
    else:
        depth_comparison = np.zeros((predicted_depth.shape[0], predicted_depth.shape[1]), dtype=np.uint8)
        depth_comparison[:, :] = predicted_depth

    return depth_comparison


def color_a_semantic_image(a_semantic_array):
    output = np.zeros(shape=(a_semantic_array.shape[0], a_semantic_array.shape[1], 3), dtype=np.uint8)
    for key in config.NUTFUSER_SEMANTIC_COLOR:
        output[a_semantic_array == key] = config.NUTFUSER_SEMANTIC_COLOR[key]
    return output


def create_semantic_comparison(predicted_semantic, label_semantic=None, concatenate_vertically=True):
    predicted_semantic = process_array_and_tensor(predicted_semantic, denormalize=False, data_dims=2, channels=1,
                                                  dtype=np.uint8, argmax=True)
    if label_semantic is not None:
        label_semantic = process_array_and_tensor(label_semantic, denormalize=False, data_dims=2, channels=1,
                                                  dtype=np.uint8, argmax=False)

        if concatenate_vertically:
            semantic_comparison = np.zeros((predicted_semantic.shape[0] * 2, predicted_semantic.shape[1], 3),
                                           dtype=np.uint8)
            semantic_comparison[0:predicted_semantic.shape[0], :] = color_a_semantic_image(predicted_semantic)
            semantic_comparison[label_semantic.shape[0]:, :] = color_a_semantic_image(label_semantic)
        else:
            semantic_comparison = np.zeros((predicted_semantic.shape[0], predicted_semantic.shape[1] * 2, 3),
                                           dtype=np.uint8)
            semantic_comparison[:, 0:predicted_semantic.shape[1]] = np.rot90(color_a_semantic_image(predicted_semantic),
                                                                             1)
            semantic_comparison[:, label_semantic.shape[1]:] = color_a_semantic_image(label_semantic)
    else:
        if concatenate_vertically:
            semantic_comparison = np.zeros((predicted_semantic.shape[0], predicted_semantic.shape[1], 3),
                                           dtype=np.uint8)
            semantic_comparison[:, :] = color_a_semantic_image(predicted_semantic)
        else:
            semantic_comparison = np.zeros((predicted_semantic.shape[0], predicted_semantic.shape[1], 3),
                                           dtype=np.uint8)
            semantic_comparison[:, :] = np.rot90(color_a_semantic_image(predicted_semantic), 1)

    return semantic_comparison


def create_flow_comparison(predicted_flow, label_flow=None):
    predicted_flow = ((predicted_flow + 1) * (2 ** 15)).permute(0, 2, 3, 1)
    predicted_flow = process_array_and_tensor(predicted_flow, denormalize=False, data_dims=2, channels=2,
                                              dtype=np.float32, argmax=False)
    if label_flow is not None:
        label_flow = label_flow[:, :, :, :2]
        label_flow = process_array_and_tensor(label_flow, denormalize=False, data_dims=2, channels=2, dtype=np.float32,
                                              argmax=False)

        flow_comparison = np.zeros((predicted_flow.shape[0] * 2, predicted_flow.shape[1], 3), dtype=np.uint8)
        flow_comparison[0:predicted_flow.shape[0], :, :] = optical_flow_to_human(predicted_flow)
        flow_comparison[predicted_flow.shape[0]:, :, :] = optical_flow_to_human(label_flow)
    else:
        flow_comparison = np.zeros((predicted_flow.shape[0], predicted_flow.shape[1], 3), dtype=np.uint8)
        flow_comparison[:, :, :] = optical_flow_to_human(predicted_flow)

    return flow_comparison


def create_a_fake_rgb_comparison(rgb):
    return process_array_and_tensor(rgb, denormalize=False, data_dims=2, channels=3, dtype=np.uint8, argmax=False)


def create_a_fake_lidar_comparison(lidar):
    return process_array_and_tensor(lidar, denormalize=False, data_dims=2, channels=1, dtype=np.uint8, argmax=False)


def create_waypoints_comparison(prediction_target_speed, prediction_waypoints, actual_speed, target_point,
                                label_bev_semantic=None, label_target_speed=None, label_waypoints=None,
                                pred_bev_semantic=None):
    if label_bev_semantic is not None:
        background = process_array_and_tensor(label_bev_semantic, denormalize=False, data_dims=2, channels=1,
                                              dtype=np.uint8, argmax=False)
    elif pred_bev_semantic is not None:
        background = process_array_and_tensor(pred_bev_semantic, denormalize=False, data_dims=2, channels=1,
                                              dtype=np.uint8, argmax=True)
        background = np.rot90(background, 1)
    else:
        raise NutException(
            color_error_string("I receinved both 'label_bev_semantic' and 'pred_bev_semantic' as None! :-("))

    background_rgb = np.zeros((background.shape[0], background.shape[1], 3), dtype=np.uint8)
    background_rgb[:, :, 0] = background * 30
    background_rgb[:, :, 1] = background * 30
    background_rgb[:, :, 2] = background * 30
    black_part_for_text = np.zeros((background.shape[0], background.shape[1], 3), dtype=np.uint8)

    if label_target_speed is not None:
        label_target_speed = process_array_and_tensor(label_target_speed, denormalize=False, data_dims=1, channels=1,
                                                      dtype=np.float32, argmax=False)
    if label_waypoints is not None:
        label_waypoints = process_array_and_tensor(label_waypoints, denormalize=False, data_dims=1,
                                                   channels=label_waypoints.shape[-1], dtype=np.float32, argmax=False)
        label_waypoints = label_waypoints[:, :2]  # we drop the z
    prediction_target_speed = process_array_and_tensor(prediction_target_speed, denormalize=False, data_dims=1,
                                                       channels=1, dtype=np.float32, argmax=False, softmax=True)
    prediction_waypoints = process_array_and_tensor(prediction_waypoints, denormalize=False, data_dims=1, channels=2,
                                                    dtype=np.float32, argmax=False)
    actual_speed = process_array_and_tensor(actual_speed, denormalize=False, data_dims=1, channels=1, dtype=np.float32,
                                            argmax=False)
    target_point = process_array_and_tensor(target_point, denormalize=False, data_dims=1, channels=1, dtype=np.float32,
                                            argmax=False)
    target_point = target_point[:2]  # we drop the z

    # We draw the targetpoint
    target_point_x = target_point[0] * 256 / config.BEV_SQUARE_SIDE_IN_M
    target_point_y = target_point[1] * 256 / config.BEV_SQUARE_SIDE_IN_M
    if target_point_x > 128:
        target_point_x = 128
    elif target_point_x < -128:
        target_point_x = -128
    if target_point_y > 128:
        target_point_y = 128
    elif target_point_y < -128:
        target_point_y = -128
    background_rgb = cv2.circle(background_rgb, (int(128 - target_point_x), int(128 - target_point_y)), 5,
                                (0, 255, 255), -1)

    # We draw the waypoints
    for i in range(prediction_waypoints.shape[0]):
        background_rgb = cv2.circle(background_rgb,
                                    (int(128 - prediction_waypoints[i, 0] * 256 / config.BEV_SQUARE_SIDE_IN_M),
                                     int(128 - prediction_waypoints[i, 1] * 256 / config.BEV_SQUARE_SIDE_IN_M)),
                                    3, (0, 0, 255), -1)
        if label_waypoints is not None:
            background_rgb = cv2.circle(background_rgb,
                                        (int(128 - label_waypoints[i, 0] * 256 / config.BEV_SQUARE_SIDE_IN_M),
                                         int(128 - label_waypoints[i, 1] * 256 / config.BEV_SQUARE_SIDE_IN_M)),
                                        2, (0, 255, 0), -1)

    space_from_left = 10
    space_from_top = 30
    # We draw the actual speed
    cv2.putText(black_part_for_text, f"{float(actual_speed * 3.6):.2f} km/h", (space_from_left, space_from_top),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    speeds = [0, 7, 18, 29]
    if label_target_speed is not None:
        # We draw the label target speed
        list_label_target_speed = [float(el) for el in label_target_speed]
        index_best_label_target_speed = np.argmax(label_target_speed)
        text_label_target_speed = ""
        for i in range(4):
            text_label_target_speed += f"{list_label_target_speed[i]:.2f}"
            if i != 3:
                text_label_target_speed += ","
            text_label_target_speed += " "
        cv2.putText(black_part_for_text, text_label_target_speed, (space_from_left, space_from_top * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(black_part_for_text, f"{speeds[index_best_label_target_speed]:.2f} km/h",
                    (space_from_left, space_from_top * 3), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0, 255, 0),
                    thickness=1, lineType=cv2.LINE_AA)

    # We draw the predicted target speed
    list_predicted_target_speed = [float(el) for el in prediction_target_speed]
    index_best_predicted_target_speed = np.argmax(prediction_target_speed)
    text_predicted_target_speed = ""
    for i in range(4):
        text_predicted_target_speed += f"{list_predicted_target_speed[i]:.2f}"
        if i != 3:
            text_predicted_target_speed += ","
        text_predicted_target_speed += " "
    cv2.putText(black_part_for_text, text_predicted_target_speed, (space_from_left, space_from_top * 4),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(black_part_for_text, f"{speeds[index_best_predicted_target_speed]:.2f} km/h",
                (space_from_left, space_from_top * 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(255, 0, 0),
                thickness=1, lineType=cv2.LINE_AA)

    waypoints_comparison = np.zeros((background_rgb.shape[0], background_rgb.shape[1] * 2, 3), dtype=np.uint8)
    waypoints_comparison[:, 0:background_rgb.shape[1]] = background_rgb
    waypoints_comparison[:, background_rgb.shape[1]:] = black_part_for_text

    return waypoints_comparison


def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
    :param lidar: (N,3) numpy, LiDAR point cloud
    :return: (2, H, W) numpy, LiDAR as sparse image
    """
    MAX_HIST_POINTS = 5

    def splat_points(point_cloud):
        # 256 x 256 grid
        xbins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_IMAGE_W + 1)
        ybins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_SQUARE_SIDE_IN_M / 2, config.BEV_IMAGE_H + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist[hist > MAX_HIST_POINTS] = MAX_HIST_POINTS
        overhead_splat = hist / MAX_HIST_POINTS
        # The transpose here is an efficient axis swap.
        # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
        # (x height channel, y width channel)
        return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < -2.5 + config.MAXIMUM_LIDAR_HEIGHT]
    lidar = lidar[lidar[..., 2] > -2.5 + config.MINIMUM_LIDAR_HEIGHT]
    features = splat_points(lidar)
    features = np.stack([features], axis=-1)
    features = np.transpose(features, (2, 0, 1))
    features *= 255
    features = features.astype(np.uint8)
    return features


def load_model_given_weights(weights_path):
    try:
        weights = torch.load(weights_path)
    except Exception as e:
        raise NutException(color_error_string(f"Impossible to load weights located in '{weights_path}'!"))

    a_config_file = GlobalConfig()
    predicting_flow = None
    just_a_backbone = None
    tfpp_original = None
    if "flow_decoder.deconv1.0.weight" in weights.keys():
        # Predicting also flow
        a_config_file.use_flow = True
        predicting_flow = True
    else:
        # Not predicting flow
        a_config_file.use_flow = False
        predicting_flow = False
    if "wp_decoder.encoder.weight" not in weights.keys() and "checkpoint_decoder.decoder.weight" not in weights.keys():
        # Just a Backbone
        a_config_file.use_controller_input_prediction = False
        just_a_backbone = True
        # So we suppose it's not the original tfpp network
        tfpp_original = False
    else:
        # A full Network
        a_config_file.use_controller_input_prediction = True
        just_a_backbone = False
        # We use the extra sensor file to understand if it's an original fpp Network
        # because nutfuser do not use Commands!
        extra_sensor_num = weights["extra_sensor_encoder.0.weight"].shape[1]
        if extra_sensor_num == 7:
            # An original tfpp Network
            a_config_file.use_flow = False
            a_config_file.num_bev_semantic_classes = 11
            a_config_file.num_semantic_classes = 7
            del a_config_file.detailed_loss_weights["loss_flow"]
            a_config_file.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            a_config_file.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            tfpp_original = True
        else:
            a_config_file.use_discrete_command = False
            tfpp_original = False

    print(f"PREDICT FLOW:\t\t{predicting_flow}")
    print(f"JUST A BACKBONE:\t{just_a_backbone}")
    print(f"ORIGINAL TFPP:\t\t{tfpp_original}")

    if tfpp_original:
        raise NutException(color_error_string(f"You have given me original tfpp weights but I cannot deal with them!"))

    model = LidarCenterNet(a_config_file)

    model.cuda()

    try:
        model.load_state_dict(weights, strict=False)
    except Exception as e:
        raise NutException(color_error_string(f"Weight in '{weights_path}' not compatible with the model!"))

    return model, predicting_flow, just_a_backbone, tfpp_original


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def unify_routes_xml(xmls_folder_path):
    routes = ET.Element("routes")

    xml_files = os.listdir(xmls_folder_path)
    if "evaluation.xml" in xml_files:
        xml_files.remove("evaluation.xml")

    for file in xml_files:
        single_route_tree = ET.parse(os.path.join(xmls_folder_path, file))
        single_route = single_route_tree.getroot().getchildren()[0]
        routes.append(single_route)

    tree = ET.ElementTree(routes)
    indent(routes)
    tree.write(os.path.join(xmls_folder_path, f"evaluation.xml"))


def get_configs_as_dict():
    vars_name = {item: getattr(config, item) for item in dir(config) if not item.startswith("__")
                 and not isinstance(getattr(config, item), ModuleType)}
    return vars_name


def create_ground_truth(input_size, num_classes, batches_of_boxes, downsample_ratio=4, yaw_classes=12):
    """
    Creates ground truth heatmaps, size maps, offset maps, and yaw maps for CenterNet training.

    Parameters:
    - input_size: (width, height) of the input image
    - num_classes: number of object classes
    - boxes: list of bounding boxes, each box is [x_min, y_min, x_max, y_max]
    - labels: list of labels corresponding to each bounding box
    - yaws: list of yaw angles (in radians) corresponding to each bounding box
    - downsample_ratio: the ratio by which the output feature map is downsampled

    Returns:
    - heatmap: the ground truth heatmap
    - size_map: the ground truth size map
    - offset_map: the ground truth offset map
    - yaw_map: the ground truth yaw map
    """
    output_size = (input_size[0] // downsample_ratio, input_size[1] // downsample_ratio)
    batches = batches_of_boxes.shape[0]

    heatmap =           torch.zeros((batches, output_size[1], output_size[0], num_classes), dtype=torch.float32)
    size_map =          torch.zeros((batches, output_size[1], output_size[0], 2),           dtype=torch.float32)
    offset_map =        torch.zeros((batches, output_size[1], output_size[0], 2),           dtype=torch.float32)
    yaw_class_map =     torch.zeros((batches, output_size[1], output_size[0]),              dtype=torch.long)
    yaw_res_map =       torch.zeros((batches, output_size[1], output_size[0]),              dtype=torch.float32)
    pixel_weight =      torch.zeros((batches, output_size[1], output_size[0], 2),           dtype=torch.int)
    num_ob_bbs =        torch.zeros(batches,                                                dtype=torch.int)

    def gaussian_radius(det_size, min_overlap=0.7):
        _height, _width = det_size
        a1 = 1
        b1 = (_height + _width)
        c1 = _width * _height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (_height + _width)
        c2 = (1 - min_overlap) * _width * _height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = 2 * min_overlap * (_height + _width)
        c3 = (min_overlap - 1) * _width * _height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def draw_gaussian(batch, cls_id, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = torch.exp(- ((torch.arange(diameter) - radius) ** 2) / (2 * (radius / 3) ** 2))
        gaussian = torch.outer(gaussian, gaussian)
        x, y = int(center[0]), int(center[1])
        _height, _width = output_size[1], output_size[0]

        left, right = min(x, radius), min(_width - x, radius + 1)
        top, bottom = min(y, radius), min(_height - y, radius + 1)

        masked_heatmap = heatmap[batch, y - top:y + bottom, x - left:x + right, cls_id]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def angle2class(angle, yaw_classes=yaw_classes):
        """
        Convert continuous angle to a discrete class and a small regression number from class center angle to current angle.
        Args:
            angle (float): Angle is from 0-2pi (or -pi~pi),
              class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).
            yaw_classes: number of classes to devide the full angle
        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * torch.pi)
        angle_per_class = 2 * torch.pi / float(yaw_classes)
        shifted_angle = (angle + angle_per_class / 2) % (2 * torch.pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
        return int(angle_cls), angle_res

    for batch in range(batches_of_boxes.shape[0]):
        num_ob_bbs[batch] = batches_of_boxes[batch].shape[0]
        for box in batches_of_boxes[batch]:
            center_x = box[0]
            center_y = box[1]
            width = box[2]
            height = box[3]
            yaw = box[4]
            label = box[5]
            cls_id = int(label)

            if width > 0 and height > 0:
                center_x_s, center_y_s = center_x / downsample_ratio, center_y / downsample_ratio
                if center_x_s >= output_size[0]:
                    center_x_s = output_size[0]-1
                if center_y_s >= output_size[1]:
                    center_y_s = output_size[1]-1
                center = (center_x_s, center_y_s)
                radius = gaussian_radius((height / downsample_ratio, width / downsample_ratio))
                radius = max(0, int(radius))

                draw_gaussian(batch, cls_id, center, radius)

                size_map[batch, int(center_y_s), int(center_x_s), :] = torch.tensor([width, height])
                offset_map[batch, int(center_y_s), int(center_x_s), :] = \
                    torch.tensor([int(center_x_s) - int(center_x_s), int(center_y_s) - int(center_y_s)])

                yaw_class, yaw_res = angle2class(yaw)
                yaw_class_map[batch, int(center_y_s), int(center_x_s)] = yaw_class
                yaw_res_map[batch, int(center_y_s), int(center_x_s)] = yaw_res

                pixel_weight[batch, int(center_y_s), int(center_x_s), :] = 1

    return heatmap, size_map, offset_map, yaw_class_map, yaw_res_map, pixel_weight, num_ob_bbs


def decode_predictions(heatmap, size_map, offset_map, yaw_class_map, yaw_res_map, score_threshold=0.1,
                       downsample_ratio=4, yaw_classes=12):
    """
    Decodes the heatmaps, size maps, offset maps, and yaw maps back to bounding boxes, class labels, and yaw angles.

    Parameters:
    - heatmap: the predicted heatmap
    - size_map: the predicted size map
    - offset_map: the predicted offset map
    - yaw_map: the predicted yaw map
    - score_threshold: threshold for heatmap scores to consider a detection
    - downsample_ratio: the ratio by which the output feature map is downsampled

    Returns:
    - boxes: list of bounding boxes [x_min, y_min, x_max, y_max]
    - labels: list of class labels corresponding to each bounding box
    - yaws: list of yaw angles corresponding to each bounding box
    - score: list of the scores for each bounding boxes
    """
    def class2angle(angle_cls, angle_res, limit_period=True):
        """
        Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(yaw_classes)
        angle_center = float(angle_cls) * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            if angle > np.pi:
                angle -= 2 * np.pi
        return angle

    boxes = []

    heatmap[heatmap < 0] = 0
    heatmap[heatmap > 1] = 1

    batches = heatmap.shape[0]

    for batch in range(batches):
        for cls_id in range(heatmap[batch].shape[0]):
            y_indices, x_indices = torch.where(heatmap[batch, cls_id, :, :] >= score_threshold)

            for y, x in zip(y_indices, x_indices):
                score = heatmap[batch, cls_id, y, x]
                if score < score_threshold:
                    continue

                width, height = size_map[batch, :, y, x]
                if width < 2 or height < 2:
                    continue
                offset_x, offset_y = offset_map[batch, :, y, x]
                yaw = class2angle(yaw_class_map[batch, y, x], yaw_res_map[batch, y, x])

                center_x = (x + offset_x) * downsample_ratio
                center_y = (y + offset_y) * downsample_ratio

                boxes.append([center_x, center_y, width, height, yaw, cls_id, score])

    return torch.tensor(boxes)


def draw_bounding_boxes(image, boxes, thickness=2):
    """
    Draws bounding boxes on an image.

    Parameters:
    - image: the input image
    - boxes: list of bounding boxes, each box is (center_x, center_y, width, height, yaw)
    - color: the color of the bounding box (default is green)
    - thickness: the thickness of the bounding box lines (default is 2)

    Returns:
    - image: the image with bounding boxes drawn
    """
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255)
    ]

    for i in range(boxes.shape[0]):
        center_x = boxes[i][0]
        center_y = boxes[i][1]
        width = boxes[i][2]
        height = boxes[i][3]
        yaw = boxes[i][4]
        cls_id = boxes[i][5]

        color = colors[int(cls_id)]
        center = (int(center_x), int(center_y))

        # Calculate corner points of the bounding box
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        half_width = width / 2
        half_height = height / 2

        points = torch.tensor([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])

        rotation_matrix = torch.tensor([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])

        rotated_points = torch.tensor([(rotation_matrix @ points[i]).tolist() for i in range(4)])
        translated_points = rotated_points + torch.tensor(center)

        pts = translated_points.numpy().astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw the bounding box
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    return image


def save_bbs_in_json(path, bbs):
    if len(bbs) > config.NUM_OF_BBS_PER_FRAME:
        bbs = bbs[:config.NUM_OF_BBS_PER_FRAME]
    else:
        bbs += [[-1000, -1000, 0, 0, 0, 0] for _ in range(config.NUM_OF_BBS_PER_FRAME-len(bbs))]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(bbs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import neural_networks.tfpp.config as tf_config

    a_config = tf_config.GlobalConfig()
    import torch
    # Example usage:
    input_size = (512, 512)
    num_classes = 2
    boxes = torch.tensor([
                            [[100, 100, 50, 50, 0.5, 0],
                             [200, 200, 50, 50, 0.0, 0],
                             [300, 300, 50, 50, -0.5, 1]],
                         ])  # Example bbs

    heatmap, size_map, offset_map, yaw_class_map, yaw_res_map, pixel_weight, num_ob_bbs =\
        create_ground_truth(input_size, num_classes, boxes, downsample_ratio=4)

    import matplotlib.pyplot as plt
    imgplot = plt.imshow(pixel_weight[0, :, :, 0])
    plt.show()
    imgplot = plt.imshow(heatmap[0, :, :, 0])
    plt.show()
    imgplot = plt.imshow(heatmap[0, :, :, 1])
    plt.show()

    # Example usage:
    # Assuming heatmap, size_map, offset_map, yaw_map are the predicted maps from the model
    # heatmap, size_map, offset_map, yaw_map = model_output

    boxes = decode_predictions(heatmap, size_map, offset_map, yaw_class_map, yaw_res_map, score_threshold=0.9,
                               downsample_ratio=4, yaw_classes=12)

    print(boxes.shape)

    image = np.zeros((512, 512, 3))
    image = draw_bounding_boxes(image, boxes)

    cv2.imshow("bbs", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()





