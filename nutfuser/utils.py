from termcolor import colored
import numpy as np
import math
import cv2
import os
import torch 

from tqdm import tqdm
from datetime import datetime
from pynvml import *
from nutfuser.raft_flow_colormap import flow_to_image
import nutfuser.config as config



def color_error_string(string):
    return colored(string, "red", attrs=["bold"]) # , "blink"

def color_info_string(string):
    return colored(string, "yellow", attrs=["bold"])

def color_info_success(string):
    return colored(string, "green", attrs=["bold"])

def get_a_title(string, color):
    line = "#"*(len(string)+2)
    final_string = line + "\n#" + string + "#\n" + line
    return colored(final_string, color, attrs=["bold"])


def convert_gps_to_carla(gps_array):
    position_carla_coord = gps_array * np.array([111324.60662786, 111319.490945, 1])
    position_carla_coord = np.concatenate([  np.expand_dims( position_carla_coord[:, 1], 1),
                                             np.expand_dims(-position_carla_coord[:, 0], 1),
                                             np.expand_dims( position_carla_coord[:, 2], 1)],
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
        points[:, 1] = (points[:, 1] - min_y + (den_x-den_y)/2) / denominator
    else:
        points[:, 0] = (points[:, 0] - min_x + (den_y-den_x)/2) / denominator
        points[:, 1] = (points[:, 1] - min_y) / denominator
    return points, origin, den_x, den_y, min_x, min_y

def calculate_point_each_meter(points, den_x, den_y):
    denominator = max(den_x, den_y)
    points_in_m = points * denominator
    points_with_equal_distance = []
    subframe_position = []
    tmp_total_distance = 0
    for i in range(len(points_in_m) - 1):
        distance = math.sqrt((points_in_m[i, 0] - points_in_m[i+1, 0])**2 + (points_in_m[i, 1] - points_in_m[i+1, 1])**2)
        tmp_total_distance += distance
        if tmp_total_distance > 1.0:
            points_with_equal_distance.append(points[i])
            subframe_position.append(i)
            tmp_total_distance = 0
    return np.array(points_with_equal_distance), subframe_position

def optical_flow_to_human(flow):
    flow = (flow - 2**15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]
    
    output = flow_to_image(flow, clip_flow=60)
    output *= 255
    return output

def optical_flow_to_human_with_path(optical_flow_path):
    flow = cv2.imread(optical_flow_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow.astype(np.float32)
    flow = flow[:, :, :2]
    
    return optical_flow_to_human(flow)


def optical_flow_to_human_slow(optical_flow_path):
    """
    Not used anymore! Keeping just for history!
    """
    flow = cv2.imread(optical_flow_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow.astype(np.float32)
    flow = flow[:, :, :2]
    flow = (flow - 2**15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]
    flow[:, :, 0] /= H * 0.5
    flow[:, :, 1] /= W * 0.5
    output = np.zeros((H, W, 3), dtype=np.uint8)
    rad2ang = 180./math.pi
    for i in range(H):
        for j in range(W):
            vx = flow[i, j, 0]
            vy = flow[i, j, 1]
            angle = 180. + math.atan2(vy, vx)*rad2ang
            if angle < 0:
                angle = 360. + angle
                pass
            angle = math.fmod(angle, 360.)
            norm = math.sqrt(vx*vx + vy*vy)
            shift = 0.999
            a = 1/math.log(0.1 + shift)
            raw_intensity = a*math.log(norm + shift)
            if raw_intensity < 0.:
                intensity = 0.
            elif raw_intensity > 1.:
                intensity = 1.
            else:
                intensity = raw_intensity
            S = 1.
            V = intensity
            H_60 = angle*1./60.
            C = V * S
            X = C*(1. - abs(math.fmod(H_60, 2.) - 1.))
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
            R = int((r+m)*255)
            G = int((g+m)*255)
            B = int((b+m)*255)
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

    folder_that_should_be_there =  [(f"rgb_A_{i}", ".jpg") for i in camera_indexes] +\
                        [(f"rgb_B_{i}", ".jpg") for i in camera_indexes] +\
                        [(f"depth_{i}", ".png") for i in camera_indexes] +\
                        [(f"optical_flow_{i}", ".png") for i in camera_indexes] +\
                        [(f"semantic_{i}", ".png") for i in camera_indexes] +\
                        [("bev_semantic", ".png"),      ("bev_lidar", ".png")]
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
                    raise Exception(color_error_string(f"Missing frame {i} in '{os.path.join(dataset_path, folder)}'\n[{all_frames}]"))
        except:
            raise Exception(color_error_string(f"Some strange frame name inside '{os.path.join(dataset_path, folder)}'\n[{all_frames}]"))
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
        video = cv2.VideoWriter(os.path.join(folders_path, f"{folder}.mp4"), fourcc, 15, (example_of_frame.shape[1], example_of_frame.shape[0]))
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
        video = cv2.VideoWriter(os.path.join(where_to_save, f"comparison_{folder}.mp4"), fourcc, 15, (example_of_frame.shape[1], int((example_of_frame.shape[0]/2)*3)))
        for frame in tqdm(int_all_frames, desc=f"comparison_{folder}"):
            img_A = cv2.imread(os.path.join(full_folder_path_A, f"{frame}{exctention}"))
            img_B = cv2.imread(os.path.join(full_folder_path_B, f"{frame}{exctention}"))
            img = np.zeros(shape=(int((img_A.shape[0]/2)*3), img_A.shape[1], img_A.shape[2]), dtype=img_A.dtype)
            img[:img_A.shape[0], :] = img_A
            img[img_A.shape[0]//2:, :] = img_B
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
        used_total = {i:0 for i in range(len(gpus))}
        total_total = {i:0 for i in range(len(gpus))}
        free_total = {i:0 for i in range(len(gpus))}
        while True:
            for i in range(100):
                for i, gpu in enumerate(gpus):
                    info = nvmlDeviceGetMemoryInfo(gpu)
                    used_total[i] += info.used/10**9
                    total_total[i] += info.total/10**9
                    free_total[i] += info.free/10**9
                time.sleep(delay_in_seconds/100)
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
        xbins = np.linspace(-32, 32, 256+1)
        ybins = np.linspace(-32, 32, 256+1)
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

def process_array_and_tensor(an_array_or_tensor, denormalize=False, data_dims=2, channels=1, dtype=None, argmax=False, softmax=False):
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
            raise NutException(f"Asked to denormalize but seems that the tensor/array is not normalized! [max = {maximum:.3f} > 1.0]")
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
            raise NutException(f"Wrong number of channels! Expected {channels} but found {an_array_or_tensor.shape[-1]}!")

    actual_dims = len(an_array_or_tensor.shape)
    if actual_dims < expected_dims:
        raise NutException(f"The expected dimension [{expected_dims}] is bigger than the actual one {actual_dims} after some change!")

    if isinstance(an_array_or_tensor, torch.Tensor):
        if softmax:
            an_array_or_tensor = an_array_or_tensor.contiguous().detach().cpu()
            an_array_or_tensor = torch.nn.functional.softmax(an_array_or_tensor, dim=0)
            return (an_array_or_tensor.numpy()).astype(dtype)
        if dtype is not None:
            return (an_array_or_tensor.contiguous().detach().cpu().numpy()).astype(dtype)
        else:
            return an_array_or_tensor.contiguous().detach().cpu().numpy()

def create_depth_comparison(predicted_depth, label_depth):
    predicted_depth = process_array_and_tensor(predicted_depth, denormalize=True, data_dims=2, channels=1, dtype=np.uint8, argmax=False)
    label_depth = process_array_and_tensor(label_depth, denormalize=False, data_dims=2, channels=1, dtype=np.uint8, argmax=False)
    
    depth_comparison = np.zeros((predicted_depth.shape[0]*2, predicted_depth.shape[1]), dtype=np.uint8)
    depth_comparison[0:predicted_depth.shape[0], :] = predicted_depth
    depth_comparison[label_depth.shape[0]:, :] = label_depth

    return depth_comparison

def color_a_semantic_image(a_semantic_array):
    output = np.zeros(shape=(a_semantic_array.shape[0], a_semantic_array.shape[1], 3), dtype=np.uint8)
    for key in config.NUTFUSER_SEMANTIC_COLOR:
        output[a_semantic_array==key] = config.NUTFUSER_SEMANTIC_COLOR[key]
    return output

def create_semantic_comparison(predicted_semantic, label_semantic, concatenate_vertically=True):
    predicted_semantic = process_array_and_tensor(predicted_semantic, denormalize=False, data_dims=2, channels=1, dtype=np.uint8, argmax=True)
    label_semantic = process_array_and_tensor(label_semantic, denormalize=False, data_dims=2, channels=1, dtype=np.uint8, argmax=False)
    
    if concatenate_vertically:
        semantic_comparison = np.zeros((predicted_semantic.shape[0]*2, predicted_semantic.shape[1], 3), dtype=np.uint8)
        semantic_comparison[0:predicted_semantic.shape[0], :] = color_a_semantic_image(predicted_semantic)
        semantic_comparison[label_semantic.shape[0]:, :] = color_a_semantic_image(label_semantic)
    else:
        semantic_comparison = np.zeros((predicted_semantic.shape[0], predicted_semantic.shape[1]*2, 3), dtype=np.uint8)
        semantic_comparison[:, 0:predicted_semantic.shape[1]] = np.rot90(color_a_semantic_image(predicted_semantic), 1)
        semantic_comparison[:, label_semantic.shape[1]:] = color_a_semantic_image(label_semantic)

    return semantic_comparison

def create_flow_comparison(predicted_flow, label_flow):
    predicted_flow = ((predicted_flow + 1)*(2**15)).permute(0, 2, 3, 1)
    label_flow = label_flow[:, :, :, :2]
    predicted_flow = process_array_and_tensor(predicted_flow, denormalize=False, data_dims=2, channels=2, dtype=np.float32, argmax=False)
    label_flow = process_array_and_tensor(label_flow, denormalize=False, data_dims=2, channels=2, dtype=np.float32, argmax=False)

    flow_comparison = np.zeros((predicted_flow.shape[0]*2, predicted_flow.shape[1], 3), dtype=np.uint8)
    flow_comparison[0:predicted_flow.shape[0], :, :] = optical_flow_to_human(predicted_flow)
    flow_comparison[predicted_flow.shape[0]:, :, :] = optical_flow_to_human(label_flow)

    return flow_comparison

def create_a_fake_rgb_comparison(rgb):
    return process_array_and_tensor(rgb, denormalize=False, data_dims=2, channels=3, dtype=np.uint8, argmax=False)

def create_a_fake_lidar_comparison(lidar):
    return process_array_and_tensor(lidar, denormalize=False, data_dims=2, channels=1, dtype=np.uint8, argmax=False)

def create_waypoints_comparison(label_bev_semantic, label_target_speed, label_waypoints, prediction_target_speed, prediction_waypoints, actual_speed, target_point):
    label_bev_semantic = process_array_and_tensor(label_bev_semantic, denormalize=False, data_dims=2, channels=1, dtype=np.uint8, argmax=False)
    label_bev_semantic_rgb = np.zeros((label_bev_semantic.shape[0], label_bev_semantic.shape[1], 3), dtype=np.uint8)
    label_bev_semantic_rgb[:, :, 0] = label_bev_semantic * 30
    label_bev_semantic_rgb[:, :, 1] = label_bev_semantic * 30
    label_bev_semantic_rgb[:, :, 2] = label_bev_semantic * 30
    black_part_for_text = np.zeros((label_bev_semantic.shape[0], label_bev_semantic.shape[1], 3), dtype=np.uint8)

    label_target_speed = process_array_and_tensor(label_target_speed, denormalize=False, data_dims=1, channels=1, dtype=np.float32, argmax=False)
    label_waypoints = process_array_and_tensor(label_waypoints, denormalize=False, data_dims=1, channels=3, dtype=np.float32, argmax=False)
    label_waypoints = label_waypoints[:, :2] # we drop the z
    prediction_target_speed = process_array_and_tensor(prediction_target_speed, denormalize=False, data_dims=1, channels=1, dtype=np.float32, argmax=False, softmax=True)
    prediction_waypoints = process_array_and_tensor(prediction_waypoints, denormalize=False, data_dims=1, channels=2, dtype=np.float32, argmax=False)
    actual_speed = process_array_and_tensor(actual_speed, denormalize=False, data_dims=1, channels=1, dtype=np.float32, argmax=False)
    target_point = process_array_and_tensor(target_point, denormalize=False, data_dims=1, channels=1, dtype=np.float32, argmax=False)
    target_point = target_point[:2] # we drop the z

    # We draw the targetpoint
    target_point_x = target_point[0]*256/config.BEV_SQUARE_SIDE_IN_M
    target_point_y = target_point[1]*256/config.BEV_SQUARE_SIDE_IN_M
    if target_point_x > 128:
        target_point_x = 128
    elif target_point_x < -128:
        target_point_x = -128
    if target_point_y > 128:
        target_point_y = 128
    elif target_point_y < -128:
        target_point_y = -128
    label_bev_semantic_rgb = cv2.circle(label_bev_semantic_rgb, (int(128-target_point_x), int(128-target_point_y)), 5, (0, 255, 255), -1)

    # We draw the waypoints
    for i in range(label_waypoints.shape[0]):
        label_bev_semantic_rgb = cv2.circle(label_bev_semantic_rgb,
                                            (   int(128-prediction_waypoints[i, 0]*256/config.BEV_SQUARE_SIDE_IN_M),
                                                int(128-prediction_waypoints[i, 1]*256/config.BEV_SQUARE_SIDE_IN_M)),
                                            3, (0, 0, 255), -1)
        label_bev_semantic_rgb = cv2.circle(label_bev_semantic_rgb, 
                                            (   int(128-label_waypoints[i, 0]*256/config.BEV_SQUARE_SIDE_IN_M),
                                                int(128-label_waypoints[i, 1]*256/config.BEV_SQUARE_SIDE_IN_M)),
                                            2, (0, 255, 0), -1)
    
    space_from_left = 10
    space_from_top = 30
    # We draw the actual speed
    cv2.putText(black_part_for_text, f"{float(actual_speed*3.6):.2f} km/h", (space_from_left, space_from_top), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.6, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    speeds = [0, 7, 18, 29]
    # We draw the label target speed
    list_label_target_speed = [float(el) for el in label_target_speed]
    index_best_label_target_speed = np.argmax(label_target_speed)
    text_label_target_speed = ""
    for i in range(4):
        text_label_target_speed += f"{list_label_target_speed[i]:.2f}"
        if i != 3:
            text_label_target_speed += ","
        text_label_target_speed += " "
    cv2.putText(black_part_for_text, text_label_target_speed, (space_from_left, space_from_top*2), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.45, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(black_part_for_text, f"{speeds[index_best_label_target_speed]:.2f} km/h", (space_from_left, space_from_top*3), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.45, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # We draw the predicted target speed
    list_predicted_target_speed = [float(el) for el in prediction_target_speed]
    index_best_predicted_target_speed = np.argmax(prediction_target_speed)
    text_predicted_target_speed = ""
    for i in range(4):
        text_predicted_target_speed += f"{list_predicted_target_speed[i]:.2f}"
        if i != 3:
            text_predicted_target_speed += ","
        text_predicted_target_speed += " "
    cv2.putText(black_part_for_text, text_predicted_target_speed, (space_from_left, space_from_top*4), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.45, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(black_part_for_text, f"{speeds[index_best_predicted_target_speed]:.2f} km/h", (space_from_left, space_from_top*5), cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.45, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    waypoints_comparison = np.zeros((label_bev_semantic.shape[0], label_bev_semantic.shape[1]*2, 3), dtype=np.uint8)
    waypoints_comparison[:, 0:label_bev_semantic.shape[1]] = label_bev_semantic_rgb
    waypoints_comparison[:, label_bev_semantic.shape[1]:] =black_part_for_text

    return waypoints_comparison

