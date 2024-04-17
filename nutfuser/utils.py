from termcolor import colored
import numpy as np
import math
import cv2
import os
from tqdm import tqdm
from datetime import datetime
from pynvml import *


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
    flow[:, :, 0] /= H * 0.5
    flow[:, :, 1] /= W * 0.5
    output = np.zeros((H, W, 3), dtype=np.float32)
    rad2ang = 180./np.pi
    angle = 180. + np.arctan2(flow[:, :, 1], flow[:, :, 0]) * rad2ang
    angle[angle < 0] += 360
    angle = np.fmod(angle, 360.)
    H_60 = angle / 60.
    norm = np.sqrt(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2))
    raw_intensity = 1/np.log(0.1 + 0.999) * np.log(norm + 0.999)
    raw_intensity[raw_intensity < 0.] = 0.
    raw_intensity[raw_intensity > 1.] = 1.
    C = raw_intensity
    X = raw_intensity * (np.ones_like(H_60) - np.abs(np.fmod(H_60, 2,) - np.ones_like(H_60)))
    H_60 = H_60.astype(int)
    output[H_60 == 0] = np.stack([C, X, np.zeros_like(C)], axis = -1)[H_60 == 0]
    output[H_60 == 1] = np.stack([X, C, np.zeros_like(C)], axis = -1)[H_60 == 1]
    output[H_60 == 2] = np.stack([np.zeros_like(C), C, X], axis = -1)[H_60 == 2]
    output[H_60 == 3] = np.stack([np.zeros_like(C), X, C], axis = -1)[H_60 == 3]
    output[H_60 == 4] = np.stack([X, np.zeros_like(C), C], axis = -1)[H_60 == 4]
    output[H_60 == 5] = np.stack([C, np.zeros_like(C), X], axis = -1)[H_60 == 5]
    output[H_60 >= 6] = np.stack([np.ones_like(C), np.ones_like(C), np.ones_like(C)], axis = -1)[H_60 >= 6]

    output *= 255
    # output = output.astype(np.uint8)
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
