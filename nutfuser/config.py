import math
import os
import time

# GENERAL
DATASET_PATH = "/home/enrico/Downloads/nut_dataset_test" # "/leonardo_work/IscrC_SSNeRF/nut_dataset"
VALIDATION_DATASET_PATH = "/home/enrico/Downloads/nut_dataset_old/nut_dataset_test"

# TAKE DATA
CARLA_FPS = 20
IMAGE_W = 1024
IMAGE_H = 256
BEV_IMAGE_W = 256
BEV_IMAGE_H = 256
BEV_ALTITUDE = 1000
BEV_BOTTOM_ALTITUDE = 17
BEV_SQUARE_SIDE_IN_M = 40
MAXIMUM_DISTANCE_FROM_VEHICLE_IN_BEV = BEV_SQUARE_SIDE_IN_M/math.sqrt(2)
BEV_FOV_IN_DEGREES = (2 * math.atan(BEV_SQUARE_SIDE_IN_M / (2 * BEV_ALTITUDE)))/math.pi * 180
BEV_BOTTOM_FOV_IN_DEGREES = (2 * math.atan(BEV_SQUARE_SIDE_IN_M / (2 * BEV_BOTTOM_ALTITUDE)))/math.pi * 180
AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE = 5 # minimum is 3
# MAX_NUM_OF_SAVED_FRAME = 10 # 1000*34 almost 50 GB
NUM_OF_WAYPOINTS = 10
DISTANCE_BETWEEN_WAYPOINTS = 1 # m
MINIMUM_LIDAR_HEIGHT = 0.3 # m
MAXIMUM_LIDAR_HEIGHT = 3 # m
FRAME_TO_KEEP_GOING_AFTER_THE_END = AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE*60 # we want to be sure to explore enaught space fro the last waypoint ground truth
HOW_MANY_CARLA_FRAME_FOR_CALCULATING_SPEEDS = 3
TOWN_DICT = {1:  "Town01_Opt", 2:  "Town02_Opt", 3:  "Town03_Opt", 4:  "Town04_Opt",
             5:  "Town05_Opt", 6:  "Town06_Opt", 7:  "Town07_Opt", 10: "Town10HD_Opt",
             12: "Town12",     13: "Town13",     15: "Town15"}
SELECTED_TOWN_NAME = "Town15"

# DATA LOADER
JOB_TMP_DIR = None
JOB_TMP_DIR_NAME = "nut_tmp"
DATASET_FOLDER_STRUCT = [("rgb_A_0", ".jpg"),           ("rgb_A_1", ".jpg"),        ("rgb_A_2", ".jpg"),        ("rgb_A_3", ".jpg"),
                         ("rgb_B_0", ".jpg"),           ("rgb_B_1", ".jpg"),        ("rgb_B_2", ".jpg"),        ("rgb_B_3", ".jpg"),
                         ("depth_0", ".png"),           ("depth_1", ".png"),        ("depth_2", ".png"),        ("depth_3", ".png"),
                         ("optical_flow_0", ".png"),    ("optical_flow_1", ".png"), ("optical_flow_2", ".png"), ("optical_flow_3", ".png"),
                         ("semantic_0", ".png"),        ("semantic_1", ".png"),     ("semantic_2", ".png"),     ("semantic_3", ".png"),
                         ("bev_semantic", ".png"),      ("bev_lidar", ".png")]
CACHE_SIZE_LIMIT =  0 # int(0.1 * 1e9) # Bytes (GB without the 1e9)

# TRAIN
BATCH_SIZE = 10
LEARNING_RATE = 7.5e-5
LEARNING_RATE_DECAY = 0.1
START_EPOCH = 0
PRETRAINED_WEIGHT_PATH = None
PRETRAINED_OPTIMIZER_PATH = None
PRETRAINED_SCHEDULER_PATH = None
PRETRAINED_SCALER_PATH = None
TRAIN_LOG_DIR = os.path.join(os.getcwd(), "logs")
REDUCE_LR_FIRST_TIME = 30
REDUCE_LR_SECOND_TIME = 40

# TMP
START_TIME = time.time()
