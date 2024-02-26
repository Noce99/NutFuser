import math

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
TMP_DATASET_PATH = "/leonardo_work/IscrC_SSNeRF/nut_dataset"
AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE = 30 # minimum is 3

TOWN_DICT = {1:  "Town01_Opt", 2:  "Town02_Opt", 3:  "Town03_Opt", 4:  "Town04_Opt",
             5:  "Town05_Opt", 6:  "Town06_Opt", 7:  "Town07_Opt", 10: "Town10HD_Opt",
             11: "Town11",     13: "Town13",     15: "Town15"}
SELECTED_TOWN_NAME = "Town15"
