from termcolor import colored
import numpy as np
import math

def color_error_string(string):
    return colored(string, "red", attrs=["bold"]) # , "blink"

def color_info_string(string):
    return colored(string, "yellow", attrs=["bold"])

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
