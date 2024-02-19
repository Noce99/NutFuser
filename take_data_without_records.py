import carla 
import os
import math
import shutil
import signal
import time 
import numpy as np
import cv2

IMAGE_W = 1024
IMAGE_H = 256
BEV_IMAGE_W = 256
BEV_IMAGE_H = 256
BEV_ALTITUDE = 1024
BEV_SQUARE_SIDE_IN_M = 40
BEV_FOV_IN_DEGREES = (2 * math.atan(BEV_SQUARE_SIDE_IN_M / (2 * BEV_ALTITUDE)))/math.pi * 180
TMP_DATASET_PATH = "/home/enrico/Projects/Carla/tmp_experiment"
AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE = 5

data_last_frame = {}
take_new_data = {}
data_last_frame["lidar"] = np.zeros((BEV_IMAGE_H, BEV_IMAGE_W), dtype=np.uint8)
take_new_data["lidar"] = True
data_last_frame["bev_semantic"] = np.zeros((BEV_IMAGE_H, BEV_IMAGE_W), dtype=np.uint8)
take_new_data["bev_semantic"] = True
for i in range(8):
    """
    from 0 to 3 -> amazing cameras
    from 4 to 7 -> obvious view
    """
    data_last_frame[f"rgb_{i}"] = np.zeros((IMAGE_H, IMAGE_W, 4), dtype=np.uint8)
    take_new_data[f"rgb_{i}"] = True
    data_last_frame[f"depth_{i}"] = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
    take_new_data[f"depth_{i}"] = True
    data_last_frame[f"semantic_{i}"] = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
    take_new_data[f"semantic_{i}"] = True

# Connect the client and set up bp library
client = carla.Client('localhost', 2000)
client.set_timeout(120.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# LIDAR callback
def lidar_callback(point_cloud):
    global data_last_frame
    def lidar_to_histogram_features(lidar):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
        :param lidar: (N,3) numpy, LiDAR point cloud
        :return: (2, H, W) numpy, LiDAR as sparse image
        """
        MAX_HIST_POINTS = 5
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(-BEV_SQUARE_SIDE_IN_M/2, BEV_SQUARE_SIDE_IN_M/2, BEV_IMAGE_W)
            ybins = np.linspace(-BEV_SQUARE_SIDE_IN_M/2, BEV_SQUARE_SIDE_IN_M/2, BEV_IMAGE_H)
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > MAX_HIST_POINTS] = MAX_HIST_POINTS
            overhead_splat = hist / MAX_HIST_POINTS
            # The transpose here is an efficient axis swap.
            # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
            # (x height channel, y width channel)
            return overhead_splat.T

        # Remove points above the vehicle
        lidar = lidar[lidar[..., 2] < 100]
        lidar = lidar[lidar[..., 2] > -2.2]
        features = splat_points(lidar)
        features = np.stack([features], axis=-1)
        features = np.transpose(features, (2, 0, 1))
        features *= 255
        features = features.astype(np.uint8)
        return features
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    data_last_frame[f"lidar"] = lidar_to_histogram_features(data[:, :3])[0]

# CAMERAS callback
def rgb_callback(image, number):
    if take_new_data[f"rgb_{number}"]:
        global data_last_frame
        data_last_frame[f"rgb_{number}"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        take_new_data[f"rgb_{number}"] = False

# DEPTH callback
def depth_callback(depth, number):
    if take_new_data[f"depth_{number}"]:
        global data_last_frame
        depth.convert(carla.ColorConverter.LogarithmicDepth)
        data_last_frame[f"depth_{number}"] = np.reshape(np.copy(depth.raw_data), (depth.height, depth.width, 4))
        take_new_data[f"depth_{number}"] = False

# SEMATIC callback
def semantic_callback(semantic, number):
    if take_new_data[f"semantic_{number}"]:
        global data_last_frame
        # semantic.convert(carla.ColorConverter.CityScapesPalette)
        data_last_frame[f"semantic_{number}"] = np.reshape(np.copy(semantic.raw_data), (semantic.height, semantic.width, 4))
        take_new_data[f"semantic_{number}"] = False

# BEV SEMANTIC callback
def bev_semantic_callback(bev_semantic):
    if take_new_data["bev_semantic"]:
        global data_last_frame
        bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
        data_last_frame["bev_semantic"] = np.reshape(np.copy(bev_semantic.raw_data), (bev_semantic.height, bev_semantic.width, 4))
        data_last_frame["bev_semantic"] = cv2.resize(data_last_frame["bev_semantic"], (BEV_IMAGE_H, BEV_IMAGE_W, 4), interpolation= cv2.INTER_LINEAR)
        take_new_data["bev_semantic"] = False
        
# Search the CAR
hero = None
while hero is None:
    print("Waiting for the ego vehicle...")
    possible_vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in possible_vehicles:
        if vehicle.attributes['role_name'] == 'hero':
            print("Ego vehicle found")
            hero = vehicle
            break
    time.sleep(1)

# LIDAR
lidar_bp = bp_lib.find('sensor.lidar.ray_cast') 
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('noise_stddev', '0.0')
lidar_bp.set_attribute('upper_fov', '0.0')
lidar_bp.set_attribute('lower_fov', '-25.0')
lidar_bp.set_attribute('channels', '32.0')
lidar_bp.set_attribute('rotation_frequency', '20.0')
lidar_bp.set_attribute('points_per_second', '600000')
lidar_init_trans = carla.Transform(
    carla.Location(x=0, y=0, z=2.5),
    carla.Rotation(pitch=0, roll=0, yaw=0)
)

# RGB CAMERAS
camera_bp = bp_lib.find("sensor.camera.rgb") 
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("image_size_x", f"{IMAGE_W}")
camera_bp.set_attribute("image_size_y", f"{IMAGE_H}")

# DEPTH CAMERAS
depth_bp = bp_lib.find("sensor.camera.depth") 
depth_bp.set_attribute("fov", "90")
depth_bp.set_attribute("image_size_x", f"{IMAGE_W}")
depth_bp.set_attribute("image_size_y", f"{IMAGE_H}")

# SEMANTIC CAMERAS
semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation") 
semantic_bp.set_attribute("fov", "90")
semantic_bp.set_attribute("image_size_x", f"{IMAGE_W}")
semantic_bp.set_attribute("image_size_y", f"{IMAGE_H}")

# BEV SEMANTIC
bev_semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation") 
bev_semantic_bp.set_attribute("fov", f"{BEV_FOV_IN_DEGREES}")
bev_semantic_bp.set_attribute("image_size_x", f"{BEV_IMAGE_W*4}")
bev_semantic_bp.set_attribute("image_size_y", f"{BEV_IMAGE_H*4}")

# Amazing CAMERAS
transformations = []
transformations.append(carla.Transform(carla.Location(x=-2.5, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=-0.0, roll=0, yaw=40)))
transformations.append(carla.Transform(carla.Location(x=+2.5, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=-0.0, roll=0, yaw=140)))
transformations.append(carla.Transform(carla.Location(x=-2.5, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=-0.0, roll=0, yaw=320)))
transformations.append(carla.Transform(carla.Location(x=+2.5, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=-0.0, roll=0, yaw=220)))
# Obvious CAMERAS
transformations.append(carla.Transform(carla.Location(x=1.0, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=0.0, roll=0, yaw=0)))
transformations.append(carla.Transform(carla.Location(x=0.0, y=-1.0, z=2.0),
                                     carla.Rotation(pitch=-5.0, roll=0, yaw=90)))
transformations.append(carla.Transform(carla.Location(x=-1.0, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=0.0, roll=0, yaw=180)))
transformations.append(carla.Transform(carla.Location(x=+0.0, y=0.0, z=2.0),
                                     carla.Rotation(pitch=-20, roll=0, yaw=270)))

bev_transformation = carla.Transform(carla.Location(x=+0.0, y=0.0, z=BEV_ALTITUDE),
                                     carla.Rotation(pitch=-90, roll=0, yaw=0))

sensors = {}
sensors["lidar"] = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=hero)
sensors["bev_semantic"] = world.spawn_actor(bev_semantic_bp, bev_transformation, attach_to=hero)
for i in range(8):
    sensors[f"rgb_{i}"] = world.spawn_actor(camera_bp, transformations[i] , attach_to=hero)
    sensors[f"depth_{i}"] = world.spawn_actor(depth_bp, transformations[i], attach_to=hero)
    sensors[f"semantic_{i}"] = world.spawn_actor(semantic_bp, transformations[i], attach_to=hero)

# Connect Sensor and Callbacks
sensors["lidar"].listen(lambda data: lidar_callback(data))
sensors["bev_semantic"].listen(lambda data: bev_semantic_callback(data))
for i in range(8):
    sensors[f"rgb_{i}"].listen(lambda image, i=i: rgb_callback(image, i))
    sensors[f"depth_{i}"].listen(lambda depth, i=i: depth_callback(depth, i))
    sensors[f"semantic_{i}"].listen(lambda semantic, i=i: semantic_callback(semantic, i))

# Create Directory Branches
if os.path.isdir(TMP_DATASET_PATH):
    shutil.rmtree(TMP_DATASET_PATH)
os.mkdir(TMP_DATASET_PATH)

rgb_folders_name =      [f"fancy_rgb_{i}"   for i in range(4)] +\
                        [f"normy_rgb_{i}"   for i in range(4)]
depth_folders_name =    [f"fancy_depth_{i}" for i in range(4)] +\
                        [f"normy_depth_{i}" for i in range(4)]
semantic_folders_name = [f"fancy_semantic_{i}" for i in range(4)] +\
                        [f"normy_semantic_{i}" for i in range(4)]

paths = {}
paths["lidar"] = os.path.join(TMP_DATASET_PATH, "bev_lidar")
paths["bev_semantic"] = os.path.join(TMP_DATASET_PATH, "bev_semantic")
for i in range(8):
    paths[f"rgb_{i}"] = os.path.join(TMP_DATASET_PATH, rgb_folders_name[i])
    paths[f"depth_{i}"] = os.path.join(TMP_DATASET_PATH, depth_folders_name[i])
    paths[f"semantic_{i}"] = os.path.join(TMP_DATASET_PATH, semantic_folders_name[i])

for key_path in paths:
    os.mkdir(paths[key_path])

def cntrl_c(_, __):
    sensors["lidar"].stop()
    sensors["lidar"].destroy()
    sensors["bev_semantic"].stop()
    sensors["bev_semantic"].destroy()
    for i in range(8):
        sensors[f"rgb_{i}"].stop()
        sensors[f"rgb_{i}"].destroy()
        sensors[f"depth_{i}"].stop()
        sensors[f"depth_{i}"].destroy()
        sensors[f"semantic_{i}"].stop()
        sensors[f"semantic_{i}"].destroy()
    exit()

signal.signal(signal.SIGINT, cntrl_c)

carla_frame = 0
saved_frame = 0
print("Starting Loop!")
while True:
    if carla_frame > AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE:
        carla_frame = 0
        cv2.imwrite(os.path.join(paths["lidar"], f"{saved_frame}.png"), data_last_frame["lidar"])
        cv2.imwrite(os.path.join(paths["bev_semantic"], f"{saved_frame}.png"), data_last_frame["bev_semantic"])
        for i in range(8):
            cv2.imwrite(os.path.join(paths[f"rgb_{i}"], f"{saved_frame}.jpg"), data_last_frame[f"rgb_{i}"])
            cv2.imwrite(os.path.join(paths[f"depth_{i}"], f"{saved_frame}.png"), data_last_frame[f"depth_{i}"][:, :, 0])
            cv2.imwrite(os.path.join(paths[f"semantic_{i}"], f"{saved_frame}.jpg"), data_last_frame[f"semantic_{i}"][:, :, 2])
        for key_take_new_data in take_new_data:
            take_new_data[key_take_new_data] = True
        saved_frame += 1
    carla_frame += 1
    world.wait_for_tick()
    # if saved_frame == 40:
    #    cntrl_c(None, None)

