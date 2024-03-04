import sys
sys.path.append("/leonardo_work/IscrC_SSNeRF/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla 
import os
import math
import shutil
import signal
import time 
import numpy as np
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

from get_bbs import get_bbs_as_bev_image

if len(sys.argv) == 1:
    print("No argumnets I will setup Town15, port 20000 and job_id to 0!")
    town_int = 15
    carla_port = 20000
    job_id = 0
else:
    try:
        town_int = int(sys.argv[1])
        if town_int not in config.TOWN_DICT.keys():
            print(f"This town does not exist [{town_int}], I will setup Town15!")
            town_int = 15
        else:
            print(f"The argument was an integer! [{town_int}] -> {config.TOWN_DICT[town_int]}")
    except Exception as e:
        print(e)
        print(f"The argument was not an integer [{sys.argv[1]}], I will setup Town15!")
        town_int = 15
    if len(sys.argv) == 2:
        print("No port given! I will set it to 20000!")
        carla_port = 20000
    else:
        carla_port = int(sys.argv[2])
    if len(sys.argv) == 3:
        print("No id given! I will set it to 0!")
        job_id = 0
    else:
        job_id = sys.argv[3]

data_last_frame = {}
take_new_data = {}
data_last_frame["lidar"] = np.zeros((config.BEV_IMAGE_H, config.BEV_IMAGE_W), dtype=np.uint8)
take_new_data["lidar"] = True
data_last_frame["bev_semantic"] = np.zeros((config.BEV_IMAGE_H, config.BEV_IMAGE_W), dtype=np.uint8)
take_new_data["bev_semantic"] = True
data_last_frame["bottom_bev_semantic"] = np.zeros((config.BEV_IMAGE_H, config.BEV_IMAGE_W), dtype=np.uint8)
take_new_data["bottom_bev_semantic"] = True
UNIFIED_BEV_SEMANTIC = None
for i in range(4):
    """
    from 0 to 3 -> obvious view
    """
    data_last_frame[f"rgb_{i}"] = np.zeros((config.IMAGE_H, config.IMAGE_W, 4), dtype=np.uint8)
    take_new_data[f"rgb_{i}"] = True
    data_last_frame[f"depth_{i}"] = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)
    take_new_data[f"depth_{i}"] = True
    data_last_frame[f"semantic_{i}"] = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)
    take_new_data[f"semantic_{i}"] = True
    data_last_frame[f"optical_flow_{i}"] = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)
    take_new_data[f"optical_flow_{i}"] = True
#    data_last_frame[f"optical_flow_human_{i}"] = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)

# Connect the client and set up bp library
client = carla.Client('localhost', carla_port)
client.set_timeout(20.0)
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
# In this case, the simulator will take 20 steps (1/0.05) to recreate one second of
# the simulated world.
settings.substepping = True
settings.max_substep_delta_time = 0.01
settings.max_substeps = 10
# fixed_delta_seconds <= max_substep_delta_time * max_substeps
# In order to have an optimal physical simulation,
# the substep delta time should at least be below 0.01666 and ideally below 0.01.
world.apply_settings(settings)
bp_lib = world.get_blueprint_library()

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
            xbins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_IMAGE_W)
            ybins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_IMAGE_H)
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
    print(f"OCIIOOOOOOO {data_last_frame['lidar'].shape} dovrebbe essere [256, 256]!!!!!!!")

# CAMERAS callback
def rgb_callback(image, number):
    if take_new_data[f"rgb_{number}"]:
        global data_last_frame
        data_last_frame[f"rgb_{number}"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        take_new_data[f"rgb_{number}"] = False

# DEPTH callback
def depth_callback(depth, number):
    if take_new_data[f"depth_{number}"]:
        take_new_data[f"depth_{number}"] = False
        global data_last_frame
        depth.convert(carla.ColorConverter.LogarithmicDepth)
        depth = np.reshape(np.copy(depth.raw_data), (depth.height, depth.width, 4))
        data_last_frame[f"depth_{number}"] = depth[: , :, 0]

# SEMATIC callback
def semantic_callback(semantic, number):
    if take_new_data[f"semantic_{number}"]:
        take_new_data[f"semantic_{number}"] = False
        global data_last_frame
        # semantic.convert(carla.ColorConverter.CityScapesPalette)
        semantic = np.reshape(np.copy(semantic.raw_data), (semantic.height, semantic.width, 4))
        data_last_frame[f"semantic_{number}"] = semantic[:, :, 2]

# OPTICAL FLOW callback
def optical_flow_callback(optical_flow, number):
    if take_new_data[f"optical_flow_{number}"]:
        take_new_data[f"optical_flow_{number}"] = False
#        optical_flow_human = optical_flow.get_color_coded_flow() # JUST TEMPORANEALLY
#        optical_flow_human = np.reshape(np.copy(optical_flow_human.raw_data), (config.IMAGE_H, config.IMAGE_W, 4))
#        optical_flow_human[: , :, 3] = 255
        global data_last_frame
        array_optical_flow = np.copy(np.frombuffer(optical_flow.raw_data, dtype=np.float32))
        optical_flow =  np.reshape(array_optical_flow, (optical_flow.height, optical_flow.width, 2))
        optical_flow[:, :, 0] *= config.IMAGE_H * 0.5
        optical_flow[:, :, 1] *= config.IMAGE_W * 0.5
        optical_flow = 64.0 * optical_flow + 2**15 # This means maximum pixel distance of 512 
        valid = np.ones([optical_flow.shape[0], optical_flow.shape[1], 1])
        optical_flow = np.concatenate([optical_flow, valid], axis=-1).astype(np.uint16)
        # print(f"0 [{np.min(optical_flow[:, :, 0])}; {np.max(optical_flow[:, :, 0])}]")
        # print(f"1 [{np.min(optical_flow[:, :, 1])}; {np.max(optical_flow[:, :, 1])}]")
        data_last_frame[f"optical_flow_{number}"] = optical_flow
#        data_last_frame[f"optical_flow_human_{number}"] = optical_flow_human

# BEV SEMANTIC callback
def bev_semantic_callback(bev_semantic):
    if take_new_data["bev_semantic"]:
        take_new_data["bev_semantic"] = False
        global data_last_frame
        # bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
        bev_semantic = np.reshape(np.copy(bev_semantic.raw_data), (bev_semantic.height, bev_semantic.width, 4))
        bev_semantic = bev_semantic[:, :, 2]
        data_last_frame["bev_semantic"] = cv2.resize(bev_semantic, (config.BEV_IMAGE_H, config.BEV_IMAGE_W), interpolation= cv2.INTER_NEAREST_EXACT)


# BOTTOM BEV SEMANTIC callback
def bev_bottom_semantic_callback(bev_semantic):
    if take_new_data["bottom_bev_semantic"]:
        take_new_data["bottom_bev_semantic"] = False
        global data_last_frame
        # bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
        bev_semantic = np.reshape(np.copy(bev_semantic.raw_data), (bev_semantic.height, bev_semantic.width, 4))
        bev_semantic = bev_semantic[:, :, 2]
        bev_semantic = cv2.resize(bev_semantic, (config.BEV_IMAGE_H, config.BEV_IMAGE_W), interpolation= cv2.INTER_NEAREST_EXACT)
        data_last_frame["bottom_bev_semantic"] = np.flip(bev_semantic, 1)



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
camera_bp.set_attribute("image_size_x", f"{config.IMAGE_W}")
camera_bp.set_attribute("image_size_y", f"{config.IMAGE_H}")

# DEPTH CAMERAS
depth_bp = bp_lib.find("sensor.camera.depth") 
depth_bp.set_attribute("fov", "90")
depth_bp.set_attribute("image_size_x", f"{config.IMAGE_W}")
depth_bp.set_attribute("image_size_y", f"{config.IMAGE_H}")

# SEMANTIC CAMERAS
semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation") 
semantic_bp.set_attribute("fov", "90")
semantic_bp.set_attribute("image_size_x", f"{config.IMAGE_W}")
semantic_bp.set_attribute("image_size_y", f"{config.IMAGE_H}")

# OPTICAL FLOW
optical_flow_bp = bp_lib.find("sensor.camera.optical_flow") 
optical_flow_bp.set_attribute("fov", f"90")
optical_flow_bp.set_attribute("image_size_x", f"{config.IMAGE_W}")
optical_flow_bp.set_attribute("image_size_y", f"{config.IMAGE_H}")

# BEV SEMANTIC
bev_semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation") 
bev_semantic_bp.set_attribute("fov", f"{config.BEV_FOV_IN_DEGREES}")
bev_semantic_bp.set_attribute("image_size_x", f"{config.BEV_IMAGE_W*4}")
bev_semantic_bp.set_attribute("image_size_y", f"{config.BEV_IMAGE_H*4}")

# BOTTOM BEV SEMANTIC
bottom_bev_semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation") 
bottom_bev_semantic_bp.set_attribute("fov", f"{config.BEV_BOTTOM_FOV_IN_DEGREES}")
bottom_bev_semantic_bp.set_attribute("image_size_x", f"{config.BEV_IMAGE_W*4}")
bottom_bev_semantic_bp.set_attribute("image_size_y", f"{config.BEV_IMAGE_H*4}")

# Amazing CAMERAS
transformations = []

# Obvious CAMERAS
transformations.append(carla.Transform(carla.Location(x=1.0, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=0.0, roll=0, yaw=0)))
transformations.append(carla.Transform(carla.Location(x=0.0, y=-1.0, z=2.0),
                                     carla.Rotation(pitch=-5.0, roll=0, yaw=90)))
transformations.append(carla.Transform(carla.Location(x=-1.0, y=+0.0, z=2.0),
                                     carla.Rotation(pitch=0.0, roll=0, yaw=180)))
transformations.append(carla.Transform(carla.Location(x=+0.0, y=1.0, z=2.0),
                                     carla.Rotation(pitch=-5.0, roll=0, yaw=270)))

bev_transformation = carla.Transform(carla.Location(x=+0.0, y=0.0, z=config.BEV_ALTITUDE),
                                     carla.Rotation(pitch=-90, roll=0, yaw=0))

bottom_bev_transformation = carla.Transform(carla.Location(x=+0.0, y=0.0, z=-config.BEV_BOTTOM_ALTITUDE),
                                     carla.Rotation(pitch=90, roll=180, yaw=0))

sensors = {}
sensors["lidar"] = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=hero)
sensors["bev_semantic"] = world.spawn_actor(bev_semantic_bp, bev_transformation, attach_to=hero)
sensors["bottom_bev_semantic"] = world.spawn_actor(bottom_bev_semantic_bp, bottom_bev_transformation, attach_to=hero)
for i in range(4):
    sensors[f"rgb_{i}"] = world.spawn_actor(camera_bp, transformations[i] , attach_to=hero)
    sensors[f"depth_{i}"] = world.spawn_actor(depth_bp, transformations[i], attach_to=hero)
    sensors[f"semantic_{i}"] = world.spawn_actor(semantic_bp, transformations[i], attach_to=hero)
    sensors[f"optical_flow_{i}"] = world.spawn_actor(optical_flow_bp, transformations[i], attach_to=hero)

# Connect Sensor and Callbacks
sensors["lidar"].listen(lambda data: lidar_callback(data))
sensors["bev_semantic"].listen(lambda data: bev_semantic_callback(data))
sensors["bottom_bev_semantic"].listen(lambda data: bev_bottom_semantic_callback(data))
for i in range(4):
    sensors[f"rgb_{i}"].listen(lambda image, i=i: rgb_callback(image, i))
    sensors[f"depth_{i}"].listen(lambda depth, i=i: depth_callback(depth, i))
    sensors[f"semantic_{i}"].listen(lambda semantic, i=i: semantic_callback(semantic, i))
    sensors[f"optical_flow_{i}"].listen(lambda optical_flow, i=i: optical_flow_callback(optical_flow, i))
# world.on_tick(lambda world_snapshot: bbs_callback(world_snapshot)) # On Tick callback for bounding boxes

# Create Directory Branches

#if os.path.isdir(config.DATASET_PATH):
#    shutil.rmtree(config.DATASET_PATH)
now = datetime.now()
current_time = now.strftime("%d_%m_%Y_%H:%M:%S")

config.DATASET_PATH = os.path.join(config.DATASET_PATH, f"{job_id}_{current_time}_{config.TOWN_DICT[town_int]}")
os.mkdir(config.DATASET_PATH)

rgb_A_folders_name =        [f"rgb_A_{i}"           for i in range(4)]
rgb_B_folders_name =        [f"rgb_B_{i}"           for i in range(4)]
depth_folders_name =        [f"depth_{i}"           for i in range(4)]
semantic_folders_name =     [f"semantic_{i}"        for i in range(4)]
optical_flow_folders_name = [f"optical_flow_{i}"    for i in range(4)]

paths = {}
paths["lidar"] = os.path.join(config.DATASET_PATH, "bev_lidar")
paths["bev_semantic"] = os.path.join(config.DATASET_PATH, "bev_semantic")
for i in range(4):
    paths[f"rgb_A_{i}"] = os.path.join(config.DATASET_PATH, rgb_A_folders_name[i])
    paths[f"rgb_B_{i}"] = os.path.join(config.DATASET_PATH, rgb_B_folders_name[i])
    paths[f"depth_{i}"] = os.path.join(config.DATASET_PATH, depth_folders_name[i])
    paths[f"semantic_{i}"] = os.path.join(config.DATASET_PATH, semantic_folders_name[i])
    paths[f"optical_flow_{i}"] = os.path.join(config.DATASET_PATH, optical_flow_folders_name[i])

for key_path in paths:
    os.mkdir(paths[key_path])

def cntrl_c(_, __):
    sensors["lidar"].stop()
    sensors["lidar"].destroy()
    sensors["bev_semantic"].stop()
    sensors["bev_semantic"].destroy()
    sensors["bottom_bev_semantic"].stop()
    sensors["bottom_bev_semantic"].destroy()
    for i in range(4):
        sensors[f"rgb_{i}"].stop()
        sensors[f"rgb_{i}"].destroy()
        sensors[f"depth_{i}"].stop()
        sensors[f"depth_{i}"].destroy()
        sensors[f"semantic_{i}"].stop()
        sensors[f"semantic_{i}"].destroy()
        sensors[f"optical_flow_{i}"].stop()
        sensors[f"optical_flow_{i}"].destroy()
    exit()

def merge_semantics():
    """
    We keep:
    0 -> uknown
    1 -> road
    2 -> terrain where the car should not go
    3 -> line_on_asphalt + written staff on the ground
    4 -> vehicles
    5 -> pedestrian
    """
    global UNIFIED_BEV_SEMANTIC
    UNIFIED_BEV_SEMANTIC = np.zeros((config.BEV_IMAGE_H, config.BEV_IMAGE_W), dtype=np.uint8)

    # road
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 1] = 1
    # terrain where the car should not go
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 2] = 2
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 10] = 2
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 25] = 2
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 23] = 2
    # line_on_asphalt + written staff on the ground
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 24] = 3
    # vehicles
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 13] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 14] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 15] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 16] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 17] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 18] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 19] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 13] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 14] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 15] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 16] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 17] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 18] = 4
    UNIFIED_BEV_SEMANTIC[data_last_frame["bottom_bev_semantic"] == 19] = 4
    # pedestrian
    UNIFIED_BEV_SEMANTIC[data_last_frame["bev_semantic"] == 12] = 5

def unify_semantic_tags():
    """
    We keep:
    0 -> uknown
    1 -> road
    2 -> terrain where the car should not go
    3 -> line_on_asphalt + written staff on the ground
    4 -> vehicles
    5 -> pedestrian
    6 -> sign
    7 -> traffic lights
    """
    for i in range(4):
        UNIFIED_SEMANTIC = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)
        # road
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 1] = 1
        # terrain where the car should not go
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 2] = 2
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 10] = 2
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 25] = 2
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 23] = 2
        # line_on_asphalt + written staff on the ground
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 24] = 3
        # vehicles
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 13] = 4
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 14] = 4
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 15] = 4
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 16] = 4
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 17] = 4
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 18] = 4
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 19] = 4
        # pedestrian
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 12] = 5
        # sign
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 8] = 6
        # traffic lights
        UNIFIED_SEMANTIC[data_last_frame[f"semantic_{i}"] == 7] = 7
        data_last_frame[f"semantic_{i}"] = UNIFIED_SEMANTIC

signal.signal(signal.SIGINT, cntrl_c)

carla_frame = 0
saved_frame = 0
world.tick()
for key_take_new_data in take_new_data:
            take_new_data[key_take_new_data] = True
world.tick()
print("Starting Data Loop!")
while True:
    if carla_frame == 0:
        for key_take_new_data in take_new_data:
            if key_take_new_data[:3] == "rgb":
                take_new_data[key_take_new_data] = True
    elif carla_frame == 1:
        for i in range(4):
            cv2.imwrite(os.path.join(paths[f"rgb_A_{i}"], f"{saved_frame}.jpg"), data_last_frame[f"rgb_{i}"])
        for key_take_new_data in take_new_data:
            take_new_data[key_take_new_data] = True
    elif carla_frame == config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE:
        carla_frame = -1
        cv2.imwrite(os.path.join(paths["lidar"], f"{saved_frame}.png"), data_last_frame["lidar"])
        merge_semantics()
        cv2.imwrite(os.path.join(paths["bev_semantic"], f"{saved_frame}.png"), UNIFIED_BEV_SEMANTIC)
        unify_semantic_tags()
        for i in range(4):
            cv2.imwrite(os.path.join(paths[f"rgb_B_{i}"], f"{saved_frame}.jpg"), data_last_frame[f"rgb_{i}"])
            cv2.imwrite(os.path.join(paths[f"depth_{i}"], f"{saved_frame}.png"), data_last_frame[f"depth_{i}"])
            cv2.imwrite(os.path.join(paths[f"semantic_{i}"], f"{saved_frame}.png"), data_last_frame[f"semantic_{i}"])
            cv2.imwrite(os.path.join(paths[f"optical_flow_{i}"], f"{saved_frame}.png"), data_last_frame[f"optical_flow_{i}"])
#            cv2.imwrite(os.path.join(paths[f"optical_flow_{i}"], f"h_{saved_frame}.png"), data_last_frame[f"optical_flow_human_{i}"])
        saved_frame += 1
    carla_frame += 1
    if saved_frame > 400:
        exit()
    # world.tick()
    world.wait_for_tick()

