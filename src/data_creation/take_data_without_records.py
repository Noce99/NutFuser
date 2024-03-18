import sys
# sys.path.append("/leonardo_work/IscrC_SSNeRF/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
# sys.path.append("/home/enrico/Progetti/Carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import os
import math
import shutil
import signal
import time 
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

STARTING_FRAME = None
PATHS = {}
ALREADY_OBTAINED_DATA_FROM_SENSOR_A = [False for _ in range(18)] # rgb_A, depth, semantic, optical_flow, lidar, bev_semantic
ALREADY_OBTAINED_DATA_FROM_SENSOR_B = [False for _ in range(4)] # rgb_B
GPS_POSITIONS = []

def take_data_backbone(carla_egg_path, town_id, rpc_port, job_id, ego_vehicle_found_event, starting_data_loop_event, finished_taking_data_event, you_can_tick_event):
    
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass
    
    take_new_data = {}
    take_new_data["lidar"] = True
    take_new_data["bev_semantic"] = True
    take_new_data["bottom_bev_semantic"] = True
    UNIFIED_BEV_SEMANTIC = None
    for i in range(4):
        """
        from 0 to 3 -> obvious view
        """
        take_new_data[f"rgb_{i}"] = True
        take_new_data[f"depth_{i}"] = True
        take_new_data[f"semantic_{i}"] = True
        take_new_data[f"optical_flow_{i}"] = True

    # Connect the client and set up bp library
    client = carla.Client('localhost', rpc_port)
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
    ego_vehicle_found_event.set()

    global STARTING_FRAME
    STARTING_FRAME = world.tick()

    # LIDAR callback
    def lidar_callback(data):
        def lidar_to_histogram_features(lidar):
            """
            Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
            :param lidar: (N,3) numpy, LiDAR point cloud
            :return: (2, H, W) numpy, LiDAR as sparse image
            """
            MAX_HIST_POINTS = 5
            def splat_points(point_cloud):
                # 256 x 256 grid
                xbins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_IMAGE_W+1)
                ybins = np.linspace(-config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_SQUARE_SIDE_IN_M/2, config.BEV_IMAGE_H+1)
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
        
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            lidar_data = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
            lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 4), 4))
            lidar_data = lidar_to_histogram_features(lidar_data[:, :3])[0]
            lidar_data = np.rot90(lidar_data)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["lidar"], f"{saved_frame}.png"), lidar_data)

        
    # CAMERAS callback
    def rgb_callback(data, number):
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"rgb_B_{number}"], f"{saved_frame}.jpg"), rgb)
        elif (data.frame + 1 - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE + 1
            cv2.imwrite(os.path.join(PATHS[f"rgb_A_{number}"], f"{saved_frame}.jpg"), rgb)

    # DEPTH callback
    def depth_callback(data, number):
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            data.convert(carla.ColorConverter.LogarithmicDepth)
            depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            depth = depth[: , :, 0]
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"depth_{number}"], f"{saved_frame}.png"), depth)

    # SEMATIC callback
    def semantic_callback(data, number):
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            # semantic.convert(carla.ColorConverter.CityScapesPalette)
            semantic = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            semantic = semantic[:, :, 2]
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"semantic_{number}"], f"{saved_frame}.png"), unify_semantic_tags(semantic))

    # OPTICAL FLOW callback
    def optical_flow_callback(data, number):
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            optical_flow = np.copy(np.frombuffer(data.raw_data, dtype=np.float32))
            optical_flow =  np.reshape(optical_flow, (data.height, data.width, 2))
            optical_flow[:, :, 0] *= config.IMAGE_H * 0.5
            optical_flow[:, :, 1] *= config.IMAGE_W * 0.5
            optical_flow = 64.0 * optical_flow + 2**15 # This means maximum pixel distance of 512 
            valid = np.ones([optical_flow.shape[0], optical_flow.shape[1], 1])
            optical_flow = np.concatenate([optical_flow, valid], axis=-1).astype(np.uint16)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"optical_flow_{number}"], f"{saved_frame}.png"), optical_flow)
            
    # BEV SEMANTIC callback
    def bev_semantic_callback(data):
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            # bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
            top_bev_semantic = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            top_bev_semantic = top_bev_semantic[:, :, 2]
            top_bev_semantic = cv2.resize(top_bev_semantic, (config.BEV_IMAGE_H, config.BEV_IMAGE_W), interpolation= cv2.INTER_NEAREST_EXACT)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["bev_semantic"], f"top_{saved_frame}.png"), top_bev_semantic)

    # BOTTOM BEV SEMANTIC callback
    def bev_bottom_semantic_callback(data):
        if (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            take_new_data["bottom_bev_semantic"] = False
            # bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
            bottom_bev_semantic = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            bottom_bev_semantic = bottom_bev_semantic[:, :, 2]
            bottom_bev_semantic = cv2.resize(bottom_bev_semantic, (config.BEV_IMAGE_H, config.BEV_IMAGE_W), interpolation= cv2.INTER_NEAREST_EXACT)
            bottom_bev_semantic =  np.flip(bottom_bev_semantic, 1)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["bev_semantic"], f"bottom_{saved_frame}.png"), bottom_bev_semantic)
    
    # GPS callback
    def gps_callback(data):
        print(f"gps {data.frame}")
        GPS_POSITIONS.append((data.latitude, data.longitude, data.altitude))

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

    # GPS
    gps_bp = bp_lib.find("sensor.other.gnss")

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
    sensors["gps"] = world.spawn_actor(gps_bp, lidar_init_trans, attach_to=hero)

    # Connect Sensor and Callbacks
    sensors["lidar"].listen(lambda data: lidar_callback(data))
    sensors["bev_semantic"].listen(lambda data: bev_semantic_callback(data))
    sensors["bottom_bev_semantic"].listen(lambda data: bev_bottom_semantic_callback(data))
    for i in range(4):
        sensors[f"rgb_{i}"].listen(lambda image, i=i: rgb_callback(image, i))
        sensors[f"depth_{i}"].listen(lambda depth, i=i: depth_callback(depth, i))
        sensors[f"semantic_{i}"].listen(lambda semantic, i=i: semantic_callback(semantic, i))
        sensors[f"optical_flow_{i}"].listen(lambda optical_flow, i=i: optical_flow_callback(optical_flow, i))
    sensors["gps"].listen(lambda data: gps_callback(data))

    # Create Directory Branches

    #if os.path.isdir(config.DATASET_PATH):
    #    shutil.rmtree(config.DATASET_PATH)
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H:%M:%S")

    config.DATASET_PATH = os.path.join(config.DATASET_PATH, f"{job_id}_{current_time}_{config.TOWN_DICT[town_id]}")
    os.mkdir(config.DATASET_PATH)

    rgb_A_folders_name =        [f"rgb_A_{i}"           for i in range(4)]
    rgb_B_folders_name =        [f"rgb_B_{i}"           for i in range(4)]
    depth_folders_name =        [f"depth_{i}"           for i in range(4)]
    semantic_folders_name =     [f"semantic_{i}"        for i in range(4)]
    optical_flow_folders_name = [f"optical_flow_{i}"    for i in range(4)]

    global PATHS
    PATHS["lidar"] = os.path.join(config.DATASET_PATH, "bev_lidar")
    PATHS["bev_semantic"] = os.path.join(config.DATASET_PATH, "bev_semantic")
    for i in range(4):
        PATHS[f"rgb_A_{i}"] = os.path.join(config.DATASET_PATH, rgb_A_folders_name[i])
        PATHS[f"rgb_B_{i}"] = os.path.join(config.DATASET_PATH, rgb_B_folders_name[i])
        PATHS[f"depth_{i}"] = os.path.join(config.DATASET_PATH, depth_folders_name[i])
        PATHS[f"semantic_{i}"] = os.path.join(config.DATASET_PATH, semantic_folders_name[i])
        PATHS[f"optical_flow_{i}"] = os.path.join(config.DATASET_PATH, optical_flow_folders_name[i])

    for key_path in PATHS:
        os.mkdir(PATHS[key_path])

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

    def merge_semantics(bev_semantic, bottom_bev_semantic):
        """
        We keep:
        0 -> uknown
        1 -> road
        2 -> terrain where the car should not go
        3 -> line_on_asphalt + written staff on the ground
        4 -> vehicles
        5 -> pedestrian
        """
        unified_bev_semantic = np.zeros((config.BEV_IMAGE_H, config.BEV_IMAGE_W), dtype=np.uint8)

        # road
        unified_bev_semantic[bev_semantic == 1] = 1
        # terrain where the car should not go
        unified_bev_semantic[bev_semantic == 2] = 2
        unified_bev_semantic[bev_semantic == 10] = 2
        unified_bev_semantic[bev_semantic == 25] = 2
        unified_bev_semantic[bev_semantic == 23] = 2
        # line_on_asphalt + written staff on the ground
        unified_bev_semantic[bev_semantic == 24] = 3
        # vehicles
        unified_bev_semantic[bev_semantic == 13] = 4
        unified_bev_semantic[bev_semantic == 14] = 4
        unified_bev_semantic[bev_semantic == 15] = 4
        unified_bev_semantic[bev_semantic == 16] = 4
        unified_bev_semantic[bev_semantic == 17] = 4
        unified_bev_semantic[bev_semantic == 18] = 4
        unified_bev_semantic[bev_semantic == 19] = 4
        unified_bev_semantic[bottom_bev_semantic == 13] = 4
        unified_bev_semantic[bottom_bev_semantic == 14] = 4
        unified_bev_semantic[bottom_bev_semantic == 15] = 4
        unified_bev_semantic[bottom_bev_semantic == 16] = 4
        unified_bev_semantic[bottom_bev_semantic == 17] = 4
        unified_bev_semantic[bottom_bev_semantic == 18] = 4
        unified_bev_semantic[bottom_bev_semantic == 19] = 4
        # pedestrian
        unified_bev_semantic[bev_semantic == 12] = 5

        return unified_bev_semantic

    def unify_semantic_tags(semantic):
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
        unified_semantic = np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)
        # road
        unified_semantic[semantic == 1] = 1
        # terrain where the car should not go
        unified_semantic[semantic == 2] = 2
        unified_semantic[semantic == 10] = 2
        unified_semantic[semantic == 25] = 2
        unified_semantic[semantic == 23] = 2
        # line_on_asphalt + written staff on the ground
        unified_semantic[semantic == 24] = 3
        # vehicles
        unified_semantic[semantic == 13] = 4
        unified_semantic[semantic == 14] = 4
        unified_semantic[semantic == 15] = 4
        unified_semantic[semantic == 16] = 4
        unified_semantic[semantic == 17] = 4
        unified_semantic[semantic == 18] = 4
        unified_semantic[semantic == 19] = 4
        # pedestrian
        unified_semantic[semantic == 12] = 5
        # sign
        unified_semantic[semantic == 8] = 6
        # traffic lights
        unified_semantic[semantic == 7] = 7
        return unified_semantic

    signal.signal(signal.SIGINT, cntrl_c)

    print("Starting Data Loop!")
    starting_data_loop_event.set()
    you_can_tick_event.set()
    with tqdm(total=config.MAX_NUM_OF_SAVED_FRAME*config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE) as pbar:
        while True:
            world_snapshot = world.wait_for_tick()
            if (world_snapshot.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
                while True:
                    if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR_A) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR_A):
                        break
            elif (world_snapshot.frame + 1 - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
                while True:
                    if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR_B) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR_B):
                        break
            you_can_tick_event.set()
            pbar.update(1)
            if (world_snapshot.frame - STARTING_FRAME) > config.MAX_NUM_OF_SAVED_FRAME*config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE:
                break
    world.wait_for_tick()
    finished_taking_data_event.set()
    print("Unifing Top and Bottom BEV!")
    for i in tqdm(range(1, config.MAX_NUM_OF_SAVED_FRAME+1)):
        top_bev_semantic = cv2.imread(os.path.join(PATHS["bev_semantic"], f"top_{i}.png"))[:, :, 0]
        bottom_bev_semantic = cv2.imread(os.path.join(PATHS["bev_semantic"], f"bottom_{i}.png"))[:, :, 0]
        bev_semantic = merge_semantics(top_bev_semantic, bottom_bev_semantic)
        cv2.imwrite(os.path.join(PATHS["bev_semantic"], f"{i}.png"), bev_semantic)
        os.remove(os.path.join(PATHS["bev_semantic"], f"top_{i}.png"))
        os.remove(os.path.join(PATHS["bev_semantic"], f"bottom_{i}.png"))
    print("Saving GPS locations!")
    gps_array = np.array(GPS_POSITIONS)
    np.save(os.path.join(os.path.join(config.DATASET_PATH), "gps.npy"), gps_array)

