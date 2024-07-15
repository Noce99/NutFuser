import sys
# sys.path.append("/leonardo_work/IscrC_SSNeRF/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
# sys.path.append("/home/enrico/Progetti/Carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import os
import math
import signal
import time
import numpy as np
import cv2
from tqdm import tqdm
import json

from nutfuser import config
from nutfuser import utils
from nutfuser.data_creation.weather import get_a_random_weather

from carla_birdeye_view import (
    BirdViewProducer,
    BirdView,
    BirdViewCropType,
)
from carla_birdeye_view.mask import PixelDimensions

STARTING_FRAME = None
PATHS = {}
ALREADY_OBTAINED_DATA_FROM_SENSOR_A = []
ALREADY_OBTAINED_DATA_FROM_SENSOR_B = []
ALREADY_OBTAINED_ALL_GPS = False
FRAME_GPS_POSITIONS = []
ALL_GPS_POSITIONS = []
FRAME_COMPASS = []

DISABLE_ALL_SENSORS = False
KEEP_GPS = False


def take_data_backbone(carla_egg_path, town_id, rpc_port, job_id, ego_vehicle_found_event, finished_taking_data_event,
                       you_can_tick_event, how_many_frames, where_to_save, back_camera, lateral_cameras, tfpp_inputs):
    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass

    # Setup witch cameras is present
    cameras_indexes = [0]
    if back_camera:
        cameras_indexes.append(2)
    if lateral_cameras:
        cameras_indexes.append(1)
        cameras_indexes.append(3)
    global ALREADY_OBTAINED_DATA_FROM_SENSOR_A
    ALREADY_OBTAINED_DATA_FROM_SENSOR_A = {f"rgb_A_{i}": False for i in cameras_indexes}
    global ALREADY_OBTAINED_DATA_FROM_SENSOR_B
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = {f"rgb_B_{i}": False for i in cameras_indexes}
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = dict({f"depth_{i}": False for i in cameras_indexes},
                                               **ALREADY_OBTAINED_DATA_FROM_SENSOR_B)
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = dict({f"semantic_{i}": False for i in cameras_indexes},
                                               **ALREADY_OBTAINED_DATA_FROM_SENSOR_B)
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = dict({f"optical_flow_{i}": False for i in cameras_indexes},
                                               **ALREADY_OBTAINED_DATA_FROM_SENSOR_B)
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = dict(
        {"lidar": False, "bev_semantic_top": False, "bev_semantic_bottom": False, "frame_gps_position": False,
         "compass": False}, **ALREADY_OBTAINED_DATA_FROM_SENSOR_B)
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = dict({"bev_semantic_2": False}, **ALREADY_OBTAINED_DATA_FROM_SENSOR_B)

    if tfpp_inputs:
        ALREADY_OBTAINED_DATA_FROM_SENSOR_B["rgb_tfpp"] = False
        ALREADY_OBTAINED_DATA_FROM_SENSOR_B["lidar_tfpp"] = False

    global ALREADY_OBTAINED_ALL_GPS
    ALREADY_OBTAINED_ALL_GPS = True

    # Connect the client and set up bp library
    client = carla.Client('localhost', rpc_port)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1. / config.CARLA_FPS
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

    # I will create the bev semantic generator
    birdview_producer = BirdViewProducer(
        client,
        PixelDimensions(width=config.BEV_IMAGE_W, height=config.BEV_IMAGE_H),
        pixels_per_meter=int(config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M),
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        render_lanes_on_junctions=True,
    )

    # I will set a random weather
    a_random_weather, weather_dict = get_a_random_weather()
    world.set_weather(a_random_weather)

    # LIDAR callback
    def lidar_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            lidar_data_raw = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
            lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))

            # MY LIDAR
            lidar_data = utils.lidar_to_histogram_features(lidar_data_raw[:, :3])[0]
            lidar_data = np.rot90(lidar_data)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["lidar"], f"{saved_frame}.png"), lidar_data)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["lidar"] = True

            # TFPP ORIGINAL LIDAR
            if tfpp_inputs:
                lidar_data = utils.lidar_to_histogram_features_tfpp_original(lidar_data_raw[:, :3])[0]
                cv2.imwrite(os.path.join(PATHS["lidar_tfpp"], f"{saved_frame}.png"), lidar_data)
                ALREADY_OBTAINED_DATA_FROM_SENSOR_B["lidar_tfpp"] = True

    # CAMERAS callback
    def rgb_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"rgb_B_{number}"], f"{saved_frame}.jpg"), rgb)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B[f"rgb_B_{number}"] = True
        elif not DISABLE_ALL_SENSORS and (
                data.frame + 1 - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"rgb_A_{number}"], f"{saved_frame}.jpg"), rgb)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_A[f"rgb_A_{number}"] = True

    # DEPTH callback
    def depth_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            data.convert(carla.ColorConverter.LogarithmicDepth)
            depth = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            depth = depth[:, :, 0]
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"depth_{number}"], f"{saved_frame}.png"), depth)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B[f"depth_{number}"] = True

    # SEMATIC callback
    def semantic_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            # semantic.convert(carla.ColorConverter.CityScapesPalette)
            semantic = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            semantic = semantic[:, :, 2]
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"semantic_{number}"], f"{saved_frame}.png"), unify_semantic_tags(semantic))
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B[f"semantic_{number}"] = True

    # OPTICAL FLOW callback
    def optical_flow_callback(data, number):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            optical_flow = np.copy(np.frombuffer(data.raw_data, dtype=np.float32))
            optical_flow = np.reshape(optical_flow, (data.height, data.width, 2))
            optical_flow[:, :, 0] *= config.IMAGE_H * 0.5
            optical_flow[:, :, 1] *= config.IMAGE_W * 0.5
            optical_flow = 64.0 * optical_flow + 2 ** 15  # This means maximum pixel distance of 512 (2**15/64=512)
            valid = np.ones([optical_flow.shape[0], optical_flow.shape[1], 1])
            optical_flow = np.concatenate([optical_flow, valid], axis=-1).astype(np.uint16)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS[f"optical_flow_{number}"], f"{saved_frame}.png"), optical_flow)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B[f"optical_flow_{number}"] = True

    # BEV SEMANTIC callback
    def bev_semantic_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            # bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
            top_bev_semantic = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            top_bev_semantic = top_bev_semantic[:, :, 2]
            top_bev_semantic = cv2.resize(top_bev_semantic, (config.BEV_IMAGE_H, config.BEV_IMAGE_W),
                                          interpolation=cv2.INTER_NEAREST_EXACT)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["bev_semantic"], f"top_{saved_frame}.png"), top_bev_semantic)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["bev_semantic_top"] = True

    # BOTTOM BEV SEMANTIC callback
    def bev_bottom_semantic_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            # bev_semantic.convert(carla.ColorConverter.CityScapesPalette)
            bottom_bev_semantic = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            bottom_bev_semantic = bottom_bev_semantic[:, :, 2]
            bottom_bev_semantic = cv2.resize(bottom_bev_semantic, (config.BEV_IMAGE_H, config.BEV_IMAGE_W),
                                             interpolation=cv2.INTER_NEAREST_EXACT)
            bottom_bev_semantic = np.flip(bottom_bev_semantic, 1)
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["bev_semantic"], f"bottom_{saved_frame}.png"), bottom_bev_semantic)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["bev_semantic_bottom"] = True

    # GPS callback
    def gps_callback(data):
        global ALREADY_OBTAINED_ALL_GPS
        if not DISABLE_ALL_SENSORS or KEEP_GPS:
            ALL_GPS_POSITIONS.append((data.latitude, data.longitude, data.altitude))
            ALREADY_OBTAINED_ALL_GPS = True
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            FRAME_GPS_POSITIONS.append((data.latitude, data.longitude, data.altitude))
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["frame_gps_position"] = True

    # IMU callback
    def imu_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            FRAME_COMPASS.append(data.compass)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["compass"] = True

    # TFPP rgb callback
    def rgb_tfpp_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            rgb = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
            saved_frame = (data.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["rgb_tfpp"], f"{saved_frame}.jpg"), rgb)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["rgb_tfpp"] = True

    # SEMANTIC BEV 2.0 callback
    def bev_semantic_callback_2(my_world_snapshot):
        if not DISABLE_ALL_SENSORS and (
                my_world_snapshot.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            birdview, bbs = birdview_producer.produce(agent_vehicle=hero)
            index_image = BirdViewProducer.as_carla_semantic(birdview)
            saved_frame = (my_world_snapshot.frame - STARTING_FRAME) // config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
            cv2.imwrite(os.path.join(PATHS["bev_semantic_2"], f"{saved_frame}.png"), index_image)
            utils.save_bbs_in_json(os.path.join(PATHS["bounding_boxes"], f"{saved_frame}.json"), bbs)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["bev_semantic_2"] = True

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
    bev_semantic_bp.set_attribute("image_size_x", f"{config.BEV_IMAGE_W * 4}")
    bev_semantic_bp.set_attribute("image_size_y", f"{config.BEV_IMAGE_H * 4}")

    # BOTTOM BEV SEMANTIC
    bottom_bev_semantic_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    bottom_bev_semantic_bp.set_attribute("fov", f"{config.BEV_BOTTOM_FOV_IN_DEGREES}")
    bottom_bev_semantic_bp.set_attribute("image_size_x", f"{config.BEV_IMAGE_W * 4}")
    bottom_bev_semantic_bp.set_attribute("image_size_y", f"{config.BEV_IMAGE_H * 4}")

    # GPS
    gps_bp = bp_lib.find("sensor.other.gnss")

    # IMU
    imu_bp = bp_lib.find("sensor.other.imu")

    # RGB TFPP
    rgb_tfpp_bp = bp_lib.find("sensor.camera.rgb")
    rgb_tfpp_bp.set_attribute("fov", "110")
    rgb_tfpp_bp.set_attribute("image_size_x", f"{config.IMAGE_W}")
    rgb_tfpp_bp.set_attribute("image_size_y", f"{config.IMAGE_H}")

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

    # SEMANTIC BEV
    bev_transformation = carla.Transform(carla.Location(x=+0.0, y=0.0, z=config.BEV_ALTITUDE),
                                         carla.Rotation(pitch=-90, roll=0, yaw=0))

    bottom_bev_transformation = carla.Transform(carla.Location(x=+0.0, y=0.0, z=-config.BEV_BOTTOM_ALTITUDE),
                                                carla.Rotation(pitch=90, roll=180, yaw=0))

    # RGBTFPP
    rgb_tfpp_transformation = carla.Transform(carla.Location(x=-1.5, y=0, z=2.0),
                                              carla.Rotation(pitch=0.0, roll=0, yaw=0))

    sensors = {}
    sensors["lidar"] = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=hero)
    sensors["bev_semantic"] = world.spawn_actor(bev_semantic_bp, bev_transformation, attach_to=hero)
    sensors["bottom_bev_semantic"] = world.spawn_actor(bottom_bev_semantic_bp, bottom_bev_transformation,
                                                       attach_to=hero)
    for i in cameras_indexes:
        sensors[f"rgb_{i}"] = world.spawn_actor(camera_bp, transformations[i], attach_to=hero)
        sensors[f"depth_{i}"] = world.spawn_actor(depth_bp, transformations[i], attach_to=hero)
        sensors[f"semantic_{i}"] = world.spawn_actor(semantic_bp, transformations[i], attach_to=hero)
        sensors[f"optical_flow_{i}"] = world.spawn_actor(optical_flow_bp, transformations[i], attach_to=hero)
    sensors["gps"] = world.spawn_actor(gps_bp, lidar_init_trans, attach_to=hero)
    sensors["imu"] = world.spawn_actor(imu_bp, lidar_init_trans, attach_to=hero)
    if tfpp_inputs:
        sensors["rgb_tfpp"] = world.spawn_actor(rgb_tfpp_bp, rgb_tfpp_transformation, attach_to=hero)

    # Connect Sensor and Callbacks
    sensors["lidar"].listen(lambda data: lidar_callback(data))
    sensors["bev_semantic"].listen(lambda data: bev_semantic_callback(data))
    sensors["bottom_bev_semantic"].listen(lambda data: bev_bottom_semantic_callback(data))
    for i in cameras_indexes:
        sensors[f"rgb_{i}"].listen(lambda image, j=i: rgb_callback(image, j))
        sensors[f"depth_{i}"].listen(lambda depth, j=i: depth_callback(depth, j))
        sensors[f"semantic_{i}"].listen(lambda semantic, j=i: semantic_callback(semantic, j))
        sensors[f"optical_flow_{i}"].listen(lambda optical_flow, j=i: optical_flow_callback(optical_flow, j))
    sensors["gps"].listen(lambda data: gps_callback(data))
    sensors["imu"].listen(lambda data: imu_callback(data))
    if tfpp_inputs:
        sensors["rgb_tfpp"].listen(lambda data: rgb_tfpp_callback(data))
    world.on_tick(bev_semantic_callback_2)

    rgb_A_folders_name = [f"rgb_A_{i}" for i in cameras_indexes]
    rgb_B_folders_name = [f"rgb_B_{i}" for i in cameras_indexes]
    depth_folders_name = [f"depth_{i}" for i in cameras_indexes]
    semantic_folders_name = [f"semantic_{i}" for i in cameras_indexes]
    optical_flow_folders_name = [f"optical_flow_{i}" for i in cameras_indexes]

    global PATHS
    PATHS["lidar"] = os.path.join(where_to_save, "bev_lidar")
    PATHS["bev_semantic"] = os.path.join(where_to_save, "bev_semantic")
    for i in cameras_indexes:
        PATHS[f"rgb_A_{i}"] = os.path.join(where_to_save, rgb_A_folders_name[i])
        PATHS[f"rgb_B_{i}"] = os.path.join(where_to_save, rgb_B_folders_name[i])
        PATHS[f"depth_{i}"] = os.path.join(where_to_save, depth_folders_name[i])
        PATHS[f"semantic_{i}"] = os.path.join(where_to_save, semantic_folders_name[i])
        PATHS[f"optical_flow_{i}"] = os.path.join(where_to_save, optical_flow_folders_name[i])
    if tfpp_inputs:
        PATHS["rgb_tfpp"] = os.path.join(where_to_save, "rgb_tfpp")
        PATHS["lidar_tfpp"] = os.path.join(where_to_save, "lidar_tfpp")
    PATHS["bev_semantic_2"] = os.path.join(where_to_save, "bev_semantic_2")
    PATHS["bounding_boxes"] = os.path.join(where_to_save, "bounding_boxes")

    for key_path in PATHS:
        os.mkdir(PATHS[key_path])

    def cntrl_c(_, __):
        sensors["lidar"].stop()
        sensors["lidar"].destroy()
        sensors["bev_semantic"].stop()
        sensors["bev_semantic"].destroy()
        sensors["bottom_bev_semantic"].stop()
        sensors["bottom_bev_semantic"].destroy()
        for i in cameras_indexes:
            sensors[f"rgb_{i}"].stop()
            sensors[f"rgb_{i}"].destroy()
            sensors[f"depth_{i}"].stop()
            sensors[f"depth_{i}"].destroy()
            sensors[f"semantic_{i}"].stop()
            sensors[f"semantic_{i}"].destroy()
            sensors[f"optical_flow_{i}"].stop()
            sensors[f"optical_flow_{i}"].destroy()
        if tfpp_inputs:
            sensors["rgb_tfpp"].stop()
            sensors["rgb_tfpp"].destroy()
        exit()

    def merge_semantics(bev_semantic, bottom_bev_semantic):
        """
        We keep:
        0 -> unknown
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
        0 -> unknown
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

    # Let's Run Some Carla's Step to let everything be set up
    global DISABLE_ALL_SENSORS
    global KEEP_GPS
    global STARTING_FRAME
    KEEP_GPS = False
    DISABLE_ALL_SENSORS = True
    for _ in tqdm(range(10), desc=utils.color_info_string("Warming Up...")):
        you_can_tick_event.set()
        world_snapshot = world.wait_for_tick()
        STARTING_FRAME = world_snapshot.frame
        time.sleep(0.3)

    time.sleep(3)
    STARTING_FRAME += 1
    DISABLE_ALL_SENSORS = False

    you_can_tick_event.set()
    for _ in tqdm(range(how_many_frames * config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE),
                  desc=utils.color_info_string("Taking data...")):
        world_snapshot = world.wait_for_tick()
        if (world_snapshot.frame - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            while True:
                if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR_B.values()) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR_B):
                    break
            # print("Obtained all the sensors data! B")
        elif (world_snapshot.frame + 1 - STARTING_FRAME) % config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            while True:
                if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR_A.values()) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR_A):
                    break
            # print("Obtained all the sensors data! A")
        you_can_tick_event.set()
        for key in ALREADY_OBTAINED_DATA_FROM_SENSOR_A:
            ALREADY_OBTAINED_DATA_FROM_SENSOR_A[key] = False
        for key in ALREADY_OBTAINED_DATA_FROM_SENSOR_B:
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B[key] = False
    DISABLE_ALL_SENSORS = True
    KEEP_GPS = True
    last_real_coordinate = []
    cumulative_distance = 0
    distance_we_need = config.DISTANCE_BETWEEN_TARGETPOINTS + 0.3 * config.DISTANCE_BETWEEN_TARGETPOINTS
    distance_bar = tqdm(total=distance_we_need,
                        desc=utils.color_info_string("I get the last gps data..."),
                        bar_format="{l_bar}{bar}| {n:.3f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    while True:
        world.wait_for_tick()
        while True:
            if ALREADY_OBTAINED_ALL_GPS:
                break
        last_real_coordinate.append(utils.convert_gps_to_carla(np.array([ALL_GPS_POSITIONS[-1]]))[0])
        if len(last_real_coordinate) > 1:
            a_distance = math.sqrt((last_real_coordinate[-2][0] - last_real_coordinate[-1][0]) ** 2 +
                                   (last_real_coordinate[-2][1] - last_real_coordinate[-1][1]) ** 2)
            cumulative_distance += a_distance
            if cumulative_distance >= distance_we_need:
                break
            distance_bar.update(a_distance)
        you_can_tick_event.set()
        ALREADY_OBTAINED_ALL_GPS = False
    distance_bar.close()
    del last_real_coordinate
    del cumulative_distance
    del distance_bar
    # BEV SEMANTIC
    for i in tqdm(range(0, how_many_frames), desc=utils.color_info_string("Unifing Top and Bottom BEV...")):
        top_bev_semantic = cv2.imread(os.path.join(PATHS["bev_semantic"], f"top_{i}.png"))[:, :, 0]
        bottom_bev_semantic = cv2.imread(os.path.join(PATHS["bev_semantic"], f"bottom_{i}.png"))[:, :, 0]
        bev_semantic = merge_semantics(top_bev_semantic, bottom_bev_semantic)
        cv2.imwrite(os.path.join(PATHS["bev_semantic"], f"{i}.png"), bev_semantic)
        os.remove(os.path.join(PATHS["bev_semantic"], f"top_{i}.png"))
        os.remove(os.path.join(PATHS["bev_semantic"], f"bottom_{i}.png"))
    # GPS LOCATION
    print(utils.color_info_string("Saving GPS locations!"))
    all_gps_positions_array = np.array(ALL_GPS_POSITIONS)
    frame_gps_positions_array = np.array(FRAME_GPS_POSITIONS)
    all_carla_positions_array = utils.convert_gps_to_carla(all_gps_positions_array)
    frame_carla_positions_array = utils.convert_gps_to_carla(frame_gps_positions_array)
    np.save(os.path.join(str(os.path.join(where_to_save)), "all_gps_positions.npy"), all_gps_positions_array)
    np.save(os.path.join(str(os.path.join(where_to_save)), "frame_gps_positions.npy"), frame_gps_positions_array)
    # COMPASS
    print(utils.color_info_string("Saving Compass data!"))
    frame_compass_array = np.array(FRAME_COMPASS)
    np.save(os.path.join(str(os.path.join(where_to_save)), "frame_compass.npy"), frame_compass_array)
    # NEXT 10 WAYPOINTS of 1 M distance
    frame_waypoints = np.zeros((len(frame_carla_positions_array), config.NUM_OF_WAYPOINTS, 3))
    frame_waypoints_not_rotated = np.zeros((len(frame_carla_positions_array), config.NUM_OF_WAYPOINTS, 3))
    for frame_id in tqdm(range(len(frame_carla_positions_array)), desc=utils.color_info_string("Saving Waypoints...")):
        all_id = frame_id * config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
        # print(f"{frame_carla_positions_array[frame_id]} =? {all_carla_positions_array[all_id]}")
        distance = 0

        for i in range(1, config.NUM_OF_WAYPOINTS + 1):
            while True:
                all_id += 1
                old_distance = distance
                distance += math.sqrt(
                    (all_carla_positions_array[all_id, 0] - all_carla_positions_array[all_id - 1, 0]) ** 2
                    + (all_carla_positions_array[all_id, 1] - all_carla_positions_array[all_id - 1, 1]) ** 2)
                # print(distance)
                if distance > i * config.DISTANCE_BETWEEN_WAYPOINTS:
                    big_delta_distance = distance - old_distance
                    small_delta_distance = i * config.DISTANCE_BETWEEN_WAYPOINTS - old_distance
                    percentage = small_delta_distance / big_delta_distance
                    # print(f"big = {big_delta_distance}, small = {small_delta_distance}, percentage = {percentage}")
                    point_x = all_carla_positions_array[all_id - 1, 0] + percentage * (
                            all_carla_positions_array[all_id, 0] - all_carla_positions_array[all_id - 1, 0])
                    point_y = all_carla_positions_array[all_id - 1, 1] + percentage * (
                            all_carla_positions_array[all_id, 1] - all_carla_positions_array[all_id - 1, 1])
                    point_z = all_carla_positions_array[all_id - 1, 2] + percentage * (
                            all_carla_positions_array[all_id, 2] - all_carla_positions_array[all_id - 1, 2])
                    carla_coordinate_waypoint = np.array([point_x, point_y, point_z])
                    # carla_coordinate_waypoint = all_carla_positions_array[all_id]
                    vehicle_origin_carla_coordinate_waypoint = carla_coordinate_waypoint - frame_carla_positions_array[
                        frame_id]
                    theta = -frame_compass_array[frame_id] + math.pi
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                                [np.sin(theta), np.cos(theta), 0],
                                                [0, 0, 1]])
                    vehicle_waypoint = rotation_matrix @ vehicle_origin_carla_coordinate_waypoint
                    # TO UNDERSTAND
                    # distance_1 = math.sqrt((carla_coordinate_waypoint[0] - frame_carla_positions_array[frame_id, 0])**2 + (carla_coordinate_waypoint[1] - frame_carla_positions_array[frame_id, 1])**2)
                    # distance_2 = math.sqrt((vehicle_origin_carla_coordinate_waypoint[0])**2 + (vehicle_origin_carla_coordinate_waypoint[1])**2)
                    # distance_3 = math.sqrt((vehicle_waypoint[0])**2 + (vehicle_waypoint[1])**2)
                    # print(f"CORD: [{i}] {carla_coordinate_waypoint} -> {vehicle_origin_carla_coordinate_waypoint} -> {vehicle_waypoint}")
                    # print(f"DIST: [{i}] {distance_1} -> {distance_2} -> {distance_3}")
                    frame_waypoints[frame_id, i - 1] = vehicle_waypoint
                    frame_waypoints_not_rotated[frame_id, i - 1] = vehicle_origin_carla_coordinate_waypoint
                    break
    np.save(os.path.join(str(os.path.join(where_to_save)), "frame_waypoints.npy"), frame_waypoints)
    # NEXT TARGETPOINT of 30/50 m distance
    frame_targetpoints = np.zeros((len(frame_carla_positions_array), 3))

    def get_new_targetpoint(frame_id):
        all_id = frame_id * config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE
        distance = 0
        old_distance = 0
        while True:
            all_id += 1
            old_distance = distance
            distance += math.sqrt((all_carla_positions_array[all_id, 0] - all_carla_positions_array[all_id - 1, 0]) ** 2
                                  + (all_carla_positions_array[all_id, 1] -
                                     all_carla_positions_array[all_id - 1, 1]) ** 2)
            if distance > config.DISTANCE_BETWEEN_TARGETPOINTS:
                big_delta_distance = distance - old_distance
                small_delta_distance = config.DISTANCE_BETWEEN_TARGETPOINTS - old_distance
                percentage = small_delta_distance / big_delta_distance
                point_x = all_carla_positions_array[all_id - 1, 0] + percentage * (
                        all_carla_positions_array[all_id, 0] - all_carla_positions_array[all_id - 1, 0])
                point_y = all_carla_positions_array[all_id - 1, 1] + percentage * (
                        all_carla_positions_array[all_id, 1] - all_carla_positions_array[all_id - 1, 1])
                point_z = all_carla_positions_array[all_id - 1, 2] + percentage * (
                        all_carla_positions_array[all_id, 2] - all_carla_positions_array[all_id - 1, 2])
                carla_coordinate_targetpoint = np.array([point_x, point_y, point_z])
                return carla_coordinate_targetpoint

    carla_coordinate_targetpoint = None
    for frame_id in tqdm(range(len(frame_carla_positions_array)),
                         desc=utils.color_info_string("Saving Targetpoints...")):
        if carla_coordinate_targetpoint is None:
            carla_coordinate_targetpoint = get_new_targetpoint(frame_id)
        vehicle_origin_carla_coordinate_targetpoint = carla_coordinate_targetpoint - frame_carla_positions_array[
            frame_id]
        theta = -frame_compass_array[frame_id] + math.pi
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        vehicle_targetpoint = rotation_matrix @ vehicle_origin_carla_coordinate_targetpoint
        frame_targetpoints[frame_id] = vehicle_targetpoint
        # let's check if we need to calculate the next targetpoint
        distance_from_target_point = math.sqrt(
            vehicle_origin_carla_coordinate_targetpoint[0] ** 2 + vehicle_origin_carla_coordinate_targetpoint[1] ** 2)
        if distance_from_target_point < config.MINIMUM_DISTANCE_FOR_NEXT_TARGETPOINT:
            carla_coordinate_targetpoint = None
    np.save(os.path.join(os.path.join(where_to_save), "frame_targetpoints.npy"), frame_targetpoints)
    # SPEEDS
    previous_speeds = []  # the speed given as input to the model
    next_speeds = []  # the ground truth speed that the model have to predict
    accelerations = []  # the prediction of the model regarding speed

    def get_speed(index):
        curr_x = all_carla_positions_array[index, 0]
        curr_y = all_carla_positions_array[index, 1]
        next_x = all_carla_positions_array[1 + index, 0]
        next_y = all_carla_positions_array[1 + index, 1]
        ellapsed_distance = math.sqrt((curr_x - next_x) ** 2 + (curr_y - next_y) ** 2)
        ellapsed_time = 1. / config.CARLA_FPS
        speed = ellapsed_distance / ellapsed_time
        return speed

    # we calculate the first frame assuming the start speed is 0
    previous_speeds.append(0)
    # print(f"[PREV_{0}] speed = {0} m/s -> {0} km/h")
    speeds = []
    for i in range(0, config.HOW_MANY_CARLA_FRAME_FOR_CALCULATING_SPEEDS + 1, +1):
        speed = get_speed(i)
        speeds.append(speed)
    starting_next_speed = sum(speeds) / len(speeds)
    # print(f"[NEXT_{0}] speed = {starting_next_speed:.4f} m/s -> {starting_next_speed*3.6:.4f} km/h")
    next_speed_as_probability = [0., 0., 0., 0.]
    if starting_next_speed < 3.0:
        next_speed_as_probability[0] = 1.0
    elif starting_next_speed < 12.5:
        next_speed_as_probability[1] = 1.0
    elif starting_next_speed < 23.5:
        next_speed_as_probability[2] = 1.0
    else:
        next_speed_as_probability[3] = 1.0
    next_speeds.append(next_speed_as_probability)
    # CALCULATING ACCELERATION AS PROBABILITY
    acceleration_as_probability = [0., 0., 0.]  # DECREASING, SAME, INCREASING SPEED
    speed_difference = starting_next_speed - previous_speeds[0]
    if abs(speed_difference) < config.MINIMUM_SPEED_DIFFERENCE_TO_PUT_ACCELERATION_TO_0:
        acceleration_as_probability[1] = 1.0
    elif speed_difference < 0.:
        acceleration_as_probability[0] = 1.0
    elif speed_difference > 0.:
        acceleration_as_probability[2] = 1.0
    accelerations.append(acceleration_as_probability)
    # we calculate the speed for the following frames
    for frame_index in tqdm(range(1, len(frame_gps_positions_array)), desc=utils.color_info_string("Saving Speeds...")):
        speeds = []
        for i in range(-config.HOW_MANY_CARLA_FRAME_FOR_CALCULATING_SPEEDS - 1, 0):
            speed = get_speed(frame_index * config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE + i)
            speeds.append(speed)
        previous_speed = sum(speeds) / len(speeds)
        # print(f"[PREV_{frame_index}] speed = {previous_speed:.4f} m/s -> {previous_speed*3.6:.4f} km/h")
        previous_speeds.append(previous_speed)
        speeds = []
        for i in range(0, config.HOW_MANY_CARLA_FRAME_FOR_CALCULATING_SPEEDS + 1, +1):
            speed = get_speed(frame_index * config.AMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE + i)
            speeds.append(speed)
        next_speed = sum(speeds) / len(speeds)
        # print(f"[NEXT_{frame_index}] speed = {next_speed:.4f} m/s -> {next_speed*3.6:.4f} km/h")
        # CALCULATE NEXT SPEED PROBABILITY VECTOR AS IN TRANSFUSER
        next_speed_as_probability = [0., 0., 0., 0.]
        if next_speed < 3.0:
            next_speed_as_probability[0] = 1.0
        elif next_speed < 12.5:
            next_speed_as_probability[1] = 1.0
        elif next_speed < 23.5:
            next_speed_as_probability[2] = 1.0
        else:
            next_speed_as_probability[3] = 1.0
        next_speeds.append(next_speed_as_probability)
        # CALCULATING ACCELERATION AS PROBABILITY
        acceleration_as_probability = [0., 0., 0.]  # DECREASING, SAME, INCREASING SPEED
        speed_difference = next_speed - previous_speed
        if abs(speed_difference) < 0.1:  # IT'S A RANDOM NUMBER
            acceleration_as_probability[1] = 1.0
        elif speed_difference < 0.:
            acceleration_as_probability[0] = 1.0
        elif speed_difference > 0.:
            acceleration_as_probability[2] = 1.0
        accelerations.append(acceleration_as_probability)
    previous_speeds_array = np.array(previous_speeds)
    next_speeds_array = np.array(next_speeds)
    acceleration_array = np.array(accelerations)
    # we save the speed data
    np.save(os.path.join(str(os.path.join(where_to_save)), "previous_speeds.npy"), previous_speeds_array)
    np.save(os.path.join(str(os.path.join(where_to_save)), "next_speeds.npy"), next_speeds_array)
    np.save(os.path.join(str(os.path.join(where_to_save)), "accelerations.npy"), acceleration_array)

    # LET'S PRINT A INFO JSON FILE
    run_info = {
        "carla_egg_path": carla_egg_path,
        "town_id": town_id,
        "rpc_port": rpc_port,
        "job_id": job_id,
        "how_many_frames": how_many_frames,
        "where_to_save": where_to_save,
        "back_camera": back_camera,
        "lateral_cameras": lateral_cameras,
        "tfpp_inputs": tfpp_inputs
    }
    data_info_json = {
        "weather": weather_dict,
        "config": utils.get_configs_as_dict(),
        "run_info": run_info,
    }
    data_info_file_name = "data_info.json"
    with open(os.path.join(where_to_save, data_info_file_name), "w") as out_file:
        json.dump(data_info_json, out_file, indent=4, sort_keys=True)

    finished_taking_data_event.set()
