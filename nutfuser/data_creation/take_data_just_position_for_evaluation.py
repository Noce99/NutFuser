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
import xml.etree.ElementTree as ET
from nutfuser import config
from nutfuser import utils
from nutfuser.data_creation.weather import get_a_random_weather

STARTING_FRAME = None
PATHS = {}
ALREADY_OBTAINED_DATA_FROM_SENSOR_B = []
FRAME_GPS_POSITIONS = []
ALL_GPS_POSITIONS = []
FRAME_COMPASS = []


DISABLE_ALL_SENSORS = False
KEEP_GPS = False
RGB = None

def take_data_just_position_for_evaluation(carla_egg_path, town_id, rpc_port, job_id, ego_vehicle_found_event, finished_taking_data_event, you_can_tick_event, how_many_frames, where_to_save, show_rgb):

    sys.path.append(carla_egg_path)
    try:
        import carla
    except:
        pass
    
    global ALREADY_OBTAINED_DATA_FROM_SENSOR_B
    ALREADY_OBTAINED_DATA_FROM_SENSOR_B = {"frame_gps_position":False, "compass":False}
    
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

    # CAMERAS callback
    def rgb_callback(data):
        global RGB
        RGB = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))[:, :, :3]

    # GPS callback
    def gps_callback(data):
        if not DISABLE_ALL_SENSORS or KEEP_GPS:
            ALL_GPS_POSITIONS.append((data.latitude, data.longitude, data.altitude))
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            FRAME_GPS_POSITIONS.append((data.latitude, data.longitude, data.altitude))
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["frame_gps_position"] = True

    # IMU callback
    def imu_callback(data):
        if not DISABLE_ALL_SENSORS and (data.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            FRAME_COMPASS.append(data.compass)
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B["compass"] = True


    if show_rgb:
        # RGB CAMERAS
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("image_size_x", f"{config.IMAGE_W}")
        camera_bp.set_attribute("image_size_y", f"{config.IMAGE_H}")

    # GPS
    gps_bp = bp_lib.find("sensor.other.gnss")

    # IMU
    imu_bp = bp_lib.find("sensor.other.imu")

    # POSITIONS OF SENSORS
    basic_position = carla.Transform(
        carla.Location(x=0, y=0, z=2.5),
        carla.Rotation(pitch=0, roll=0, yaw=0)
    )
    if show_rgb:
        rgb_transform = carla.Transform(    carla.Location(x=1.0, y=+0.0, z=2.0),
                                            carla.Rotation(pitch=0.0, roll=0, yaw=0))  
    sensors = {} 
    sensors["gps"] = world.spawn_actor(gps_bp, basic_position, attach_to=hero)
    sensors["imu"] = world.spawn_actor(imu_bp, basic_position, attach_to=hero)
    if show_rgb:
        sensors[f"rgb"] = world.spawn_actor(camera_bp, rgb_transform, attach_to=hero)

    # Connect Sensor and Callbacks
    sensors["gps"].listen(lambda data: gps_callback(data))
    sensors["imu"].listen(lambda data: imu_callback(data))
    if show_rgb:
        sensors["rgb"].listen(lambda image: rgb_callback(image))

    def cntrl_c(_, __):
        if show_rgb:
            sensors["rgb"].stop()
            sensors["rgb"].destroy()
        sensors["gps"].stop()
        sensors["gps"].destroy()
        sensors["imu"].stop()
        sensors["imu"].destroy()
        exit()

    signal.signal(signal.SIGINT, cntrl_c)

    # Let's Run Some Carla's Step to let everithing to be setted up
    global DISABLE_ALL_SENSORS
    global KEEP_GPS
    global STARTING_FRAME
    global RGB
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
    for _ in tqdm(range(how_many_frames*config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE), desc=utils.color_info_string("Taking data...")):
        world_snapshot = world.wait_for_tick()
        if (world_snapshot.frame - STARTING_FRAME) % config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE == 0:
            while True:
                if sum(ALREADY_OBTAINED_DATA_FROM_SENSOR_B.values()) == len(ALREADY_OBTAINED_DATA_FROM_SENSOR_B):
                    break
            if show_rgb:
                cv2.imshow('rgb', RGB)  
                cv2.waitKey(10)
            # print("Obtained all the sensors data! B")
        you_can_tick_event.set()
        for key in ALREADY_OBTAINED_DATA_FROM_SENSOR_B:
            ALREADY_OBTAINED_DATA_FROM_SENSOR_B[key] = False
    DISABLE_ALL_SENSORS = True
    KEEP_GPS = True
    # GPS LOCATION
    print(utils.color_info_string("Saving GPS locations!"))
    all_gps_positions_array = np.array(ALL_GPS_POSITIONS)
    frame_gps_positions_array = np.array(FRAME_GPS_POSITIONS)
    all_carla_positions_array = utils.convert_gps_to_carla(all_gps_positions_array)
    frame_carla_positions_array = utils.convert_gps_to_carla(frame_gps_positions_array)
    # np.save(os.path.join(os.path.join(where_to_save), "all_gps_positions.npy"), all_gps_positions_array)
    # np.save(os.path.join(os.path.join(where_to_save), "frame_gps_positions.npy"), frame_gps_positions_array)
    # COMPASS
    print(utils.color_info_string("Saving Compass data!"))
    frame_compass_array = np.array(FRAME_COMPASS)
    # np.save(os.path.join(os.path.join(where_to_save), "frame_compass.npy"), frame_compass_array)
    
    last_point = frame_carla_positions_array[0]
    cumulative_distance = 0
    routes = ET.Element("routes")
    route = ET.SubElement(routes, "route", {"id":f"{job_id}", "town":config.TOWN_DICT[town_id]})
    waypoints = ET.SubElement(route, "waypoints")
    for i in tqdm(range(1, frame_carla_positions_array.shape[0]), desc=utils.color_info_string("Saving Targetpoints...")):
        cumulative_distance += math.sqrt((frame_carla_positions_array[i][0] - last_point[0])**2 +
                                         (frame_carla_positions_array[i][1] - last_point[1])**2 )
        last_point = frame_carla_positions_array[i]
        if cumulative_distance > config.DISTANCE_BETWEEN_TARGETPOINTS_IN_EVALUATION_ROUTES:
            cumulative_distance = 0
            position_dict = {}
            position_dict["x"] = f"{frame_carla_positions_array[i][0]}"
            position_dict["y"] = f"{frame_carla_positions_array[i][1]}"
            position_dict["z"] = "0.0"
            ET.SubElement(waypoints, "position", position_dict)
    ET.SubElement(route, "scenarios")

    tree = ET.ElementTree(routes)
    utils.indent(routes)

    tree.write(os.path.join(where_to_save, f"{job_id}_{config.TOWN_DICT[town_id]}.xml"))

    finished_taking_data_event.set()
