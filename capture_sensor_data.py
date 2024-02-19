#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to the capture sensor data script, a script that provides users with a baseline for data collection,
which they can later modify to their specific needs, easying the process of creating a database.

This script will start with a CARLA recorder log, spawning the desired sensor configuration at the ego vehicle,
and saving their data into a folder. The exact parameters can be found at the very top of the script and include:

> SENSORS: List of all the sensors tha will be spawned in the simulation
> WEATHER: Weather of the simulation
> RECORDER_INFO: List of all the CARLA recorder logs that will be run. Each recorder has three elements:
    - folder: path to the folder with the `*.log` recorder file, and a `log.json` file, which has all the ego vehicle information
    - start_time: start time of the recorder
    - duration: duration of the recorder
"""

import time
import os
import carla
import argparse
import random
import json
import threading
import glob

import numpy as np
import cv2

from queue import Queue, Empty

################### User simulation configuration ####################
# 1) Choose the sensors
SENSORS = [
    [
        'CameraFront',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
    ],
    [
        'CameraRight',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0
        },
    ],
    [
        'CameraBack',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0
        },
    ],
    [
        'CameraLeft',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 270.0
        },
    ],
    [
        'DepthFront',
        {
            'bp': 'sensor.camera.depth',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
    ],
    [
        'DepthRight',
        {
            'bp': 'sensor.camera.depth',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0
        },
    ],
    [
        'DepthBack',
        {
            'bp': 'sensor.camera.depth',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0
        },
    ],
    [
        'DepthLeft',
        {
            'bp': 'sensor.camera.depth',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 270.0
        },
    ],
    [
        'FlowFront',
        {
            'bp': 'sensor.camera.optical_flow',
            'image_size_x': 1080, 'image_size_y': 720, 'fov': 90,
            'x': 0.0, 'y': 0.0, 'z': 2.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
    ],
    [
        'SemanticBEV',
        {
            'bp': 'sensor.camera.semantic_segmentation',
            'image_size_x': 1080, 'image_size_y': 1080, 'fov': 100,
            'x': 0.0, 'y': 0.0, 'z': 15, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0
        },
    ],
    [
        'LidarTest',
        {
            'bp': 'sensor.lidar.ray_cast',
            'x': 0.7, 'y': 0.0, 'z': 2.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
            'range': 100, 'upper_fov': 10, 'lower_fov': -30, 'points_per_second': 500000,
            'rotation_frequency': 20, 'channels': 64
        }
    ]
]
# 2) Choose a weather
WEATHER = carla.WeatherParameters(
    sun_azimuth_angle=-1.0, sun_altitude_angle=70.0,
    cloudiness=30.0, precipitation=0.0, precipitation_deposits=80.0, wetness=15.0,
    wind_intensity=10.0,
    fog_density=2.0, fog_distance=0.0, fog_falloff=0.0)

# 3) Choose a recorder file (folder path, start time, duration)
RECORDER_INFO = [
    {
        'folder': "/home/enrico/Projects/Carla/LEADERBOARD_STUFF/Scenario_Logs/ParkingExit",
        'start_time': 0,
        'duration': 17
    },
    {
        'folder': "/home/enrico/Projects/Carla/LEADERBOARD_STUFF/Scenario_Logs/HighwayExit",
        'start_time': 0,
        'duration': 27
    }
]

# 4) Choose the destination folder where the sensor data will be saved
DESTINATION_FOLDER = "database"
################# End user simulation configuration ##################

FPS = 20
THREADS = 5
CURRENT_THREADS = 0

def create_folders(endpoint, sensors):
    for sensor_id, sensor_bp in sensors:
        sensor_endpoint = f"{endpoint}/{sensor_id}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)

        if 'gnss' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Altitude,Latitude,Longitude\n"
                data_file.write(data_txt)

        if 'imu' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,Gyroscope X,Gyroscope Y,Gyroscope Z\n"
                data_file.write(data_txt)

def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))

def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return

def get_ego_id(recorder_file):
    found_lincoln = False
    found_id = None

    for line in recorder_file.split("\n"):

        # Check the role_name for hero
        if found_lincoln:
            if not line.startswith("  "):
                found_lincoln = False
                found_id = None
            else:
                data = line.split(" = ")
                if 'role_name' in data[0] and 'hero' in data[1]:
                    return found_id

        # Search for all lincoln vehicles
        if not found_lincoln and line.startswith(" Create ") and 'vehicle.lincoln' in line:
            found_lincoln = True
            found_id =  int(line.split(" ")[2][:-1])

    return found_id

LIDAR_DISTANCE = 32
def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
    :param lidar: (N,3) numpy, LiDAR point cloud
    :return: (2, H, W) numpy, LiDAR as sparse image
    """
    
    # print(f"0 -> [{lidar[:, 0].min()}; {lidar[:, 0].max()}]") [-6.148300647735596; 9.741386413574219]
    # print(f"1 -> [{lidar[:, 1].min()}; {lidar[:, 1].max()}]") [-3.9062499126885086e-05; 9.49496078491211]
    # print(f"2 -> [{lidar[:, 2].min()}; {lidar[:, 2].max()}]") [-1.6756640672683716; 1.6750586032867432]
    
    def splat_points(point_cloud):
      # 256 x 256 grid
      xbins = np.linspace(-LIDAR_DISTANCE, LIDAR_DISTANCE, LIDAR_DISTANCE*2 * 4 + 1)
      ybins = np.linspace(-LIDAR_DISTANCE, LIDAR_DISTANCE, LIDAR_DISTANCE*2 * 4 + 1)
      hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
      hist[hist > 5] = 5
      overhead_splat = hist / 5
      # The transpose here is an efficient axis swap.
      # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
      # (x height channel, y width channel)
      return overhead_splat.T

    # Remove points above the vehicle
    # lidar = lidar[lidar[..., 2] < 100]
    # above = lidar[lidar[..., 2] > 0.2]
    # above_features = splat_points(above)
    # features = np.stack([above_features], axis=-1)
    features = np.stack([splat_points(lidar)], axis=-1)
    features = np.transpose(features, (2, 0, 1))
    features *= 255
    features = features.astype(np.uint8)
    return features

def save_data_to_disk(sensor_id, frame, data, imu_data, endpoint):
    """
    Saves the sensor data into file:
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'
    """
    global CURRENT_THREADS
    CURRENT_THREADS += 1
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.jpg"
        if sensor_id == "SemanticBEV":
            data.save_to_disk(sensor_endpoint, carla.libcarla.ColorConverter.CityScapesPalette)
        elif sensor_id in ["DepthFront", "DepthRight", "DepthBack", "DepthLeft"]:
            data.save_to_disk(sensor_endpoint, carla.libcarla.ColorConverter.LogarithmicDepth)
        else:
            data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.LidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.jpg"
        out = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        out = np.reshape(out, (int(out.shape[0] / 4), 4))
        bev = lidar_to_histogram_features(out[:, :3])[0]
        cv2.imwrite(sensor_endpoint, bev)
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.SemanticLidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.RadarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.csv"
        data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
        for point_data in data:
            data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
        with open(sensor_endpoint, 'w') as data_file:
            data_file.write(data_txt)

    elif isinstance(data, carla.GnssMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
            data_file.write(data_txt)

    elif isinstance(data, carla.IMUMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{imu_data[0][0]},{imu_data[0][1]},{imu_data[0][2]},{data.compass},{imu_data[1][0]},{imu_data[1][1]},{imu_data[1][2]}\n"
            data_file.write(data_txt)
    elif isinstance(data, carla.libcarla.OpticalFlowImage):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
        data = data.get_color_coded_flow()
        out = np.frombuffer(data.raw_data, dtype=np.uint8)
        out = np.reshape(out, (data.height, data.width, 4))
        cv2.imwrite(sensor_endpoint, out[:, :, :3])
    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

    CURRENT_THREADS -= 1

def extract_imu_data(log):
    records = log["records"]
    log_data = []
    for record in records:
        acceleration_data = record["state"]["acceleration"]
        acceleration_vector = [acceleration_data["x"], acceleration_data["y"], acceleration_data["z"]]

        # TODO: Remove this (don't use logs without angular velocity)
        if "angular_velcoity" in record["state"]:
            angular_data = record["state"]["angular_velocity"]
            angular_vector = [angular_data["x"], angular_data["y"], angular_data["z"]]
        else:
            angular_vector = [random.random(), random.random(), random.random()]

        log_data.append([acceleration_vector, angular_vector])

    return log_data

def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()
    print(__doc__)

    active_sensors = []

    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(120.0)
        world = client.get_world()

        for recorder_info in RECORDER_INFO:
            recorder_folder = recorder_info['folder']
            recorder_start = recorder_info['start_time']
            recorder_duration = recorder_info['duration']

            # 0) Get the recorder files
            recorder_path_list = glob.glob(f"{recorder_folder}/*.log")
            if recorder_path_list:
                recorder_path = recorder_path_list[0]
            else:
                print(f"Couldn't find the recorder file for the folder '{recorder_folder}'. Stopping...")
                continue
            recorder_log_list = glob.glob(f"{recorder_folder}/log.json")
            if recorder_log_list:
                recorder_log = recorder_log_list[0]
            else:
                recorder_log = None

            print(f"Running: {recorder_path}")
            endpoint = f"{DESTINATION_FOLDER}/{recorder_path.split('/')[-1][:-4]}"
            
            
            
            # 1) Get the recorder map and load the world
            recorder_str = client.show_recorder_file_info(recorder_path, True)
            recorder_map = recorder_str.split("\n")[1][5:]
            world = client.load_world(recorder_map)
            world.tick()

            # 2) Change the weather and synchronous mode
            world.set_weather(WEATHER)
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / FPS
            settings.synchronous_mode = True
            world.apply_settings(settings)

            for _ in range(100):
                world.tick()

            # 3) Replay the recorder
            max_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
            print(f"NOCE max duration = {max_duration}")
            if recorder_duration == 0:
                recorder_duration = max_duration
            elif recorder_start + recorder_duration > max_duration:
                print("WARNING: Found a duration that exceeds the recorder length. Reducing it...")
                recorder_duration = max_duration - recorder_start
            if recorder_start >= max_duration:
                print("WARNING: Found a start point that exceeds the recoder duration. Ignoring it...")
                continue
            print(f"Duration: {round(recorder_duration, 2)} - Frames: {round(20*recorder_duration, 0)}")

            if recorder_log:
                with open(recorder_log) as fd:
                    log_json = json.load(fd)
                imu_logs = extract_imu_data(log_json)
            else:
                imu_logs = None

            client.replay_file(recorder_path, recorder_start, recorder_duration, get_ego_id(recorder_str), False)
            with open(f"{recorder_path[:-4]}.txt", 'w') as fd:
                fd.write(recorder_str)
            world.tick()

            # 4) Link onto the ego vehicle
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

            # 5) Create the sensors, and save their data into a queue
            create_folders(endpoint, [[s[0], s[1].get('bp')] for s in SENSORS])
            blueprint_library = world.get_blueprint_library()
            sensor_queue = Queue()
            for sensor in SENSORS:

                # Extract the data from the sesor configuration
                sensor_id = sensor[0]
                attributes = sensor[1]
                blueprint_name = attributes.get('bp')
                sensor_transform = carla.Transform(
                    carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
                    carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
                )

                # Get the blueprint and add the attributes
                blueprint = blueprint_library.find(blueprint_name)
                for key, value in attributes.items():
                    if key in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        continue
                    blueprint.set_attribute(str(key), str(value))

                # Create the sensors and its callback
                sensor = world.spawn_actor(blueprint, sensor_transform, hero)
                add_listener(sensor, sensor_queue, sensor_id)
                active_sensors.append(sensor)

            world.tick()
            time.sleep(10)
            
            # 6) Run the simulation
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            sensor_amount = len(SENSORS)

            max_threads = THREADS
            results = []

            while True:
                world.tick()
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
                print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")

                # Get and save the sensor data from the queue.
                missing_sensors = sensor_amount
                while True:
                
                    frame = world.get_snapshot().frame
                    try:
                        sensor_data = sensor_queue.get(True, 2.0)
                        if sensor_data[1] != frame:
                            continue  # Ignore previous frame data
                        missing_sensors -= 1
                    except Empty:
                        raise ValueError("A sensor took too long to send their data")
                    # Get the data
                    sensor_id = sensor_data[0]
                    frame_diff = sensor_data[1] - start_frame
                    data = sensor_data[2]
                    if imu_logs:
                        imu_data = imu_logs[int(FPS*recorder_start + frame_diff)]
                    else:
                        imu_data = [[0,0,0], [0,0,0]]

                    res = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_diff, data, imu_data, endpoint))
                    results.append(res)
                    res.start()

                    if CURRENT_THREADS > max_threads:
                        for res in results:
                            res.join()
                        results = []

                    if missing_sensors <= 0:
                        break

                # world.tick()

            for res in results:
                res.join()

            for sensor in active_sensors:
                sensor.stop()
                sensor.destroy()
            active_sensors = []

            for _ in range(50):
                world.tick()

    # End the simulation
    finally:
        # stop and remove cameras
        for sensor in active_sensors:
            sensor.stop()
            sensor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
