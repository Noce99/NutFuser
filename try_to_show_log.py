import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import open3d as o3d
from matplotlib import cm

bev = np.zeros((256*2, 256*2), dtype=np.uint8)

# Connect the client and set up bp library and spawn point
client = carla.Client('localhost', 2000)
client.set_timeout(120.0)
world = client.load_world("Town13")
bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points() 


# We load the log file
client.replay_file("/home/enrico/Projects/Carla/LEADERBOARD_STUFF/Scenario_Logs/noce_town13.log",
# client.replay_file("/home/enrico/Projects/Carla/LEADERBOARD_STUFF/Scenario_Logs/ParkingExit/ParkingExit.log",
# client.replay_file("/home/enrico/Projects/Carla/carla_garage/leaderboard/data/training/routes/ll/Town04_ll.xml",
0,
20,
0,
False)

world.tick()
world.tick()


# Find out the vehicle
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

# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

# LIDAR and RADAR callbacks

def lidar_callback(point_cloud):

    global bev

    def lidar_to_histogram_features(lidar):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
        :param lidar: (N,3) numpy, LiDAR point cloud
        :return: (2, H, W) numpy, LiDAR as sparse image
        """
        
        # print(f"0 -> [{lidar[:, 0].min()}; {lidar[:, 0].max()}]") [-6.148300647735596; 9.741386413574219]
        # print(f"1 -> [{lidar[:, 1].min()}; {lidar[:, 1].max()}]") [-3.9062499126885086e-05; 9.49496078491211]
        # print(f"2 -> [{lidar[:, 2].min()}; {lidar[:, 2].max()}]") [-1.6756640672683716; 1.6750586032867432]
        LIDAR_DISTANCE = 32
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
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    
    bev = lidar_to_histogram_features(data[:, :3])[0]

    width = int(bev.shape[1] * 2)
    height = int(bev.shape[0] * 2)
    dim = (width, height)
    bev = cv2.resize(bev, dim, interpolation = cv2.INTER_AREA)

def radar_callback(data):
    radar_data = np.zeros((len(data), 4))
    
    for i, detection in enumerate(data):
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)
        
        radar_data[i, :] = [x, y, z, detection.velocity]
        
    intensity = np.abs(radar_data[:, -1])
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]
    
    points = radar_data[:, :-1]
    points[:, :1] = -points[:, :1]
    
# Camera callback
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 

# Set up LIDAR and RADAR, parameters are to assisst visualisation

lidar_bp = bp_lib.find('sensor.lidar.ray_cast') 
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('noise_stddev', '0.1')
lidar_bp.set_attribute('upper_fov', '15.0')
lidar_bp.set_attribute('lower_fov', '-25.0')
lidar_bp.set_attribute('channels', '64.0')
lidar_bp.set_attribute('rotation_frequency', '20.0')
lidar_bp.set_attribute('points_per_second', '500000')
    
lidar_init_trans = carla.Transform(carla.Location(z=2))
lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=hero)

# radar_bp = bp_lib.find('sensor.other.radar') 
# radar_bp.set_attribute('horizontal_fov', '30.0')
# radar_bp.set_attribute('vertical_fov', '30.0')
# radar_bp.set_attribute('points_per_second', '10000')
# radar_init_trans = carla.Transform(carla.Location(z=2))
# radar = world.spawn_actor(radar_bp, radar_init_trans, attach_to=hero)

# Spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb') 
camera_init_trans = carla.Transform(carla.Location(z=2.5, x=-3), carla.Rotation())
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=hero)

# Set up dictionary for camera data
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
camera_data = {'image': np.zeros((image_h, image_w, 4))} 

# Start sensors
lidar.listen(lambda data: lidar_callback(data))
# radar.listen(lambda data: radar_callback(data))
camera.listen(lambda image: camera_callback(image, camera_data))

# OpenCV window for camera
cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera', camera_data['image'])
cv2.waitKey(1)

# OpenCV window for lidar
cv2.namedWindow('Lidar', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Lidar', bev)
cv2.waitKey(1)

# Update geometry and camera in game loop
while True:

    cv2.imshow('RGB Camera', camera_data['image'])
    cv2.imshow('Lidar', bev)

    # Break if user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Close displayws and stop sensors
cv2.destroyAllWindows()
# radar.stop()
# radar.destroy()
lidar.stop()
lidar.destroy()
camera.stop()
camera.destroy()

for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for actor in world.get_actors().filter('*sensor*'):
    actor.destroy()
