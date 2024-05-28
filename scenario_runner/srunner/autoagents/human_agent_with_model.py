#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

from __future__ import print_function

import json

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

from srunner.autoagents.autonomous_agent import AutonomousAgent
import cv2
import numpy as np
from nutfuser import utils
from nutfuser import config
from tqdm import tqdm
import torch


class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        """
        Run the GUI
        """
        # process sensor data
        image_center = input_data['Front'][1][:, :, -2::-1]

        # display image
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def quit_interface(self):
        """
        Stops the pygame window
        """
        pygame.quit()


class HumanAgentWithModel(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False
    prev_timestamp = 0

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.agent_engaged = False
        self.camera_width = 800
        self.camera_height = 500

        self._hic = HumanInterface(self.camera_width, self.camera_height)
        self._controller = KeyboardControl(path_to_conf_file)
        self.prev_timestamp = 0

        # NUT
        self.next_waypoint_index = 0
        self.waypoint_tqdm_bar = None
        self.last_location = None
        self.last_front_image = None
        self.device = torch.device(f'cuda:{0}')

        # Let's load the Neural Network
        backbone = "/home/enrico/Projects/Carla/NutFuser/train_logs/test_my_data_30_04_2024_15:56:39/model_0030.pth"
        full_net = "/home/enrico/Projects/Carla/NutFuser/train_logs/test_my_data_01_05_2024_17:52:10/model_0030.pth"
        weight_path = full_net
        self.model, self.predicting_flow, self.just_a_backbone, self.tfpp_original = utils.load_model_given_weights(weight_path)
        self.model.eval()
        # END NUT

    def sensors(self):

        sensors = [
            # (1)
            {'type': 'sensor.camera.rgb',
                'x': 1.0, 'y': 0.0, 'z': 2.0,
                'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                'width': 1024, 'height': 256, 'fov': 90,
                'id': 'Front'},
            # (2)
            {'type': 'sensor.lidar.ray_cast',
                'x': 0.0, 'y': 0.0, 'z': 2.5,
                'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                'range': 100, 'noise_stddev': 0.0, 'upper_fov': 0.0,
                'lower_fov': -25.0, 'channels': 32, 'rotation_frequency': 20.0,
                'points_per_second': 600000,
                'id': 'Lidar'},
            # (3)
            {'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 2.5,
                'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                'id': 'GPS'},
            # (4)
            {'type': 'sensor.other.imu', 
                'x': 0.0, 'y': 0.0, 'z': 2.5,
                'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
                'id': 'IMU'},
            # THE FOLLOWING IS USED JUST FOR VISUALIZATION
            # (5)
            {'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 20.0,
                'pitch': -90.0, 'roll': 0.0, 'yaw': 0.0,
                'width': 500, 'height': 500, 'fov': 90,
                'id': 'Bev_RGB'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        # NUT
        if self.waypoint_tqdm_bar is None:
            self.waypoint_tqdm_bar = tqdm(total=len(self._global_plan_world_coord))
            self.waypoint_tqdm_bar.update(1)
                # Execute one step of navigation.
        
        # @@@@@@@@@@@@@@@@
        # @ GET THE DATA @
        # @@@@@@@@@@@@@@@@
        # Input data is a dict of tuple, each tuple contains as first element
        # the frame id and as second element the data

        front_image =   input_data['Front'][1]     # (1)
        lidar_data =    input_data['Lidar'][1]     # (2)
        gps_data =      input_data['GPS'][1]       # (3)
        imu_data =      input_data['IMU'][1]       # (4)
        bev_rgb_image = input_data['Bev_RGB'][1]   # (5) (not used by the net)

        # Process image
        front_image = front_image[:, :, :3]
        # Process GPS
        gps_data = utils.convert_gps_to_carla(np.expand_dims(gps_data, 0))[0]
        car_location = (gps_data[0], gps_data[1], gps_data[2]) # m
        # Process Compass
        compass = - imu_data[-1] + np.pi # rad
        # Process Lidar Bev
        lidar_bev = utils.lidar_to_histogram_features(lidar_data[:, :3])[0]
        lidar_bev = np.rot90(lidar_bev)
        # Process Speed
        if self.last_location is None:
            speed = np.array([0])
        else:
            curr_x = car_location[0]
            curr_y = car_location[1]
            last_x = self.last_location[0]
            last_y = self.last_location[1]
            ellapsed_distance = np.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
            ellapsed_time = 1. / 20.
            speed = np.array([ellapsed_distance / ellapsed_time])
        # Process Waypoint Point
        if self.next_waypoint_index >= len(self._global_plan_world_coord):
            return carla.VehicleControl()
        next_waypoint_location = self._global_plan_world_coord[self.next_waypoint_index][0].location
        print(f"next_waypoint_location = [{next_waypoint_location.x}; {next_waypoint_location.y}]")
        print(f"car_location = [{car_location[0]}; {car_location[1]}]")
        target_point = np.array([next_waypoint_location.x - car_location[0], next_waypoint_location.y - car_location[1]])
        rotation_matrix = np.array([[np.cos(compass),     -np.sin(compass)],
                                    [np.sin(compass),      np.cos(compass)]])
        target_point = rotation_matrix@target_point
        distance_car_next_waypoint = np.sqrt(target_point[0]**2 + target_point[1]**2)
        if distance_car_next_waypoint < config.MINIMUM_DISTANCE_FOR_NEXT_TARGETPOINT:
            self.waypoint_tqdm_bar.update(1)
            self.next_waypoint_index += 1
        if self.next_waypoint_index >= len(self._global_plan_world_coord):
            self.waypoint_tqdm_bar.close()

        if self.last_front_image is None:
            self.last_front_image = front_image
            return carla.VehicleControl()

        # @@@@@@@@@@@@@@@@@
        # @ SHOW THE DATA @
        # @@@@@@@@@@@@@@@@@
        # Show the input of the network
        # cv2.imshow('FRONT RGB', front_image)  
        cv2.imshow('BEV RGB', bev_rgb_image)  
        # cv2.imshow('BEV LIDAR', lidar_bev)  

        # @@@@@@@@@@@@@@@@@@@@@
        # @ EXECUTE THE MODEL @
        # @@@@@@@@@@@@@@@@@@@@@

        # RGB
        rgb_a = torch.from_numpy(self.last_front_image)[None, :].permute(0, 3, 1, 2).contiguous().to(self.device, dtype=torch.float32)
        rgb_b = torch.from_numpy(front_image)[None, :].permute(0, 3, 1, 2).contiguous().to(self.device, dtype=torch.float32)
        if self.predicting_flow:
            rgb = torch.concatenate([rgb_a, rgb_b], dim=1)
        else:
            rgb = rgb_a
        
        # LIDAR
        lidar_bev = torch.from_numpy(lidar_bev.copy())[None, None, :].contiguous().to(self.device, dtype=torch.float32)
        
        # TARGET POINT
        target_point = torch.from_numpy(target_point)[None, :].to(self.device, dtype=torch.float32)

        # SPEED
        speed_for_model = torch.from_numpy(speed)[None, :].to(self.device, dtype=torch.float32)

        predictions = self.model(   rgb=rgb,
                                    lidar_bev=lidar_bev,
                                    target_point=target_point,
                                    ego_vel=speed_for_model,
                                    command=None)

        pred_target_speed = predictions[1]
        pred_waypoints = predictions[2]
        pred_semantic = predictions[3]
        pred_bev_semantic = predictions[4]
        pred_depth = predictions[5]
        if self.predicting_flow:
           pred_flow = predictions[-1]
        
        # Get Images from Model Output
        depth_image = utils.create_depth_comparison(predicted_depth=pred_depth)
        semantic_image = utils.create_semantic_comparison(predicted_semantic=pred_semantic)
        bev_semantic_image = utils.create_semantic_comparison(predicted_semantic=pred_bev_semantic, concatenate_vertically=False)
        waypoints_image = utils.create_waypoints_comparison(    pred_bev_semantic=pred_bev_semantic,
                                                                prediction_target_speed=pred_target_speed,
                                                                prediction_waypoints=pred_waypoints,
                                                                actual_speed=speed_for_model,
                                                                target_point=target_point
                                                            )
        if self.predicting_flow:
            flow_image = utils.create_flow_comparison(predicted_flow=pred_flow, label_flow=None)
        
        # Get controls from model output
        steer, throttle, brake = self.model.control_pid(torch.flip(pred_waypoints, dims=(2, )), speed_for_model)
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @ SHOW THE OUTPUT OF THE NETWORK @
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # cv2.imshow('FRONT SEMANTIC', semantic_image)  
        # cv2.imshow('BEV SEMANTIC', bev_semantic_image)  
        # cv2.imshow('FRONT DEPTH', depth_image)
        cv2.imshow('WAYPOINTS', waypoints_image)
        if self.predicting_flow:
            # cv2.imshow('FRONT FLOW', flow_image)
            pass
        cv2.waitKey(10) 


        # @@@@@@@@@@@@@@@@@@@@@@
        # @ SET LAST VARIABLES @
        # @@@@@@@@@@@@@@@@@@@@@@
        self.last_location = car_location
        self.last_front_image = front_image

        # END NUT
        self.agent_engaged = True
        self._hic.run_interface(input_data)

        control = self._controller.parse_events(timestamp - self.prev_timestamp)
        self.prev_timestamp = timestamp

        return control

    def destroy(self):
        """
        Cleanup
        """
        self._hic.quit_interface = True


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self, path_to_conf_file):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # Get the mode
        if path_to_conf_file:

            with (open(path_to_conf_file, "r")) as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {'records': []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except ValueError:
                        # Moving to Python 3.5+ this can be replaced with json.JSONDecodeError
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):
        """
        Parses the json file into a list of carla.VehicleControl
        """

        # transform strs into VehicleControl commands
        for entry in self._records['records']:
            control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                           steer=entry['control']['steer'],
                                           brake=entry['control']['brake'],
                                           hand_brake=entry['control']['hand_brake'],
                                           reverse=entry['control']['reverse'],
                                           manual_gear_shift=entry['control']['manual_gear_shift'],
                                           gear=entry['control']['gear'])
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.6
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):
        """
        Gets the control corresponding to the current frame
        """

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        """
        Saves the list of control into a json file
        """

        new_record = {
            'control': {
                'throttle': self._control.throttle,
                'steer': self._control.steer,
                'brake': self._control.brake,
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            }
        }

        self._log_data['records'].append(new_record)

    def __del__(self):
        """
        Delete method
        """
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)
