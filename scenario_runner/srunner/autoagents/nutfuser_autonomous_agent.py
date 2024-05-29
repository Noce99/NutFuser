#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import cv2
import numpy as np
from nutfuser import utils
from nutfuser import config
from tqdm import tqdm
import torch

class NutfuserAutonomousAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._agent = None
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


    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

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
        if self.waypoint_tqdm_bar is None:
            self.waypoint_tqdm_bar = tqdm(total=len(self._global_plan_world_coord))
            # self.waypoint_tqdm_bar.update(1)
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
            self.next_waypoint_index = len(self._global_plan_world_coord) - 1
        next_waypoint_location = self._global_plan_world_coord[self.next_waypoint_index][0].location
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
            return carla.VehicleControl(steer=float(0), throttle=float(0), brake=float(0))

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
        numpy_lidar_bev = lidar_bev
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
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @ LET'S CONTROL THE VEHICLE @
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # Get controls from model output
        pred_waypoints_for_pid = torch.flip(pred_waypoints, dims=(2, ))
        steer, throttle, brake = self.model.control_pid(pred_waypoints_for_pid, speed_for_model)

        # There I create some rectangles in front of the car
        lidar_bev = np.float32(numpy_lidar_bev)

        """
        rectangle_top_left =        (   int(lidar_bev.shape[0]/2 - config.LIDAR_RECATNGLE_WIDTH/2),
                                        int(lidar_bev.shape[1]/2 - config.LIDAR_RECTANGLE_HEIGHT))
        rectangle_bottom_right =    (   int(lidar_bev.shape[0]/2 + config.LIDAR_RECATNGLE_WIDTH/2),
                                        int(lidar_bev.shape[1]/2))

        rectangle = lidar_bev[rectangle_top_left[1]:rectangle_bottom_right[1],
                              rectangle_top_left[0]:rectangle_bottom_right[0]]

        rectangle /= 255
        ammount_of_obtacles_in_front = np.sum(rectangle > 0.5)
        lidar_bev = cv2.rectangle(  lidar_bev,
                                    rectangle_top_left,
                                    rectangle_bottom_right,
                                    255,
                                    3)
        """
        ammount_of_obtacles_in_front = 0
        processed_pred_waypoints = utils.process_array_and_tensor(pred_waypoints, denormalize=False, data_dims=1, channels=2, dtype=np.float32, argmax=False)
        for i in range(processed_pred_waypoints.shape[0]):
            wp_x = int(128-processed_pred_waypoints[i, 0]*256/config.BEV_SQUARE_SIDE_IN_M)
            wp_y = int(128-processed_pred_waypoints[i, 1]*256/config.BEV_SQUARE_SIDE_IN_M)

            rectangle_top_left =        (   int(wp_x - 10),
                                            int(wp_y - 10))
            rectangle_bottom_right =    (   int(wp_x + 10),
                                            int(wp_y + 10))
            """
            lidar_bev = cv2.rectangle(  lidar_bev,
                                        rectangle_top_left,
                                        rectangle_bottom_right,
                                        255,
                                        1)
            """
            rectangle = lidar_bev[  rectangle_top_left[1]:rectangle_bottom_right[1],
                                    rectangle_top_left[0]:rectangle_bottom_right[0]]
            rectangle /= 255
            ammount_of_obtacles_in_front += np.sum(rectangle > 0.5)


        # There I use the speed prediction and I choose to follow or not it
        vel_indx = torch.argmax(pred_target_speed[0], 0)
        if vel_indx == 0 and ammount_of_obtacles_in_front > config.MINIMUM_AMMOUNT_OF_OBSTACLES_IN_FRONT: # the car want to stop
            control = carla.VehicleControl(steer=float(0), throttle=float(0), brake=float(1.0))
        else:
            control = carla.VehicleControl(steer=float(-steer), throttle=float(throttle), brake=float(brake))
        
        
        # control = carla.VehicleControl(steer=float(-steer), throttle=float(throttle), brake=float(brake))

        # @@@@@@@@@@@@@@@@@
        # @ SHOW THE DATA @
        # @@@@@@@@@@@@@@@@@
        # Show the input of the network
        cv2.imshow('FRONT RGB', front_image)  
        cv2.imshow('BEV RGB', bev_rgb_image)  
        cv2.imshow('BEV LIDAR', lidar_bev)  

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

        return control
        # return self._agent.run_step()
    
    def destroy(self):
        print("Destroying NutfuserAutonomousAgent!")
        cv2.destroyAllWindows() 
        self.waypoint_tqdm_bar.close()

