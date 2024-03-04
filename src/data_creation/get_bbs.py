import carla
import math
import time 
import numpy as np
import config
import cv2

def from_carla_to_bev(hero_transform, object_transform):

    def transform_coordinate(x, y):
        x_in_hero = x - hero_x
        y_in_hero = y - hero_y
        x_in_hero, y_in_hero = rotate_point(x_in_hero,
                                            y_in_hero,
                                            - (hero_transform.rotation.yaw / 180 * math.pi) - math.pi/2)
        bev_x = int((2*x_in_hero / config.BEV_SQUARE_SIDE_IN_M) * config.BEV_IMAGE_W/2 + config.BEV_IMAGE_W/2)
        bev_y = int((2*y_in_hero / config.BEV_SQUARE_SIDE_IN_M) * config.BEV_IMAGE_H/2 + config.BEV_IMAGE_H/2)
        return bev_x, bev_y

    def rotate_point(x, y, theta):
        new_x = x*math.cos(theta) - y*math.sin(theta)
        new_y = x*math.sin(theta) + y*math.cos(theta)
        return new_x, new_y

    hero_x = hero_transform.location.x
    hero_y = hero_transform.location.y
    obj_x = object_transform.location.x
    obj_x_half_size = object_transform.extent.x
    obj_y_half_size = object_transform.extent.y
    obj_y = object_transform.location.y
    obj_theta = object_transform.rotation.yaw / 180 * math.pi
    distance = math.sqrt((hero_x-obj_x)**2 + (hero_y-obj_y)**2)
    if distance > config.MAXIMUM_DISTANCE_FROM_VEHICLE_IN_BEV:
        return None
    points = []
    points.append((- obj_x_half_size, + obj_y_half_size))
    points.append((+ obj_x_half_size, + obj_y_half_size))
    points.append((+ obj_x_half_size, - obj_y_half_size))
    points.append((- obj_x_half_size, - obj_y_half_size))
    for i in range(4):
        points[i] = rotate_point(points[i][0], points[i][1], + obj_theta)
        points[i] = transform_coordinate(obj_x + points[i][0], obj_y + points[i][1])
    return points

def get_bbs_as_bev_image(hero_transform, bbs):
    surroundings = np.zeros((config.BEV_IMAGE_W, config.BEV_IMAGE_H), np.uint8)
    for bb in bbs:
        # bev_coord = from_carla_to_bev(hero.get_transform(), bb)
        bev_coord = from_carla_to_bev(hero_transform, bb)
        if bev_coord is not None:
            pts = np.array(bev_coord)
            pts = pts.reshape((-1, 1, 2))
            surroundings = cv2.fillPoly(surroundings,
                                        pts = [pts],
                                        color = 255)
    return surroundings

if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(120.0)
    world = client.get_world()

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
    
    hero_transform = hero.get_transform()
    car_bbs =           world.get_level_bbs(carla.CityObjectLabel.Car) +\
                        world.get_level_bbs(carla.CityObjectLabel.Bicycle) +\
                        world.get_level_bbs(carla.CityObjectLabel.Bus) +\
                        world.get_level_bbs(carla.CityObjectLabel.Motorcycle) +\
                        world.get_level_bbs(carla.CityObjectLabel.Truck) +\
                        world.get_level_bbs(carla.CityObjectLabel.Train)

    cv2.imwrite("test_bb_representation.jpg", get_bbs_as_bev_image(hero_transform, car_bbs))

