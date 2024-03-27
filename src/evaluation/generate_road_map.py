import sys
import os
sys.path.append("/home/enrico/Progetti/Carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import cv2
import numpy as np
import pygame
from collections import deque

COLOR_WHITE = (255, 255, 255)

def carla_rot_to_mat(carla_rotation):
  """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation
    :return: np.array rotation matrix
  """
  roll = np.deg2rad(carla_rotation.roll)
  pitch = np.deg2rad(carla_rotation.pitch)
  yaw = np.deg2rad(carla_rotation.yaw)

  yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
  pitch_matrix = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
  roll_matrix = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])

  rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
  return rotation_matrix

def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
  """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
  rotation = carla_rot_to_mat(ref_rot_in_global)
  np_vec_in_global = np.array([[target_vec_in_global.x], [target_vec_in_global.y], [target_vec_in_global.z]])
  np_vec_in_ref = rotation.T.dot(np_vec_in_global)
  target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
  return target_vec_in_ref

def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
  """
    :param target_loc_in_global: carla.Location in global coordinate (world, actor)
    :param ref_trans_in_global: carla.Transform in global coordinate (world, actor)
    :return: carla.Location in ref coordinate
    """
  x = target_loc_in_global.x - ref_trans_in_global.location.x
  y = target_loc_in_global.y - ref_trans_in_global.location.y
  z = target_loc_in_global.z - ref_trans_in_global.location.z
  vec_in_global = carla.Vector3D(x=x, y=y, z=z)
  vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)

  target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
  return target_loc_in_ref

def _get_traffic_light_waypoints(traffic_light, carla_map):
  """
    get area of a given traffic light
    adapted from "carla-simulator/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py"
    """
  base_transform = traffic_light.get_transform()
  tv_loc = traffic_light.trigger_volume.location
  tv_ext = traffic_light.trigger_volume.extent

  # Discretize the trigger box into points
  x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes
  area = []
  for x in x_values:
    point_location = base_transform.transform(tv_loc + carla.Location(x=x))
    area.append(point_location)

  # Get the waypoints of these points, removing duplicates
  ini_wps = []
  for pt in area:
    wpx = carla_map.get_waypoint(pt)
    # As x_values are arranged in order, only the last one has to be checked
    if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
      ini_wps.append(wpx)

  # Leaderboard: Advance them until the intersection
  stopline_wps = []
  stopline_vertices = []
  junction_wps = []
  for wpx in ini_wps:
    # Below: just use trigger volume, otherwise it's on the zebra lines.
    # stopline_wps.append(wpx)
    # vec_forward = wpx.transform.get_forward_vector()
    # vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

    # loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
    # loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
    # stopline_vertices.append([loc_left, loc_right])

    while not wpx.is_intersection:
      next_wp = wpx.next(0.5)[0]
      if next_wp and not next_wp.is_intersection:
        wpx = next_wp
      else:
        break
    junction_wps.append(wpx)

    stopline_wps.append(wpx)
    vec_forward = wpx.transform.get_forward_vector()
    vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

    loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
    loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
    stopline_vertices.append([loc_left, loc_right])

  # all paths at junction for this traffic light
  junction_paths = []
  path_wps = []
  wp_queue = deque(junction_wps)
  while len(wp_queue) > 0:
    current_wp = wp_queue.pop()
    path_wps.append(current_wp)
    next_wps = current_wp.next(1.0)
    for next_wp in next_wps:
      if next_wp.is_junction:
        wp_queue.append(next_wp)
      else:
        junction_paths.append(path_wps)
        path_wps = []

  return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices, junction_paths

class TrafficLightHandler:
  """
  Class used to generate stop lines for the traffic lights.
  """
  num_tl = 0
  list_tl_actor = []
  list_tv_loc = []
  list_stopline_wps = []
  list_stopline_vtx = []
  list_junction_paths = []
  carla_map = None

  @staticmethod
  def reset(world):
    TrafficLightHandler.carla_map = world.get_map()

    TrafficLightHandler.num_tl = 0
    TrafficLightHandler.list_tl_actor = []
    TrafficLightHandler.list_tv_loc = []
    TrafficLightHandler.list_stopline_wps = []
    TrafficLightHandler.list_stopline_vtx = []
    TrafficLightHandler.list_junction_paths = []

    all_actors = world.get_actors()
    for actor in all_actors:
      if 'traffic_light' in actor.type_id:
        tv_loc, stopline_wps, stopline_vtx, junction_paths = _get_traffic_light_waypoints(
            actor, TrafficLightHandler.carla_map)

        TrafficLightHandler.list_tl_actor.append(actor)
        TrafficLightHandler.list_tv_loc.append(tv_loc)
        TrafficLightHandler.list_stopline_wps.append(stopline_wps)
        TrafficLightHandler.list_stopline_vtx.append(stopline_vtx)
        TrafficLightHandler.list_junction_paths.append(junction_paths)

        TrafficLightHandler.num_tl += 1

  @staticmethod
  def get_light_state(vehicle, offset=0.0, dist_threshold=15.0):
    '''
        vehicle: carla.Vehicle
        '''
    vec_tra = vehicle.get_transform()
    veh_dir = vec_tra.get_forward_vector()

    hit_loc = vec_tra.transform(carla.Location(x=offset))
    hit_wp = TrafficLightHandler.carla_map.get_waypoint(hit_loc)

    light_loc = None
    light_state = None
    light_id = None
    for i in range(TrafficLightHandler.num_tl):
      traffic_light = TrafficLightHandler.list_tl_actor[i]
      tv_loc = 0.5*TrafficLightHandler.list_stopline_wps[i][0].transform.location \
          + 0.5*TrafficLightHandler.list_stopline_wps[i][-1].transform.location

      distance = np.sqrt((tv_loc.x - hit_loc.x)**2 + (tv_loc.y - hit_loc.y)**2)
      if distance > dist_threshold:
        continue

      for wp in TrafficLightHandler.list_stopline_wps[i]:

        wp_dir = wp.transform.get_forward_vector()
        dot_ve_wp = veh_dir.x * wp_dir.x + veh_dir.y * wp_dir.y + veh_dir.z * wp_dir.z

        wp_1 = wp.previous(4.0)[0]
        same_road = (hit_wp.road_id == wp.road_id) and (hit_wp.lane_id == wp.lane_id)
        same_road_1 = (hit_wp.road_id == wp_1.road_id) and (hit_wp.lane_id == wp_1.lane_id)

        # if (wp.road_id != wp_1.road_id) or (wp.lane_id != wp_1.lane_id):
        #     print(f'Traffic Light Problem: {wp.road_id}={wp_1.road_id}, {wp.lane_id}={wp_1.lane_id}')

        if (same_road or same_road_1) and dot_ve_wp > 0:
          # This light is red and is affecting our lane
          loc_in_ev = loc_global_to_ref(wp.transform.location, vec_tra)
          light_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
          light_state = traffic_light.state
          light_id = traffic_light.id
          break

    return light_state, light_loc, light_id

  @staticmethod
  def get_junctoin_paths(veh_loc, color=0, dist_threshold=50.0):
    if color == 0:
      tl_state = carla.TrafficLightState.Green
    elif color == 1:
      tl_state = carla.TrafficLightState.Yellow
    elif color == 2:
      tl_state = carla.TrafficLightState.Red

    junctoin_paths = []
    for i in range(TrafficLightHandler.num_tl):
      traffic_light = TrafficLightHandler.list_tl_actor[i]
      tv_loc = TrafficLightHandler.list_tv_loc[i]
      if tv_loc.distance(veh_loc) > dist_threshold:
        continue
      if traffic_light.state != tl_state:
        continue

      junctoin_paths += TrafficLightHandler.list_junction_paths[i]

    return junctoin_paths

  @staticmethod
  def get_stopline_vtx(veh_loc, color, dist_threshold=50.0, close_traffic_lights=None):
    if color == 0:
      tl_state = carla.TrafficLightState.Green
    elif color == 1:
      tl_state = carla.TrafficLightState.Yellow
    elif color == 2:
      tl_state = carla.TrafficLightState.Red

    stopline_vtx = []
    for i in range(TrafficLightHandler.num_tl):
      traffic_light = TrafficLightHandler.list_tl_actor[i]
      tv_loc = TrafficLightHandler.list_tv_loc[i]
      if tv_loc.distance(veh_loc) > dist_threshold:
        continue
      if traffic_light.state != tl_state:
        continue
      if close_traffic_lights is not None:
        for close_tl in close_traffic_lights:
          if traffic_light.id == int(close_tl[2]) and close_tl[3]:
            stopline_vtx += TrafficLightHandler.list_stopline_vtx[i]
            break
      else:
        stopline_vtx += TrafficLightHandler.list_stopline_vtx[i]

class MapImage(object):
  """
  Functionality used to store CARLA maps as h5 files.
  """

  @staticmethod
  def draw_map_image(carla_map_local, pixels_per_meter_local, precision=0.05):

    waypoints = carla_map_local.generate_waypoints(2)
    margin = 100
    max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
    max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
    min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
    min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

    world_offset = np.array([min_x, min_y], dtype=np.float32)
    width_in_meters = max(max_x - min_x, max_y - min_y)
    width_in_pixels = round(pixels_per_meter_local * width_in_meters)

    road_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    shoulder_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    parking_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    sidewalk_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_yellow_broken_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_yellow_solid_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_white_broken_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_white_solid_surface = pygame.Surface((width_in_pixels, width_in_pixels))
    lane_marking_all_surface = pygame.Surface((width_in_pixels, width_in_pixels))

    topology = [x[0] for x in carla_map_local.get_topology()]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    for waypoint in topology:
      waypoints = [waypoint]
      # Generate waypoints of a road id. Stop when road id differs
      nxt = waypoint.next(precision)
      if len(nxt) > 0:
        nxt = nxt[0]
        while nxt.road_id == waypoint.road_id:
          waypoints.append(nxt)
          nxt = nxt.next(precision)
          if len(nxt) > 0:
            nxt = nxt[0]
          else:
            break
      # Draw Shoulders, Parkings and Sidewalks
      shoulder = [[], []]
      parking = [[], []]
      sidewalk = [[], []]

      for w in waypoints:
        # Classify lane types until there are no waypoints by going left
        l = w.get_left_lane()
        while l and l.lane_type != carla.LaneType.Driving:
          if l.lane_type == carla.LaneType.Shoulder:
            shoulder[0].append(l)
          if l.lane_type == carla.LaneType.Parking:
            parking[0].append(l)
          if l.lane_type == carla.LaneType.Sidewalk:
            sidewalk[0].append(l)
          l = l.get_left_lane()
        # Classify lane types until there are no waypoints by going right
        r = w.get_right_lane()
        while r and r.lane_type != carla.LaneType.Driving:
          if r.lane_type == carla.LaneType.Shoulder:
            shoulder[1].append(r)
          if r.lane_type == carla.LaneType.Parking:
            parking[1].append(r)
          if r.lane_type == carla.LaneType.Sidewalk:
            sidewalk[1].append(r)
          r = r.get_right_lane()

      MapImage.draw_lane(road_surface, waypoints, COLOR_WHITE, pixels_per_meter_local, world_offset)

      MapImage.draw_lane(sidewalk_surface, sidewalk[0], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(sidewalk_surface, sidewalk[1], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(shoulder_surface, shoulder[0], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(shoulder_surface, shoulder[1], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(parking_surface, parking[0], COLOR_WHITE, pixels_per_meter_local, world_offset)
      MapImage.draw_lane(parking_surface, parking[1], COLOR_WHITE, pixels_per_meter_local, world_offset)

      if not waypoint.is_junction:
        MapImage.draw_lane_marking_single_side(lane_marking_yellow_broken_surface, lane_marking_yellow_solid_surface,
                                               lane_marking_white_broken_surface, lane_marking_white_solid_surface,
                                               lane_marking_all_surface, waypoints, -1, pixels_per_meter_local,
                                               world_offset)
        MapImage.draw_lane_marking_single_side(lane_marking_yellow_broken_surface, lane_marking_yellow_solid_surface,
                                               lane_marking_white_broken_surface, lane_marking_white_solid_surface,
                                               lane_marking_all_surface, waypoints, 1, pixels_per_meter_local,
                                               world_offset)

    # stoplines
    stopline_surface = pygame.Surface((width_in_pixels, width_in_pixels))

    for stopline_vertices in TrafficLightHandler.list_stopline_vtx:
      for loc_left, loc_right in stopline_vertices:
        stopline_points = [
            MapImage.world_to_pixel(loc_left, pixels_per_meter_local, world_offset),
            MapImage.world_to_pixel(loc_right, pixels_per_meter_local, world_offset)
        ]
        MapImage.draw_line(stopline_surface, stopline_points, 2)

    # np.uint8 mask
    def _make_mask(x):
      return pygame.surfarray.array3d(x)[..., 0].astype(np.uint8)

    # make a dict
    dict_masks_local = {
        'road': _make_mask(road_surface),
        'shoulder': _make_mask(shoulder_surface),
        'parking': _make_mask(parking_surface),
        'sidewalk': _make_mask(sidewalk_surface),
        'lane_marking_yellow_broken': _make_mask(lane_marking_yellow_broken_surface),
        'lane_marking_yellow_solid': _make_mask(lane_marking_yellow_solid_surface),
        'lane_marking_white_broken': _make_mask(lane_marking_white_broken_surface),
        'lane_marking_white_solid': _make_mask(lane_marking_white_solid_surface),
        'lane_marking_all': _make_mask(lane_marking_all_surface),
        'stopline': _make_mask(stopline_surface),
        'world_offset': world_offset,
        'pixels_per_meter': pixels_per_meter_local,
        'width_in_meters': width_in_meters,
        'width_in_pixels': width_in_pixels
    }
    return dict_masks_local

  @staticmethod
  def draw_lane_marking_single_side(lane_marking_yellow_broken_surface, lane_marking_yellow_solid_surface,
                                    lane_marking_white_broken_surface, lane_marking_white_solid_surface,
                                    lane_marking_all_surface, waypoints, sign, pixels_per_meter_local, world_offset):
    """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
        the waypoint based on the sign parameter"""
    lane_marking = None

    previous_marking_type = carla.LaneMarkingType.NONE
    previous_marking_color = carla.LaneMarkingColor.Other
    current_lane_marking = carla.LaneMarkingType.NONE

    markings_list = []
    temp_waypoints = []
    for sample in waypoints:
      lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

      if lane_marking is None:
        continue

      if current_lane_marking != lane_marking.type:
        # Get the list of lane markings to draw
        markings = MapImage.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign,
                                              pixels_per_meter_local, world_offset)
        current_lane_marking = lane_marking.type

        # Append each lane marking in the list
        for marking in markings:
          markings_list.append(marking)

        temp_waypoints = temp_waypoints[-1:]

      else:
        temp_waypoints.append((sample))
        previous_marking_type = lane_marking.type
        previous_marking_color = lane_marking.color

    # Add last marking
    last_markings = MapImage.get_lane_markings(previous_marking_type, previous_marking_color, temp_waypoints, sign,
                                               pixels_per_meter_local, world_offset)
    for marking in last_markings:
      markings_list.append(marking)

    # Once the lane markings have been simplified to Solid or Broken lines, we draw them
    for markings in markings_list:
      if markings[1] == carla.LaneMarkingColor.White and markings[0] == carla.LaneMarkingType.Solid:
        MapImage.draw_line(lane_marking_white_solid_surface, markings[2], 1)
      elif markings[1] == carla.LaneMarkingColor.Yellow and markings[0] == carla.LaneMarkingType.Solid:
        MapImage.draw_line(lane_marking_yellow_solid_surface, markings[2], 1)
      elif markings[1] == carla.LaneMarkingColor.White and markings[0] == carla.LaneMarkingType.Broken:
        MapImage.draw_line(lane_marking_white_broken_surface, markings[2], 1)
      elif markings[1] == carla.LaneMarkingColor.Yellow and markings[0] == carla.LaneMarkingType.Broken:
        MapImage.draw_line(lane_marking_yellow_broken_surface, markings[2], 1)

      MapImage.draw_line(lane_marking_all_surface, markings[2], 1)

  @staticmethod
  def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign, pixels_per_meter_local, world_offset):
    """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
            as a combination of Broken and Solid lines"""
    margin = 0.25
    marking_1 = [
        MapImage.world_to_pixel(MapImage.lateral_shift(w.transform, sign * w.lane_width * 0.5), pixels_per_meter_local,
                                world_offset) for w in waypoints
    ]

    if lane_marking_type in (carla.LaneMarkingType.Broken, carla.LaneMarkingType.Solid):
      return [(lane_marking_type, lane_marking_color, marking_1)]
    else:
      marking_2 = [
          MapImage.world_to_pixel(MapImage.lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2)),
                                  pixels_per_meter_local, world_offset) for w in waypoints
      ]
      if lane_marking_type == carla.LaneMarkingType.SolidBroken:
        return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
      elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
        return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
      elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
        return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
      elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
        return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
    return [(carla.LaneMarkingType.NONE, lane_marking_color, marking_1)]

  @staticmethod
  def draw_line(surface, points, width):
    """Draws solid lines in a surface given a set of points, width and color"""
    if len(points) >= 2:
      pygame.draw.lines(surface, COLOR_WHITE, False, points, width)

  @staticmethod
  def draw_lane(surface, wp_list, color, pixels_per_meter_local, world_offset):
    """Renders a single lane in a surface and with a specified color"""
    lane_left_side = [MapImage.lateral_shift(w.transform, -w.lane_width * 0.5) for w in wp_list]
    lane_right_side = [MapImage.lateral_shift(w.transform, w.lane_width * 0.5) for w in wp_list]

    polygon = lane_left_side + list(reversed(lane_right_side))
    polygon = [MapImage.world_to_pixel(x, pixels_per_meter_local, world_offset) for x in polygon]

    if len(polygon) > 2:
      pygame.draw.polygon(surface, color, polygon, 5)
      pygame.draw.polygon(surface, color, polygon)

  @staticmethod
  def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()

  @staticmethod
  def world_to_pixel(location, pixels_per_meter_local, world_offset):
    """Converts the world coordinates to pixel coordinates"""
    x = pixels_per_meter_local * (location.x - world_offset[0])
    y = pixels_per_meter_local * (location.y - world_offset[1])
    return [round(y), round(x)]
  
if __name__ == '__main__':
    pixels_per_meter = 1
    map = "Town15"

    client = carla.Client('localhost', 2000)
    client.set_timeout(30)
    world = client.load_world(map)

    dict_masks = MapImage.draw_map_image(world.get_map(), pixels_per_meter)

    cv2.imwrite(os.path.join("maps_from_top", f"{map}_road.jpg"), dict_masks['road'])
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_shoulder.jpg"), dict_masks['shoulder']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_parking.jpg"), dict_masks['parking']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_sidewalk.jpg"), dict_masks['sidewalk']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_lane_marking_yellow_broken.jpg"), dict_masks['lane_marking_yellow_broken']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_lane_marking_yellow_solid.jpg"), dict_masks['lane_marking_yellow_solid']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_lane_marking_white_broken.jpg"), dict_masks['lane_marking_white_broken']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_lane_marking_white_solid.jpg"), dict_masks['lane_marking_white_solid']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_lane_marking_all.jpg"), dict_masks['lane_marking_all']) 
    cv2.imwrite(os.path.join("maps_from_top", f"{map}_stopline.jpg"), dict_masks['stopline']) 