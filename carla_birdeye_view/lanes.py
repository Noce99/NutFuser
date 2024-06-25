import carla
import numpy as np
import cv2
from enum import IntEnum

"""
This file is mostly a copy&paste from "Learning by Cheating" code.
It requires refactor.
"""


class LaneSide(IntEnum):
    LEFT = -1
    RIGHT = 1


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def draw_solid_line(canvas, color, closed, points, width):
    """Draws solid lines in a surface given a set of points, width and color"""
    if len(points) >= 2:
        cv2.polylines(
            img=canvas,
            pts=np.int32([points]),
            isClosed=closed,
            color=color,
            thickness=width,
        )


def draw_broken_line(canvas, color, closed, points, width):
    """Draws broken lines in a surface given a set of points, width and color"""
    # Select which lines are going to be rendered from the set of lines
    broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

    # Draw selected lines
    for line in broken_lines:
        cv2.polylines(
            img=canvas,
            pts=np.int32([line]),
            isClosed=closed,
            color=color,
            thickness=width,
        )


def get_lane_markings(
    lane_marking_type,
    lane_marking_color,
    waypoints,
    side: LaneSide,
    location_to_pixel_func,
):
    """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken),
    it converts them as a combination of Broken and Solid lines.
    """
    margin = 0.25
    sign = side.value
    marking_1 = [
        location_to_pixel_func(lateral_shift(w.transform, sign * w.lane_width * 0.5))
        for w in waypoints
    ]
    if lane_marking_type == carla.LaneMarkingType.Broken or (
        lane_marking_type == carla.LaneMarkingType.Solid
    ):
        return [(lane_marking_type, lane_marking_color, marking_1)]
    else:
        marking_2 = [
            location_to_pixel_func(
                lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2))
            )
            for w in waypoints
        ]
        if lane_marking_type == carla.LaneMarkingType.SolidBroken:
            return [
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
            return [
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
            return [
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
            ]
        elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
            return [
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
            ]
    return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]


def draw_lane_marking_single_side(
    surface, waypoints, side: LaneSide, location_to_pixel_func, color
):
    """Draws the lane marking given a set of waypoints and decides
    whether drawing the right or left side of the waypoint based on the sign parameter
    """
    previous_marking_type = carla.LaneMarkingType.NONE
    markings_list = []
    temp_waypoints = []
    current_lane_marking = carla.LaneMarkingType.NONE
    for sample in waypoints:
        lane_marking = (
            sample.left_lane_marking
            if side is LaneSide.LEFT
            else sample.right_lane_marking
        )

        if lane_marking is None:
            continue

        marking_type = lane_marking.type
        marking_color = lane_marking.color

        if current_lane_marking != marking_type:
            # Get the list of lane markings to draw
            markings = get_lane_markings(
                previous_marking_type,
                color,  # lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                side,
                location_to_pixel_func,
            )
            current_lane_marking = marking_type

            # Append each lane marking in the list
            for marking in markings:
                markings_list.append(marking)

            temp_waypoints = temp_waypoints[-1:]

        else:
            temp_waypoints.append((sample))
            previous_marking_type = marking_type

    # Add last marking
    last_markings = get_lane_markings(
        previous_marking_type,
        color,  # lane_marking_color_to_tango(previous_marking_color),
        temp_waypoints,
        side,
        location_to_pixel_func,
    )

    for marking in last_markings:
        markings_list.append(marking)

    # Once the lane markings have been simplified to Solid or Broken lines, we draw them
    for markings in markings_list:
        if markings[0] == carla.LaneMarkingType.Solid:
            draw_solid_line(surface, markings[1], False, markings[2], 1)
        elif markings[0] == carla.LaneMarkingType.Broken:
            draw_broken_line(surface, markings[1], False, markings[2], 1)


def nut_lanes_draw(passable_line, unpassable_line, waypoints, location_to_pixel_func, color, map):
    """
    road_ids = []
    lane_ids = []
    section_ids = []
    for el in waypoints:
        if el.road_id not in road_ids:
            road_ids.append(el.road_id)
        if el.lane_id not in lane_ids:
            lane_ids.append(el.lane_id)
        if el.section_id not in section_ids:
            section_ids.append(el.section_id)

    print(f"{road_ids} {lane_ids} {section_ids}")
    """
    if not waypoints[0].is_junction:

        list_of_waypoints_in_pixels_coord_left = [
            location_to_pixel_func(lateral_shift(w.transform, -1 * w.lane_width * 0.5))
            for w in waypoints
        ]
        list_of_waypoints_in_pixels_coord_right = [
            location_to_pixel_func(lateral_shift(w.transform, 1 * w.lane_width * 0.5))
            for w in waypoints
        ]
        if waypoints[0].left_lane_marking.type == carla.LaneMarkingType.Broken:
            draw_solid_line(passable_line, color, False, list_of_waypoints_in_pixels_coord_left, 3)
        else:
            draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_left, 3)
        if waypoints[0].right_lane_marking.type == carla.LaneMarkingType.Broken:
            draw_solid_line(passable_line, color, False, list_of_waypoints_in_pixels_coord_right, 3)
        else:
            draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_right, 3)

    else:

        """
        road_id = waypoints[0].road_id
        lane_id = waypoints[0].lane_id
        get_if_there_is_a_lane_on_the_right = map.get_waypoint_xodr(road_id, lane_id-1, 0.0)
        if get_if_there_is_a_lane_on_the_right is None or get_if_there_is_a_lane_on_the_right.lane_type != carla.LaneType.Driving :
            list_of_waypoints_in_pixels_coord_right = [
                location_to_pixel_func(lateral_shift(w.transform, 1 * w.lane_width * 0.5))
                for w in waypoints
            ]
            if waypoints[0].right_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
                draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_right, 3)
            else:
                draw_solid_line(passable_line, color, False, list_of_waypoints_in_pixels_coord_right, 1)

        """
        """
        list_of_waypoints_in_pixels_coord_left = [
            location_to_pixel_func(lateral_shift(w.transform, -1 * w.lane_width * 0.5))
            for w in waypoints
        ]
        list_of_waypoints_in_pixels_coord_right = [
            location_to_pixel_func(lateral_shift(w.transform, 1 * w.lane_width * 0.5))
            for w in waypoints
        ]

        if waypoints[0].lane_change == carla.LaneChange.Both:
            can_turn_right =    True
            can_turn_left =     True
        elif waypoints[0].lane_change == carla.LaneChange.Right:
            can_turn_right =    True
            can_turn_left =     False
        elif waypoints[0].lane_change == carla.LaneChange.Left:
            can_turn_right =    False
            can_turn_left =     True
        else:
            can_turn_right =    False
            can_turn_left =     False

        if waypoints[0].left_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid] and \
                not can_turn_left:
            draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_left, 3)

        if waypoints[0].right_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]and \
                not can_turn_right:
            draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_right, 3)

        """
        """

        junction = waypoints[0].get_junction()
        junction_bb = junction.bounding_box

        center = junction_bb.location
        sizes = junction_bb.extent

        corners = [
            carla.Location(x=center.x-sizes.x, y=center.y-sizes.y),
            carla.Location(x=center.x+sizes.x, y=center.y-sizes.y),
            carla.Location(x=center.x+sizes.x, y=center.y+sizes.y),
            carla.Location(x=center.x-sizes.x, y=center.y+sizes.y),
        ]
        corners = [location_to_pixel_func(loc) for loc in corners]

        # cv.fillPoly(img=passable_line, pts=np.int32([corners]), color=color)

        other_waypoints = junction.get_waypoints(carla.LaneType.Any)
        for w in other_waypoints:
            waypoints_list = w[0].next_until_lane_end(1)
            assert w[0].lane_type == w[1].lane_type
            if w[0].road_id != w[1].road_id:
                print(f"{w[0].road_id} -> {w[1].road_id}")
            if w[0].lane_type == carla.LaneType.Driving:
                for ww in waypoints_list:
                    center = location_to_pixel_func(ww.transform.location)
                    cv.circle(unpassable_line, center, 3, 1, -1)

        """
        """
        list_of_waypoints_in_pixels_coord_left = []
        list_of_waypoints_in_pixels_coord_right = []
        for i, w in enumerate(waypoints):
            if i > 0:
                previous_is_a_junction = waypoints[i-1].is_junction
            else:
                previous_is_a_junction = False
            if i < len(waypoints)-1:
                next_is_a_junction = waypoints[i+1].is_junction
            else:
                next_is_a_junction = False
            if not w.is_junction or not previous_is_a_junction or not next_is_a_junction:
                list_of_waypoints_in_pixels_coord_left.append(location_to_pixel_func(
                    lateral_shift(w.transform, -1 * w.lane_width * 0.5)))
                list_of_waypoints_in_pixels_coord_right.append(location_to_pixel_func(
                    lateral_shift(w.transform, 1 * w.lane_width * 0.5)))


        if waypoints[0].left_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
            draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_left, 3)
        else:
            draw_solid_line(passable_line, color, False, list_of_waypoints_in_pixels_coord_left, 1)
        if waypoints[0].right_lane_marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
            draw_solid_line(unpassable_line, color, False, list_of_waypoints_in_pixels_coord_right, 3)
        else:
            draw_solid_line(passable_line, color, False, list_of_waypoints_in_pixels_coord_right, 1)
        """
