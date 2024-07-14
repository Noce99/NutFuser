import math

import carla
import logging
import numpy as np
import cv2
import pathlib
import os

from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import List
from filelock import FileLock

from carla_birdeye_view import actors, cache
from carla_birdeye_view.actors import SegregatedActors
from carla_birdeye_view.colors import RGB
from carla_birdeye_view.mask import (
    PixelDimensions,
    Coord,
    CroppingRect,
    MapMaskGenerator,
    Mask,
    COLOR_ON,
    RenderingWindow,
    Dimensions,
)

LOGGER = logging.getLogger(__name__)


BirdView = np.ndarray  # [np.uint8] with shape (height, width, channel)
RgbCanvas = np.ndarray  # [np.uint8] with shape (height, width, 3)


class BirdViewCropType(Enum):
    FRONT_AND_REAR_AREA = auto()  # Freeway mode
    FRONT_AREA_ONLY = auto()  # Like in "Learning by Cheating"


DEFAULT_HEIGHT = 336*4  # its 84m when density is 4px/m
DEFAULT_WIDTH = 150*4  # its 37.5m when density is 4px/m
DEFAULT_CROP_TYPE = BirdViewCropType.FRONT_AND_REAR_AREA


class BirdViewMasks(IntEnum):
    PEDESTRIANS =       10
    RED_LIGHTS =        9
    YELLOW_LIGHTS =     8
    GREEN_LIGHTS =      7
    AGENT =             6
    VEHICLES =          5
    CENTERLINES =       4
    PASSABLE_LANES =    3
    UNPASSABLE_LANES =  2
    CROSSWALKS =        1
    ROAD =              0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))


RGB_BY_MASK = {
    BirdViewMasks.PEDESTRIANS:      RGB.VIOLET,
    BirdViewMasks.RED_LIGHTS:       RGB.RED,
    BirdViewMasks.YELLOW_LIGHTS:    RGB.YELLOW,
    BirdViewMasks.GREEN_LIGHTS:     RGB.GREEN,
    BirdViewMasks.AGENT:            RGB.CHAMELEON,
    BirdViewMasks.VEHICLES:         RGB.ORANGE,
    BirdViewMasks.CENTERLINES:      RGB.CHOCOLATE,
    BirdViewMasks.PASSABLE_LANES:   RGB.WHITE,
    BirdViewMasks.UNPASSABLE_LANES: RGB.RED,
    BirdViewMasks.CROSSWALKS:       RGB.SKY_BLUE,
    BirdViewMasks.ROAD:             RGB.DIM_GRAY,
}

INDEX_BY_MASK = {
    BirdViewMasks.PEDESTRIANS:      1,
    BirdViewMasks.RED_LIGHTS:       2,
    BirdViewMasks.YELLOW_LIGHTS:    3,
    BirdViewMasks.GREEN_LIGHTS:     4,
    BirdViewMasks.AGENT:            5,
    BirdViewMasks.VEHICLES:         6,
    BirdViewMasks.CENTERLINES:      7,
    BirdViewMasks.PASSABLE_LANES:   8,
    BirdViewMasks.UNPASSABLE_LANES: 9,
    BirdViewMasks.CROSSWALKS:       10,
    BirdViewMasks.ROAD:             11,
}

inv_INDEX_BY_MASK = {v: k for k, v in INDEX_BY_MASK.items()}

def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated


def circle_circumscribed_around_rectangle(rect_size: Dimensions) -> float:
    """Returns radius of that circle."""
    a = rect_size.width / 2
    b = rect_size.height / 2
    return float(np.sqrt(np.power(a, 2) + np.power(b, 2)))


def square_fitting_rect_at_any_rotation(rect_size: Dimensions) -> float:
    """Preview: https://pasteboard.co/J1XK62H.png"""
    radius = circle_circumscribed_around_rectangle(rect_size)
    side_length_of_square_circumscribed_around_circle = radius * 2
    return side_length_of_square_circumscribed_around_circle


class BirdViewProducer:
    """Responsible for producing top-down view on the map, following agent's vehicle.

    About BirdView:
    - top-down view, fixed directly above the agent (including vehicle rotation), cropped to desired size
    - consists of stacked layers (masks), each filled with ones and zeros (depends on MaskMaskGenerator implementation).
        Example layers: road, vehicles, pedestrians. 0 indicates -> no presence in that pixel, 1 -> presence
    - convertible to RGB image
    - Rendering full road and lanes masks is computationally expensive, hence caching mechanism is used
    """

    def __init__(
        self,
        client: carla.Client,
        target_size: PixelDimensions,
        render_lanes_on_junctions: bool,
        pixels_per_meter: int = 4,
        crop_type: BirdViewCropType = BirdViewCropType.FRONT_AND_REAR_AREA,
    ) -> None:
        self.client = client
        self.target_size = target_size
        self.pixels_per_meter = pixels_per_meter
        self._crop_type = crop_type

        if crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            rendering_square_size = round(
                square_fitting_rect_at_any_rotation(self.target_size)
            )
        elif crop_type is BirdViewCropType.FRONT_AREA_ONLY:
            # We must keep rendering size from FRONT_AND_REAR_AREA (in order to avoid rotation issues)
            enlarged_size = PixelDimensions(
                width=target_size.width, height=target_size.height * 2
            )
            rendering_square_size = round(
                square_fitting_rect_at_any_rotation(enlarged_size)
            )
        else:
            raise NotImplementedError
        self.rendering_area = PixelDimensions(
            width=rendering_square_size, height=rendering_square_size
        )
        self._world = client.get_world()
        self._map = self._world.get_map()
        self.masks_generator = MapMaskGenerator(
            client,
            pixels_per_meter=pixels_per_meter,
            render_lanes_on_junctions=render_lanes_on_junctions,
        )

        carla_birdeye_view = pathlib.Path(__file__).parent.resolve()
        cache_path = os.path.join(f"{carla_birdeye_view}", f"{self.parametrized_cache_path()}")
        if Path(cache_path).is_file():
            LOGGER.info(f"Loading cache from {cache_path}")
            static_cache = np.load(cache_path)
            self.full_road_cache = static_cache[0]
            self.full_passable_lanes_cache = static_cache[1]
            self.full_unpassable_lanes_cache = static_cache[2]
            self.full_centerlines_cache = static_cache[3]
            self.full_crosswalks_cache = static_cache[4]
            self.full_parked_vehicle = static_cache[5]
            LOGGER.info(f"Loaded static layers from cache file: {cache_path}")
        else:
            LOGGER.warning(
                f"Cache file does not exist, generating cache at {cache_path}"
            )
            self.full_road_cache = self.masks_generator.road_mask()
            self.full_passable_lanes_cache, \
                self.full_unpassable_lanes_cache = self.masks_generator.lanes_mask()
            self.full_centerlines_cache = self.masks_generator.centerlines_mask()
            self.full_crosswalks_cache = self.masks_generator.crosswalks_mask()
            self.full_parked_vehicle = self.masks_generator.parked_vehicle_mask()

            self.full_crosswalks_cache[self.full_road_cache == 0] = 0

            static_cache = np.stack(
                [
                    self.full_road_cache,
                    self.full_passable_lanes_cache,
                    self.full_unpassable_lanes_cache,
                    self.full_centerlines_cache,
                    self.full_crosswalks_cache,
                    self.full_parked_vehicle
                ]
            )
            np.save(cache_path, static_cache, allow_pickle=False)
            LOGGER.info(f"Saved static layers to cache file: {cache_path}")

    def parametrized_cache_path(self) -> str:
        cache_dir = Path("birdview_v3_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        opendrive_content_hash = cache.generate_opendrive_content_hash(self._map)
        cache_filename = (
            f"{self._map.name}__"
            f"px_per_meter={self.pixels_per_meter}__"
            f"opendrive_hash={opendrive_content_hash}__"
            f"margin={mask.MAP_BOUNDARY_MARGIN}.npy"
        )
        return str(cache_dir / cache_filename)

    def produce(self, agent_vehicle: carla.Actor) -> BirdView:
        all_actors = actors.query_all(world=self._world)
        segregated_actors = actors.segregate_by_type(actors=all_actors)
        agent_vehicle_loc = agent_vehicle.get_location()

        # Reusing already generated static masks for whole map
        self.masks_generator.disable_local_rendering_mode()
        agent_global_px_pos = self.masks_generator.location_to_pixel(agent_vehicle_loc)

        cropping_rect = CroppingRect(
            x=int(agent_global_px_pos.x - self.rendering_area.width / 2),
            y=int(agent_global_px_pos.y - self.rendering_area.height / 2),
            width=self.rendering_area.width,
            height=self.rendering_area.height,
        )

        masks = np.zeros(
            shape=(
                len(BirdViewMasks),
                self.rendering_area.height,
                self.rendering_area.width,
            ),
            dtype=np.uint8,
        )
        masks[BirdViewMasks.ROAD.value] = self.full_road_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.PASSABLE_LANES.value] = self.full_passable_lanes_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.UNPASSABLE_LANES.value] = self.full_unpassable_lanes_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.CENTERLINES.value] = self.full_centerlines_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.CROSSWALKS.value] = self.full_crosswalks_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.VEHICLES.value] = self.full_parked_vehicle[
            cropping_rect.vslice, cropping_rect.hslice
        ]

        # Dynamic masks
        rendering_window = RenderingWindow(
            origin=agent_vehicle_loc, area=self.rendering_area
        )
        self.masks_generator.enable_local_rendering_mode(rendering_window)
        masks = self._render_actors_masks(agent_vehicle, segregated_actors, masks)
        cropped_masks = self.apply_agent_following_transformation_to_masks(
            agent_vehicle, masks
        )
        ordered_indices = [mask.value for mask in BirdViewMasks.bottom_to_top()]

        # Let's create the Bounding Boxes around Cars and Walkers
        actor_sum = []

        for el in segregated_actors.vehicles:
            actor_sum.append((el, 1))
        for el in segregated_actors.pedestrians:
            actor_sum.append((el, 0))
        bbs = []

        center = agent_vehicle.get_transform().transform(carla.Location(x=0, y=0))
        rotation_test = agent_vehicle.get_transform().transform(carla.Location(x=1, y=0))
        yaw_agent = - math.atan2(rotation_test.x - center.x, rotation_test.y - center.y) + math.pi / 2

        for actor, label in actor_sum:
            center = actor.get_transform().transform(carla.Location(x=0, y=0))
            rotation_test = actor.get_transform().transform(carla.Location(x=1, y=0))
            yaw = - math.atan2(rotation_test.x - center.x, rotation_test.y - center.y) + math.pi / 2 - yaw_agent

            if (math.sqrt(
                    (center.x - agent_vehicle_loc.x)**2 +
                    (center.y - agent_vehicle_loc.y)**2) * self.pixels_per_meter >
                    self.target_size.width * math.sqrt(2)):
                continue
            width = actor.bounding_box.extent.x * 2
            if width < 1.5:
                width = 1.5
            height = actor.bounding_box.extent.y * 2
            if height < 1.5:
                height = 1.5

            # FROM m to px
            center_x = (center.x - agent_vehicle_loc.x) * self.pixels_per_meter + self.target_size.width / 2
            center_y = (center.y - agent_vehicle_loc.y) * self.pixels_per_meter + self.target_size.height / 2
            center_x_inverted = self.target_size.height - center_x
            width *= self.pixels_per_meter
            height *= self.pixels_per_meter

            bbs.append([int(center_x_inverted), int(center_y), width, height, yaw, label])

        return cropped_masks[:, :, ordered_indices], bbs

    @staticmethod
    def as_rgb(birdview: BirdView) -> RgbCanvas:
        h, w, d = birdview.shape
        assert d == len(BirdViewMasks)
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            rgb_color = RGB_BY_MASK[mask_type]
            mask = birdview[:, :, mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas

    @staticmethod
    def as_carla_semantic(birdview: BirdView) -> RgbCanvas:
        h, w, d = birdview.shape
        assert d == len(BirdViewMasks)
        bw_canvas = np.zeros(shape=(h, w, 1), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            index_color = INDEX_BY_MASK[mask_type]
            mask = birdview[:, :, mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            bw_canvas[nonzero_indices(mask)] = index_color
        return bw_canvas

    def _render_actors_masks(
        self,
        agent_vehicle: carla.Actor,
        segregated_actors: SegregatedActors,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Fill masks with ones and zeros (more precisely called as "bitmask").
        Although numpy dtype is still the same, additional semantic meaning is being added.
        """
        lights_masks = self.masks_generator.traffic_lights_masks(
            segregated_actors.traffic_lights
        )
        red_lights_mask, yellow_lights_mask, green_lights_mask = lights_masks
        masks[BirdViewMasks.RED_LIGHTS.value] = red_lights_mask
        masks[BirdViewMasks.YELLOW_LIGHTS.value] = yellow_lights_mask
        masks[BirdViewMasks.GREEN_LIGHTS.value] = green_lights_mask
        masks[BirdViewMasks.AGENT.value] = self.masks_generator.agent_vehicle_mask(
            agent_vehicle
        )
        masks[BirdViewMasks.VEHICLES.value] = self.masks_generator.vehicles_mask(
            segregated_actors.vehicles
        ) + masks[BirdViewMasks.VEHICLES.value]
        masks[BirdViewMasks.PEDESTRIANS.value] = self.masks_generator.pedestrians_mask(
            segregated_actors.pedestrians
        )
        return masks

    def apply_agent_following_transformation_to_masks(
        self, agent_vehicle: carla.Actor, masks: np.ndarray
    ) -> np.ndarray:
        """Returns image of shape: height, width, channels"""
        agent_transform = agent_vehicle.get_transform()
        angle = (
            agent_transform.rotation.yaw + 90
        )  # vehicle's front will point to the top

        # Rotating around the center
        crop_with_car_in_the_center = masks
        masks_n, h, w = crop_with_car_in_the_center.shape
        rotation_center = Coord(x=w // 2, y=h // 2)

        # warpAffine from OpenCV requires the first two dimensions to be in order: height, width, channels
        crop_with_centered_car = np.transpose(
            crop_with_car_in_the_center, axes=(1, 2, 0)
        )
        rotated = rotate(crop_with_centered_car, angle, center=rotation_center)

        half_width = self.target_size.width // 2
        hslice = slice(rotation_center.x - half_width, rotation_center.x + half_width)

        if self._crop_type is BirdViewCropType.FRONT_AREA_ONLY:
            vslice = slice(
                rotation_center.y - self.target_size.height, rotation_center.y
            )
        elif self._crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            half_height = self.target_size.height // 2
            vslice = slice(
                rotation_center.y - half_height, rotation_center.y + half_height
            )
        else:
            raise NotImplementedError
        assert (
            vslice.start > 0 and hslice.start > 0
        ), "Trying to access negative indexes is not allowed, check for calculation errors!"
        car_on_the_bottom = rotated[vslice, hslice]
        return car_on_the_bottom
