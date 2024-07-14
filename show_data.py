import pygame
import sys
import os
import cv2
import numpy as np
import argparse
import math
import json

from nutfuser import config
from nutfuser import utils
from carla_birdeye_view import RGB_BY_MASK, inv_INDEX_BY_MASK

BACK_GROUD_COLOR = (0, 0, 255)
SECTION_TITLE_COLOR = (255, 0, 0)
SIDE_SPACE_SIZE = 300

FONT = None
SMALL_FONT = None

FRAME = 0
MAX_FRAME = 400
ACTUAL_SECTION = 0

CAMERAS_INDEXES = []
SPEEDS = [0, 7, 18, 29]


def set_section_title(screen, title):
    txt_img = FONT.render(title, True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 20))


def show_frame_num_and_speeds(screen, dataset_path):
    previous_speeds_path = os.path.join(dataset_path, "previous_speeds.npy")
    next_speeds_path = os.path.join(dataset_path, "next_speeds.npy")
    acceleration_path = os.path.join(dataset_path, "accelerations.npy")
    if os.path.isfile(previous_speeds_path) and os.path.isfile(next_speeds_path):
        previous_speeds_array = np.load(previous_speeds_path)
        next_speeds_array = np.load(next_speeds_path)
        next_speed = SPEEDS[np.argmax(next_speeds_array[FRAME])]
        txt_img = SMALL_FONT.render(f"actual : {previous_speeds_array[FRAME]*3.6:.4f} km/h", True, SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 80))
        txt_img = SMALL_FONT.render(f"next : {next_speed*3.6:.4f} km/h", True, SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 100))
    else:
        print("No speed files found!")
    txt_img = SMALL_FONT.render(f"FPS : {FRAME}/{MAX_FRAME}", True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 60))
    if os.path.isfile(acceleration_path):
        acceleration_array = np.load(acceleration_path)
        max_index = np.argmax(acceleration_array[FRAME])
        representative_char = None
        if max_index == 0:
            representative_char = "-"
        elif max_index == 1:
            representative_char = "/"
        elif max_index == 2:
            representative_char = "+"
        assert representative_char is not None

        txt_img = SMALL_FONT.render(f"acceleration : {representative_char}", True,
                                    SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 120))


titles_dict = {0: "RGB_A", 1: "RGB_B", 2: "DEPTH", 3: "SEMANTIC", 4: "OPTICAL_FLOW", 5: "BEV", 6: "ALL GPS",
               7: "BEV Semantic 2 + BBS", 8: "TESTS"}
folders_dict = {0: "rgb_A", 1: "rgb_B", 2: "depth", 3: "semantic", 4: "optical_flow"}


def print_section(screen, number, dataset_path):
    global ACTUAL_SECTION
    ACTUAL_SECTION = number
    screen.fill(BACK_GROUD_COLOR)
    set_section_title(screen, titles_dict[number])
    show_frame_num_and_speeds(screen, dataset_path)

    if number in [0, 1, 2, 3, 4]:
        extention = ".png"
        if number in [0, 1]:
            extention = ".jpg"
        for i in CAMERAS_INDEXES:
            folder_path = os.path.join(dataset_path, f"{folders_dict[number]}_{i}")
            img_path = os.path.join(folder_path, f"{FRAME}{extention}")
            if number in [0, 1, 2]:
                img = pygame.image.load(img_path)
            elif number == 3:
                my_array = cv2.imread(img_path)
                my_array = my_array * 20
                my_array = np.rot90(my_array)
                img = pygame.surfarray.make_surface(my_array)
            elif number == 4:
                my_array = utils.optical_flow_to_human_with_path(img_path)
                my_array = np.rot90(my_array)
                my_array = np.flip(my_array, 0)
                img = pygame.surfarray.make_surface(my_array)
            img.convert()
            rect = img.get_rect()
            x = SIDE_SPACE_SIZE + config.IMAGE_W//2
            y = config.IMAGE_H * i + config.IMAGE_H//2
            rect.center = x, y
            screen.blit(img, rect)
    elif number == 5:
        # LIDAR
        folder_path = os.path.join(dataset_path, "bev_lidar")
        img_path = os.path.join(folder_path, f"{FRAME}.png")
        my_array = cv2.imread(img_path)
        my_array = cv2.resize(my_array, (config.BEV_IMAGE_H*2, config.BEV_IMAGE_W*2), interpolation= cv2.INTER_NEAREST_EXACT)
        my_array = np.rot90(my_array, k=1)
        my_array = np.flip(my_array, axis = 0)
        img = pygame.surfarray.make_surface(my_array)
        img.convert()
        rect = img.get_rect()
        x = config.BEV_IMAGE_W
        y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
        rect.center = x, y
        screen.blit(img, rect)
        # BEV SEMANTIC
        folder_path = os.path.join(dataset_path, "bev_semantic")
        img_path = os.path.join(folder_path, f"{FRAME}.png")
        my_array = cv2.imread(img_path)
        my_array = cv2.resize(my_array, (config.BEV_IMAGE_H*2, config.BEV_IMAGE_W*2), interpolation= cv2.INTER_NEAREST_EXACT)
        my_array = my_array * 20
        my_array = np.rot90(my_array, k=1)
        my_array = np.flip(my_array, axis = 0)
        img = pygame.surfarray.make_surface(my_array)
        img.convert()
        rect = img.get_rect()
        x = config.BEV_IMAGE_W*2 + config.BEV_IMAGE_W
        y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
        rect.center = x, y
        screen.blit(img, rect)
        # WAYPOINTS
        frame_waypoints_path = os.path.join(dataset_path, "frame_waypoints.npy")
        if os.path.isfile(frame_waypoints_path):
            frame_waypoints = np.load(frame_waypoints_path)
            for i in range(frame_waypoints.shape[1]):
                xx = x - frame_waypoints[FRAME, i, 0]*config.BEV_IMAGE_W/config.BEV_SQUARE_SIDE_IN_M * 2
                yy = y - frame_waypoints[FRAME, i, 1]*config.BEV_IMAGE_W/config.BEV_SQUARE_SIDE_IN_M * 2
                pygame.draw.circle(screen, (255, 0, 0), (xx, yy), 4)
        # TARGETPOINTS
        frame_targetpoints_path = os.path.join(dataset_path, "frame_targetpoints.npy")
        if os.path.isfile(frame_targetpoints_path):
            frame_targetpoints = np.load(frame_targetpoints_path)
            xx = x - frame_targetpoints[FRAME, 0]*config.BEV_IMAGE_W/config.BEV_SQUARE_SIDE_IN_M * 2
            yy = y - frame_targetpoints[FRAME, 1]*config.BEV_IMAGE_W/config.BEV_SQUARE_SIDE_IN_M * 2
            pygame.draw.circle(screen, (0, 255, 0), (xx, yy), 6)
            font = pygame.font.Font("freesansbold.ttf", 32)
            distance_of_targetpoint = math.sqrt(frame_targetpoints[FRAME, 0]**2 + frame_targetpoints[FRAME, 1]**2)
            text = font.render(f"targetpoint distance = {distance_of_targetpoint:.2f}", True, (0, 255, 0))
            textRect = text.get_rect()
            textRect.center = (x, y + config.BEV_IMAGE_H)
            screen.blit(text, textRect)
    elif number == 6:
        # ALL GPS
        all_gps_lat_lon_path = os.path.join(dataset_path, "all_gps_positions.npy")
        frame_gps_lat_lon_path = os.path.join(dataset_path, "frame_gps_positions.npy")
        all_gps_lat_lon_data = np.load(all_gps_lat_lon_path)
        frame_gps_lat_lon_data = np.load(frame_gps_lat_lon_path)
        all_gps_positions, origin, den_x, den_y, min_x, min_y = utils.lat_lon_to_normalize_carla_cords(all_gps_lat_lon_data)
        frame_gps_positions, _, _, _, _, _ = utils.lat_lon_to_normalize_carla_cords(frame_gps_lat_lon_data, origin, den_x, den_y, min_x, min_y)
        # visualization follow
        font = pygame.font.Font("freesansbold.ttf", 32)
        border = 20
        window_W, window_H = pygame.display.get_surface().get_size()
        window_size = min(window_W, window_H) - border*2
        if window_W > window_H:
            H_to_sum = 0
            W_to_sum = (window_W - window_H) / 2
        else:
            H_to_sum = (window_H - window_W) / 2
            W_to_sum = 0
        for point in all_gps_positions:
            pygame.draw.circle(screen, (255, 0, 0),
                               (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum),
                               1)
        for i, point in enumerate(frame_gps_positions):
            if i == (FRAME-1):
                radius = 5
                color = (255, 0, 255)
            else:
                radius = 3
                color = (0, 255, 0)
            pygame.draw.circle(screen, color, (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum), radius)
            if i == 0 or i == len(frame_gps_positions)-1:
                if i == 0:
                    text = font.render(f"START", True, (0, 255, 0))
                else:
                    text = font.render(f"FINISH", True, (0, 255, 0))
                textRect = text.get_rect()
                textRect.center = (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum + 30)
                screen.blit(text, textRect)
    elif number == 7:
        if os.path.isdir(os.path.join(dataset_path, "bev_semantic_2")):
            folder_path = os.path.join(dataset_path, "bev_semantic_2")
            img_path = os.path.join(folder_path, f"{FRAME}.png")
            my_array = cv2.imread(img_path)
            my_array = cv2.resize(my_array, (config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2),
                                  interpolation=cv2.INTER_NEAREST_EXACT)
            my_array = my_array
            my_array = np.rot90(my_array, k=1)
            my_array = np.flip(my_array, axis=0)
            my_rgb_array = np.zeros((config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2, 3), dtype=np.uint8)
            for key in inv_INDEX_BY_MASK:
                color = RGB_BY_MASK[inv_INDEX_BY_MASK[key]]
                for i in range(3):
                    my_rgb_array[:, :, i][my_array[:, :, 0] == key] = color[i]
            img = pygame.surfarray.make_surface(my_rgb_array)
            img.convert()
            rect = img.get_rect()
            x = config.BEV_IMAGE_W * 2 + config.BEV_IMAGE_W
            y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
            rect.center = x, y
            screen.blit(img, rect)
        if os.path.isdir(os.path.join(dataset_path, "bounding_boxes")):
            folder_path = os.path.join(dataset_path, "bounding_boxes")
            frame_path = os.path.join(folder_path, f"{FRAME}.json")
            with open(frame_path) as json_data:
                bbs = json.loads(json_data.read())
            for bb in bbs:
                bb[0] *= 2
                bb[1] *= 2
                bb[2] *= 2
                bb[3] *= 2
            my_rgb_array = np.zeros((config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2, 3), dtype=np.uint8)
            my_rgb_array = utils.draw_bounding_boxes(my_rgb_array, bbs)
            img = pygame.surfarray.make_surface(my_rgb_array)
            img.convert()
            rect = img.get_rect()
            x = config.BEV_IMAGE_W
            y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
            rect.center = x, y
            screen.blit(img, rect)



    elif number == 8:
        pass
    pygame.display.update()


def custom_check_dataset_folder(dataset_path):
    global CAMERAS_INDEXES
    CAMERAS_INDEXES, max_index = utils.check_dataset_folder(dataset_path)
    global MAX_FRAME
    if MAX_FRAME is None:
        raise Exception(utils.color_error_string(f"WTF?! MAX_FRAME is None?"))
    MAX_FRAME = max_index


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--dataset_path',
        help='Path to the Dataset!',
        required=True,
        type=str
    )
    args = argparser.parse_args()
    custom_check_dataset_folder(args.dataset_path)

    pygame.init()
    FONT = pygame.font.SysFont(None, 48)
    SMALL_FONT = pygame.font.SysFont(None, 28)
    screen = pygame.display.set_mode((config.IMAGE_W + SIDE_SPACE_SIZE, config.IMAGE_H*4))
    pygame.display.set_caption("Show Carla's Data")
    running = True
    screen.fill(BACK_GROUD_COLOR)
    print_section(screen, 0, args.dataset_path)
    pygame.display.update()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    FRAME += 1
                    if FRAME > MAX_FRAME:
                        FRAME = MAX_FRAME
                    print_section(screen, ACTUAL_SECTION, args.dataset_path)
                elif event.key == pygame.K_LEFT:
                    FRAME -= 1
                    if FRAME < 0:
                        FRAME = 0
                    print_section(screen, ACTUAL_SECTION, args.dataset_path)
                elif event.key == pygame.K_l:
                    FRAME += 100
                    if FRAME > MAX_FRAME:
                        FRAME = MAX_FRAME
                    print_section(screen, ACTUAL_SECTION, args.dataset_path)
                elif event.key == pygame.K_k:
                    FRAME -= 100
                    if FRAME < 0:
                        FRAME = 0
                    print_section(screen, ACTUAL_SECTION, args.dataset_path)
                elif event.key == pygame.K_1:
                    print_section(screen, 0, args.dataset_path)
                elif event.key == pygame.K_2:
                    print_section(screen, 1, args.dataset_path)
                elif event.key == pygame.K_3:
                    print_section(screen, 2, args.dataset_path)
                elif event.key == pygame.K_4:
                    print_section(screen, 3, args.dataset_path)
                elif event.key == pygame.K_5:
                    print_section(screen, 4, args.dataset_path)
                elif event.key == pygame.K_6:
                    print_section(screen, 5, args.dataset_path)
                elif event.key == pygame.K_7:
                    print_section(screen, 6, args.dataset_path)
                elif event.key == pygame.K_8:
                    print_section(screen, 7, args.dataset_path)
                elif event.key == pygame.K_9:
                    print_section(screen, 8, args.dataset_path)
    pygame.quit()
