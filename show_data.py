import pygame
import sys
import os
import cv2
import numpy as np
import argparse
import math
import json
import pathlib
from datetime import datetime

import torch

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
        txt_img = SMALL_FONT.render(f"actual : {previous_speeds_array[FRAME] * 3.6:.4f} km/h", True,
                                    SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 80))
        txt_img = SMALL_FONT.render(f"next : {next_speed * 3.6:.4f} km/h", True, SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 100))
    else:
        print("No speed files found!")
    txt_img = SMALL_FONT.render(f"FPS : {FRAME}/{MAX_FRAME}", True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 60))
    if os.path.isfile(acceleration_path):
        acceleration_array = np.load(acceleration_path)
        previous_speeds_array = np.load(previous_speeds_path)
        max_index = np.argmax(acceleration_array[FRAME])
        if previous_speeds_array[FRAME] < 0.5:
            max_index = 0
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
               7: "BEV Semantic 2 + BBS", 8: "TESTS", 9:"BEV COMPARISON"}
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
            x = SIDE_SPACE_SIZE + config.IMAGE_W // 2
            y = config.IMAGE_H * i + config.IMAGE_H // 2
            rect.center = x, y
            screen.blit(img, rect)
    elif number == 5:
        # LIDAR
        folder_path = os.path.join(dataset_path, "bev_lidar")
        img_path = os.path.join(folder_path, f"{FRAME}.png")
        my_array = cv2.imread(img_path)
        my_array = cv2.resize(my_array, (config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2),
                              interpolation=cv2.INTER_NEAREST_EXACT)
        my_array = np.rot90(my_array, k=1)
        my_array = np.flip(my_array, axis=0)
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
        my_array = cv2.resize(my_array, (config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2),
                              interpolation=cv2.INTER_NEAREST_EXACT)
        my_array = my_array * 20
        my_array = np.rot90(my_array, k=1)
        my_array = np.flip(my_array, axis=0)
        img = pygame.surfarray.make_surface(my_array)
        img.convert()
        rect = img.get_rect()
        x = config.BEV_IMAGE_W * 2 + config.BEV_IMAGE_W
        y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
        rect.center = x, y
        screen.blit(img, rect)
        # WAYPOINTS
        frame_waypoints_path = os.path.join(dataset_path, "frame_waypoints.npy")
        if os.path.isfile(frame_waypoints_path):
            frame_waypoints = np.load(frame_waypoints_path)
            for i in range(frame_waypoints.shape[1]):
                xx = x - frame_waypoints[FRAME, i, 0] * config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M * 2
                yy = y - frame_waypoints[FRAME, i, 1] * config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M * 2
                pygame.draw.circle(screen, (255, 0, 0), (xx, yy), 4)
        # TARGETPOINTS
        frame_targetpoints_path = os.path.join(dataset_path, "frame_targetpoints.npy")
        if os.path.isfile(frame_targetpoints_path):
            frame_targetpoints = np.load(frame_targetpoints_path)
            xx = x - frame_targetpoints[FRAME, 0] * config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M * 2
            yy = y - frame_targetpoints[FRAME, 1] * config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M * 2
            pygame.draw.circle(screen, (0, 255, 0), (xx, yy), 6)
            font = pygame.font.Font("freesansbold.ttf", 32)
            distance_of_targetpoint = math.sqrt(frame_targetpoints[FRAME, 0] ** 2 + frame_targetpoints[FRAME, 1] ** 2)
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
        all_gps_positions, origin, den_x, den_y, min_x, min_y = utils.lat_lon_to_normalize_carla_cords(
            all_gps_lat_lon_data)
        frame_gps_positions, _, _, _, _, _ = utils.lat_lon_to_normalize_carla_cords(frame_gps_lat_lon_data, origin,
                                                                                    den_x, den_y, min_x, min_y)
        # visualization follow
        font = pygame.font.Font("freesansbold.ttf", 32)
        border = 20
        window_W, window_H = pygame.display.get_surface().get_size()
        window_size = min(window_W, window_H) - border * 2
        if window_W > window_H:
            H_to_sum = 0
            W_to_sum = (window_W - window_H) / 2
        else:
            H_to_sum = (window_H - window_W) / 2
            W_to_sum = 0
        for point in all_gps_positions:
            pygame.draw.circle(screen, (255, 0, 0),
                               (point[0] * window_size + border + W_to_sum, point[1] * window_size + border + H_to_sum),
                               1)
        for i, point in enumerate(frame_gps_positions):
            if i == (FRAME - 1):
                radius = 5
                color = (255, 0, 255)
            else:
                radius = 3
                color = (0, 255, 0)
            pygame.draw.circle(screen, color,
                               (point[0] * window_size + border + W_to_sum, point[1] * window_size + border + H_to_sum),
                               radius)
            if i == 0 or i == len(frame_gps_positions) - 1:
                if i == 0:
                    text = font.render(f"START", True, (0, 255, 0))
                else:
                    text = font.render(f"FINISH", True, (0, 255, 0))
                textRect = text.get_rect()
                textRect.center = (
                point[0] * window_size + border + W_to_sum, point[1] * window_size + border + H_to_sum + 30)
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
            my_rgb_array = utils.draw_bounding_boxes(my_rgb_array, torch.tensor(bbs))
            img = pygame.surfarray.make_surface(my_rgb_array)
            img.convert()
            rect = img.get_rect()
            x = config.BEV_IMAGE_W
            y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
            rect.center = x, y
            screen.blit(img, rect)
    elif number == 8:
        image = save_image(dataset_path)
        image = np.rot90(image, k=1)
        image = np.flip(image, axis=0)
        image = cv2.resize(image, (int(image.shape[1]*0.6), int(image.shape[0]*0.6)),
                           interpolation=cv2.INTER_NEAREST_EXACT)
        img = pygame.surfarray.make_surface(image)
        img.convert()
        rect = img.get_rect()
        x = SIDE_SPACE_SIZE * 0.8 + config.IMAGE_W // 2
        y = config.IMAGE_H * 4 // 2
        rect.center = x, y
        screen.blit(img, rect)
    elif number == 9:
        if os.path.isdir(os.path.join(dataset_path, "bev_semantic_2")):
            folder_path = os.path.join(dataset_path, "bev_semantic_2")
            img_path = os.path.join(folder_path, f"{FRAME}.png")
            my_array = cv2.imread(img_path)
            my_array = utils.color_semantic_2(my_array)
            my_array = np.rot90(my_array, k=1)
            my_array = np.flip(my_array, axis=0)
            my_array = cv2.resize(my_array, (config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2),
                                  interpolation=cv2.INTER_NEAREST_EXACT)
            img = pygame.surfarray.make_surface(my_array)
            img.convert()
            rect = img.get_rect()
            x = config.BEV_IMAGE_W
            y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
            rect.center = x, y
            screen.blit(img, rect)
        if os.path.isdir(os.path.join(dataset_path, "bev_semantic")):
            folder_path = os.path.join(dataset_path, "bev_semantic")
            img_path = os.path.join(folder_path, f"{FRAME}.png")
            my_array = cv2.imread(img_path)
            my_array = utils.color_semantic(my_array)
            my_array = np.rot90(my_array, k=1)
            my_array = np.flip(my_array, axis=0)
            my_array = cv2.resize(my_array, (config.BEV_IMAGE_H * 2, config.BEV_IMAGE_W * 2),
                                  interpolation=cv2.INTER_NEAREST_EXACT)
            img = pygame.surfarray.make_surface(my_array)
            img.convert()
            rect = img.get_rect()
            x = config.BEV_IMAGE_W * 2 + config.BEV_IMAGE_W
            y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
            rect.center = x, y
            screen.blit(img, rect)
    pygame.display.update()


def save_image(dataset_path):
    # RGB
    folder_path = os.path.join(dataset_path, f"rgb_A_0")
    img_path = os.path.join(folder_path, f"{FRAME}.jpg")
    rgb = cv2.imread(img_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    print(f"RGB = {rgb.shape}")
    # RGB 2
    folder_path = os.path.join(dataset_path, f"rgb_B_0")
    img_path = os.path.join(folder_path, f"{FRAME}.jpg")
    rgb_2 = cv2.imread(img_path)
    rgb_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2RGB)
    print(f"RGB 2 = {rgb.shape}")
    # DEPTH
    folder_path = os.path.join(dataset_path, f"depth_0")
    img_path = os.path.join(folder_path, f"{FRAME}.png")
    depth = cv2.imread(img_path)
    print(f"Depth = {depth.shape}")
    # SEMANTIC
    folder_path = os.path.join(dataset_path, f"semantic_0")
    img_path = os.path.join(folder_path, f"{FRAME}.png")
    semantic = cv2.imread(img_path)
    semantic = utils.color_semantic(semantic)
    print(f"Semantic = {semantic.shape}")
    # OPTICAL FLOW
    folder_path = os.path.join(dataset_path, f"optical_flow_0")
    img_path = os.path.join(folder_path, f"{FRAME}.png")
    flow = utils.optical_flow_to_human_with_path(img_path)
    print(f"Flow = {flow.shape}")
    # LIDAR
    folder_path = os.path.join(dataset_path, "bev_lidar")
    img_path = os.path.join(folder_path, f"{FRAME}.png")
    bev_lidar = cv2.imread(img_path)
    print(f"Bev Lidar = {bev_lidar.shape}")
    # BEV SEMANTIC
    folder_path = os.path.join(dataset_path, "bev_semantic")
    img_path = os.path.join(folder_path, f"{FRAME}.png")
    bev_semantic = cv2.imread(img_path)
    bev_semantic = utils.color_semantic(bev_semantic)
    print(f"Bev Semantic = {bev_semantic.shape}")
    # BEV SEMANTIC 2
    folder_path = os.path.join(dataset_path, "bev_semantic_2")
    img_path = os.path.join(folder_path, f"{FRAME}.png")
    bev_semantic_2 = cv2.imread(img_path)
    bev_semantic_2 = utils.color_semantic_2(bev_semantic_2)
    print(f"Bev Semantic 2 = {bev_semantic_2.shape}")
    # BOUNDING BOXES
    folder_path = os.path.join(dataset_path, "bounding_boxes")
    frame_path = os.path.join(folder_path, f"{FRAME}.json")
    with open(frame_path) as json_data:
        bbs = json.loads(json_data.read())
    bbs_image = np.zeros((config.BEV_IMAGE_H, config.BEV_IMAGE_W, 3), dtype=np.uint8)
    bbs_image = utils.draw_bounding_boxes(bbs_image, torch.tensor(bbs))
    bbs_image = np.rot90(bbs_image, k=3)
    bbs_image = np.fliplr(bbs_image)
    print(f"Bounding Boxes = {bbs_image.shape}")
    # WAYPOINTS
    frame_waypoints_path = os.path.join(dataset_path, "frame_waypoints.npy")
    if os.path.isfile(frame_waypoints_path):
        frame_waypoints = np.load(frame_waypoints_path)
        for i in range(frame_waypoints.shape[1]):
            xx = ((config.BEV_IMAGE_W / 2) - frame_waypoints[FRAME, i, 0] *
                  config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M)
            yy = ((config.BEV_IMAGE_H / 2) - frame_waypoints[FRAME, i, 1] *
                  config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M)
            cv2.circle(bev_lidar, (int(xx), int(yy)), 3, (255, 0, 0), -1)
    # TARGET POINTS
    frame_targetpoints_path = os.path.join(dataset_path, "frame_targetpoints.npy")
    frame_targetpoints = np.load(frame_targetpoints_path)
    xx = ((config.BEV_IMAGE_W / 2) - frame_targetpoints[FRAME, 0] *
          config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M)
    yy = ((config.BEV_IMAGE_H / 2) - frame_targetpoints[FRAME, 1] *
          config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M)
    if xx < 0:
        xx = 0
    if yy < 0:
        yy = 0
    if xx > config.BEV_IMAGE_W:
        xx = config.BEV_IMAGE_W
    if yy > config.BEV_IMAGE_H:
        yy = config.BEV_IMAGE_H
    target_point_x = int(xx)
    target_point_y = int(yy)
    cv2.circle(bev_lidar, (int(xx), int(yy)), 4, (0, 255, 0), -1)
    # PREVIOUS SPEED
    previous_speeds_path = os.path.join(dataset_path, "previous_speeds.npy")
    previous_speeds_array = np.load(previous_speeds_path)
    previous_speed = previous_speeds_array[FRAME]
    # NEXT SPEED
    next_speeds_path = os.path.join(dataset_path, "next_speeds.npy")
    next_speeds_array = np.load(next_speeds_path)
    next_speed = SPEEDS[np.argmax(next_speeds_array[FRAME])]
    # NEXT ACCELERATION
    acceleration_path = os.path.join(dataset_path, "accelerations.npy")
    acceleration_array = np.load(acceleration_path)
    max_index = np.argmax(acceleration_array[FRAME])
    representative_text = None
    if max_index == 0:
        representative_text = "Negative"
    elif max_index == 1:
        representative_text = "Zero"
    elif max_index == 2:
        representative_text = "Positive"
    assert representative_text is not None
    next_acceleration = representative_text

    # RENDER EVERYTHING
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    last_written_part_width = config.BEV_IMAGE_H // 2
    big_image = np.zeros((config.BEV_IMAGE_H*6 + last_written_part_width, config.IMAGE_W, 3), dtype=np.uint8)
    big_image[config.BEV_IMAGE_H * 0:config.BEV_IMAGE_H * 1, :] = rgb
    big_image[config.BEV_IMAGE_H * 1:config.BEV_IMAGE_H * 2, :] = rgb_2
    big_image[config.BEV_IMAGE_H * 2:config.BEV_IMAGE_H * 3, :] = depth
    big_image[config.BEV_IMAGE_H * 3:config.BEV_IMAGE_H * 4, :] = semantic
    big_image[config.BEV_IMAGE_H * 4:config.BEV_IMAGE_H * 5, :] = flow
    big_image[config.BEV_IMAGE_H * 5:config.BEV_IMAGE_H * 6,
              config.BEV_IMAGE_W * 0:config.BEV_IMAGE_W * 1] = bev_lidar
    big_image[config.BEV_IMAGE_H * 5:config.BEV_IMAGE_H * 6,
              config.BEV_IMAGE_W * 1:config.BEV_IMAGE_W * 2] = bev_semantic
    big_image[config.BEV_IMAGE_H * 5:config.BEV_IMAGE_H * 6,
              config.BEV_IMAGE_W * 2:config.BEV_IMAGE_W * 3] = bev_semantic_2
    big_image[config.BEV_IMAGE_H * 5:config.BEV_IMAGE_H * 6,
              config.BEV_IMAGE_W * 3:config.BEV_IMAGE_W * 4] = bbs_image
    org = (50, config.BEV_IMAGE_H * 6 + last_written_part_width//3 * 1)
    big_image = cv2.putText(big_image, f"Actual Speed: {previous_speed * 3.6:.2f} km/h", org, font,
                            font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)
    org = (50,  config.BEV_IMAGE_H * 6 + last_written_part_width//3 * 2)
    big_image = cv2.putText(big_image, f"Target Speed: {next_speed * 3.6:.2f} km/h", org, font,
                            font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)
    org = (550, config.BEV_IMAGE_H * 6 + last_written_part_width // 3 * 2)
    # big_image = cv2.putText(big_image, f"Target Acceleration: {next_acceleration}", org, font,
    #                         font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)

    # PRINT LABELS
    labels = "ABCDE"
    for i in range(5):
        org = (15, config.BEV_IMAGE_H * i + 30)
        cv2.circle(big_image, (org[0]+10, org[1]-10), 20, (255, 255, 255), -1)
        big_image = cv2.putText(big_image, f"{labels[i]}", org, font,
                                font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)
    labels = "FJKL"
    for i in range(4):
        org = (config.BEV_IMAGE_W * i + 15, config.BEV_IMAGE_H * 5 + 30)
        cv2.circle(big_image, (org[0] + 10, org[1] - 10), 20, (255, 255, 255), -1)
        big_image = cv2.putText(big_image, f"{labels[i]}", org, font,
                                font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)

    if os.path.isfile(frame_waypoints_path):
        frame_waypoints = np.load(frame_waypoints_path)
        xx = ((config.BEV_IMAGE_W / 2) - frame_waypoints[FRAME, 0, 0] *
              config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M)
        yy = ((config.BEV_IMAGE_H / 2) - frame_waypoints[FRAME, 0, 1] *
              config.BEV_IMAGE_W / config.BEV_SQUARE_SIDE_IN_M)
        org = (int(xx)-10, config.BEV_IMAGE_H * 5 + int(yy)+40)
        cv2.circle(big_image, (org[0] + 10, org[1] - 10), 20, (255, 255, 255), -1)
        big_image = cv2.putText(big_image, "G", org, font,
                                font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)

    org = (target_point_x, config.BEV_IMAGE_H * 5 + target_point_y)
    if target_point_x < config.BEV_IMAGE_W // 2:
        org = (org[0] + 30, org[1])
    else:
        org = (org[0] - 40, org[1])
    cv2.circle(big_image, (org[0] + 10, org[1] - 10), 20, (255, 255, 255), -1)
    big_image = cv2.putText(big_image, "H", org, font,
                            font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)

    org = (10, config.BEV_IMAGE_H * 6 + last_written_part_width//3 * 1)
    cv2.circle(big_image, (org[0] + 10, org[1] - 10), 20, (255, 255, 255), -1)
    big_image = cv2.putText(big_image, "M", org, font,
                            font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)

    org = (10, config.BEV_IMAGE_H * 6 + last_written_part_width // 3 * 2)
    cv2.circle(big_image, (org[0] + 10, org[1] - 10), 20, (255, 255, 255), -1)
    big_image = cv2.putText(big_image, "N", org, font,
                            font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)

    # org = (510, config.BEV_IMAGE_H * 6 + last_written_part_width // 3 * 2)
    # cv2.circle(big_image, (org[0] + 10, org[1] - 10), 20, (255, 255, 255), -1)
    # big_image = cv2.putText(big_image, "O", org, font,
    #                         font_scale, SECTION_TITLE_COLOR, font_thickness, cv2.LINE_AA)
    return big_image


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
    screen = pygame.display.set_mode((config.IMAGE_W + SIDE_SPACE_SIZE, config.IMAGE_H * 4))
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
                elif event.key == pygame.K_0:
                    print_section(screen, 9, args.dataset_path)
                elif event.key == pygame.K_s:
                    image = save_image(args.dataset_path)
                    path = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets", "show_data.jpg")
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path, image)
    pygame.quit()
