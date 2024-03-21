import pygame
import sys
import os
import cv2
import numpy as np
import argparse
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config
import utils

BACK_GROUD_COLOR = (0, 0, 255)
SECTION_TITLE_COLOR = (255, 0, 0)
SIDE_SPACE_SIZE = 300

FONT = None
SMALL_FONT = None

FRAME = 0
MAX_FRAME = 400
ACTUAL_SECTION = 0

def set_section_title(screen, title):
    txt_img = FONT.render(title, True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 20))

def show_frame_num_and_speeds(screen, dataset_path):
    previous_speeds_path = os.path.join(dataset_path, "previous_speeds.npy")
    next_speeds_path = os.path.join(dataset_path, "next_speeds.npy")
    if os.path.isfile(previous_speeds_path) and os.path.isfile(next_speeds_path):
        previous_speeds_array = np.load(previous_speeds_path)
        next_speeds_array = np.load(next_speeds_path)
        txt_img = SMALL_FONT.render(f"actual : {previous_speeds_array[FRAME]*3.6:.4f} km/h", True, SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 80))
        txt_img = SMALL_FONT.render(f"next : {next_speeds_array[FRAME]*3.6:.4f} km/h", True, SECTION_TITLE_COLOR)
        screen.blit(txt_img, (20, 100))
    else:
        print("No speed files found!")
    txt_img = SMALL_FONT.render(f"FPS : {FRAME}/{MAX_FRAME}", True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 60))

titles_dict = {0: "RGB_A", 1: "RGB_B", 2: "DEPTH", 3: "SEMANTIC", 4: "OPTICAL_FLOW", 5: "BEV", 6: "ALL GPS", 7: "FRAME GPS", 8:"TESTS"}
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
        for i in range(4):
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
                my_array = optical_flow_to_human_slow(img_path)
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
        # SEMANTIC
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
        frame_waypoints_path = os.path.join(dataset_path, "frame_waypoints.npy")
        if os.path.isfile(frame_waypoints_path):
            frame_waypoints = np.load(frame_waypoints_path)
            for i in range(frame_waypoints.shape[1]):
                xx = x - frame_waypoints[FRAME, i, 0]*config.BEV_IMAGE_W/config.BEV_SQUARE_SIDE_IN_M * 2
                yy = y - frame_waypoints[FRAME, i, 1]*config.BEV_IMAGE_W/config.BEV_SQUARE_SIDE_IN_M * 2
                pygame.draw.circle(screen, (255, 0, 0), (xx, yy), 4)

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
            pygame.draw.circle(screen, (255, 0, 0), (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum), 1)
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
        # FRAME GPS
        all_gps_lat_lon_path = os.path.join(dataset_path, "all_gps_positions.npy")
        frame_gps_lat_lon_path = os.path.join(dataset_path, "frame_gps_positions.npy")
        all_gps_lat_lon_data = np.load(all_gps_lat_lon_path)
        frame_gps_lat_lon_data = np.load(frame_gps_lat_lon_path)
        how_many_frame_to_get = 40
        all_gps_lat_lon_data = all_gps_lat_lon_data[(FRAME)*config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE:(FRAME+how_many_frame_to_get+1)*config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE, :]
        frame_gps_lat_lon_data = frame_gps_lat_lon_data[FRAME-1:FRAME+how_many_frame_to_get, :]
        all_gps_positions, origin, den_x, den_y, min_x, min_y = utils.lat_lon_to_normalize_carla_cords(all_gps_lat_lon_data)
        frame_gps_positions, _, _, _, _, _ = utils.lat_lon_to_normalize_carla_cords(frame_gps_lat_lon_data, origin, den_x, den_y, min_x, min_y)
        point_each_meter, subframe_position_point_each_meter = utils.calculate_point_each_meter(all_gps_positions, den_x, den_y)
        exact_number_of_frame_to_get = int(subframe_position_point_each_meter[10] / 30)
        all_gps_lat_lon_data = all_gps_lat_lon_data[:(exact_number_of_frame_to_get+1)*config.AMMOUNT_OF_CARLA_FRAME_AFTER_WE_SAVE, :]
        frame_gps_lat_lon_data = frame_gps_lat_lon_data[:exact_number_of_frame_to_get+1, :]
        all_gps_positions, origin, den_x, den_y, min_x, min_y = utils.lat_lon_to_normalize_carla_cords(all_gps_lat_lon_data)
        frame_gps_positions, _, _, _, _, _ = utils.lat_lon_to_normalize_carla_cords(frame_gps_lat_lon_data, origin, den_x, den_y, min_x, min_y)
        point_each_meter, subframe_position_point_each_meter = utils.calculate_point_each_meter(all_gps_positions, den_x, den_y)
        # now we visualize
        font = pygame.font.Font("freesansbold.ttf", 32)
        border = 100
        window_W, window_H = pygame.display.get_surface().get_size()
        window_size = min(window_W, window_H) - border*2
        if window_W > window_H:
            H_to_sum = 0
            W_to_sum = (window_W - window_H) / 2
        else:
            H_to_sum = (window_H - window_W) / 2
            W_to_sum = 0
        for i, point in enumerate(frame_gps_positions):
            pygame.draw.circle(screen, (0, 255, 0), (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum), 5)
            text = font.render(f"{FRAME+i}", True, (0, 255, 0))
            textRect = text.get_rect()
            textRect.center = (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum + 30)
            screen.blit(text, textRect)
        for i, point in enumerate(point_each_meter):
            if i < 10:
                color = (255, 0, 0)
                radius = 3
            else:
                color = (255, 255, 0)
                radius = 2
            pygame.draw.circle(screen, color, (point[0]*window_size+border+W_to_sum, point[1]*window_size+border+H_to_sum), radius)
    elif number == 8:
        # WAYPOINTS
        frame_waypoints_path = os.path.join(dataset_path, "frame_waypoints.npy")
        if os.path.isfile(frame_waypoints_path):
            frame_waypoints = np.load(frame_waypoints_path)
            for i in range(frame_waypoints.shape[1]):
                pygame.draw.circle(screen, (255, 0, 0), (frame_waypoints[FRAME-1, i, 0]*30+600, -frame_waypoints[FRAME-1, i, 1]*30+600), 4)

    pygame.display.update()

def check_dataset_folder(dataset_path):
    all_files = os.listdir(dataset_path)
    max_index = None
    for folder, extention in config.DATASET_FOLDER_STRUCT:
        if folder not in all_files:
            raise Exception(utils.color_error_string(f"Cannot find out {folder} in '{dataset_path}'")) 
        all_frames = os.listdir(os.path.join(dataset_path, folder))
        try:
            all_index = [int(frame[:-len(extention)]) for frame in all_frames]
            if max_index is None:
                max_index = max(all_index)
            for i in range(0, max_index):
                if i not in all_index:
                    raise Exception(utils.color_error_string(f"Missing frame {i} in '{os.path.join(dataset_path, folder)}'\n[{all_frames}]")) 
        except:
            raise Exception(utils.color_error_string(f"Some strange frame name inside '{os.path.join(dataset_path, folder)}'\n[{all_frames}]")) 
    global MAX_FRAME
    if MAX_FRAME is None:
        raise Exception(utils.color_error_string(f"WTF?! MAX_FRAME is None?")) 
    MAX_FRAME = max_index

def optical_flow_to_human_fast(optical_flow_path):
    flow = cv2.imread(optical_flow_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow.astype(np.float32)
    flow = flow[:, :, :2]
    flow = (flow - 2**15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]
    flow[:, :, 0] /= H * 0.5
    flow[:, :, 1] /= W * 0.5

    output = np.zeros((H, W, 3), dtype=np.uint8)
    output[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    output[..., 0] = ang*180/np.pi/2
    output[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return bgr

def optical_flow_to_human_slow(optical_flow_path):
    flow = cv2.imread(optical_flow_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow.astype(np.float32)
    flow = flow[:, :, :2]
    flow = (flow - 2**15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]
    flow[:, :, 0] /= H * 0.5
    flow[:, :, 1] /= W * 0.5
    output = np.zeros((H, W, 3), dtype=np.uint8)
    rad2ang = 180./math.pi
    for i in range(H):
        for j in range(W):
            vx = flow[i, j, 0] 
            vy = flow[i, j, 1]
            angle = 180. + math.atan2(vy, vx)*rad2ang
            if angle < 0:
                angle = 360. + angle
                pass
            angle = math.fmod(angle, 360.)
            norm = math.sqrt(vx*vx + vy*vy)
            shift = 0.999
            a = 1/math.log(0.1 + shift)
            raw_intensity = a*math.log(norm + shift)
            if raw_intensity < 0.:
                intensity = 0.
            elif raw_intensity > 1.:
                intensity = 1.
            else:
                intensity = raw_intensity
            S = 1.
            V = intensity
            H_60 = angle*1./60.
            C = V * S
            X = C*(1. - abs(math.fmod(H_60, 2.) - 1.))
            m = V - C
            r = 0.
            g = 0.
            b = 0.
            angle_case = int(H_60)
            if angle_case == 0:
                r = C
                g = X
                b = 0
            elif angle_case == 1:
                r = X
                g = C
                b = 0
            elif angle_case == 2:
                r = 0
                g = C
                b = X
            elif angle_case == 3:
                r = 0
                g = X
                b = C
            elif angle_case == 4:
                r = X
                g = 0
                b = C
            elif angle_case == 5:
                r = C
                g = 0
                b = X
            else:
                r = 1
                g = 1
                b = 1
            R = int((r+m)*255)
            G = int((g+m)*255)
            B = int((b+m)*255)
            output[i, j, 0] = R
            output[i, j, 1] = G
            output[i, j, 2] = B
    return output

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--dataset_path',
        help='Path to the Dataset!',
        required=True,
        type=str
    )
    args = argparser.parse_args()
    check_dataset_folder(args.dataset_path)

    pygame.init()
    FONT = pygame.font.SysFont(None, 48)
    SMALL_FONT = pygame.font.SysFont(None, 28)
    screen = pygame.display.set_mode((config.IMAGE_W + SIDE_SPACE_SIZE, config.IMAGE_H*4))
    pygame.display.set_caption("Show Carla's Data")
    running = True
    screen.fill(BACK_GROUD_COLOR)
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
                
