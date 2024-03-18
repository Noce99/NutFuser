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

FRAME = 1
MAX_FRAME = 400
ACTUAL_SECTION = 0

def set_section_title(screen, title):
    txt_img = FONT.render(title, True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 20))

titles_dict = {0: "RGB_A", 1: "RGB_B", 2: "DEPTH", 3: "SEMANTIC", 4: "OPTICAL_FLOW", 5: "BEV", 6: "GPS"}
folders_dict = {0: "rgb_A", 1: "rgb_B", 2: "depth", 3: "semantic", 4: "optical_flow"}

def print_section(screen, number, dataset_path):
    global ACTUAL_SECTION
    ACTUAL_SECTION = number
    screen.fill(BACK_GROUD_COLOR)
    set_section_title(screen, titles_dict[number])
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
                my_array = optical_flow_to_human(img_path)
                my_array = np.rot90(my_array)
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
        img = pygame.surfarray.make_surface(my_array)
        img.convert()
        rect = img.get_rect()
        x = config.BEV_IMAGE_W*2 + config.BEV_IMAGE_W
        y = SIDE_SPACE_SIZE + config.BEV_IMAGE_H
        rect.center = x, y
        screen.blit(img, rect)
    elif number == 6:
        # GPS
        gps_path = os.path.join(dataset_path, "gps.npy")
        gps_data = np.load(gps_path)
        points_in_m = analyze_gps(gps_data)
        for point in points_in_m:
            pygame.draw.circle(screen, (255, 0, 0), (point[0]*600+100, point[1]*600+100), 1)
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
            for i in range(1, max_index+1):
                if i not in all_index:
                    raise Exception(utils.color_error_string(f"Missing frame {i} in '{os.path.join(dataset_path, folder)}'\n[{all_frames}]")) 
        except:
            raise Exception(utils.color_error_string(f"Some strange frame name inside '{os.path.join(dataset_path, folder)}'\n[{all_frames}]")) 
    global MAX_FRAME
    if MAX_FRAME is None:
        raise Exception(utils.color_error_string(f"WTF?! MAX_FRAME is None?")) 
    MAX_FRAME = max_index

def optical_flow_to_human(optical_flow_path):
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

def analyze_gps(gps_array):
    def convert_gps_to_carla(gps_array):
        gps = gps_array * np.array([111324.60662786, 111319.490945, 1])
        gps = np.concatenate([  np.expand_dims(gps[:, 1], 1),
                                np.expand_dims(-gps[:, 0], 1),
                                np.expand_dims(gps[:, 2], 1)], 1)
        return gps
    points_in_m = convert_gps_to_carla(gps_array)
    points_in_m -= points_in_m[0]
    max_x = np.max(points_in_m[:, 0])
    min_x = np.min(points_in_m[:, 0])
    max_y = np.max(points_in_m[:, 1])
    min_y = np.min(points_in_m[:, 1])
    points_in_m[:, 0] = (points_in_m[:, 0] - min_x) / (max_x - min_x)
    points_in_m[:, 1] = (points_in_m[:, 1] - min_y) / (max_y - min_y)
    return points_in_m

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
                    if FRAME < 1:
                        FRAME = 1
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
    pygame.quit()
                
