import pygame
import config
import os
import cv2
import numpy as np

config.TMP_DATASET_PATH = "/home/enrico/Downloads/tmp_experiment"

BACK_GROUD_COLOR = (0, 0, 255)
SECTION_TITLE_COLOR = (255, 0, 0)
SIDE_SPACE_SIZE = 300

FONT = None

FRAME = 0
MAX_FRAME = 400
ACTUAL_SECTION = 0

def set_section_title(screen, title):
    txt_img = FONT.render(title, True, SECTION_TITLE_COLOR)
    screen.blit(txt_img, (20, 20))

titles_dict = {0: "RGB_A", 1: "RGB_B", 2: "DEPTH", 3: "SEMANTIC", 4: "OPTICAL_FLOW", 5: "BEV"}
folders_dict = {0: "rgb_A", 1: "rgb_B", 2: "depth", 3: "semantic", 4: "optical_flow_human"}

def print_section(screen, number):
    global ACTUAL_SECTION
    ACTUAL_SECTION = number
    screen.fill(BACK_GROUD_COLOR)
    set_section_title(screen, titles_dict[number])
    if number in [0, 1, 2, 3, 4]:
        extention = ".png"
        if number in [0, 1]:
            extention = ".jpg"
        for i in range(4):
            folder_path = os.path.join(config.TMP_DATASET_PATH, f"{folders_dict[number]}_{i}")
            img_path = os.path.join(folder_path, f"{FRAME}{extention}")
            if number != 3:
                img = pygame.image.load(img_path)
            else:
                my_array = cv2.imread(img_path)
                my_array = my_array * 20
                img = pygame.surfarray.make_surface(my_array)
            img.convert()
            rect = img.get_rect()
            x = SIDE_SPACE_SIZE + config.IMAGE_W//2
            y = config.IMAGE_H * i + config.IMAGE_H//2 
            rect.center = x, y
            screen.blit(img, rect)
    elif number == 5:
        # LIDAR
        folder_path = os.path.join(config.TMP_DATASET_PATH, "bev_lidar")
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
        folder_path = os.path.join(config.TMP_DATASET_PATH, "bev_semantic")
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
    pygame.display.update()

if __name__ == "__main__":
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
                    print("Pressed Key Right")
                    FRAME += 1
                    if FRAME > MAX_FRAME:
                        FRAME = MAX_FRAME
                    print_section(screen, ACTUAL_SECTION)
                elif event.key == pygame.K_LEFT:
                    print("Pressed Key Left")
                    FRAME -= 1
                    if FRAME < 0:
                        FRAME = 0
                    print_section(screen, ACTUAL_SECTION)
                elif event.key == pygame.K_1:
                    print("Pressed 1")
                    print_section(screen, 0)
                elif event.key == pygame.K_2:
                    print("Pressed 2")
                    print_section(screen, 1)
                elif event.key == pygame.K_3:
                    print("Pressed 3")
                    print_section(screen, 2)
                elif event.key == pygame.K_4:
                    print("Pressed 4")
                    print_section(screen, 3)
                elif event.key == pygame.K_5:
                    print("Pressed 5")
                    print_section(screen, 4)
                elif event.key == pygame.K_6:
                    print("Pressed 6")
                    print_section(screen, 5)
    pygame.quit()
                
