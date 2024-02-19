import cv2
import os
import numpy as np
from tqdm import tqdm

IMAGE_W = 1024
IMAGE_H = 256
VIDEO_W = IMAGE_W*2+20
VIDEO_H = IMAGE_H*4+5*3
TMP_DATASET_PATH = "/home/enrico/Projects/Carla/tmp_experiment"

frame_num = 0
path_camera_1 = os.path.join(TMP_DATASET_PATH, "camera_1")
path_camera_2 = os.path.join(TMP_DATASET_PATH, "camera_2")
path_camera_3 = os.path.join(TMP_DATASET_PATH, "camera_3")
path_camera_4 = os.path.join(TMP_DATASET_PATH, "camera_4")
path_camera_5 = os.path.join(TMP_DATASET_PATH, "camera_5")
path_camera_6 = os.path.join(TMP_DATASET_PATH, "camera_6")
path_camera_7 = os.path.join(TMP_DATASET_PATH, "camera_7")
path_camera_8 = os.path.join(TMP_DATASET_PATH, "camera_8")

path_output_video = os.path.join(TMP_DATASET_PATH, "video_presentation.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20 
outputvideo = cv2.VideoWriter(path_output_video, fourcc, fps, (VIDEO_W, VIDEO_H))

frames_name = [f for f in os.listdir(path_camera_1)]
def get_int_from_file_name(file_name):
    return int(file_name[:-4])
frames_name = sorted(frames_name, key=get_int_from_file_name)

for i in tqdm(range(len(frames_name))):
    frame_name = frames_name[i]
    frame = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    subframe_1 = cv2.imread(os.path.join(path_camera_1, frame_name))
    subframe_2 = cv2.imread(os.path.join(path_camera_2, frame_name))
    subframe_3 = cv2.imread(os.path.join(path_camera_3, frame_name))
    subframe_4 = cv2.imread(os.path.join(path_camera_4, frame_name))
    subframe_5 = cv2.imread(os.path.join(path_camera_5, frame_name))
    subframe_6 = cv2.imread(os.path.join(path_camera_6, frame_name))
    subframe_7 = cv2.imread(os.path.join(path_camera_7, frame_name))
    subframe_8 = cv2.imread(os.path.join(path_camera_8, frame_name))
    frame[256*0+5*0:256*1+5*0, 1024*0+20*0:1024*1+20*0, :] = subframe_1
    frame[256*1+5*1:256*2+5*1, 1024*0+20*0:1024*1+20*0, :] = subframe_3
    frame[256*2+5*2:256*3+5*2, 1024*0+20*0:1024*1+20*0, :] = subframe_2
    frame[256*3+5*3:256*4+5*3, 1024*0+20*0:1024*1+20*0, :] = subframe_4
    frame[256*0+5*0:256*1+5*0, 1024*1+20*1:1024*2+20*1, :] = subframe_5
    frame[256*1+5*1:256*2+5*1, 1024*1+20*1:1024*2+20*1, :] = subframe_6
    frame[256*2+5*2:256*3+5*2, 1024*1+20*1:1024*2+20*1, :] = subframe_7
    frame[256*3+5*3:256*4+5*3, 1024*1+20*1:1024*2+20*1, :] = subframe_8
    outputvideo.write(frame) 
outputvideo.release()
