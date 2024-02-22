import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import shutil
import math
import numpy as np

def readFlowKITTI(filename):

    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    # flow = flow[:,:,::-1].astype(np.float32)
    flow = flow.astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def convert_folder(num):
    print(f"Converting folder number {num}")
    PATH_OPTICAL_FLOW = f"/home/enrico/Downloads/tmp_experiment/optical_flow_{num}"
    PATH_OUTPUT_DIR = f"/home/enrico/Downloads/tmp_experiment/optical_flow_human_{num}"
    optical_flow_file_list = os.listdir(PATH_OPTICAL_FLOW)
    if os.path.isdir(PATH_OUTPUT_DIR):
        shutil.rmtree(PATH_OUTPUT_DIR)
    os.mkdir(PATH_OUTPUT_DIR)

    for a_file in optical_flow_file_list:
        if a_file[0] == "h":
            continue
        print(f"\tStarting ({a_file})...")
        flow, valid = readFlowKITTI(f"{PATH_OPTICAL_FLOW}/{a_file}")
        #hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        #hsv[:, :, 1] = 255
        #mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        #hsv[:, :, 0] = ang * 180 / np.pi / 2
        #hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        H = flow.shape[0]
        W = flow.shape[1]
        flow[:, :, 0] /= H * 0.5
        flow[:, :, 1] /= W * 0.5
        output = np.zeros((H, W, 4), dtype=np.uint8)
        rad2ang = 180./math.pi
        xs = []
        ys = []
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
                output[i, j, 3] = 255
        cv2.imwrite(f"{PATH_OUTPUT_DIR}/{a_file}", output)
        
for i in range(4):
    convert_folder(i)
