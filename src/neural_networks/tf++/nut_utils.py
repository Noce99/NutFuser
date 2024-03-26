from termcolor import colored
import numpy as np

def color_error_string(string):
    return colored(string, "red", attrs=["bold"]) # , "blink"

def color_info_string(string):
    return colored(string, "yellow", attrs=["bold"])

def optical_flow_to_human(flow):
    flow = (flow - 2**15) / 64.0
    H = flow.shape[0]
    W = flow.shape[1]
    flow[:, :, 0] /= H * 0.5
    flow[:, :, 1] /= W * 0.5
    output = np.zeros((H, W, 3), dtype=np.float32)
    rad2ang = 180./np.pi
    angle = 180. + np.arctan2(flow[:, :, 1], flow[:, :, 0]) * rad2ang
    angle[angle < 0] += 360
    angle = np.fmod(angle, 360.)
    H_60 = angle / 60.
    norm = np.sqrt(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2))
    raw_intensity = 1/np.log(0.1 + 0.999) * np.log(norm + 0.999)
    raw_intensity[raw_intensity < 0.] = 0.
    raw_intensity[raw_intensity > 1.] = 1.
    C = raw_intensity
    X = raw_intensity * (np.ones_like(H_60) - np.abs(np.fmod(H_60, 2,) - np.ones_like(H_60)))
    H_60 = H_60.astype(int)
    output[H_60 == 0] = np.stack([C, X, np.zeros_like(C)], axis = -1)[H_60 == 0]
    output[H_60 == 1] = np.stack([X, C, np.zeros_like(C)], axis = -1)[H_60 == 1]
    output[H_60 == 2] = np.stack([np.zeros_like(C), C, X], axis = -1)[H_60 == 2]
    output[H_60 == 3] = np.stack([np.zeros_like(C), X, C], axis = -1)[H_60 == 3]
    output[H_60 == 4] = np.stack([X, np.zeros_like(C), C], axis = -1)[H_60 == 4]
    output[H_60 == 5] = np.stack([C, np.zeros_like(C), X], axis = -1)[H_60 == 5]
    output[H_60 >= 6] = np.stack([np.ones_like(C), np.ones_like(C), np.ones_like(C)], axis = -1)[H_60 >= 6]

    output *= 255
    output = output.astype(np.uint8)
    return output