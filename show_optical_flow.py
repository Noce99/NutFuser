import cv2
import numpy as np

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    #flow = flow[:,:,::-1].astype(np.float32)
    #flow, valid = flow[:, :, :2], flow[:, :, 2]
    # flow = (flow - 2**15) / 64.0
    flow = flow / 255
    valid = None
    return flow, valid

flow, valid = readFlowKITTI("/home/enrico/Projects/Carla/tmp_experiment/optical_flow_0/10.png")


hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
hsv[:, :, 1] = 255

print(np.min(flow[:, :, 0]))
print(np.max(flow[:, :, 0]))
print(np.min(flow[:, :, 1]))
print(np.max(flow[:, :, 1]))

mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
print(np.min(mag))
print(np.max(mag))
print(np.min(ang))
print(np.max(ang))
hsv[:, :, 0] = ang * 180 / np.pi / 2
hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("colored flow", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()