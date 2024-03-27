import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve().parent.resolve()))

from model import LidarCenterNet
from tfpp_config import GlobalConfig
from data_loader import backbone_dataset

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import time


MODEL_PATH = "/home/enrico/Projects/Carla/NutFuser/src/neural_networks/train_logs/cvlab_pc_small_dataset/model_0030.pth"

def use_model(model, data, witch_direction: int):
    rgb = data[f"rgb_A_{witch_direction}"].permute(0, 3, 1, 2).contiguous().to(0, dtype=torch.float32)
    semantic_label = F.one_hot(data[f"semantic_{witch_direction}"][:, :, :, 0].type(torch.LongTensor), 8).permute(0, 3, 1, 2).contiguous().to(0, dtype=torch.float32)
    bev_semantic_label = F.one_hot(torch.rot90(data["bev_semantic"], 3+witch_direction, [1, 2])[:, :, :, 0].type(torch.LongTensor), 6).permute(0, 3, 1, 2).contiguous().to(0, dtype=torch.float32)
    depth_label = (data[f"depth_{witch_direction}"][:, :, :, 0]/255).contiguous().to(0, dtype=torch.float32)
    lidar = torch.rot90(data["bev_lidar"], witch_direction, [1, 2])[:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(0, dtype=torch.float32)

    # print(data["bev_lidar"].shape)
    cv2.imshow("bev_lidar", torch.rot90(data["bev_lidar"], witch_direction, [1, 2]).detach().cpu().numpy()[0, :])

    start = time.time()
    pred_wp,\
    pred_target_speed,\
    pred_checkpoint,\
    pred_semantic, \
    pred_bev_semantic, \
    pred_depth, \
    pred_bounding_box, _, \
    pred_wp_1, \
    selected_path = model(  rgb=rgb,
                            lidar_bev=lidar,
                            target_point=None,
                            ego_vel=None,
                            command=None)
    end = time.time()
    print(f"Inference Time = {end-start} [{1/(end-start):.2f} FPS]")

    pred_depth = (pred_depth[0, :, :].detach().cpu().numpy()*255).astype(np.uint8)
    pred_semantic = torch.argmax(pred_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
    pred_bev_semantic = torch.argmax(pred_bev_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)

    depth_comparison = np.zeros((pred_depth.shape[0]*2, pred_depth.shape[1]), dtype=np.uint8)
    depth_comparison[0:pred_depth.shape[0], :] = pred_depth
    depth_comparison[pred_depth.shape[0]:, :] = data[f"depth_{witch_direction}"][0, :, :, 0]

    semantic_comparison = np.zeros((pred_semantic.shape[0]*2, pred_semantic.shape[1]), dtype=np.uint8)
    semantic_comparison[0:pred_semantic.shape[0], :] = pred_semantic
    semantic_comparison[pred_depth.shape[0]:, :] = data[f"semantic_{witch_direction}"][0, :, :, 0]

    bev_semantic_comparison = np.zeros((pred_bev_semantic.shape[0]*2, pred_bev_semantic.shape[1]), dtype=np.uint8)
    bev_semantic_comparison[0:pred_bev_semantic.shape[0], :] = pred_bev_semantic
    bev_semantic_comparison[pred_bev_semantic.shape[0]:, :] = np.rot90(data["bev_semantic"][0, :, :, 0], 3)

    cv2.imshow("rgb", data[f"rgb_A_{witch_direction}"][0, :].detach().cpu().numpy().astype(np.uint8))

    # cv2.imshow("bev_lidar", torch.rot90(data["bev_lidar"], witch_direction, [1, 2])[0, :, :, 0].detach().cpu().numpy().astype(np.uint8).shape)

    cv2.imshow("depth", depth_comparison)

    cv2.imshow("semantic", semantic_comparison*30)

    cv2.imshow("bev_semantic", bev_semantic_comparison*30)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = LidarCenterNet(GlobalConfig())
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(0)
    model.eval()

    my_dataset = backbone_dataset(rank=0)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)

    i = 0
    for data in my_dataloader:
        use_model(model, data, 1)
        i += 1
        if i > 5:
            exit()
