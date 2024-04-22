import argparse
import pathlib
import os
import torch
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import cv2

from nutfuser import utils
from nutfuser import config
 

sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), "nutfuser", "neural_networks", "tfpp"))
from nutfuser.neural_networks.tfpp.model import LidarCenterNet
from nutfuser.neural_networks.tfpp.model_original import LidarCenterNet as LidarCenterNetOriginal
from nutfuser.neural_networks.tfpp.config import GlobalConfig
from nutfuser.neural_networks.tfpp.nut_data import backbone_dataset
from nutfuser.neural_networks.tfpp.nut_utils import optical_flow_to_human

def get_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--just_backbone',
        help='Set if you want to evaluate just the backbone!',
        action='store_true'
    )
    argparser.add_argument(
        '--use_flow',
        help='Set if you want to predict also the flow!',
        action='store_true'
    )
    argparser.add_argument(
        '--original_tfpp',
        help='Set if you are giving weights from the original tfpp!',
        action='store_true'
    )
    argparser.add_argument(
        "--weights_path",
        help="Path to the pretrained weights!",
        required=True,
        type=str
    )
    argparser.add_argument(
        '--where_to_save',
        help='Where to save the data!',
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets", "evaluation_output"),
        type=str
    )
    argparser.add_argument(
        "--data_folder",
        help="Where to take the data for validation!",
        required=True,
        type=str
    )
    args = argparser.parse_args()
    # THERE I CHECK THE ARGUMENTS
    if not os.path.isfile(args.weights_path):
        raise  utils.NutException(utils.color_error_string(
            f"The file '{args.weights_path}' does not exist!"))
    # THERE I CHECK THE DATASET
    for folder in tqdm(os.listdir(args.data_folder)):
        if not os.path.isdir(os.path.join(args.data_folder, folder)):
            raise  utils.NutException(utils.color_error_string(
                f"The folder '{os.path.join(args.data_folder, folder)}' does not exist!"))
        if os.path.isdir(os.path.join(args.data_folder, folder)):
                utils.check_dataset_folder(os.path.join(args.data_folder, folder))
    if args.use_flow:
        args.use_flow = 1
    else:
        args.use_flow = 0
    return args

if __name__=="__main__":
    args = get_arguments()
    a_config_file = GlobalConfig()
    a_config_file.use_flow = args.use_flow
    if args.just_backbone:
        a_config_file.use_controller_input_prediction = 0
    if args.original_tfpp:
        print("Original!")
        a_config_file.use_flow = False
        a_config_file.num_bev_semantic_classes = 11
        a_config_file.num_semantic_classes = 7
        del a_config_file.detailed_loss_weights["loss_flow"]
        a_config_file.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        a_config_file.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        model = LidarCenterNetOriginal(a_config_file)
    else:
        model = LidarCenterNet(a_config_file)
    model.cuda()

    try:
        model.load_state_dict(torch.load(args.weights_path), strict=False)
    except Exception as e:
        print(utils.color_error_string(f"Impossible to load weights located in '{args.weights_path}'"))
        print(utils.color_info_string(repr(e)))
        exit()

    dataset = backbone_dataset(rank=0, dataset_path=args.data_folder, use_cache=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device(f'cuda:{0}')


    if not os.path.isdir(args.where_to_save):
        os.mkdir(args.where_to_save)
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    output_dir = os.path.join(args.where_to_save, current_time)
    os.mkdir(output_dir)

    rgb_path = os.path.join(output_dir, "rgb")
    depth_path = os.path.join(output_dir, "depth")
    semantic_path = os.path.join(output_dir, "semantic")
    bev_semantic_path = os.path.join(output_dir, "bev_semantic")

    if args.use_flow:
        flow_path = os.path.join(output_dir, "optical_flow")

    os.mkdir(rgb_path)
    os.mkdir(depth_path)
    os.mkdir(semantic_path)
    os.mkdir(bev_semantic_path)
    if args.use_flow:
        os.mkdir(flow_path)

    for i, data in enumerate(tqdm(dataloader)):
        target_point = None
        command = None
        ego_vel = None

        rgb_a = data["rgb_A_0"].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
        rgb_b = data["rgb_B_0"].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
        if args.use_flow:
            rgb = torch.concatenate([rgb_a, rgb_b], dim=1)
        else:
            rgb = rgb_a
        semantic_label = F.one_hot(data["semantic_0"][:, :, :, 0].type(torch.LongTensor), 8).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
        bev_semantic_label = F.one_hot(torch.rot90(data["bev_semantic"], 3, [1, 2])[:, :, :, 0].type(torch.LongTensor), 6).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
        depth_label = (data["depth_0"][:, :, :, 0]/255).contiguous().to(device, dtype=torch.float32)
        lidar = data["bev_lidar"][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
        flow_label = (data["optical_flow_0"][:, :, :, :2] / 2**15 - 1).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)

        predictions = model(      rgb=rgb,
                                lidar_bev=lidar,
                                target_point=target_point,
                                ego_vel=ego_vel,
                                command=command)
        
        if not args.original_tfpp:
            pred_wp,\
            pred_target_speed,\
            pred_checkpoint,\
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            pred_bounding_box, _, \
            pred_wp_1, \
            selected_path, \
            pred_flow = predictions
        else:
            pred_wp,\
            pred_target_speed,\
            pred_checkpoint,\
            pred_semantic, \
            pred_bev_semantic, \
            pred_depth, \
            pred_bounding_box, _, \
            pred_wp_1, \
            selected_path = predictions
        
        pred_depth = (pred_depth[0, :, :].detach().cpu().numpy()*255).astype(np.uint8)
        pred_semantic = torch.argmax(pred_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
        semantic_label = torch.argmax(semantic_label[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
        pred_bev_semantic = torch.argmax(pred_bev_semantic[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
        bev_semantic_label = torch.argmax(bev_semantic_label[0, :], dim=0).detach().cpu().numpy().astype(np.uint8)
        depth_label = (depth_label[0, :, :].detach().cpu().numpy()*255).astype(np.uint8)
        if args.use_flow:
            pred_flow = ((pred_flow + 1)*(2**15)).permute(0, 2, 3, 1)[0, :, :, :].contiguous().detach().cpu().numpy()
            flow_label = ((flow_label + 1)*(2**15)).permute(0, 2, 3, 1)[0, :, :, :].contiguous().detach().cpu().numpy()
        
        rgb_comparison = rgb_a[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

        depth_comparison = np.zeros((pred_depth.shape[0]*2, pred_depth.shape[1]), dtype=np.uint8)
        depth_comparison[0:pred_depth.shape[0], :] = pred_depth
        depth_comparison[pred_depth.shape[0]:, :] = depth_label

        semantic_comparison = np.zeros((pred_semantic.shape[0]*2, pred_semantic.shape[1]), dtype=np.uint8)
        semantic_comparison[0:pred_semantic.shape[0], :] = pred_semantic * 30
        semantic_comparison[pred_depth.shape[0]:, :] = semantic_label * 30

        bev_semantic_comparison = np.zeros((pred_bev_semantic.shape[0]*2, pred_bev_semantic.shape[1]), dtype=np.uint8)
        bev_semantic_comparison[0:pred_bev_semantic.shape[0], :] = pred_bev_semantic * 30
        bev_semantic_comparison[pred_bev_semantic.shape[0]:, :] = bev_semantic_label * 30

        if args.use_flow:
            flow_comparison = np.zeros((pred_flow.shape[0]*2, pred_flow.shape[1], 3), dtype=np.uint8)
            # print(f"PRED\t0 : [{np.min(pred_flow[:, :, 0])}; {np.max(pred_flow[:, :, 0])}] 1 : [{np.min(pred_flow[:, :, 1])}; {np.max(pred_flow[:, :, 1])}]")
            # print(f"LABEL\t0 : [{np.min(flow_label[:, :, 0])}; {np.max(flow_label[:, :, 0])}] 1 : [{np.min(flow_label[:, :, 1])}; {np.max(flow_label[:, :, 1])}]")
            flow_comparison[0:pred_flow.shape[0], :, :] = optical_flow_to_human(pred_flow)
            flow_comparison[pred_flow.shape[0]:, :, :] = optical_flow_to_human(flow_label[:, :, :2])
        
        cv2.imwrite(os.path.join(rgb_path, f"{i}.png"), rgb_comparison)
        cv2.imwrite(os.path.join(depth_path, f"{i}.png"), depth_comparison)
        cv2.imwrite(os.path.join(semantic_path, f"{i}.png"), semantic_comparison)
        cv2.imwrite(os.path.join(bev_semantic_path, f"{i}.png"), bev_semantic_comparison)

        if args.use_flow:
            cv2.imwrite(os.path.join(flow_path, f"{i}.png"), flow_comparison)






    