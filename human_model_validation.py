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
from nutfuser.utils import optical_flow_to_human

def get_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--weights_path",
        help="Path to the pretrained weights!",
        required=True,
        type=str
    )
    argparser.add_argument(
        '--second_weights_path',
        help=f'{utils.color_info_string("Optional")} A second path to weights used to have a comparison! (default: None)',
        default=None,
        type=str
    )
    argparser.add_argument(
        '--where_to_save',
        help=f"Where to save the data! (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), 'datasets', 'evaluation_output')})",
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets", "evaluation_output"),
        type=str
    )
    argparser.add_argument(
        "--dataset_folder",
        help=f"Where to take the data for validation! (default: {os.path.join(pathlib.Path(__file__).parent.resolve(), 'datasets', 'evaluation_dataset')})",
        required=False,
        default=os.path.join(pathlib.Path(__file__).parent.resolve(), "datasets", "evaluation_dataset"),
        type=str
    )
    args = argparser.parse_args()
    # THERE I CHECK THE ARGUMENTS
    if not os.path.isfile(args.weights_path):
        raise  utils.NutException(utils.color_error_string(
            f"The file '{args.weights_path}' does not exist!"))
    if args.second_weights_path is not None:
        if not os.path.isfile(args.second_weights_path):
            raise  utils.NutException(utils.color_error_string(
                f"The file '{args.second_weights_path}' does not exist!"))
    # THERE I CHECK THE DATASET
    for folder in tqdm(os.listdir(args.dataset_folder)):
        if not os.path.isdir(os.path.join(args.dataset_folder, folder)):
            raise  utils.NutException(utils.color_error_string(
                f"The folder '{os.path.join(args.dataset_folder, folder)}' does not exist!"))
        if os.path.isdir(os.path.join(args.dataset_folder, folder)):
                utils.check_dataset_folder(os.path.join(args.dataset_folder, folder))
    return args

def infering_the_model(dataset_path, where_to_save, weights_path):
        print(utils.color_info_string(f"Infering {weights_path}"))
        try:
            weights = torch.load(weights_path)
        except Exception as e:
            print(utils.color_error_string(f"Impossible to load weights located in '{weights_path}'!"))
            print(utils.color_info_string(repr(e)))
            exit()

        a_config_file = GlobalConfig()
        predicting_flow = None
        just_a_backbone = None
        tfpp_original = None
        if "flow_decoder.deconv1.0.weight" in weights.keys():
            # Predicting also flow
            a_config_file.use_flow = True
            predicting_flow = True
        else:
            # Not predicting flow
            a_config_file.use_flow = False
            predicting_flow = False
        if "wp_decoder.encoder.weight" not in weights.keys():
            # Just a Backbone
            a_config_file.use_controller_input_prediction = False
            just_a_backbone = True
            # So we suppose it's not the original tfpp network
            tfpp_original = False
        else:
            # A full Network
            a_config_file.use_controller_input_prediction = True
            just_a_backbone = False
            # We use the xtra sensor file to understand if it's an original fpp Network
            # because nutfuser do not use Commands!
            extra_sensor_num = weights["extra_sensor_encoder.0.weight"].shape[1]
            if extra_sensor_num == 7:
                # An original tfpp Network
                a_config_file.use_flow = False
                a_config_file.num_bev_semantic_classes = 11
                a_config_file.num_semantic_classes = 7
                del a_config_file.detailed_loss_weights["loss_flow"]
                a_config_file.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                a_config_file.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                tfpp_original = True
            else:
                tfpp_original = False
        
        print(f"PREDICT FLOW:\t\t{predicting_flow}")
        print(f"JUST A BACKBONE:\t{just_a_backbone}")
        print(f"ORIGINAL TFPP:\t\t{tfpp_original}")

        if tfpp_original:
            model = LidarCenterNetOriginal(a_config_file)
        else:
            model = LidarCenterNet(a_config_file)

        model.cuda()
        model.eval()

        try:
            model.load_state_dict(weights, strict=False)
        except Exception as e:
            print(utils.color_error_string(f"Weight in '{weights_path}' not compatible with the model!"))
            print(utils.color_info_string(repr(e)))
            exit()
        dataset = backbone_dataset(rank=0, dataset_path=dataset_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        device = torch.device(f'cuda:{0}')


        if not os.path.isdir(where_to_save):
            os.mkdir(where_to_save)
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
        name = ""
        if tfpp_original:
            name += "tfpp_"
        else:
            name += "nutfuser_"
        if just_a_backbone:
            name += "backbone_"
        else:
            name += "whole_"
        if predicting_flow:
            name += "flow_"
        else:
            name += "NOflow_"
        name_and_time = name + current_time
        output_dir = os.path.join(where_to_save, name_and_time)
        os.mkdir(output_dir)

        rgb_path = os.path.join(output_dir, "rgb")
        lidar_path = os.path.join(output_dir, "lidar")
        depth_path = os.path.join(output_dir, "depth")
        semantic_path = os.path.join(output_dir, "semantic")
        bev_semantic_path = os.path.join(output_dir, "bev_semantic")

        if predicting_flow:
            flow_path = os.path.join(output_dir, "optical_flow")

        os.mkdir(rgb_path)
        os.mkdir(lidar_path)
        os.mkdir(depth_path)
        os.mkdir(semantic_path)
        os.mkdir(bev_semantic_path)
        if predicting_flow:
            os.mkdir(flow_path)

        for i, data in enumerate(tqdm(dataloader, desc="Inferencing the NN")):

            rgb_a = data["rgb_A_0"].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
            rgb_b = data["rgb_B_0"].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
            if predicting_flow:
                rgb = torch.concatenate([rgb_a, rgb_b], dim=1)
            else:
                rgb = rgb_a
            semantic_label = F.one_hot(data["semantic_0"][:, :, :, 0].type(torch.LongTensor), 8).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
            bev_semantic_label = F.one_hot(torch.rot90(data["bev_semantic"], 3, [1, 2])[:, :, :, 0].type(torch.LongTensor), 6).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
            depth_label = (data["depth_0"][:, :, :, 0]/255).contiguous().to(device, dtype=torch.float32)
            lidar = data["bev_lidar"][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
            flow_label = (data["optical_flow_0"][:, :, :, :2] / 2**15 - 1).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
            if tfpp_original:
                rgb = data["rgb_tfpp"].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
                lidar = data["lidar_tfpp"][:, :, :, 0][:, :, :, None].permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.float32)
                lidar /= 255

            if just_a_backbone:
                target_point = None
                command = None
                ego_vel = None
            else:
                target_point = data["targetpoint"].to(device, dtype=torch.float32)
                if target_point.shape[1] == 3:
                    target_point = target_point[:, :-1]
                command = None
                ego_vel = data["target_speed"].to(device, dtype=torch.float32)
                ego_vel = ego_vel[None, :]

            
            predictions = model(    rgb=rgb,
                                    lidar_bev=lidar,
                                    target_point=target_point,
                                    ego_vel=ego_vel,
                                    command=command)
            
            if not tfpp_original:
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
            if predicting_flow:
                pred_flow = ((pred_flow + 1)*(2**15)).permute(0, 2, 3, 1)[0, :, :, :].contiguous().detach().cpu().numpy()
                flow_label = ((flow_label + 1)*(2**15)).permute(0, 2, 3, 1)[0, :, :, :].contiguous().detach().cpu().numpy()
            
            if not tfpp_original:
                rgb_comparison = rgb_a[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            else:
                rgb_comparison = rgb[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            

            lidar_comparison = lidar[0, 0, :, :].detach().cpu().numpy().astype(np.uint8)*255

            depth_comparison = np.zeros((pred_depth.shape[0]*2, pred_depth.shape[1]), dtype=np.uint8)
            depth_comparison[0:pred_depth.shape[0], :] = pred_depth
            depth_comparison[pred_depth.shape[0]:, :] = depth_label

            semantic_comparison = np.zeros((pred_semantic.shape[0]*2, pred_semantic.shape[1]), dtype=np.uint8)
            semantic_comparison[0:pred_semantic.shape[0], :] = pred_semantic * 30
            semantic_comparison[pred_depth.shape[0]:, :] = semantic_label * 30

            bev_semantic_comparison = np.zeros((pred_bev_semantic.shape[0]*2, pred_bev_semantic.shape[1]), dtype=np.uint8)
            bev_semantic_comparison[0:pred_bev_semantic.shape[0], :] = pred_bev_semantic * 30
            bev_semantic_comparison[pred_bev_semantic.shape[0]:, :] = bev_semantic_label * 30

            if predicting_flow:
                flow_comparison = np.zeros((pred_flow.shape[0]*2, pred_flow.shape[1], 3), dtype=np.uint8)
                flow_comparison[0:pred_flow.shape[0], :, :] = optical_flow_to_human(pred_flow)
                flow_comparison[pred_flow.shape[0]:, :, :] = optical_flow_to_human(flow_label[:, :, :2])
            
            cv2.imwrite(os.path.join(rgb_path, f"{i}.jpg"), rgb_comparison)
            cv2.imwrite(os.path.join(lidar_path, f"{i}.png"), lidar_comparison)
            cv2.imwrite(os.path.join(depth_path, f"{i}.png"), depth_comparison)
            cv2.imwrite(os.path.join(semantic_path, f"{i}.png"), semantic_comparison)
            cv2.imwrite(os.path.join(bev_semantic_path, f"{i}.png"), bev_semantic_comparison)

            if predicting_flow:
                cv2.imwrite(os.path.join(flow_path, f"{i}.png"), flow_comparison)

        print(utils.color_info_success(f"Saved everything in '{output_dir}'"))
        return output_dir, name

if __name__=="__main__":

    # if containing "flow_decoder.deconv1.0.weight" means that it has flow prediction!
    # if not havin "wp_decoder.encoder.weight" means that it is just a backbone
    # if NOT just a backbone and the shape of "extra_sensor_encoder.0.weight" is [128, 7] it is an Original TFPP!

    args = get_arguments()
    output_dir, name = infering_the_model(args.dataset_folder, args.where_to_save, args.weights_path)
    if args.second_weights_path is not None:
        output_dir_2, name_2 = infering_the_model(args.dataset_folder, args.where_to_save, args.second_weights_path)
    
    # CREATE SINGLE VIDEO
    utils.create_validation_video(output_dir)
    if args.second_weights_path is not None:
        utils.create_validation_video(output_dir_2)
    
    # COMPARISON
    if args.second_weights_path is not None:
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
        comparison_folder_name = "comparison_" + name + "_" + name_2 + current_time
        comparison_path = os.path.join(args.where_to_save, comparison_folder_name)
        os.mkdir(comparison_path)
        utils.create_compariso_validation_video(output_dir, output_dir_2, comparison_path)



        






    