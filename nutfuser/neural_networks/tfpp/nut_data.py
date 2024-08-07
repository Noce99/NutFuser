import nutfuser.utils as utils

import os
import collections
import json

from torch.utils.data import Dataset
from tabulate import tabulate
import cv2
import numpy as np

DataFolder = collections.namedtuple('DataFolder', ['path', 'elements'])


class backbone_dataset(Dataset):
    def __init__(self, rank, dataset_path, tfpp_config=None):
        self.rank = rank
        self.dataset_path = dataset_path
        self.tfpp_config = tfpp_config
        if self.tfpp_config is not None:
            self.use_abstract_bev_semantic = self.tfpp_config.use_abstract_bev_semantic
        else:
            self.use_abstract_bev_semantic = False

        self.cache = None
        self.data_folders = []
        self.bytes_that_I_need = None
        self.max_chars_data_folder_name_length = None
        self.ram_that_I_will_use_in_bytes = None
        # The real data corpus:
        self.data_path = None
        self.data_id = None
        # end

        self.folders_that_should_be_there = None

        self.target_points = {}
        self.waypoints = {}
        self.previous_speeds = {}
        self.next_speeds = {}
        self.acceleration = {}

        self.total_number_of_frames = self.check_dataset_and_get_data_folders()
        self.bytes_per_frame = self.get_one_frame_memory_size()
        if self.rank == 0:
            self.print_dataset_summary_table()
        self.make_data()
        del self.data_folders
        print(utils.color_info_string(f"I will use {self.ram_that_I_will_use_in_bytes / 1e6:.3f} MB of RAM"))

    def check_dataset_and_get_data_folders(self):
        # Let's check if the dataset folder really exist
        if not os.path.isdir(self.dataset_path):
            raise Exception(utils.color_error_string(f"No Dataset folder found in '{self.dataset_path}'!"))

        # I get the first folder I found (random_data_folder). I expect this to be a Data Folder.
        random_data_folder = None
        for el in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, el)):
                random_data_folder = os.path.join(self.dataset_path, el)
                break
        # I get all the directories and files inside the random_data_folder
        subfolders = os.listdir(random_data_folder)
        # There I found out how many cameras we have, for each of them I expect a rgb_A_i folder.
        # Ex: With 3 cameras we expect to have: rgb_A_0, rgb_A_1 and rgb_A_2
        camera_indexes = [int(el[len("rgb_A_"):]) for el in subfolders if "rgb_A_" in el]

        # Knowing how many cameras are there I create the folders_that_should_be_there containing all
        # the folders that I expect to be contained by each Data Folder.
        self.folders_that_should_be_there = \
            [(f"rgb_A_{i}", ".jpg") for i in camera_indexes] + \
            [(f"rgb_B_{i}", ".jpg") for i in camera_indexes] + \
            [(f"depth_{i}", ".png") for i in camera_indexes] + \
            [(f"optical_flow_{i}", ".png") for i in camera_indexes] + \
            [(f"semantic_{i}", ".png") for i in camera_indexes] + \
            [("bev_semantic", ".png"), ("bev_lidar", ".png"), ("bounding_boxes", ".json")]
        # We check if there is also the rgb_tfpp folder we add it to the folders_that_should_be_there
        if "rgb_tfpp" in subfolders:
            self.folders_that_should_be_there += [("rgb_tfpp", ".jpg")]
        if "lidar_tfpp" in subfolders:
            self.folders_that_should_be_there += [("lidar_tfpp", ".png")]
        if self.use_abstract_bev_semantic:
            self.folders_that_should_be_there.remove(("bev_semantic", ".png"))
            self.folders_that_should_be_there += [("bev_semantic_2", ".png")]

        def check_that_all_sensors_have_the_same_amount_of_frame(data_folder_path):
            """
            This function check for each subfolder of the given Data Folder Path if
            all the folders that we expect to be here exists and calculate how many
            frames they contain.
            """
            min_number_of_frames = None
            for data_subfolder in self.folders_that_should_be_there:
                subfolder_to_check = str(os.path.join(data_folder_path, data_subfolder[0]))
                files = os.listdir(subfolder_to_check)
                if min_number_of_frames is None:
                    min_number_of_frames = len(files)
                else:
                    if len(files) < min_number_of_frames:
                        min_number_of_frames = len(files)
            return min_number_of_frames

        total_number_of_frames = 0
        all_possible_data_folders = os.listdir(self.dataset_path)
        for data_folder in all_possible_data_folders:
            data_folder_path = os.path.join(self.dataset_path, data_folder)
            subfolders = os.listdir(data_folder_path)
            good = True
            for el in self.folders_that_should_be_there:
                if el[0] not in subfolders:
                    print(utils.color_info_string(f"Skipping '{data_folder_path}' because '{el[0]}'"
                                                  f" subfolder is missing."))
                    good = False
                    break
            if not good:
                continue
            elements = check_that_all_sensors_have_the_same_amount_of_frame(data_folder_path)
            total_number_of_frames += elements
            self.data_folders.append(DataFolder(data_folder, elements))
            if self.max_chars_data_folder_name_length is None:
                self.max_chars_data_folder_name_length = len(data_folder)
            elif len(data_folder) > self.max_chars_data_folder_name_length:
                self.max_chars_data_folder_name_length = len(data_folder)
        return total_number_of_frames

    def get_one_frame_memory_size(self):
        """
        I get the first Data Folder and going through all the elements of each subfolder and dividing
        the result by the number of elements contained in that Data Folder I found out a per frame
        memory usage.
        """
        # First of all we check if we have found out some wheel built data folder
        if len(self.data_folders) == 0:
            raise utils.NutException(
                utils.color_error_string(f"Not well built data folders found in {self.dataset_path}"))
        # We get the first Data Folder as an example
        a_data_folder = self.data_folders[0]
        total_size = 0
        for root, dirs, files in os.walk(os.path.join(str(self.dataset_path), a_data_folder.path), topdown=False):
            if root == a_data_folder.path:
                continue
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        return total_size / a_data_folder.elements

    def print_dataset_summary_table(self):
        a_table_head = ["Dataset Summary", ""]
        a_table = [
            ["Number of Frames", self.total_number_of_frames],
            ["Number of Data Folders", len(self.data_folders)],
            ["Number of Frames per Folder in average", f"{self.total_number_of_frames / len(self.data_folders):.2f}"],
            ["Memory Size per Frame", f"{self.bytes_per_frame / 1e6:.2f} MB"],
            ["Approximately Size of Dataset", f"{self.bytes_per_frame * self.total_number_of_frames / 1e9:.2f} GB"],
            ["Use Abstract Bev Semantic", f"{self.use_abstract_bev_semantic}"]
        ]
        print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    def make_data(self):
        # Let's create the numpy arrays
        self.data_path = np.zeros(shape=self.total_number_of_frames,
                                  dtype=np.dtype(('U', self.max_chars_data_folder_name_length))
                                  )  # Unicode String containing exactly the needed amount of chars
        self.ram_that_I_will_use_in_bytes = self.data_path.size * self.data_path.itemsize
        self.data_id = np.zeros(shape=self.total_number_of_frames, dtype=np.dtype('uint32'))  # MAX id is 4.294.967.295
        self.ram_that_I_will_use_in_bytes += self.data_id.size * self.data_id.itemsize

        # Let's fill them with data
        index = 0
        for data_folder in self.data_folders:
            for ii in range(data_folder.elements):
                self.data_path[index] = data_folder.path
                self.data_id[index] = ii
                index += 1

        # Let's read:
        # - input speed (input of the model)
        # - target point (input of the model)
        # - target speed (output of the model)
        # - waypoints data (output of the model)

        for data_folder in self.data_folders:
            target_points_path = str(os.path.join(self.dataset_path, data_folder.path, "frame_targetpoints.npy"))
            waypoints_path = str(os.path.join(self.dataset_path, data_folder.path, "frame_waypoints.npy"))
            previous_speeds_path = str(os.path.join(self.dataset_path, data_folder.path, "previous_speeds.npy"))
            next_speeds_path = str(os.path.join(self.dataset_path, data_folder.path, "next_speeds.npy"))
            acceleration_path = str(os.path.join(self.dataset_path, data_folder.path, "accelerations.npy"))

            self.target_points[data_folder.path] = np.load(target_points_path)
            self.ram_that_I_will_use_in_bytes += (self.target_points[data_folder.path].size *
                                                  self.target_points[data_folder.path].itemsize)
            self.waypoints[data_folder.path] = np.load(waypoints_path)
            self.ram_that_I_will_use_in_bytes += (self.waypoints[data_folder.path].size *
                                                  self.waypoints[data_folder.path].itemsize)
            self.previous_speeds[data_folder.path] = np.load(previous_speeds_path)
            self.ram_that_I_will_use_in_bytes += (self.previous_speeds[data_folder.path].size *
                                                  self.previous_speeds[data_folder.path].itemsize)
            self.next_speeds[data_folder.path] = np.load(next_speeds_path)
            self.ram_that_I_will_use_in_bytes += (self.next_speeds[data_folder.path].size *
                                                  self.next_speeds[data_folder.path].itemsize)
            self.acceleration[data_folder.path] = np.load(acceleration_path)
            self.ram_that_I_will_use_in_bytes += (self.acceleration[data_folder.path].size *
                                                  self.acceleration[data_folder.path].itemsize)

    def __len__(self):
        return self.total_number_of_frames

    def __getitem__(self, idx):
        path = str(os.path.join(self.dataset_path, self.data_path[idx]))
        id = self.data_id[idx]
        data = {}
        for data_folder in self.folders_that_should_be_there:
            data_name, data_extension = data_folder
            if "optical_flow" in data_name:
                data[data_name] = cv2.imread(str(os.path.join(path, data_name, f"{id}{data_extension}")),
                                             cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                data[data_name] = data[data_name].astype(np.float32)
            elif "bev_semantic_2" in data_name:
                data["bev_semantic"] = cv2.imread(os.path.join(path, data_name, f"{id}{data_extension}"))
            elif "bounding_boxes" in data_name:
                with open(os.path.join(path, data_name, f"{id}{data_extension}")) as json_data:
                    bbs = json.loads(json_data.read())
                data["bounding_boxes"] = np.array(bbs)
            else:
                data[data_name] = cv2.imread(os.path.join(path, data_name, f"{id}{data_extension}"))
        data["input_speed"] = self.previous_speeds[self.data_path[idx]][id]
        data["target_speed"] = self.next_speeds[self.data_path[idx]][id]
        data["waypoints"] = self.waypoints[self.data_path[idx]][id]
        data["targetpoint"] = self.target_points[self.data_path[idx]][id]
        data["acceleration"] = self.acceleration[self.data_path[idx]][id]

        return data


if __name__ == "__main__":
    pass
