import nut_config
import nut_utils

import os
import shutil
import time
import collections


from diskcache import Cache # https://grantjenks.com/docs/diskcache/api.html
from torch.utils.data import Dataset
from tabulate import tabulate
import cv2
from tqdm import tqdm
import numpy as np
import torch


DataFolder = collections.namedtuple('DataFolder', ['path', 'elements'])

class backbone_dataset(Dataset):
    def __init__(self, rank, dataset_path, use_cache=True):
        self.rank = rank

        self.cache = None
        self.data_folders = []
        self.bytes_that_I_need = None
        self.max_chars_data_folder_name_lenght = None
        self.ram_for_stooring_data_retrival_information = None
        # The real data corpus:
        self.data_path = None
        self.data_id = None
        self.data_should_be_in_cache = None
        self.data_already_in_cache = None
        # end
        self.dataset_path = dataset_path
        if use_cache:
            self.cache_size_limit = nut_config.CACHE_SIZE_LIMIT
        else:
            self.cache_size_limit = 0

        self.folders_that_should_be_there = None

        self.set_up_cache()
        self.total_number_of_frames = self.check_dataset_and_get_data_folders()
        self.bytes_per_frame = self.get_one_frame_memory_size()
        if self.rank == 0:
            self.print_dataset_summary_table()
        self.last_frame_in_cache = self.get_last_frames_to_put_in_cache()
        self.make_data()
        if self.rank == 0:
            self.print_memory_summary_table()
        del self.data_folders

    def close(self):
        self.clean_cache()

    def set_up_cache(self):
        if self.rank != 0:
            time.sleep(1)
        if "TMPDIR" in os.environ:
            if self.rank == 0:
                print("Using TMPDIR as CACHE!")
            nut_config.JOB_TMP_DIR = os.path.join(os.environ["TMPDIR"], nut_config.JOB_TMP_DIR_NAME)
        else:
            if self.rank == 0:
                print("Not using TMPDIR as CACHE!")
            nut_config.JOB_TMP_DIR = os.path.join(os.getcwd(), nut_config.JOB_TMP_DIR_NAME)
        if self.rank == 0:
            if os.path.isdir(nut_config.JOB_TMP_DIR):
                shutil.rmtree(nut_config.JOB_TMP_DIR)
                os.mkdir(nut_config.JOB_TMP_DIR)
            else:
                os.mkdir(nut_config.JOB_TMP_DIR)
            print(f"CACHE PATH is '{nut_config.JOB_TMP_DIR}' [size_limit={int(self.cache_size_limit/1e9)} GB]")
        self.cache = Cache(directory=nut_config.JOB_TMP_DIR, size_limit=self.cache_size_limit)

    def clean_cache(self):
        if self.rank == 0:
            self.cache.close()
            if os.path.isdir(nut_config.JOB_TMP_DIR):
                shutil.rmtree(nut_config.JOB_TMP_DIR)

    def check_dataset_and_get_data_folders(self):
        if not os.path.isdir(self.dataset_path):
            raise Exception(nut_utils.color_error_string("No Dataset folder found!"))
        
        random_data_folder = None
        for el in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, el)):
                random_data_folder = os.path.join(self.dataset_path, el)
                break
        all_files = os.listdir(random_data_folder)
        camera_indexes = [int(el[len("rgb_A_"):]) for el in all_files if "rgb_A_" in el]

        self.folders_that_should_be_there =  [(f"rgb_A_{i}", ".jpg") for i in camera_indexes] +\
                            [(f"rgb_B_{i}", ".jpg") for i in camera_indexes] +\
                            [(f"depth_{i}", ".png") for i in camera_indexes] +\
                            [(f"optical_flow_{i}", ".png") for i in camera_indexes] +\
                            [(f"semantic_{i}", ".png") for i in camera_indexes] +\
                            [("bev_semantic", ".png"),      ("bev_lidar", ".png")]

        def check_that_all_sensors_have_the_same_ammount_of_frame(data_folder_path):
            min_number_of_frames = None
            for data_subfolder in self.folders_that_should_be_there:
                subfolder_to_check = os.path.join(data_folder_path, data_subfolder[0])
                files = os.listdir(subfolder_to_check)
                if min_number_of_frames is None:
                    min_number_of_frames = len(files)
                else:
                    if len(files) < min_number_of_frames:
                        min_number_of_frames = len(files)
                        # raise Exception(nut_utils.color_error_string(f"The folder '{root}' has {len(files)} files instead of {number_of_frames}!"))
            return min_number_of_frames

        total_number_of_frames = 0
        for root, dirs, files in os.walk(self.dataset_path, topdown=False):
            if len(dirs) == 0:
                continue
            good = True
            for el in self.folders_that_should_be_there:
                if el[0] not in dirs:
                    good = False
                    break
            if not good:
                continue
            elements = check_that_all_sensors_have_the_same_ammount_of_frame(root)
            total_number_of_frames += elements
            # In the following I'm assuming that the folder containing all the data
            # (ex: 25_02_2024_21:20:57) is a direct child of self.dataset_path!
            head, tail = os.path.split(root)
            if os.path.normpath(head) != os.path.normpath(self.dataset_path):
                raise Exception(nut_utils.color_error_string(f"The folder '{root}' is a valid dataset subfolder but it's not a direct child of '{self.dataset_path}'!"))
            self.data_folders.append(DataFolder(tail, elements))
            if self.max_chars_data_folder_name_lenght is None:
                self.max_chars_data_folder_name_lenght = len(tail)
            elif len(tail) > self.max_chars_data_folder_name_lenght:
                self.max_chars_data_folder_name_lenght = len(tail)
        return total_number_of_frames

    def get_one_frame_memory_size(self):
        if len(self.data_folders) == 0:
            print("self.data_folders is EMPTY!")
            return None
        a_data_folder = self.data_folders[0]
        total_size = 0
        for root, dirs, files in os.walk(os.path.join(self.dataset_path, a_data_folder.path), topdown=False):
            if root == a_data_folder.path:
                continue
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        return total_size/a_data_folder.elements

    def print_dataset_summary_table(self):
        a_table_head = ["Dataset Summary", ""]
        a_table = [
            ["Number of Frames", self.total_number_of_frames],
            ["Number of Data Folders", len(self.data_folders)],
            ["Number of Frames per Folder in average", f"{self.total_number_of_frames/len(self.data_folders):.2f}"],
            ["Memory Size per Frame", f"{self.bytes_per_frame/1e6:.2f} MB"],
            ["Approximally Size of Dataset", f"{self.bytes_per_frame*self.total_number_of_frames/1e9:.2f} GB"]
        ]
        print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    def get_last_frames_to_put_in_cache(self):
        self.bytes_that_I_need = self.bytes_per_frame*self.total_number_of_frames
        # I add 5 % for stay safe!
        self.bytes_that_I_need += int(0.05*self.bytes_per_frame*self.total_number_of_frames)
        if self.cache_size_limit > self.bytes_that_I_need:
            return self.total_number_of_frames
        return  int((self.cache_size_limit / self.bytes_that_I_need) * self.total_number_of_frames)

    def print_memory_summary_table(self):
        a_table_head = ["Memory Summary", "Size", "%", "Limit"]
        a_table = [
            ["CACHE that I will Use", f"{self.last_frame_in_cache*self.bytes_per_frame/1e9:.2f} GB", f"{self.last_frame_in_cache/self.total_number_of_frames*100:.2f} %", f"{self.cache_size_limit/1e9:.2f} GB"],
            ["SLOW MEMORY that I will Use", f"{(self.total_number_of_frames-self.last_frame_in_cache)*self.bytes_per_frame/1e9:.2f} GB", f"{(self.total_number_of_frames-self.last_frame_in_cache)/self.total_number_of_frames*100:.2f} %"],
            ["TOTAL that I Need", f"{self.bytes_per_frame*self.total_number_of_frames/1e9:.2f} GB", ""],
            ["RAM that I will Use for storing data retrival info", f"{self.ram_for_stooring_data_retrival_information/1e6:.2f} MB", ""]
        ]
        if self.rank == 0:
            print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))

    def make_data(self):
        # Let's create the numpy arrays
        self.data_path = np.zeros(shape=self.total_number_of_frames, dtype=np.dtype(('U', self.max_chars_data_folder_name_lenght))) # Unicode String containing exactly the needed ammount of chars
        self.ram_for_stooring_data_retrival_information = self.data_path.size * self.data_path.itemsize
        self.data_id = np.zeros(shape=self.total_number_of_frames, dtype=np.dtype('uint32')) # MAX id is 4.294.967.295
        self.ram_for_stooring_data_retrival_information += self.data_id.size * self.data_id.itemsize
        self.data_should_be_in_cache = np.zeros(shape=self.total_number_of_frames, dtype=np.dtype('?'))
        self.ram_for_stooring_data_retrival_information += self.data_should_be_in_cache.size * self.data_should_be_in_cache.itemsize
        self.data_already_in_cache = np.zeros(shape=self.total_number_of_frames, dtype=np.dtype('?'))
        self.ram_for_stooring_data_retrival_information += self.data_already_in_cache.size * self.data_already_in_cache.itemsize

        # Let's fill them with data
        index = 0
        for data_folder in self.data_folders:
            for ii in range(0, data_folder.elements):
                self.data_path[index] = data_folder.path
                self.data_id[index] = ii
                if index <= self.last_frame_in_cache:
                    self.data_should_be_in_cache[index] = True
                else:
                    self.data_should_be_in_cache[index] = False
                self.data_already_in_cache[index] = False
                index += 1

    def __len__(self):
        return self.total_number_of_frames

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_path, self.data_path[idx])
        id = self.data_id[idx]
        data_should_be_in_cache = self.data_should_be_in_cache[idx]
        data_already_in_cache = self.data_already_in_cache[idx]
        if not data_already_in_cache:
            data = {}
            for data_folder in self.folders_that_should_be_there:
                data_name, data_extension = data_folder
                if data_name[:-2] == "optical_flow":
                    data[data_name] = cv2.imread(os.path.join(self.dataset_path, path, data_name, f"{id}{data_extension}"), cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
                    data[data_name] = data[data_name].astype(np.float32)
                else:
                    data[data_name] = cv2.imread(os.path.join(self.dataset_path, path, data_name, f"{id}{data_extension}"))
            if data_should_be_in_cache:
                data_compressed = {}
                for data_folder in self.folders_that_should_be_there:
                    data_name, data_extension = data_folder
                    data_compressed[data_name] = cv2.imencode(data_extension, data[data_name])
                self.cache[idx] = data_compressed
                self.data_already_in_cache[idx] = True
        else:
            print("Trying to get data from cache!")
            data_compressed = self.cache[idx]
            data = {}
            for data_folder in self.folders_that_should_be_there:
                data_name, data_extension = data_folder
                data[data_name] = cv2.imdecode(data_compressed[data_name], cv2.IMREAD_UNCHANGED)
        for key in data:
            if data[key] is None:
                if idx > 0:
                    return self.__getitem__(idx-1)
                else:
                    return self.__getitem__(idx+1)
        return data


if __name__ == "__main__":
    my_dataset = backbone_dataset()
    for data in tqdm(my_dataset):
        pass
    my_dataset.close()
