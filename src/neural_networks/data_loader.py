import config
import utils

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
    def __init__(self, rank):
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
            config.JOB_TMP_DIR = os.path.join(os.environ["TMPDIR"], config.JOB_TMP_DIR_NAME)
        else:
            if self.rank == 0:
                print("Not using TMPDIR as CACHE!")
            config.JOB_TMP_DIR = os.path.join(os.getcwd(), config.JOB_TMP_DIR_NAME)
        if self.rank == 0:
            if os.path.isdir(config.JOB_TMP_DIR):
                shutil.rmtree(config.JOB_TMP_DIR)
                os.mkdir(config.JOB_TMP_DIR)
            else:
                os.mkdir(config.JOB_TMP_DIR)
            print(f"CACHE PATH is '{config.JOB_TMP_DIR}' [size_limit={int(config.CACHE_SIZE_LIMIT/1e9)} GB]")
        self.cache = Cache(directory=config.JOB_TMP_DIR, size_limit=config.CACHE_SIZE_LIMIT)

    def clean_cache(self):
        if self.rank == 0:
            self.cache.close()
            if os.path.isdir(config.JOB_TMP_DIR):
                shutil.rmtree(config.JOB_TMP_DIR)
    
    def check_dataset_and_get_data_folders(self):
        if not os.path.isdir(config.DATASET_PATH):
            raise Exception(utils.color_error_string("No Dataset folder found!"))
        
        def check_that_all_sensors_have_the_same_ammount_of_frame(data_folder_path):
            number_of_frames = None        
            for data_subfolder in config.DATASET_FOLDER_STRUCT:
                subfolder_to_check = os.path.join(data_folder_path, data_subfolder[0])
                files = os.listdir(subfolder_to_check)
                if number_of_frames is None:
                    number_of_frames = len(files)
                else:
                    if len(files) != number_of_frames:
                        raise Exception(utils.color_error_string(f"The folder '{root}' has {len(files)} files instead of {number_of_frames}!"))
            return number_of_frames

        total_number_of_frames = 0
        for root, dirs, files in os.walk(config.DATASET_PATH, topdown=False):
            if len(dirs) == 0:
                continue
            good = True
            for el in config.DATASET_FOLDER_STRUCT:
                if el[0] not in dirs:
                    good = False
                    break
            if not good:
                continue
            elements = check_that_all_sensors_have_the_same_ammount_of_frame(root)
            total_number_of_frames += elements
            # In the following I'm assuming that the folder containing all the data
            # (ex: 25_02_2024_21:20:57) is a direct child of config.DATASET_PATH!
            head, tail = os.path.split(root)
            if os.path.normpath(head) != os.path.normpath(config.DATASET_PATH):
                raise Exception(utils.color_error_string(f"The folder '{root}' is a valid dataset subfolder but it's not a direct child of '{config.DATASET_PATH}'!"))               
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
        for root, dirs, files in os.walk(os.path.join(config.DATASET_PATH, a_data_folder.path), topdown=False):
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
        # I add 1 % for stay safe!
        self.bytes_that_I_need += int(0.05*self.bytes_per_frame*self.total_number_of_frames)
        if config.CACHE_SIZE_LIMIT > self.bytes_that_I_need:
            return self.total_number_of_frames
        return  int((config.CACHE_SIZE_LIMIT / self.bytes_that_I_need) * self.total_number_of_frames)

    def print_memory_summary_table(self):
        a_table_head = ["Memory Summary", "Size", "%", "Limit"]
        a_table = [
            ["CACHE that I will Use", f"{self.last_frame_in_cache*self.bytes_per_frame/1e9:.2f} GB", f"{self.last_frame_in_cache/self.total_number_of_frames*100:.2f} %", f"{config.CACHE_SIZE_LIMIT/1e9:.2f} GB"],
            ["SLOW MEMORY that I will Use", f"{(self.total_number_of_frames-self.last_frame_in_cache)*self.bytes_per_frame/1e9:.2f} GB", f"{(self.total_number_of_frames-self.last_frame_in_cache)/self.total_number_of_frames*100:.2f} %"], 
            ["TOTAL that I Need", f"{self.bytes_per_frame*self.total_number_of_frames/1e9:.2f} GB", ""],
            ["RAM that I will Use for storing data retrival info", f"{self.ram_for_stooring_data_retrival_information/1e6:.2f} MB", ""]
        ]
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
            for ii in range(1, data_folder.elements+1):
                self.data_path[index] = data_folder.path
                self.data_id[index] = ii
                if index < self.last_frame_in_cache:
                    self.data_should_be_in_cache[index] = True
                else:
                    self.data_should_be_in_cache[index] = False
                self.data_already_in_cache[index] = False
                index += 1
    
    def __len__(self):
        return self.total_number_of_frames
    
    def __getitem__(self, idx):
        path = os.path.join(config.DATASET_PATH, self.data_path[idx])
        id = self.data_id[idx]
        data_should_be_in_cache = self.data_should_be_in_cache[idx]
        data_already_in_cache = self.data_already_in_cache[idx]
        if not data_already_in_cache:
            data = {}
            for data_folder in config.DATASET_FOLDER_STRUCT:
                data_name, data_extension = data_folder
                data[data_name] = cv2.imread(os.path.join(config.DATASET_PATH, path, data_name, f"{id}{data_extension}"))
                if data[data_name] is None:
                    print(utils.color_error_string(f"{os.path.join(config.DATASET_PATH, path, data_name, f'{id}{data_extension}')} is NULL!"))
            if data_should_be_in_cache:
                data_compressed = {}
                for data_folder in config.DATASET_FOLDER_STRUCT:
                    data_name, data_extension = data_folder
                    data_compressed[data_name] = cv2.imencode(data_extension, data[data_name])
                self.cache[idx] = data_compressed
                self.data_already_in_cache[idx] = True
        else:
            print("Trying to get data from cache!")
            data_compressed = self.cache[idx]
            data = {}
            for data_folder in config.DATASET_FOLDER_STRUCT:
                data_name, data_extension = data_folder
                data[data_name] = cv2.imdecode(data_compressed[data_name], cv2.IMREAD_UNCHANGED)
                if data[data_name] is None:
                    print(utils.color_error_string(f"in CACHE is NULL!"))
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