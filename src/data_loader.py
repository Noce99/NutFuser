import config
import utils

import os
import shutil
import time
import psutil
import collections


from diskcache import Cache # https://grantjenks.com/docs/diskcache/api.html
from torch.utils.data import Dataset
from tabulate import tabulate
import cv2
from tqdm import tqdm


DataFolder = collections.namedtuple('DataFolder', ['path', 'elements'])
FramePath = collections.namedtuple('FramePath', ['path', 'id'])

class backbone_dataset(Dataset):
    def __init__(self):
        self.cache = None
        self.data_folders = []
        self.bytes_that_I_need = None
        self.slow_memory_path = {}


        self.set_up_cache()
        self.total_number_of_frames = self.check_dataset_and_get_data_folders()
        self.bytes_per_frame = self.get_one_frame_memory_size()
        self.print_dataset_summary_table()
        self.last_frame_in_cache = self.get_last_frames_to_put_in_cache()
        self.print_memory_summary_table()
        self.load_data_in_cache()


    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

    def close(self):
        self.clean_cache()

    def set_up_cache(self):
        if "TMPDIR" in os.environ:
            print("Using TMPDIR as CACHE!")
            config.JOB_TMP_DIR = os.path.join(os.environ["TMPDIR"], config.JOB_TMP_DIR_NAME)
        else:
            print("Not using TMPDIR as CACHE!")
            config.JOB_TMP_DIR = os.path.join(os.getcwd(), config.JOB_TMP_DIR_NAME)
        if os.path.isdir(config.JOB_TMP_DIR):
            shutil.rmtree(config.JOB_TMP_DIR)
        os.mkdir(config.JOB_TMP_DIR)
        print(f"CACHE PATH is '{config.JOB_TMP_DIR}' [size_limit={int(config.CACHE_SIZE_LIMIT/1e9)} GB]")
        self.cache = Cache(directory=config.JOB_TMP_DIR, size_limit=config.CACHE_SIZE_LIMIT)

    def clean_cache(self):
        self.cache.close()
        if os.path.isdir(config.JOB_TMP_DIR):
            shutil.rmtree(config.JOB_TMP_DIR)
    
    def check_dataset_and_get_data_folders(self):
        if not os.path.isdir(config.DATASET_PATH):
            raise Exception(utils.color_error_string("No Dataset folder found!"))
        
        def check_that_all_sensors_have_the_same_ammount_of_frame(data_folder_path):
            number_of_frames = None
            for root, dirs, files in os.walk(data_folder_path, topdown=False):
                if root == data_folder_path:
                    continue
                if number_of_frames is None:
                    number_of_frames = len(files)
                else:
                    if len(files) != number_of_frames:
                        raise Exception(utils.color_error_string(f"The folder '{root}' has {len(files)} instead of {number_of_frames}!"))
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
            self.data_folders.append(DataFolder(root, elements))
        return total_number_of_frames
            
    def get_one_frame_memory_size(self):
        if len(self.data_folders) == 0:
            print("self.data_folders is EMPTY!")
            return None
        a_data_folder = self.data_folders[0]
        total_size = 0
        for root, dirs, files in os.walk(a_data_folder.path, topdown=False):
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
        a_table_head = ["Memory Summary", "", "", "Limit"]
        a_table = [
            ["CACHE that I will Use", f"{self.last_frame_in_cache*self.bytes_per_frame/1e9:.2f} GB", f"{self.last_frame_in_cache/self.total_number_of_frames*100:.2f} %", f"{config.CACHE_SIZE_LIMIT/1e9:.2f} GB"],
            ["SLOW MEMORY that I will Use", f"{(self.total_number_of_frames-self.last_frame_in_cache)*self.bytes_per_frame/1e9:.2f} GB", f"{(self.total_number_of_frames-self.last_frame_in_cache)/self.total_number_of_frames*100:.2f} %"], 
            ["TOTAL that I Need", f"{self.bytes_per_frame*self.total_number_of_frames/1e9:.2f} GB", ""]
        ]
        print(tabulate(a_table, headers=a_table_head, tablefmt="grid"))
    
    def load_data_in_cache(self):
        data_folder_index = None
        frame_index = None
        index = 0
        pbar = tqdm(total=self.last_frame_in_cache+1)
        quit = False
        for i, data_folder in enumerate(self.data_folders):
            for ii in range(data_folder.elements):
                images = []
                for folder in config.DATASET_FOLDER_STRUCT:
                    images.append(cv2.imencode(folder[1], cv2.imread(os.path.join(data_folder.path, folder[0], f"{ii}{folder[1]}"))))
                self.cache[index] = images
                pbar.update(1)
                index += 1
                if index > self.last_frame_in_cache:
                    data_folder_index = i
                    frame_index = ii
                    quit = True
                    break
            if quit:
                break
        pbar.close()
        pbar = tqdm(total=self.total_number_of_frames-self.last_frame_in_cache)
        first_iteration = True
        for data_folder in self.data_folders[data_folder_index:]:
            if first_iteration:
                start = frame_index
            else:
                start = 0
            for ii in range(start, data_folder.elements):
                self.slow_memory_path[index] = FramePath(data_folder.path, ii)
                pbar.update(1)
                index += 1
            first_iteration = False
        pbar.close()

if __name__ == "__main__":
    my_dataset = backbone_dataset()
    my_dataset.close()