from nutfuser.utils import check_dataset_folder, color_error_string, color_info_string
import os

if __name__ == '__main__':
    done = False
    big_folder_path = None
    while not done:
        try:
            big_folder_path = input("Training/Evaluation folder path : ")
            if not os.path.isdir(big_folder_path):
                print(color_error_string(f"Cannot find '{big_folder_path}'!"))
                continue
            done = True
        except KeyboardInterrupt:
            print(color_info_string("\nBye!\n"))
            exit()

    all_data_folder = os.listdir(big_folder_path)
    bad_data_folder = []
    for data_folder in all_data_folder:
        try:
            check_dataset_folder(os.path.join(big_folder_path, data_folder))
        except:
            bad_data_folder.append(data_folder)

    if len(bad_data_folder) == 0:
        print(color_info_string("\nNo bad data folder found! Bye!\n"))
        exit()

    print(color_error_string("\nI found out those bad data folders:\n"))
    for data_folder in bad_data_folder:
        print(color_error_string(f"\t{os.path.join(big_folder_path, data_folder)}"))

