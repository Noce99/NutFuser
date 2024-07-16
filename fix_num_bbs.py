import argparse
import json
import os
from nutfuser import utils, config


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--dataset_path',
        help='Path to the Dataset!',
        required=True,
        type=str
    )
    args = argparser.parse_args()
    dataset_path = args.dataset_path
    if not os.path.isdir(dataset_path):
        raise utils.NutException(f"The folder {dataset_path} does not exist!")
    sub_dirs = os.listdir(path=dataset_path)
    for data_folder in sub_dirs:
        if os.path.isdir(os.path.join(dataset_path, data_folder, "bounding_boxes")):
            for frame in os.listdir(path=os.path.join(dataset_path, data_folder, "bounding_boxes")):
                with open(os.path.join(dataset_path, data_folder, "bounding_boxes", frame)) as json_data:
                    bbs = json.loads(json_data.read())
                    json_data.close()
                if len(bbs) > config.NUM_OF_BBS_PER_FRAME:
                    print("TROVATO!")
                    utils.save_bbs_in_json(os.path.join(dataset_path, data_folder, "bounding_boxes", frame), bbs)
        else:
            raise utils.NutException(f"The folder {os.path.join(dataset_path, data_folder)} does not contains"
                                     f" bounding_boxes folder!")
