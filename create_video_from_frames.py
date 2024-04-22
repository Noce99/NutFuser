import argparse
from nutfuser import utils
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--folder_path',
        help='Path where the frames are!',
        required=True,
        type=str
    )
    argparser.add_argument(
        '--second_folder_path',
        help='Path where the frames are!',
        default=None,
        type=str
    )
    args = argparser.parse_args()
    if not os.path.isdir(args.folder_path):
        print(utils.color_error_string(f"The path '{args.folder_path}' is not a directory!"))
        exit()
    if args.second_folder_path is None:
        compare_models = False
    else:
        compare_models = True
    if compare_models:
        if not os.path.isdir(args.second_folder_path):
            print(utils.color_error_string(f"The path '{args.second_folder_path}' is not a directory!"))
            exit()
    if not compare_models:
        utils.create_validation_video(args.folder_path)
    else:
        utils.create_compariso_validation_video(args.folder_path, args.second_folder_path)