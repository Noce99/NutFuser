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
    args = argparser.parse_args()
    if not os.path.isdir(args.folder_path):
        print(utils.color_error_string(f"The path '{args.folder_path}' is not a directory!"))
    utils.create_validation_video(args.folder_path)