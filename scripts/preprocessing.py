from argparse import ArgumentParser

import os

def parse_args():
    parser = ArgumentParser(description="Preprocessing pipeline.")

    parser.add_argument("-s", type=str, required=True, help="Path to source directory")

    return parser.parse_args()
    

def load_data(path):
    images = load_images(os.path.join(path, "images"))
    masks = load_masks(os.path.join(path, "masks"))
    poses = load_poses(os.path.join(path, "pred_traj.txt"))

    return images, masks, poses

def run_inpainting():
    pass

def estimate_depth():
    pass

def compose_mp4_vid():
    pass

def decompose_mp4_vid():
    pass

def thicken_masks():
    pass

def resize():
    pass

def run_preprocessing_pipeline():
    args = parse_args()
    images, masks, poses = load_data(args.s)

if __name__ == "__main__":
    run_preprocessing_pipeline()