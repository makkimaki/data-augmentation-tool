import argparse
import os 

from pathlib import Path 
import torch 
from torchvision import transforms, datasets
from tqdm import tqdm 
import cv2 
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_dir", "-rdir", type=Path, help="dataset you want to apply augmentation")
    parser.add_argument("--augmentation_size", "-size", type=int, help="augmentation size")
    parser.add_argument("--destination_dir", "-ddir", type=Path, help="augmented dataset directory save path")

    return parser.parse_args()


def data_augmentation_image_maker(
        dataset_dir: Path,
        destination_dir: Path,
        image_number: int):
    
    loop_num = int()@