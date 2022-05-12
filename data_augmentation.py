import argparse
from math import degrees
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
        image_number: int,
        data_length: int):
    
    loop_num = int(image_number / data_length) + 1
    
    for loop in tqdm(range(loop_num)):
        for i in range(data_length):
            data_transform = transforms.Compose([
                transforms.RandomApply([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=(-180/loop_num*i, 180/loop_num*i), shear=(-0.5, 0.5)),
                ], p=0.9),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=100, contrast=0, saturation=0, hue=0)
                ], p=0.1)
            ])
            dataset_augmented = datasets.ImageFolder(root=dataset_dir, transform=data_transform)
            os.makedirs(f"{destination_dir}", exist_ok=True)
            # print("Type: ", type(img), "Shape: ", img.shape)
            dataset_augmented[i][0].save(f"{destination_dir}/{loop}-{i}.png")  # type: <class 'PIL.Image.Image'>
    print("total data length: ", len(os.listdir(f"{destination_dir}")))


if __name__ == "__main__":
    args = parse_arguments()
    print("Dataset root dir: ", args.dataset_root_dir)
    dataset_augmented = datasets.Imagefolder(root=args.dataset_root_dir)
    data_length = dataset_augmented.__len__()

    data_augmentation_image_maker(
        dataset_dir=args.dataset_root_dir, 
        destination_dir=args.destination_dir,
        image_number=args.augmentation_size,
        data_length=data_length
    )




