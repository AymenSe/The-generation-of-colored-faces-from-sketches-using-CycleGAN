from PIL import Image
import os
import pathlib
from torch.utils.data import Dataset
import numpy as np


class PairedCUHKDataset(Dataset):
    def __init__(self, root, split_dir='train', transform=None):
        self.root = root
        self.data_dir = pathlib.Path(root)
        self.transform = transform
        self.rgb_images = sorted(list(self.data_dir.glob(split_dir+'/photos/*.jpg')))
        self.sketch_images = sorted(list(self.data_dir.glob(split_dir+'/sketches/*.jpg'))) 
        self.length_dataset = max(len(self.sketch_images), len(self.rgb_images))
        self.rgb_len = len(self.rgb_images)
        self.sketch_len = len(self.sketch_images)
    
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        assert self.rgb_len == self.sketch_len, "Error!"
        rgb_path = self.rgb_images[index % self.rgb_len]
        sketch_path = self.sketch_images[index % self.sketch_len]

        rgb_img = np.array(Image.open(rgb_path).convert("RGB"))
        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))
        # print(rgb_img.shape)
        

        if self.transform:
            augmentations = self.transform(image=rgb_img, image0=sketch_img)
            rgb_img = augmentations["image"]
            sketch_img = augmentations["image0"]
        # print(rgb_img.shape, sketch_img.shape)

    
        return rgb_img, sketch_img