# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class iBims(Dataset):
    """Dataset class for loading iBims images and depth maps.

    This class loads images, depth maps, and their corresponding masks from
    the iBims dataset. It provides methods to access individual samples and
    the total number of samples.

    Attributes:
        samples (list): A list of tuples containing paths to images, depth maps,
                        and masks.
        normalize (callable): A function to normalize the input images.
    """

    def __init__(self, config):
        """Initializes the iBims dataset.

        Args:
            config (object): Configuration object containing the root folder path.
        """
        root_folder = config.ibims_root
        with open(os.path.join(root_folder, "imagelist.txt"), "r") as f:
            imglist = f.read().split()

        samples = []
        for basename in imglist:
            img_path = os.path.join(root_folder, "rgb", basename + ".png")
            depth_path = os.path.join(root_folder, "depth", basename + ".png")
            valid_mask_path = os.path.join(root_folder, "mask_invalid", basename + ".png")
            transp_mask_path = os.path.join(root_folder, "mask_transp", basename + ".png")

            samples.append((img_path, depth_path, valid_mask_path, transp_mask_path))

        self.samples = samples
        # self.normalize = T.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x

    def __getitem__(self, idx):
        """Fetches a sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            dict: A dictionary containing the image, depth map, image path,
                  depth path, and dataset name.
        """
        img_path, depth_path, valid_mask_path, transp_mask_path = self.samples[idx]

        img = np.asarray(Image.open(img_path), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16).astype("float") * 50.0 / 65535

        mask_valid = np.asarray(Image.open(valid_mask_path))
        mask_transp = np.asarray(Image.open(transp_mask_path))

        # depth = depth * mask_valid * mask_transp
        depth = np.where(mask_valid * mask_transp, depth, -1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)
        depth = torch.from_numpy(depth).unsqueeze(0)
        return dict(
            image=img,
            depth=depth,
            image_path=img_path,
            depth_path=depth_path,
            dataset="ibims",
        )

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)


def get_ibims_loader(config, batch_size=1, **kwargs):
    """Creates a DataLoader for the iBims dataset.

    Args:
        config (object): Configuration object containing dataset parameters.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A DataLoader for the iBims dataset.
    """
    dataloader = DataLoader(iBims(config), batch_size=batch_size, **kwargs)
    return dataloader
