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
from torchvision import transforms


class ToTensor(object):
    """Convert images and depth maps to PyTorch tensors and apply resizing.

    This class normalizes the image and resizes it to a fixed height while
    maintaining the aspect ratio.

    Attributes:
        normalize (callable): A function to normalize the image.
        resize (torchvision.transforms.Resize): A transform to resize the image.
    """

    def __init__(self):
        self.normalize = lambda x: x
        self.resize = transforms.Resize(480)

    def __call__(self, sample):
        """Convert the sample images and depth maps to tensors and resize them.

        Args:
            sample (dict): A dictionary containing "image" and "depth".

        Returns:
            dict: A dictionary containing the transformed "image", "depth", and the dataset name.
        """
        image, depth = sample["image"], sample["depth"]
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {"image": image, "depth": depth, "dataset": "diode"}

    def to_tensor(self, pic):
        """Convert a PIL image or NumPy array to a PyTorch tensor.

        Args:
            pic (PIL.Image or np.ndarray): The image to convert.

        Returns:
            torch.Tensor: The converted image as a tensor.
        """
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class DIODE(Dataset):
    """Dataset class for loading DIODE images and depth maps.

    This class loads images and their corresponding depth maps and masks from
    the specified directory.

    Attributes:
        image_files (list): List of image file paths.
        depth_files (list): List of depth file paths.
        depth_mask_files (list): List of depth mask file paths.
        transform (ToTensor): Transform to apply to the images and depth maps.
    """

    def __init__(self, data_dir_root):
        import glob

        self.image_files = glob.glob(os.path.join(data_dir_root, "*", "*", "*.png"))
        self.depth_files = [r.replace(".png", "_depth.npy") for r in self.image_files]
        self.depth_mask_files = [r.replace(".png", "_depth_mask.npy") for r in self.image_files]
        self.transform = ToTensor()

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding depth and mask.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the transformed image, depth, and valid mask.
        """
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        depth_mask_path = self.depth_mask_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        depth = np.load(depth_path)  # in meters
        valid = np.load(depth_mask_path)  # binary

        sample = dict(image=image, depth=depth, valid=valid)
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

    def __len__(self):
        """Return the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_files)


def get_diode_loader(data_dir_root, batch_size=1, **kwargs):
    """Create a DataLoader for the DIODE dataset.

    Args:
        data_dir_root (str): The root directory containing the DIODE dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A DataLoader for the DIODE dataset.
    """
    dataset = DIODE(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)
