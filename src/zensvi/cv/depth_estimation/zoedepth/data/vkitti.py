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

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    """Convert images and depth maps to PyTorch tensors and apply normalization.

    This class normalizes the image and converts it to a tensor format suitable for PyTorch.

    Attributes:
        normalize (callable): A function to normalize the image.
    """

    def __init__(self):
        """Initializes the ToTensor transform with normalization parameters."""
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        """Apply the transformation to the sample.

        Args:
            sample (dict): A dictionary containing "image" and "depth".

        Returns:
            dict: A dictionary containing the transformed "image", "depth", and the dataset name.
        """
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        return {"image": image, "depth": depth, "dataset": "vkitti"}

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


class VKITTI(Dataset):
    """Dataset class for loading VKITTI images and depth maps.

    This class loads images and their corresponding depth maps from the specified directory.

    Attributes:
        image_files (list): List of image file paths.
        depth_files (list): List of depth file paths.
        do_kb_crop (bool): Whether to perform knowledge-based cropping.
        transform (ToTensor): Transform to apply to the images and depth maps.
    """

    def __init__(self, data_dir_root, do_kb_crop=True):
        """Initializes the VKITTI dataset.

        Args:
            data_dir_root (str): The root directory containing the VKITTI dataset.
            do_kb_crop (bool): Whether to perform knowledge-based cropping. Defaults to True.
        """
        import glob

        self.image_files = glob.glob(os.path.join(data_dir_root, "test_color", "*.png"))
        self.depth_files = [r.replace("test_color", "test_depth") for r in self.image_files]
        self.do_kb_crop = do_kb_crop
        self.transform = ToTensor()

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding depth map.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the transformed image, depth, and dataset name.
        """
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        print("depth min max", depth.min(), depth.max())

        if self.do_kb_crop and False:
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth = depth[..., None]
        sample = dict(image=image, depth=depth)

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


def get_vkitti_loader(data_dir_root, batch_size=1, **kwargs):
    """Create a DataLoader for the VKITTI dataset.

    Args:
        data_dir_root (str): The root directory containing the VKITTI dataset.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        DataLoader: A DataLoader for the VKITTI dataset.
    """
    dataset = VKITTI(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_vkitti_loader(data_dir_root="/home/bhatsf/shortcuts/datasets/vkitti_test")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample["depth"].min(), sample["depth"].max())
        if i > 5:
            break
