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
"""Miscellaneous utility functions."""

from io import BytesIO

import matplotlib
import matplotlib.cm
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.utils.data.distributed
from PIL import Image
from torchvision.transforms import ToTensor


class RunningAverage:
    """Class to compute running average of values."""

    def __init__(self):
        """Initializes the RunningAverage with average and count set to zero."""
        self.avg = 0
        self.count = 0

    def append(self, value):
        """Updates the running average with a new value.

        Args:
            value (float): The new value to include in the average.
        """
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        """Returns the current running average.

        Returns:
            float: The current average value.
        """
        return self.avg


def denormalize(x):
    """Reverses the ImageNet normalization applied to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).

    Returns:
        torch.Tensor: Denormalized input tensor.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    """Class to maintain a dictionary of running averages."""

    def __init__(self):
        """Initializes the RunningAverageDict with an empty dictionary."""
        self._dict = None

    def update(self, new_dict):
        """Updates the dictionary with new values.

        Args:
            new_dict (dict): A dictionary of new values to update.
        """
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        """Returns the current values of the running averages.

        Returns:
            dict: A dictionary of current average values.
        """
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(
    value,
    vmin=None,
    vmax=None,
    cmap="gray_r",
    invalid_val=-99,
    invalid_mask=None,
    background_color=(128, 128, 128, 255),
    gamma_corrected=False,
    value_transform=None,
):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W).
        vmin (float, optional): Minimum value for color mapping. Defaults to None.
        vmax (float, optional): Maximum value for color mapping. Defaults to None.
        cmap (str, optional): Matplotlib colormap to use. Defaults to 'gray_r'.
        invalid_val (int, optional): Value of invalid pixels. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): RGB color for invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction. Defaults to False.
        value_transform (Callable, optional): Transform function for valid pixels. Defaults to None.

    Returns:
        numpy.ndarray: Colored depth map of shape (H, W, 4).
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[...]
    img[invalid_mask] = background_color

    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def count_parameters(model, include_all=False):
    """Counts the number of parameters in a model.

    Args:
        model (torch.nn.Module): The model to count parameters for.
        include_all (bool, optional): If True, include all parameters. Defaults to False.

    Returns:
        int: The total number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad or include_all)


def compute_errors(gt, pred):
    """Compute metrics for predicted values compared to ground truth.

    Args:
        gt (numpy.ndarray): Ground truth values.
        pred (numpy.ndarray): Predicted values.

    Returns:
        dict: Dictionary containing various error metrics.
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel,
    )


def compute_metrics(
    gt,
    pred,
    interpolate=True,
    garg_crop=False,
    eigen_crop=True,
    dataset="nyu",
    min_depth_eval=0.1,
    max_depth_eval=10,
    **kwargs,
):
    """Compute metrics of predicted depth maps.

    Applies cropping and masking as necessary or specified via arguments.

    Args:
        gt (torch.Tensor): Ground truth depth map.
        pred (torch.Tensor): Predicted depth map.
        interpolate (bool, optional): If True, interpolate predictions to match ground truth size. Defaults to True.
        garg_crop (bool, optional): If True, apply Garg crop. Defaults to False.
        eigen_crop (bool, optional): If True, apply Eigen crop. Defaults to True.
        dataset (str, optional): Dataset name. Defaults to "nyu".
        min_depth_eval (float, optional): Minimum depth for evaluation. Defaults to 0.1.
        max_depth_eval (float, optional): Maximum depth for evaluation. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        dict: Computed error metrics.
    """
    if "config" in kwargs:
        config = kwargs["config"]
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(pred, gt.shape[-2:], mode="bilinear", align_corners=True)

    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[
                int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
            ] = 1

        elif eigen_crop:
            if dataset == "kitti":
                eval_mask[
                    int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                ] = 1
            else:
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])


# Model uilts ################################################


def parallelize(config, model, find_unused_parameters=True):
    """Parallelizes the model for multi-GPU training.

    Args:
        config: Configuration object containing settings for parallelization.
        model (torch.nn.Module): The model to parallelize.
        find_unused_parameters (bool, optional): If True, find unused parameters. Defaults to True.

    Returns:
        torch.nn.Module: The parallelized model.
    """
    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.rank,
        )
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        config.workers = int((config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print(
            "Device",
            config.gpu,
            "Rank",
            config.rank,
            "batch size",
            config.batch_size,
            "Workers",
            config.workers,
        )
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            output_device=config.gpu,
            find_unused_parameters=find_unused_parameters,
        )

    elif config.gpu is None:
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


#################################################################################################


#####################################################################################################


class colors:
    """Class for terminal colors.

    Provides methods to reset colors and define foreground and background colors.
    Use as colors.subclass.colorname, e.g., colors.fg.red or colors.bg.green.

    Attributes:
        reset (str): Reset all colors.
        bold (str): Bold text.
        disable (str): Disable text.
        underline (str): Underline text.
        reverse (str): Reverse text.
        strikethrough (str): Strikethrough text.
        invisible (str): Invisible text.
    """

    reset = "\033[0m"
    bold = "\033[01m"
    disable = "\033[02m"
    underline = "\033[04m"
    reverse = "\033[07m"
    strikethrough = "\033[09m"
    invisible = "\033[08m"

    class fg:
        """Foreground colors."""

        black = "\033[30m"
        red = "\033[31m"
        green = "\033[32m"
        orange = "\033[33m"
        blue = "\033[34m"
        purple = "\033[35m"
        cyan = "\033[36m"
        lightgrey = "\033[37m"
        darkgrey = "\033[90m"
        lightred = "\033[91m"
        lightgreen = "\033[92m"
        yellow = "\033[93m"
        lightblue = "\033[94m"
        pink = "\033[95m"
        lightcyan = "\033[96m"

    class bg:
        """Background colors."""

        black = "\033[40m"
        red = "\033[41m"
        green = "\033[42m"
        orange = "\033[43m"
        blue = "\033[44m"
        purple = "\033[45m"
        cyan = "\033[46m"
        lightgrey = "\033[47m"


def printc(text, color):
    """Prints colored text to the console.

    Args:
        text (str): The text to print.
        color (str): The color code to use for printing.
    """
    print(f"{color}{text}{colors.reset}")


############################################


def get_image_from_url(url):
    """Fetches an image from a URL.

    Args:
        url (str): The URL of the image.

    Returns:
        PIL.Image: The fetched image in RGB format.
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def url_to_torch(url, size=(384, 384)):
    """Converts an image from a URL to a PyTorch tensor.

    Args:
        url (str): The URL of the image.
        size (tuple[int], optional): The desired size of the image. Defaults to (384, 384).

    Returns:
        torch.Tensor: The image as a PyTorch tensor.
    """
    img = get_image_from_url(url)
    img = img.resize(size, Image.ANTIALIAS)
    img = torch.from_numpy(np.asarray(img)).float()
    img = img.permute(2, 0, 1)
    img.div_(255)
    return img


def pil_to_batched_tensor(img):
    """Converts a PIL image to a batched PyTorch tensor.

    Args:
        img (PIL.Image): The input image.

    Returns:
        torch.Tensor: The image as a batched tensor.
    """
    return ToTensor()(img).unsqueeze(0)


def save_raw_16bit(depth, fpath="raw.png"):
    """Saves a depth map as a 16-bit PNG file.

    Args:
        depth (torch.Tensor or numpy.ndarray): The depth map to save.
        fpath (str, optional): The file path to save the image. Defaults to "raw.png".
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()

    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)
