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

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn

KEY_OUTPUT = "metric_depth"


def extract_key(prediction, key):
    """Extracts the value associated with the given key from the prediction.

    Args:
        prediction (dict or torch.Tensor): The prediction output, which can be a dictionary or a tensor.
        key (str): The key to extract from the prediction dictionary.

    Returns:
        torch.Tensor: The extracted value if prediction is a dictionary; otherwise, returns the prediction itself.
    """
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction


class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""

    def __init__(self, beta=0.15):
        """Initializes the SILogLoss.

        Args:
            beta (float): A hyperparameter for the loss calculation.
        """
        super(SILogLoss, self).__init__()
        self.name = "SILog"
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        """Computes the SILog loss.

        Args:
            input (torch.Tensor): The predicted depth values.
            target (torch.Tensor): The ground truth depth values.
            mask (torch.Tensor, optional): A mask to specify which pixels to consider in the loss calculation.
            interpolate (bool, optional): Whether to interpolate the input to match the target shape.
            return_interpolated (bool, optional): Whether to return the interpolated input.

        Returns:
            torch.Tensor: The computed loss. If return_interpolated is True, also returns the interpolated input.
        """
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode="bilinear", align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)

            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input


def grad(x):
    """Calculates the gradient magnitude and angle of the input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (n, c, h, w).

    Returns:
        tuple: A tuple containing the gradient magnitude and angle.
    """
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle


def grad_mask(mask):
    """Creates a mask for valid gradients.

    Args:
        mask (torch.Tensor): The input mask tensor.

    Returns:
        torch.Tensor: A tensor indicating valid gradient locations.
    """
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss."""

    def __init__(self):
        """Initializes the GradL1Loss."""
        super(GradL1Loss, self).__init__()
        self.name = "GradL1"

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        """Computes the gradient L1 loss.

        Args:
            input (torch.Tensor): The predicted depth values.
            target (torch.Tensor): The ground truth depth values.
            mask (torch.Tensor, optional): A mask to specify which pixels to consider in the loss calculation.
            interpolate (bool, optional): Whether to interpolate the input to match the target shape.
            return_interpolated (bool, optional): Whether to return the interpolated input.

        Returns:
            torch.Tensor: The computed loss. If return_interpolated is True, also returns the interpolated input.
        """
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode="bilinear", align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)

        loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        loss += nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input


class OrdinalRegressionLoss(object):
    """Ordinal regression loss for depth estimation."""

    def __init__(self, ord_num, beta, discretization="SID"):
        """Initializes the OrdinalRegressionLoss.

        Args:
            ord_num (int): The number of ordinal classes.
            beta (float): A hyperparameter for the loss calculation.
            discretization (str): The method of discretization, either "SID" or another method.
        """
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        """Creates ordinal labels from ground truth depth values.

        Args:
            gt (torch.Tensor): Ground truth depth values of shape (N, 1, H, W).

        Returns:
            tuple: A tuple containing the ordinal labels and the mask.
        """
        N, one, H, W = gt.shape

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = (
            torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False)
            .view(1, self.ord_num, 1, 1)
            .to(gt.device)
        )
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = mask > label
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """Calculates the ordinal regression loss.

        Args:
            prob (torch.Tensor): Ordinal regression probabilities of shape (N, 2 * Ord Num, H, W).
            gt (torch.Tensor): Ground truth depth values of shape (N, H, W).

        Returns:
            torch.float: The computed loss value.
        """
        valid_mask = gt > 0.0
        ord_label, mask = self._create_ord_label(gt)
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask.squeeze(1)]
        return loss.mean()


class DiscreteNLLLoss(nn.Module):
    """Cross entropy loss for discrete depth values."""

    def __init__(self, min_depth=1e-3, max_depth=10, depth_bins=64):
        """Initializes the DiscreteNLLLoss.

        Args:
            min_depth (float): Minimum depth value.
            max_depth (float): Maximum depth value.
            depth_bins (int): Number of depth bins for quantization.
        """
        super(DiscreteNLLLoss, self).__init__()
        self.name = "CrossEntropy"
        self.ignore_index = -(depth_bins + 1)
        self._loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_bins = depth_bins
        self.alpha = 1
        self.zeta = 1 - min_depth
        self.beta = max_depth + self.zeta

    def quantize_depth(self, depth):
        """Quantizes depth values into discrete bins.

        Args:
            depth (torch.Tensor): Depth values of shape (N, 1, H, W).

        Returns:
            torch.Tensor: Quantized depth values of shape (N, H, W).
        """
        depth = torch.log(depth / self.alpha) / np.log(self.beta / self.alpha)
        depth = depth * (self.depth_bins - 1)
        depth = torch.round(depth)
        depth = depth.long()
        return depth

    def _dequantize_depth(self, depth):
        """Inverse of quantization.

        Args:
            depth (torch.Tensor): Quantized depth values of shape (N, C, H, W).

        Returns:
            torch.Tensor: Dequantized depth values of shape (N, 1, H, W).
        """
        # Get the center of the bin
        pass  # Implementation not provided

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        """Computes the discrete NLL loss.

        Args:
            input (torch.Tensor): The predicted depth values.
            target (torch.Tensor): The ground truth depth values.
            mask (torch.Tensor, optional): A mask to specify which pixels to consider in the loss calculation.
            interpolate (bool, optional): Whether to interpolate the input to match the target shape.
            return_interpolated (bool, optional): Whether to return the interpolated input.

        Returns:
            torch.Tensor: The computed loss. If return_interpolated is True, also returns the interpolated input.
        """
        input = extract_key(input, KEY_OUTPUT)

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode="bilinear", align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        target = self.quantize_depth(target)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            mask = mask.long()
            input = input * mask + (1 - mask) * self.ignore_index
            target = target * mask + (1 - mask) * self.ignore_index

        input = input.flatten(2)  # N, nbins, H*W
        target = target.flatten(1)  # N, H*W
        loss = self._loss_func(input, target)

        if not return_interpolated:
            return loss
        return loss, intr_input


def compute_scale_and_shift(prediction, target, mask):
    """Computes scale and shift parameters for the prediction.

    Args:
        prediction (torch.Tensor): The predicted depth values.
        target (torch.Tensor): The ground truth depth values.
        mask (torch.Tensor): A mask to specify which pixels to consider.

    Returns:
        tuple: A tuple containing the scale and shift parameters.
    """
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


class ScaleAndShiftInvariantLoss(nn.Module):
    """Scale and shift invariant loss for depth estimation."""

    def __init__(self):
        """Initializes the ScaleAndShiftInvariantLoss."""
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask, interpolate=True, return_interpolated=False):
        """Computes the scale and shift invariant loss.

        Args:
            prediction (torch.Tensor): The predicted depth values.
            target (torch.Tensor): The ground truth depth values.
            mask (torch.Tensor): A mask to specify which pixels to consider.
            interpolate (bool, optional): Whether to interpolate the input to match the target shape.
            return_interpolated (bool, optional): Whether to return the interpolated input.

        Returns:
            torch.Tensor: The computed loss. If return_interpolated is True, also returns the interpolated input.
        """
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode="bilinear", align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        prediction, target, mask = (
            prediction.squeeze(),
            target.squeeze(),
            mask.squeeze(),
        )
        assert (
            prediction.shape == target.shape
        ), f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."

        scale, shift = compute_scale_and_shift(prediction, target, mask)

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        if not return_interpolated:
            return loss
        return loss, intr_input


if __name__ == "__main__":
    # Tests for DiscreteNLLLoss
    celoss = DiscreteNLLLoss()
    print(
        celoss(
            torch.rand(4, 64, 26, 32) * 10,
            torch.rand(4, 1, 26, 32) * 10,
        )
    )

    d = torch.Tensor([6.59, 3.8, 10.0])
    print(celoss.dequantize_depth(celoss.quantize_depth(d)))
