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
import torch.nn as nn
from torchvision.transforms import Normalize

from .dpt_dinov2.dpt import DPT_DINOv2


def denormalize(x):
    """Reverses the ImageNet normalization applied to the input.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W).

    Returns:
        torch.Tensor: Denormalized input tensor.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


def get_activation(name, bank):
    """Creates a hook to store the output of a layer.

    Args:
        name (str): Name of the layer.
        bank (dict): Dictionary to store the output.

    Returns:
        function: Hook function to capture the output.
    """

    def hook(model, input, output):
        """Hook function to save the output of the layer.

        Args:
            model: The model instance.
            input: The input to the layer.
            output: The output from the layer.
        """
        bank[name] = output

    return hook


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Initializes the Resize object.

        Args:
            width (int): Desired output width.
            height (int): Desired output height.
            resize_target (bool, optional): If True, resize the full sample (image, mask, target). Defaults to True.
            keep_aspect_ratio (bool, optional): If True, keep the aspect ratio of the input sample. Defaults to False.
            ensure_multiple_of (int, optional): Output width and height is constrained to be multiple of this parameter. Defaults to 1.
            resize_method (str, optional): Method of resizing. Options are "lower_bound", "upper_bound", "minimal". Defaults to "lower_bound".
        """
        print("Params passed to Resize transform:")
        print("\twidth: ", width)
        print("\theight: ", height)
        print("\tresize_target: ", resize_target)
        print("\tkeep_aspect_ratio: ", keep_aspect_ratio)
        print("\tensure_multiple_of: ", ensure_multiple_of)
        print("\tresize_method: ", resize_method)

        self.__width = width
        self.__height = height
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """Constrain a value to be a multiple of a specified number.

        Args:
            x (float): Value to constrain.
            min_val (int, optional): Minimum value. Defaults to 0.
            max_val (int, optional): Maximum value. Defaults to None.

        Returns:
            int: Constrained value.
        """
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        """Determine the new size based on the resizing method.

        Args:
            width (int): Original width.
            height (int): Original height.

        Returns:
            tuple: New width and height.
        """
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possible
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        """Resize the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Resized tensor.
        """
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode="bilinear", align_corners=True)


class PrepForMidas(object):
    """Prepares input for MiDaS model by resizing and normalizing."""

    def __init__(
        self,
        resize_mode="minimal",
        keep_aspect_ratio=True,
        img_size=384,
        do_resize=True,
    ):
        """Initializes the PrepForMidas object.

        Args:
            resize_mode (str, optional): Method of resizing. Defaults to "minimal".
            keep_aspect_ratio (bool, optional): If True, keep the aspect ratio of the input image. Defaults to True.
            img_size (int, tuple, optional): Size of the input image. Defaults to 384.
            do_resize (bool, optional): If True, perform resizing. Defaults to True.
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resizer = (
            Resize(
                net_w,
                net_h,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=14,
                resize_method=resize_mode,
            )
            if do_resize
            else nn.Identity()
        )

    def __call__(self, x):
        """Apply normalization and resizing to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Normalized and resized tensor.
        """
        return self.normalization(self.resizer(x))


class DepthAnythingCore(nn.Module):
    """Core class for Depth Anything model."""

    def __init__(
        self,
        midas,
        trainable=False,
        fetch_features=True,
        layer_names=("out_conv", "l4_rn", "r4", "r3", "r2", "r1"),
        freeze_bn=False,
        keep_aspect_ratio=True,
        img_size=384,
        **kwargs,
    ):
        """Initializes the DepthAnythingCore model.

        Args:
            midas (torch.nn.Module): MiDaS model.
            trainable (bool, optional): If True, train the MiDaS model. Defaults to False.
            fetch_features (bool, optional): If True, extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): If True, freeze BatchNorm layers. Defaults to False.
            keep_aspect_ratio (bool, optional): If True, keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__()
        self.core = midas
        self.output_channels = None
        self.core_out = {}
        self.trainable = trainable
        self.fetch_features = fetch_features
        self.handles = []
        self.layer_names = layer_names

        self.set_trainable(trainable)
        self.set_fetch_features(fetch_features)

        self.prep = PrepForMidas(
            keep_aspect_ratio=keep_aspect_ratio,
            img_size=img_size,
            do_resize=kwargs.get("do_resize", True),
        )

        if freeze_bn:
            self.freeze_bn()

    def set_trainable(self, trainable):
        """Set the trainable state of the model.

        Args:
            trainable (bool): If True, the model will be trainable.

        Returns:
            DepthAnythingCore: Self for chaining.
        """
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def set_fetch_features(self, fetch_features):
        """Set the feature fetching state of the model.

        Args:
            fetch_features (bool): If True, the model will fetch features.

        Returns:
            DepthAnythingCore: Self for chaining.
        """
        self.fetch_features = fetch_features
        if fetch_features:
            if len(self.handles) == 0:
                self.attach_hooks(self.core)
        else:
            self.remove_hooks()
        return self

    def freeze(self):
        """Freeze the parameters of the model."""
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        """Unfreeze the parameters of the model."""
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def freeze_bn(self):
        """Freeze BatchNorm layers in the model."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x, denorm=False, return_rel_depth=False):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            denorm (bool, optional): If True, denormalize the input. Defaults to False.
            return_rel_depth (bool, optional): If True, return relative depth along with features. Defaults to False.

        Returns:
            torch.Tensor or tuple: Output tensor or tuple of (relative depth, features).
        """
        with torch.no_grad():
            if denorm:
                x = denormalize(x)
            x = self.prep(x)

        with torch.set_grad_enabled(self.trainable):
            rel_depth = self.core(x)
            if not self.fetch_features:
                return rel_depth
        out = [self.core_out[k] for k in self.layer_names]

        if return_rel_depth:
            return rel_depth, out
        return out

    def get_rel_pos_params(self):
        """Yield relative positional parameters from the model.

        Yields:
            torch.Tensor: Relative positional parameters.
        """
        for name, p in self.core.pretrained.named_parameters():
            if "pos_embed" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        """Yield encoder parameters excluding relative positional parameters.

        Yields:
            torch.Tensor: Encoder parameters.
        """
        for name, p in self.core.pretrained.named_parameters():
            if "pos_embed" not in name:
                yield p

    def freeze_encoder(self, freeze_rel_pos=False):
        """Freeze the encoder parameters.

        Args:
            freeze_rel_pos (bool, optional): If True, freeze relative positional parameters. Defaults to False.

        Returns:
            DepthAnythingCore: Self for chaining.
        """
        if freeze_rel_pos:
            for p in self.core.pretrained.parameters():
                p.requires_grad = False
        else:
            for p in self.get_enc_params_except_rel_pos():
                p.requires_grad = False
        return self

    def attach_hooks(self, midas):
        """Attach hooks to the MiDaS model for feature extraction.

        Args:
            midas (torch.nn.Module): MiDaS model.

        Returns:
            DepthAnythingCore: Self for chaining.
        """
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(
                list(midas.depth_head.scratch.output_conv2.children())[1].register_forward_hook(
                    get_activation("out_conv", self.core_out)
                )
            )
        if "r4" in self.layer_names:
            self.handles.append(
                midas.depth_head.scratch.refinenet4.register_forward_hook(get_activation("r4", self.core_out))
            )
        if "r3" in self.layer_names:
            self.handles.append(
                midas.depth_head.scratch.refinenet3.register_forward_hook(get_activation("r3", self.core_out))
            )
        if "r2" in self.layer_names:
            self.handles.append(
                midas.depth_head.scratch.refinenet2.register_forward_hook(get_activation("r2", self.core_out))
            )
        if "r1" in self.layer_names:
            self.handles.append(
                midas.depth_head.scratch.refinenet1.register_forward_hook(get_activation("r1", self.core_out))
            )
        if "l4_rn" in self.layer_names:
            self.handles.append(
                midas.depth_head.scratch.layer4_rn.register_forward_hook(get_activation("l4_rn", self.core_out))
            )

        return self

    def remove_hooks(self):
        """Remove all hooks attached to the model."""
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        """Destructor to remove hooks when the object is deleted."""
        self.remove_hooks()

    def set_output_channels(self):
        """Set the output channels for the model."""
        self.output_channels = [256, 256, 256, 256, 256]

    @staticmethod
    def build(
        midas_model_type="dinov2_large",
        train_midas=False,
        use_pretrained_midas=True,
        fetch_features=False,
        freeze_bn=True,
        force_keep_ar=False,
        force_reload=False,
        **kwargs,
    ):
        """Build the DepthAnythingCore model.

        Args:
            midas_model_type (str, optional): Type of MiDaS model. Defaults to "dinov2_large".
            train_midas (bool, optional): If True, train the MiDaS model. Defaults to False.
            use_pretrained_midas (bool, optional): If True, use a pretrained MiDaS model. Defaults to True.
            fetch_features (bool, optional): If True, extract features. Defaults to False.
            freeze_bn (bool, optional): If True, freeze BatchNorm layers. Defaults to True.
            force_keep_ar (bool, optional): If True, force keeping aspect ratio. Defaults to False.
            force_reload (bool, optional): If True, force reloading the model. Defaults to False.
            **kwargs: Additional arguments.

        Returns:
            DepthAnythingCore: Instance of DepthAnythingCore model.
        """
        if "img_size" in kwargs:
            kwargs = DepthAnythingCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])

        depth_anything = DPT_DINOv2(out_channels=[256, 512, 1024, 1024], use_clstoken=False)

        state_dict = torch.load("./models/depth_anything_vitl14.pth", map_location="cpu")
        depth_anything.load_state_dict(state_dict)

        kwargs.update({"keep_aspect_ratio": force_keep_ar})

        depth_anything_core = DepthAnythingCore(
            depth_anything,
            trainable=train_midas,
            fetch_features=fetch_features,
            freeze_bn=freeze_bn,
            img_size=img_size,
            **kwargs,
        )

        depth_anything_core.set_output_channels()
        return depth_anything_core

    @staticmethod
    def parse_img_size(config):
        """Parse the image size from the configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            dict: Updated configuration dictionary with parsed image size.
        """
        assert "img_size" in config
        if isinstance(config["img_size"], str):
            assert "," in config["img_size"], "img_size should be a string with comma separated img_size=H,W"
            config["img_size"] = list(map(int, config["img_size"].split(",")))
            assert len(config["img_size"]) == 2, "img_size should be a string with comma separated img_size=H,W"
        elif isinstance(config["img_size"], int):
            config["img_size"] = [config["img_size"], config["img_size"]]
        else:
            assert (
                isinstance(config["img_size"], list) and len(config["img_size"]) == 2
            ), "img_size should be a list of H,W"
        return config


nchannels2models = {
    tuple([256] * 5): [
        "DPT_BEiT_L_384",
        "DPT_BEiT_L_512",
        "DPT_BEiT_B_384",
        "DPT_SwinV2_L_384",
        "DPT_SwinV2_B_384",
        "DPT_SwinV2_T_256",
        "DPT_Large",
        "DPT_Hybrid",
    ],
    (512, 256, 128, 64, 64): ["MiDaS_small"],
}

# Model name to number of output channels
MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items() for m in v}
