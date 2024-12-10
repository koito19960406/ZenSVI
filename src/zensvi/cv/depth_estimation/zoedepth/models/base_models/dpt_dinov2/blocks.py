import torch.nn as nn


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """Creates a scratch module with convolutional layers based on input and output shapes.

    Args:
        in_shape (tuple): Shape of the input tensor.
        out_shape (int): Number of output channels for the convolutional layers.
        groups (int, optional): Number of groups for the convolution. Defaults to 1.
        expand (bool, optional): If True, expands the output shapes for the layers. Defaults to False.

    Returns:
        nn.Module: A module containing the convolutional layers.
    """
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module that applies two convolutional layers with an optional batch normalization.

    Args:
        features (int): Number of input and output features.
        activation (callable): Activation function to be applied after each convolution.
        bn (bool): If True, applies batch normalization after each convolution.
    """

    def __init__(self, features, activation, bn):
        """Initializes the ResidualConvUnit.

        Args:
            features (int): Number of features.
            activation (callable): Activation function.
            bn (bool): If True, apply batch normalization.
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass through the residual convolution unit.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after applying convolutions and skip connection.
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block that combines features from multiple sources.

    Args:
        features (int): Number of input features.
        activation (callable): Activation function to be applied.
        deconv (bool): If True, uses deconvolution. Defaults to False.
        bn (bool): If True, applies batch normalization. Defaults to False.
        expand (bool): If True, expands the output features. Defaults to False.
        align_corners (bool): If True, aligns corners during interpolation. Defaults to True.
        size (tuple, optional): Target size for the output. Defaults to None.
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Initializes the FeatureFusionBlock.

        Args:
            features (int): Number of features.
            activation (callable): Activation function.
            deconv (bool): If True, uses deconvolution.
            bn (bool): If True, applies batch normalization.
            expand (bool): If True, expands the output features.
            align_corners (bool): If True, aligns corners during interpolation.
            size (tuple, optional): Target size for the output.
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass through the feature fusion block.

        Args:
            *xs: Input tensors.
            size (tuple, optional): Target size for the output. Defaults to None.

        Returns:
            tensor: Output tensor after feature fusion and processing.
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output
