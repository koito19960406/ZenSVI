import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    """Creates a feature fusion block.

    Args:
        features (int): Number of input features.
        use_bn (bool): Whether to use batch normalization.
        size (tuple, optional): Target size for the output. Defaults to None.

    Returns:
        FeatureFusionBlock: The constructed feature fusion block.
    """
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    """DPT Head for depth estimation.

    This class processes the output features from the encoder and generates depth maps.

    Args:
        in_channels (int): Number of input channels.
        features (int, optional): Number of features for the head. Defaults to 256.
        use_bn (bool, optional): Whether to use batch normalization. Defaults to False.
        out_channels (list, optional): List of output channels for each layer. Defaults to [256, 512, 1024, 1024].
        use_clstoken (bool, optional): Whether to use class token. Defaults to False.
    """

    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        """Forward pass through the DPT head.

        Args:
            out_features (list): List of output features from the encoder.
            patch_h (int): Height of the patch.
            patch_w (int): Width of the patch.

        Returns:
            Tensor: The output depth map.
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        return out


class DPT_DINOv2(nn.Module):
    """DPT DINOv2 model for depth estimation.

    This class initializes the DPT model with a specified encoder and processes input images to generate depth maps.

    Args:
        encoder (str, optional): The encoder type to use. Defaults to "vitl".
        features (int, optional): Number of features for the depth head. Defaults to 256.
        use_bn (bool, optional): Whether to use batch normalization. Defaults to False.
        out_channels (list, optional): List of output channels for each layer. Defaults to [256, 512, 1024, 1024].
        use_clstoken (bool, optional): Whether to use class token. Defaults to False.
    """

    def __init__(
        self,
        encoder="vitl",
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):

        super(DPT_DINOv2, self).__init__()

        torch.manual_seed(1)

        self.pretrained = torch.hub.load(
            "src/zensvi/cv/depth_estimation/torchhub/facebookresearch_dinov2_main",
            "dinov2_{:}14".format(encoder),
            source="local",
            pretrained=False,
        )

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        """Forward pass through the DPT DINOv2 model.

        Args:
            x (Tensor): Input tensor representing the image.

        Returns:
            Tensor: The output depth map.
        """
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)

        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
