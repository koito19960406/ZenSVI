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

import itertools

import torch
import torch.nn as nn
from zoedepth.models.base_models.depth_anything import DepthAnythingCore
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import Projector, SeedBinRegressor, SeedBinRegressorUnnormed
from zoedepth.models.layers.patch_transformer import PatchTransformerEncoder
from zoedepth.models.model_io import load_state_from_resource


class ZoeDepthNK(DepthModel):
    """ZoeDepthNK model with two metric heads and a learned router to experts."""

    def __init__(
        self,
        core,
        bin_conf,
        bin_centers_type="softplus",
        bin_embedding_dim=128,
        n_attractors=[16, 8, 4, 1],
        attractor_alpha=300,
        attractor_gamma=2,
        attractor_kind="sum",
        attractor_type="exp",
        min_temp=5,
        max_temp=50,
        memory_efficient=False,
        train_midas=True,
        is_midas_pretrained=True,
        midas_lr_factor=1,
        encoder_lr_factor=10,
        pos_enc_lr_factor=10,
        inverse_midas=False,
        **kwargs,
    ):
        """Initializes the ZoeDepthNK model.

        Args:
            core (DepthAnythingCore): The base midas model used for extraction of "relative" features.
            bin_conf (List[dict]): A list of dictionaries containing the bin configuration for each metric head.
                Each dictionary should contain the following keys:
                    - "name" (str): Typically the same as the dataset name.
                    - "n_bins" (int): Number of bins.
                    - "min_depth" (float): Minimum depth.
                    - "max_depth" (float): Maximum depth.
            bin_centers_type (str, optional): Type of activation for bin centers. Options are "normed" or "softplus". Defaults to "normed".
            bin_embedding_dim (int, optional): Dimension of the bin embedding. Defaults to 128.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation method. Options are "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; options are "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            memory_efficient (bool, optional): Whether to use a memory-efficient version of attractor layers. Defaults to False.
            train_midas (bool, optional): Whether to train the core midas model. Defaults to True.
            is_midas_pretrained (bool, optional): Indicates if the core is pretrained. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for the base midas model. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in the midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
            inverse_midas (bool, optional): Whether to use the inverse midas model. Defaults to False.
        """
        super().__init__()

        self.core = core
        self.bin_conf = bin_conf
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.memory_efficient = memory_efficient
        self.train_midas = train_midas
        self.is_midas_pretrained = is_midas_pretrained
        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.inverse_midas = inverse_midas

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features, kernel_size=1, stride=1, padding=0)

        # Transformer classifier on the bottleneck
        self.patch_transformer = PatchTransformerEncoder(btlnck_features, 1, 128, use_class_token=True)
        self.mlp_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError("bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")
        self.bin_centers_type = bin_centers_type

        # Create a map (ModuleDict) of 'name' -> seed_bin_regressor
        self.seed_bin_regressors = nn.ModuleDict(
            {
                conf["name"]: SeedBinRegressorLayer(
                    btlnck_features,
                    conf["n_bins"],
                    mlp_dim=bin_embedding_dim // 2,
                    min_depth=conf["min_depth"],
                    max_depth=conf["max_depth"],
                )
                for conf in bin_conf
            }
        )

        self.seed_projector = Projector(btlnck_features, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
        self.projectors = nn.ModuleList(
            [Projector(num_out, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2) for num_out in num_out_features]
        )

        # Create a map (ModuleDict) of 'name' -> attractors (ModuleList)
        self.attractors = nn.ModuleDict(
            {
                conf["name"]: nn.ModuleList(
                    [
                        Attractor(
                            bin_embedding_dim,
                            n_attractors[i],
                            mlp_dim=bin_embedding_dim,
                            alpha=attractor_alpha,
                            gamma=attractor_gamma,
                            kind=attractor_kind,
                            attractor_type=attractor_type,
                            memory_efficient=memory_efficient,
                            min_depth=conf["min_depth"],
                            max_depth=conf["max_depth"],
                        )
                        for i in range(len(n_attractors))
                    ]
                )
                for conf in bin_conf
            }
        )

        last_in = N_MIDAS_OUT
        # Conditional log binomial for each bin conf
        self.conditional_log_binomial = nn.ModuleDict(
            {
                conf["name"]: ConditionalLogBinomial(
                    last_in,
                    bin_embedding_dim,
                    conf["n_bins"],
                    bottleneck_factor=4,
                    min_temp=self.min_temp,
                    max_temp=self.max_temp,
                )
                for conf in bin_conf
            }
        )

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_final_centers (bool, optional): If True, returns final bin centers. Defaults to False.
            denorm (bool, optional): If True, denormalizes the output. Defaults to False.
            return_probs (bool, optional): If True, returns bin probabilities. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary of outputs with keys:
                - "rel_depth": Relative depth map of shape (B, 1, H, W).
                - "metric_depth": Metric depth map of shape (B, 1, H, W).
                - "domain_logits": Domain logits of shape (B, 2).
                - "bin_centers": Bin centers of shape (B, N, H, W). Present only if return_final_centers is True.
                - "probs": Bin probabilities of shape (B, N, H, W). Present only if return_probs is True.
        """
        b, c, h, w = x.shape
        self.orig_input_width = w
        self.orig_input_height = h
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0

        # Predict which path to take
        embedding = self.patch_transformer(x)[0]  # N, E
        domain_logits = self.mlp_classifier(embedding)  # N, 2
        domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)  # 1, 2

        # Get the path
        bin_conf_name = ["nyu", "kitti"][torch.argmax(domain_vote, dim=-1).squeeze().item()]

        try:
            conf = [c for c in self.bin_conf if c.name == bin_conf_name][0]
        except IndexError:
            raise ValueError(f"bin_conf_name {bin_conf_name} not found in bin_confs")

        min_depth = conf["min_depth"]
        max_depth = conf["max_depth"]

        seed_bin_regressor = self.seed_bin_regressors[bin_conf_name]
        _, seed_b_centers = seed_bin_regressor(x)
        if self.bin_centers_type == "normed" or self.bin_centers_type == "hybrid2":
            b_prev = (seed_b_centers - min_depth) / (max_depth - min_depth)
        else:
            b_prev = seed_b_centers
        prev_b_embedding = self.seed_projector(x)

        attractors = self.attractors[bin_conf_name]
        for projector, attractor, x in zip(self.projectors, attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b
            prev_b_embedding = b_embedding

        last = outconv_activation

        b_centers = nn.functional.interpolate(b_centers, last.shape[-2:], mode="bilinear", align_corners=True)
        b_embedding = nn.functional.interpolate(b_embedding, last.shape[-2:], mode="bilinear", align_corners=True)

        clb = self.conditional_log_binomial[bin_conf_name]
        x = clb(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        output = dict(domain_logits=domain_logits, metric_depth=out)
        if return_final_centers or return_probs:
            output["bin_centers"] = b_centers

        if return_probs:
            output["probs"] = x
        return output

    def get_lr_params(self, lr):
        """Configures learning rates for different layers of the model.

        Args:
            lr (float): Base learning rate.

        Returns:
            list: List of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:

            def get_rel_pos_params():
                """Yields parameters related to relative position."""
                for name, p in self.core.core.pretrained.named_parameters():
                    if "pos_embed" in name:
                        yield p

            def get_enc_params_except_rel_pos():
                """Yields parameters excluding relative position."""
                for name, p in self.core.core.pretrained.named_parameters():
                    if "pos_embed" not in name:
                        yield p

            encoder_params = get_enc_params_except_rel_pos()
            rel_pos_params = get_rel_pos_params()
            midas_params = self.core.core.depth_head.parameters()
            midas_lr_factor = self.midas_lr_factor if self.is_midas_pretrained else 1.0
            param_conf.extend(
                [
                    {"params": encoder_params, "lr": lr / self.encoder_lr_factor},
                    {"params": rel_pos_params, "lr": lr / self.pos_enc_lr_factor},
                    {"params": midas_params, "lr": lr / midas_lr_factor},
                ]
            )

        remaining_modules = []
        for name, child in self.named_children():
            if name != "core":
                remaining_modules.append(child)
        remaining_params = itertools.chain(*[child.parameters() for child in remaining_modules])
        param_conf.append({"params": remaining_params, "lr": lr})
        return param_conf

    def get_conf_parameters(self, conf_name):
        """Returns parameters of all ModuleDict children exclusively used for the given bin configuration.

        Args:
            conf_name (str): Name of the bin configuration.

        Returns:
            list: List of parameters for the specified configuration.
        """
        params = []
        for name, child in self.named_children():
            if isinstance(child, nn.ModuleDict):
                for bin_conf_name, module in child.items():
                    if bin_conf_name == conf_name:
                        params += list(module.parameters())
        return params

    def freeze_conf(self, conf_name):
        """Freezes all parameters of ModuleDict children exclusively used for the given bin configuration.

        Args:
            conf_name (str): Name of the bin configuration.
        """
        for p in self.get_conf_parameters(conf_name):
            p.requires_grad = False

    def unfreeze_conf(self, conf_name):
        """Unfreezes all parameters of ModuleDict children exclusively used for the given bin configuration.

        Args:
            conf_name (str): Name of the bin configuration.
        """
        for p in self.get_conf_parameters(conf_name):
            p.requires_grad = True

    def freeze_all_confs(self):
        """Freezes all parameters of all ModuleDict children."""
        for name, child in self.named_children():
            if isinstance(child, nn.ModuleDict):
                for bin_conf_name, module in child.items():
                    for p in module.parameters():
                        p.requires_grad = False

    @staticmethod
    def build(
        midas_model_type="DPT_BEiT_L_384",
        pretrained_resource=None,
        use_pretrained_midas=False,
        train_midas=False,
        freeze_midas_bn=True,
        **kwargs,
    ):
        """Builds the ZoeDepthNK model.

        Args:
            midas_model_type (str, optional): Type of midas model. Defaults to "DPT_BEiT_L_384".
            pretrained_resource (str, optional): Resource for pretrained model. Defaults to None.
            use_pretrained_midas (bool, optional): Whether to use a pretrained midas model. Defaults to False.
            train_midas (bool, optional): Whether to train the midas model. Defaults to False.
            freeze_midas_bn (bool, optional): Whether to freeze batch normalization in midas model. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            ZoeDepthNK: An instance of the ZoeDepthNK model.
        """
        core = DepthAnythingCore.build(
            midas_model_type="dinov2_large",
            use_pretrained_midas=use_pretrained_midas,
            train_midas=train_midas,
            fetch_features=True,
            freeze_bn=freeze_midas_bn,
            **kwargs,
        )

        model = ZoeDepthNK(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        """Builds the ZoeDepthNK model from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ZoeDepthNK: An instance of the ZoeDepthNK model.
        """
        return ZoeDepthNK.build(**config)
