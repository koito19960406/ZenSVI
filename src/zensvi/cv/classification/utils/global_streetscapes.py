from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

# dict2idx -------------------------------------
weather_dict2idx = {
    "label2index": {"clear": 0, "cloudy": 1, "foggy": 2, "rainy": 3, "snowy": 4},
    "index2label": {0: "clear", 1: "cloudy", 2: "foggy", 3: "rainy", 4: "snowy"},
}

glare_dict2idx = {
    "label2index": {"False": 0, "True": 1},
    "index2label": {0: "False", 1: "True"},
}

lighting_dict2idx = {
    "label2index": {"day": 0, "dusk/dawn": 1, "night": 2},
    "index2label": {0: "day", 1: "dusk/dawn", 2: "night"},
}

panorama_dict2idx = {
    "label2index": {False: 0, True: 1},
    "index2label": {0: "False", 1: "True"},
}

platform_dict2idx = {
    "label2index": {
        "cycling surface": 0,
        "driving surface": 1,
        "fields": 2,
        "railway": 3,
        "tunnel": 4,
        "walking surface": 5,
    },
    "index2label": {
        0: "cycling surface",
        1: "driving surface",
        2: "fields",
        3: "railway",
        4: "tunnel",
        5: "walking surface",
    },
}

quality_dict2idx = {
    "label2index": {"good": 0, "slightly poor": 1, "very poor": 2},
    "index2label": {0: "good", 1: "slightly poor", 2: "very poor"},
}

reflection_dict2idx = {
    "label2index": {"False": 0, "True": 1},
    "index2label": {0: "False", 1: "True"},
}

view_direction_dict2idx = {
    "label2index": {"front/back": 0, "side": 1},
    "index2label": {0: "front/back", 1: "side"},
}


# ----------------------------------------------
class GlobalStreetScapesClassificationModel(pl.LightningModule):
    """Creates a classification model for streetscape image attributes.

    This model can classify one of eight different streetscape attributes:
    weather, glare, lighting, panorama, platform, quality, reflection,
    and view direction.

    Args:
        lr (float, optional): Learning rate. Defaults to 0.0001.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        weight (str, optional): Path to custom loss weights file. Defaults to None.
        num_classes (int, optional): Number of classes for the target variable. Defaults to None.
        class_mapping (dict, optional): Mapping between class names and indices. Defaults to None.
        model (str, optional): Model architecture name. Defaults to "maxvit_t".
        **kwargs: Additional keyword arguments.

    Attributes:
        lr (float): Learning rate for optimization.
        class_mapping (dict): Mapping between class names and indices.
        model (nn.Module): The underlying neural network model.
        loss_fn (nn.Module): The loss function used for training.
    """

    def __init__(
        self,
        lr=0.0001,
        pretrained=True,
        weight=None,
        num_classes=None,
        class_mapping=None,
        model="maxvit_t",
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.class_mapping = class_mapping

        # Setup the model
        if pretrained:
            self.model = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.DEFAULT)
        else:
            self.model = torchvision.models.maxvit_t(weights=None)
        self.model.classifier[-1] = nn.Linear(in_features=512, out_features=num_classes)

        # Configure the loss function
        if weight is not None and Path(weight).exists():
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.load(weight))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.kwargs = kwargs

    def forward(self, x):
        """Performs forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        logits = self.model(x)
        return logits
