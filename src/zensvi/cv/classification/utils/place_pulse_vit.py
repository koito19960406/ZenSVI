import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image


class PlacePulseClassificationModelViT(nn.Module):
    """
    This model was developed by Ouyan (2023).
    The original code can be access here:
        https://github.com/strawmelon11/human-perception-place-pulse
    """

    def __init__(self):
        super(Net, self).__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_fc = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_fc, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_class, bias=True)
        )
        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

    def forward(self, x):
        x = self.model(x)
        return x

    def calculate_score(self, img_path, device):
        # TODO: the img conversaion is done in the other class
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.train_transform(img)
        img = img.view(1, 3, 384, 384)
        # inference
        img = img.to(device)
        pred = self.model(img)
        softmax = nn.Softmax(dim=1)
        pred = softmax(pred)[0][1].item()
        pred = round(pred*10, 2)

        return pred
