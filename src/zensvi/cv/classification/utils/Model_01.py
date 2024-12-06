import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class Net(nn.Module):
    """
    This model was developed by Ouyan (2023).
    The original code can be access here:
        https://github.com/strawmelon11/human-perception-place-pulse
    """

    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_fc = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_fc, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_classes, bias=True)
        )
        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

    def forward(self, x):
        x = self.model(x)

        scores = self.calculate_score(x)
        return scores

    def calculate_score(self, probabilities):
        softmax = nn.Softmax(dim=1)
        scores = softmax(probabilities)[:, 1]  # get all probabilities
        scores = (scores * 10).round(decimals=2)  # scale and round values

        return scores
