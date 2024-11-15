from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class Net(nn.Module):
    """Neural network model for human perception prediction using Vision Transformer.

    This model was developed by Ouyan (2023) and uses a pre-trained ViT-B-16 backbone
    with custom classification head for predicting human perception scores of places.
    The original code can be accessed at:
    https://github.com/strawmelon11/human-perception-place-pulse

    Args:
        num_classes: Number of output classes for classification.

    Attributes:
        model: The Vision Transformer model with custom classification head.
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
            nn.Linear(256, num_classes, bias=True),
        )
        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor of perception scores scaled between 0-10.
        """
        x = self.model(x)

        scores = self.calculate_score(x)
        return scores

    def calculate_score(self, probabilities):
        """Calculates final perception scores from model outputs.

        Args:
            probabilities: Raw model output logits.

        Returns:
            Tensor of perception scores scaled between 0-10 and rounded to 2 decimals.
        """
        softmax = nn.Softmax(dim=1)
        scores = softmax(probabilities)[:, 1]  # get all probabilities
        scores = (scores * 10).round(decimals=2)  # scale and round values

        return scores
