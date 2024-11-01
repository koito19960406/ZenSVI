import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

# import pytorch_lightning as pl


class PlacePulseClassificationModel(nn.Module):
    """A neural network model for Place Pulse classification tasks.

    This model uses a ResNet50 backbone followed by custom fully connected layers
    to predict scores for place pulse attributes like safety, beauty, etc.

    Args:
        lr (float, optional): Learning rate for optimization. Defaults to 0.0001.
        num_classes (int): Number of output classes.
        **kwargs: Additional keyword arguments.

    Attributes:
        resnet50 (nn.Module): ResNet50 backbone network.
        lr (float): Learning rate.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        fc4 (nn.Linear): Output fully connected layer.
    """

    def __init__(self, lr=0.0001, num_classes=None, **kwargs):
        super().__init__()
        self.resnet50 = models.resnet50(weights=None)
        self.lr = lr

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Identity()

        # Adding custom fully connected layers
        self.fc1 = nn.Linear(num_ftrs, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Predicted scores for each input image.
        """
        x = self.resnet50(x)  # Features from ResNet50

        # Passing through the custom fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Convert logits to probabilities using softmax
        probabilities = F.softmax(x, dim=1)
        scores = self.calculate_score(probabilities)
        return scores

    def calculate_score(self, probabilities):
        """Calculate weighted scores from class probabilities.

        Args:
            probabilities (torch.Tensor): Softmax probabilities of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Weighted scores for each input of shape (batch_size,).

        Raises:
            ValueError: If number of classes doesn't match number of weights.
        """
        # Define weights for each class
        weights = torch.tensor([1, 3, 5, 7, 9], device=probabilities.device)
        # weights = torch.tensor([5/6, 15/6, 25/6, 35/6, 45/6, 55/6], device=probabilities.device)

        # Ensure the weights are compatible with the probabilities
        if probabilities.shape[1] != len(weights):
            raise ValueError("The number of classes does not match the number of weights.")

        # Compute the weighted score for each image
        scores = torch.sum(probabilities * weights, dim=1)
        return scores

    # def evaluate(self, batch):
    #     x, y = batch
    #     scores = self(x)  # Directly obtain scores from the model's forward pass
    #     return {'scores': scores}
