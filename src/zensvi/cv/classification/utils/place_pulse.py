import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
# import pytorch_lightning as pl

class PlacePulseClassificationModel(nn.Module):
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
