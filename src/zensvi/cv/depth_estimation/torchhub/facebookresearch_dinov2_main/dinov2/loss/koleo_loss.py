# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist


logger = logging.getLogger("dinov2")


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al.

    This loss function is designed to encourage the spreading of vectors for similarity search.

    References:
        - Sablayrolles et al. (2018). Spreading vectors for similarity search.

    Args:
        None

    Returns:
        None
    """

    def __init__(self):
        """Initializes the KoLeoLoss module and sets up the pairwise distance metric."""
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """Computes pairwise nearest neighbors for L2-normalized vectors.

        This method uses Torch instead of Faiss to remain on the GPU.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D) where B is the batch size and D is the feature dimension.

        Returns:
            torch.Tensor: Indices of the nearest neighbors for each vector in the input tensor.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """Computes the loss based on the student output.

        Args:
            student_output (torch.Tensor): Backbone output of the student of shape (B, D).
            eps (float, optional): A small value to avoid division by zero. Default is 1e-8.

        Returns:
            torch.Tensor: The computed loss value.
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
