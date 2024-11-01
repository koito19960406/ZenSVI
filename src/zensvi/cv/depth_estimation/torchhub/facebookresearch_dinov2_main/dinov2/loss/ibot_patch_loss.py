# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("dinov2")


try:
    from xformers.ops import cross_entropy

    def lossfunc(t, s, temp):
        """Calculates the loss using cross-entropy.

        Args:
            t (torch.Tensor): Target tensor.
            s (torch.Tensor): Prediction tensor.
            temp (float): Temperature parameter for scaling.

        Returns:
            torch.Tensor: Computed loss.
        """
        s = s.float()
        t = t.float()
        if s.ndim == 2:
            return -cross_entropy(s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True).squeeze(0)
        elif s.ndim == 3:
            return -cross_entropy(s, t, temp, bw_inplace=True)

except ImportError:

    def lossfunc(t, s, temp):
        """Calculates the loss using log softmax.

        Args:
            t (torch.Tensor): Target tensor.
            s (torch.Tensor): Prediction tensor.
            temp (float): Temperature parameter for scaling.

        Returns:
            torch.Tensor: Computed loss.
        """
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


class iBOTPatchLoss(nn.Module):
    """Implements the iBOT patch loss for training.

    Attributes:
        student_temp (float): Temperature for the student model.
        center_momentum (float): Momentum for updating the center.
    """

    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        """Initializes the iBOTPatchLoss.

        Args:
            patch_out_dim (int): Output dimension of the patch.
            student_temp (float, optional): Temperature for the student model. Defaults to 0.1.
            center_momentum (float, optional): Momentum for updating the center. Defaults to 0.9.
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        """Applies softmax to the teacher patch tokens with centering.

        Args:
            teacher_patch_tokens (torch.Tensor): Teacher patch tokens.
            teacher_temp (float): Temperature for scaling.

        Returns:
            torch.Tensor: Softmax normalized teacher patch tokens.
        """
        self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        """Applies Sinkhorn-Knopp normalization to the teacher output.

        Args:
            teacher_output (torch.Tensor): Teacher output tensor.
            teacher_temp (float): Temperature for scaling.
            n_masked_patches_tensor (torch.Tensor): Number of masked patches.
            n_iterations (int, optional): Number of iterations for normalization. Defaults to 3.

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """Computes the loss between student and teacher patch tokens.

        Args:
            student_patch_tokens (torch.Tensor): Student patch tokens.
            teacher_patch_tokens (torch.Tensor): Teacher patch tokens.
            student_masks_flat (torch.Tensor): Masks for the student tokens.

        Returns:
            torch.Tensor: Computed loss.
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """Computes the masked loss between student and teacher patch tokens.

        Args:
            student_patch_tokens_masked (torch.Tensor): Masked student patch tokens.
            teacher_patch_tokens_masked (torch.Tensor): Masked teacher patch tokens.
            student_masks_flat (torch.Tensor): Masks for the student tokens.
            n_masked_patches (int, optional): Number of masked patches. Defaults to None.
            masks_weight (torch.Tensor, optional): Weights for the masks. Defaults to None.

        Returns:
            torch.Tensor: Computed masked loss.
        """
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        """Updates the center using the teacher patch tokens.

        Args:
            teacher_patch_tokens (torch.Tensor): Teacher patch tokens.
        """
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        """Reduces the center update across distributed processes.

        Args:
            teacher_patch_tokens (torch.Tensor): Teacher patch tokens.
        """
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        """Applies the center update after reduction."""
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
