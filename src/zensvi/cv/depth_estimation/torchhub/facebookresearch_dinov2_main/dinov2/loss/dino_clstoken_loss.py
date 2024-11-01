# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class DINOLoss(nn.Module):
    """DINO Loss for training student networks with teacher outputs.

    This loss function computes the cross-entropy between the softmax outputs of the teacher and student networks,
    while also maintaining a running center for the teacher outputs.

    Attributes:
        student_temp (float): Temperature parameter for scaling student outputs.
        center_momentum (float): Momentum for updating the center.
        center (torch.Tensor): Running center for teacher outputs.
        updated (bool): Flag indicating if the center has been updated.
        reduce_handle (Optional[torch.Tensor]): Handle for asynchronous reduction.
        len_teacher_output (Optional[int]): Length of the teacher output.
        async_batch_center (Optional[torch.Tensor]): Asynchronous batch center for teacher outputs.
    """

    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        """Initializes the DINOLoss module.

        Args:
            out_dim (int): Dimension of the output features.
            student_temp (float, optional): Temperature parameter for scaling student outputs. Default is 0.1.
            center_momentum (float, optional): Momentum for updating the center. Default is 0.9.
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        """Applies softmax to the teacher output after centering.

        Args:
            teacher_output (torch.Tensor): The output from the teacher network.
            teacher_temp (float): Temperature parameter for scaling teacher outputs.

        Returns:
            torch.Tensor: Softmaxed and centered teacher output.
        """
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        """Applies Sinkhorn-Knopp normalization to the teacher output.

        Args:
            teacher_output (torch.Tensor): The output from the teacher network.
            teacher_temp (float): Temperature parameter for scaling teacher outputs.
            n_iterations (int, optional): Number of iterations for normalization. Default is 3.

        Returns:
            torch.Tensor: Normalized output matrix.
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
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

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """Computes the loss based on the student and teacher outputs.

        Args:
            student_output_list (list of torch.Tensor): List of outputs from the student network.
            teacher_out_softmaxed_centered_list (list of torch.Tensor): List of softmaxed and centered outputs from the teacher network.

        Returns:
            torch.Tensor: The computed total loss.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Updates the center using the teacher output.

        Args:
            teacher_output (torch.Tensor): The output from the teacher network.
        """
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        """Reduces the center update across distributed processes.

        Args:
            teacher_output (torch.Tensor): The output from the teacher network.
        """
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        """Applies the center update after reduction."""
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
