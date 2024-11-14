# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import dinov2.distributed as distributed
import torch
from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
from dinov2.logging import MetricLogger
from torch import nn
from torchmetrics import MetricCollection

logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    """Model wrapper that normalizes the output of the given model.

    Args:
        model (nn.Module): The model to be wrapped.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        """Forward pass through the model with normalization.

        Args:
            samples (torch.Tensor): Input samples to the model.

        Returns:
            torch.Tensor: Normalized output from the model.
        """
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    """Model that extracts features from intermediate layers.

    Args:
        feature_model (nn.Module): The feature extraction model.
        n_last_blocks (int): Number of last blocks to extract features from.
        autocast_ctx: Autocast context for mixed precision.
    """

    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        """Forward pass to get features from intermediate layers.

        Args:
            images (torch.Tensor): Input images to the model.

        Returns:
            torch.Tensor: Extracted features from the specified layers.
        """
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    """Evaluate the model on the given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader: Data loader for the evaluation dataset.
        postprocessors (Dict[str, nn.Module]): Postprocessing modules for metrics.
        metrics (Dict[str, MetricCollection]): Metrics to compute.
        device (torch.device): Device to perform evaluation on.
        criterion (Optional[nn.Module]): Loss function for evaluation (default is None).

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple containing the metric logger stats and computed metrics.
    """
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    """Gather tensors from all ranks and flatten the result.

    Args:
        tensor_rank (torch.Tensor): The tensor to gather from all ranks.

    Returns:
        torch.Tensor: Flattened tensor containing data from all ranks.
    """
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    """Extract features from the dataset using the specified model.

    Args:
        model (nn.Module): The model to extract features from.
        dataset: The dataset to extract features from.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        gather_on_cpu (bool, optional): Whether to gather results on CPU (default is False).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted features and corresponding labels.
    """
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    """Extract features using a data loader.

    Args:
        model (nn.Module): The model to extract features from.
        data_loader: Data loader for the dataset.
        sample_count (int): Total number of samples in the dataset.
        gather_on_cpu (bool, optional): Whether to gather results on CPU (default is False).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Extracted features and corresponding labels.
    """
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
