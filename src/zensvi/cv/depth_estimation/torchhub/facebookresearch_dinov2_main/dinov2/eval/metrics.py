# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.utilities.data import dim_zero_cat, select_topk

logger = logging.getLogger("dinov2")


class MetricType(Enum):
    """Enumeration of different metric types for evaluation."""

    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"

    @property
    def accuracy_averaging(self):
        """Returns the accuracy averaging method associated with the metric type."""
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    """Enumeration of accuracy averaging methods."""

    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    """Builds a metric based on the specified metric type.

    Args:
        metric_type (MetricType): The type of metric to build.
        num_classes (int): The number of classes for the metric.
        ks (Optional[tuple], optional): The top-k values to consider. Defaults to (1, 5).

    Returns:
        MetricCollection: A collection of metrics based on the specified type.

    Raises:
        ValueError: If the metric type is unknown.
    """
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    """Builds a top-k accuracy metric.

    Args:
        average_type (AccuracyAveraging): The type of averaging to use.
        num_classes (int): The number of classes for the metric.
        ks (tuple, optional): The top-k values to consider. Defaults to (1, 5).

    Returns:
        MetricCollection: A collection of top-k accuracy metrics.
    """
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)


def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    """Builds a top-k ImageNet real accuracy metric.

    Args:
        num_classes (int): The number of classes for the metric.
        ks (tuple, optional): The top-k values to consider. Defaults to (1, 5).

    Returns:
        MetricCollection: A collection of top-k ImageNet real accuracy metrics.
    """
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


class ImageNetReaLAccuracy(Metric):
    """Metric for calculating ImageNet real accuracy."""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initializes the ImageNet real accuracy metric.

        Args:
            num_classes (int): The number of classes for the metric.
            top_k (int, optional): The number of top predictions to consider. Defaults to 1.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Updates the metric with predictions and targets.

        Args:
            preds (Tensor): The predicted values.
            target (Tensor): The ground truth values.
        """
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros(
            (preds_oh.shape[0], preds_oh.shape[1] + 1),
            device=target.device,
            dtype=torch.int32,
        )
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        """Computes the final metric value.

        Returns:
            Tensor: The computed metric value.
        """
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()
