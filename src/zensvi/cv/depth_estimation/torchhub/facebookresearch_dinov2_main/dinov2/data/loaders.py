# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from .datasets import ImageNet, ImageNet22k
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler

logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    """Enumeration for different types of samplers."""

    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    """Converts a boolean value to a string.

    Args:
        b (bool): The boolean value to convert.

    Returns:
        str: "yes" if b is True, "no" otherwise.
    """
    return "yes" if b else "no"


def _make_sample_transform(
    image_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """Creates a sample transformation function.

    Args:
        image_transform (Optional[Callable]): A function to transform images. (Default is None)
        target_transform (Optional[Callable]): A function to transform targets. (Default is None)

    Returns:
        Callable: A function that applies the specified transformations to a sample.
    """

    def transform(sample):
        """Applies transformations to a sample.

        Args:
            sample: A tuple containing the image and target.

        Returns:
            Tuple: Transformed image and target.
        """
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    """Parses a dataset string into a class and keyword arguments.

    Args:
        dataset_str (str): The dataset string to parse.

    Returns:
        Tuple: A tuple containing the dataset class and a dictionary of keyword arguments.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """Creates a dataset with the specified parameters.

    Args:
        dataset_str (str): A dataset string description (e.g. ImageNet:split=TRAIN).
        transform (Optional[Callable]): A transform to apply to images. (Default is None)
        target_transform (Optional[Callable]): A transform to apply to targets. (Default is None)

    Returns:
        Dataset: The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    """Creates a sampler based on the specified parameters.

    Args:
        dataset: The dataset to sample from.
        type (Optional[SamplerType]): The type of sampler to create. (Default is None)
        shuffle (bool): Whether to shuffle the samples. (Default is False)
        seed (int): The random seed to use. (Default is 0)
        size (int): The number of samples to return. (Default is -1)
        advance (int): How many samples to skip. (Default is 0)

    Returns:
        Optional[Sampler]: The created sampler or None if no sampler is needed.

    Raises:
        ValueError: If the sampler size or advance is invalid.
        NotImplementedError: If advance > 0 for epoch sampler.
    """
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size (int): The size of batches to generate.
        num_workers (int): The number of workers to use.
        shuffle (bool): Whether to shuffle samples. (Default is True)
        seed (int): The random seed to use. (Default is 0)
        sampler_type (Optional[SamplerType]): Which sampler to use. (Default is SamplerType.INFINITE)
        sampler_size (int): The number of images per epoch or -1 for the entire dataset. (Default is -1)
        sampler_advance (int): How many samples to skip. (Default is 0)
        drop_last (bool): Whether the last non-full batch of data should be dropped. (Default is True)
        persistent_workers (bool): Maintain the workers alive after a dataset has been consumed once. (Default is False)
        collate_fn (Optional[Callable[[List[T]], Any]]): Function that performs batch collation. (Default is None)

    Returns:
        DataLoader: The created data loader.
    """
    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
