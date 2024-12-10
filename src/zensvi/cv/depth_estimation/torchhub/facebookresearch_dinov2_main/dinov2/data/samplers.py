# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings
from typing import Any, Optional

import dinov2.distributed as distributed
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class EpochSampler(Sampler):
    """Sampler that yields indices for a dataset over epochs.

    Attributes:
        size (int): Total number of samples.
        sample_count (int): Number of samples to draw in each iteration.
        shuffle (bool): Whether to shuffle the samples.
        seed (int): Seed for random number generation.
        start (Optional[int]): Starting index for sampling.
        step (Optional[int]): Step size for sampling.
    """

    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        """Yield indices for sampling based on the current epoch.

        Yields:
            Indices for the samples.
        """
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        """Return the number of samples that can be drawn.

        Returns:
            int: Number of samples.
        """
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch: int):
        """Set the current epoch for sampling.

        Args:
            epoch (int): The current epoch number.
        """
        self._epoch = epoch


def _get_numpy_dtype(size: int) -> Any:
    """Get the appropriate NumPy data type based on the size.

    Args:
        size (int): The size to determine the data type for.

    Returns:
        Any: NumPy data type (np.int32 or np.int64).
    """
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    """Get the appropriate PyTorch data type based on the size.

    Args:
        size (int): The size to determine the data type for.

    Returns:
        Any: PyTorch data type (torch.int32 or torch.int64).
    """
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation.

    Args:
        size (int): The size of the permutation.
        generator (torch.Generator): The random number generator.

    Yields:
        int: Randomly permuted indices.
    """
    dtype = _get_torch_dtype(size)
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    """Sampler that yields an infinite sequence of indices.

    Attributes:
        sample_count (int): Number of samples to draw in each iteration.
        shuffle (bool): Whether to shuffle the samples.
        seed (int): Seed for random number generation.
        start (Optional[int]): Starting index for sampling.
        step (Optional[int]): Step size for sampling.
        advance (int): Number of samples to advance before yielding.
    """

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        """Yield an infinite sequence of indices for sampling.

        Yields:
            Indices for the samples.
        """
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        """Yield indices in a non-shuffled manner.

        Yields:
            Indices for the samples.
        """
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        """Yield indices in a shuffled manner.

        Yields:
            Indices for the samples.
        """
        assert self._shuffle

        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    """Shuffle a slice of a tensor.

    Args:
        tensor (torch.Tensor): The tensor to shuffle.
        start (int, optional): The starting index for the slice. Defaults to 0.
        step (int, optional): The step size for the slice. Defaults to 1.
        generator (torch.Generator): The random number generator.

    Returns:
        np.ndarray: The shuffled slice of the tensor.
    """
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0
        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    """Shuffle a slice of a tensor using a new method.

    Args:
        tensor (torch.Tensor): The tensor to shuffle.
        start (int, optional): The starting index for the slice. Defaults to 0.
        step (int, optional): The step size for the slice. Defaults to 1.
        generator (torch.Generator): The random number generator.

    Returns:
        np.ndarray: The shuffled slice of the tensor.
    """
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    """Create a seed based on the initial seed, start index, and iteration count.

    Args:
        seed (int): The initial seed.
        start (int): The starting index.
        iter_count (int): The iteration count.

    Returns:
        int: The generated seed.
    """
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    """Sampler that yields an infinite sequence of indices with sharding.

    Attributes:
        sample_count (int): Number of samples to draw in each iteration.
        shuffle (bool): Whether to shuffle the samples.
        seed (int): Seed for random number generation.
        start (Optional[int]): Starting index for sampling.
        step (Optional[int]): Step size for sampling.
        advance (int): Number of samples to advance before yielding.
        use_new_shuffle_tensor_slice (bool): Whether to use the new shuffle method.
    """

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        """Yield an infinite sequence of indices for sampling.

        Yields:
            Indices for the samples.
        """
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        """Yield indices in a non-shuffled manner.

        Yields:
            Indices for the samples.
        """
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        """Yield indices in a shuffled manner.

        Yields:
            Indices for the samples.
        """
        assert self._shuffle

        generator = torch.Generator()
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1
