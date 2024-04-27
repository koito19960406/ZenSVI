:py:mod:`dinov2.data.samplers`
==============================

.. py:module:: dinov2.data.samplers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.samplers.EpochSampler
   dinov2.data.samplers.InfiniteSampler
   dinov2.data.samplers.ShardedInfiniteSampler




.. py:class:: EpochSampler(*, size: int, sample_count: int, shuffle: bool = False, seed: int = 0, start: Optional[int] = None, step: Optional[int] = None)


   Bases: :py:obj:`torch.utils.data.sampler.Sampler`

   Base class for all Samplers.

   Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
   way to iterate over indices or lists of indices (batches) of dataset elements, and a :meth:`__len__` method
   that returns the length of the returned iterators.

   :param data_source: This argument is not used and will be removed in 2.2.0.
                       You may still have custom implementation that utilizes it.
   :type data_source: Dataset

   .. rubric:: Example

   >>> # xdoctest: +SKIP
   >>> class AccedingSequenceLengthSampler(Sampler[int]):
   >>>     def __init__(self, data: List[str]) -> None:
   >>>         self.data = data
   >>>
   >>>     def __len__(self) -> int:
   >>>         return len(self.data)
   >>>
   >>>     def __iter__(self) -> Iterator[int]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         yield from torch.argsort(sizes).tolist()
   >>>
   >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
   >>>     def __init__(self, data: List[str], batch_size: int) -> None:
   >>>         self.data = data
   >>>         self.batch_size = batch_size
   >>>
   >>>     def __len__(self) -> int:
   >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
   >>>
   >>>     def __iter__(self) -> Iterator[List[int]]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
   >>>             yield batch.tolist()

   .. note:: The :meth:`__len__` method isn't strictly required by
             :class:`~torch.utils.data.DataLoader`, but is expected in any
             calculation involving the length of a :class:`~torch.utils.data.DataLoader`.

   .. py:method:: __iter__()


   .. py:method:: __len__()


   .. py:method:: set_epoch(epoch)


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



.. py:class:: InfiniteSampler(*, sample_count: int, shuffle: bool = False, seed: int = 0, start: Optional[int] = None, step: Optional[int] = None, advance: int = 0)


   Bases: :py:obj:`torch.utils.data.sampler.Sampler`

   Base class for all Samplers.

   Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
   way to iterate over indices or lists of indices (batches) of dataset elements, and a :meth:`__len__` method
   that returns the length of the returned iterators.

   :param data_source: This argument is not used and will be removed in 2.2.0.
                       You may still have custom implementation that utilizes it.
   :type data_source: Dataset

   .. rubric:: Example

   >>> # xdoctest: +SKIP
   >>> class AccedingSequenceLengthSampler(Sampler[int]):
   >>>     def __init__(self, data: List[str]) -> None:
   >>>         self.data = data
   >>>
   >>>     def __len__(self) -> int:
   >>>         return len(self.data)
   >>>
   >>>     def __iter__(self) -> Iterator[int]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         yield from torch.argsort(sizes).tolist()
   >>>
   >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
   >>>     def __init__(self, data: List[str], batch_size: int) -> None:
   >>>         self.data = data
   >>>         self.batch_size = batch_size
   >>>
   >>>     def __len__(self) -> int:
   >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
   >>>
   >>>     def __iter__(self) -> Iterator[List[int]]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
   >>>             yield batch.tolist()

   .. note:: The :meth:`__len__` method isn't strictly required by
             :class:`~torch.utils.data.DataLoader`, but is expected in any
             calculation involving the length of a :class:`~torch.utils.data.DataLoader`.

   .. py:method:: __iter__()


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



.. py:class:: ShardedInfiniteSampler(*, sample_count: int, shuffle: bool = False, seed: int = 0, start: Optional[int] = None, step: Optional[int] = None, advance: int = 0, use_new_shuffle_tensor_slice: bool = False)


   Bases: :py:obj:`torch.utils.data.sampler.Sampler`

   Base class for all Samplers.

   Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
   way to iterate over indices or lists of indices (batches) of dataset elements, and a :meth:`__len__` method
   that returns the length of the returned iterators.

   :param data_source: This argument is not used and will be removed in 2.2.0.
                       You may still have custom implementation that utilizes it.
   :type data_source: Dataset

   .. rubric:: Example

   >>> # xdoctest: +SKIP
   >>> class AccedingSequenceLengthSampler(Sampler[int]):
   >>>     def __init__(self, data: List[str]) -> None:
   >>>         self.data = data
   >>>
   >>>     def __len__(self) -> int:
   >>>         return len(self.data)
   >>>
   >>>     def __iter__(self) -> Iterator[int]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         yield from torch.argsort(sizes).tolist()
   >>>
   >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
   >>>     def __init__(self, data: List[str], batch_size: int) -> None:
   >>>         self.data = data
   >>>         self.batch_size = batch_size
   >>>
   >>>     def __len__(self) -> int:
   >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
   >>>
   >>>     def __iter__(self) -> Iterator[List[int]]:
   >>>         sizes = torch.tensor([len(x) for x in self.data])
   >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
   >>>             yield batch.tolist()

   .. note:: The :meth:`__len__` method isn't strictly required by
             :class:`~torch.utils.data.DataLoader`, but is expected in any
             calculation involving the length of a :class:`~torch.utils.data.DataLoader`.

   .. py:method:: __iter__()


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



