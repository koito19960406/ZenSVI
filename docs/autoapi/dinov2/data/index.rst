:py:mod:`dinov2.data`
=====================

.. py:module:: dinov2.data


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   datasets/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   adapters/index.rst
   augmentations/index.rst
   collate/index.rst
   loaders/index.rst
   masking/index.rst
   samplers/index.rst
   transforms/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.DatasetWithEnumeratedTargets
   dinov2.data.SamplerType
   dinov2.data.MaskingGenerator
   dinov2.data.DataAugmentationDINO



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.data.make_data_loader
   dinov2.data.make_dataset
   dinov2.data.collate_data_and_cast



.. py:class:: DatasetWithEnumeratedTargets(dataset)


   Bases: :py:obj:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:method:: get_image_data(index: int) -> bytes


   .. py:method:: get_target(index: int) -> Tuple[Any, int]


   .. py:method:: __getitem__(index: int) -> Tuple[Any, Tuple[Any, int]]


   .. py:method:: __len__() -> int


   .. py:method:: __add__(other: Dataset[T_co]) -> ConcatDataset[T_co]


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



.. py:function:: make_data_loader(*, dataset, batch_size: int, num_workers: int, shuffle: bool = True, seed: int = 0, sampler_type: Optional[SamplerType] = SamplerType.INFINITE, sampler_size: int = -1, sampler_advance: int = 0, drop_last: bool = True, persistent_workers: bool = False, collate_fn: Optional[Callable[[List[T]], Any]] = None)

   Creates a data loader with the specified parameters.

   :param dataset: A dataset (third party, LaViDa or WebDataset).
   :param batch_size: The size of batches to generate.
   :param num_workers: The number of workers to use.
   :param shuffle: Whether to shuffle samples.
   :param seed: The random seed to use.
   :param sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
   :param sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
   :param sampler_advance: How many samples to skip (when applicable).
   :param drop_last: Whether the last non-full batch of data should be dropped.
   :param persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
   :param collate_fn: Function that performs batch collation


.. py:function:: make_dataset(*, dataset_str: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None)

   Creates a dataset with the specified parameters.

   :param dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
   :param transform: A transform to apply to images.
   :param target_transform: A transform to apply to targets.

   :returns: The created dataset.


.. py:class:: SamplerType


   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.

   .. py:attribute:: DISTRIBUTED
      :value: 0

      

   .. py:attribute:: EPOCH
      :value: 1

      

   .. py:attribute:: INFINITE
      :value: 2

      

   .. py:attribute:: SHARDED_INFINITE
      :value: 3

      

   .. py:attribute:: SHARDED_INFINITE_NEW
      :value: 4

      

   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __dir__()

      Returns all members and all public methods


   .. py:method:: __format__(format_spec)

      Returns format using actual value type unless __str__ has been overridden.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __reduce_ex__(proto)

      Helper for pickle.


   .. py:method:: name()

      The name of the Enum member.


   .. py:method:: value()

      The value of the Enum member.



.. py:function:: collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None)


.. py:class:: MaskingGenerator(input_size, num_masking_patches=None, min_num_patches=4, max_num_patches=None, min_aspect=0.3, max_aspect=None)


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: get_shape()


   .. py:method:: __call__(num_masking_patches=0)



.. py:class:: DataAugmentationDINO(global_crops_scale, local_crops_scale, local_crops_number, global_crops_size=224, local_crops_size=96)


   Bases: :py:obj:`object`

   .. py:method:: __call__(image)



