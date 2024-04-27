:py:mod:`dinov2.data.loaders`
=============================

.. py:module:: dinov2.data.loaders


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.loaders.SamplerType



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.data.loaders.make_dataset
   dinov2.data.loaders.make_data_loader



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.data.loaders.logger
   dinov2.data.loaders.T


.. py:data:: logger

   

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



.. py:function:: make_dataset(*, dataset_str: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None)

   Creates a dataset with the specified parameters.

   :param dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
   :param transform: A transform to apply to images.
   :param target_transform: A transform to apply to targets.

   :returns: The created dataset.


.. py:data:: T

   

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


