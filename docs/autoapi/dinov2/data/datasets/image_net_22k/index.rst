:py:mod:`dinov2.data.datasets.image_net_22k`
============================================

.. py:module:: dinov2.data.datasets.image_net_22k


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.datasets.image_net_22k.ImageNet22k




.. py:class:: ImageNet22k(*, root: str, extra: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, mmap_cache_size: int = _DEFAULT_MMAP_CACHE_SIZE)


   Bases: :py:obj:`dinov2.data.datasets.extended.ExtendedVisionDataset`

   Base Class For making datasets which are compatible with torchvision.
   It is necessary to override the ``__getitem__`` and ``__len__`` method.

   :param root: Root directory of dataset. Only used for `__repr__`.
   :type root: string, optional
   :param transforms: A function/transforms that takes in
                      an image and a label and returns the transformed versions of both.
   :type transforms: callable, optional
   :param transform: A function/transform that  takes in an PIL image
                     and returns a transformed version. E.g, ``transforms.RandomCrop``
   :type transform: callable, optional
   :param target_transform: A function/transform that takes in the
                            target and transforms it.
   :type target_transform: callable, optional

   .. note::

       :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.

   .. py:attribute:: Labels

      

   .. py:method:: find_class_id(class_index: int) -> str


   .. py:method:: get_image_data(index: int) -> bytes


   .. py:method:: get_target(index: int) -> Any


   .. py:method:: get_targets() -> numpy.ndarray


   .. py:method:: get_class_id(index: int) -> str


   .. py:method:: get_class_ids() -> numpy.ndarray


   .. py:method:: __getitem__(index: int) -> Tuple[Any, Any]

      :param index: Index
      :type index: int

      :returns: Sample and meta data, optionally transformed by the respective transforms.
      :rtype: (Any)


   .. py:method:: __len__() -> int


   .. py:method:: dump_extra(root: Optional[str] = None) -> None


   .. py:method:: __repr__() -> str

      Return repr(self).


   .. py:method:: extra_repr() -> str


   .. py:method:: __add__(other: Dataset[T_co]) -> ConcatDataset[T_co]


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



