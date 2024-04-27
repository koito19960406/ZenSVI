:py:mod:`dinov2.data.datasets.extended`
=======================================

.. py:module:: dinov2.data.datasets.extended


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.datasets.extended.ExtendedVisionDataset




.. py:class:: ExtendedVisionDataset(*args, **kwargs)


   Bases: :py:obj:`torchvision.datasets.VisionDataset`

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

   .. py:method:: get_image_data(index: int) -> bytes
      :abstractmethod:


   .. py:method:: get_target(index: int) -> Any
      :abstractmethod:


   .. py:method:: __getitem__(index: int) -> Tuple[Any, Any]

      :param index: Index
      :type index: int

      :returns: Sample and meta data, optionally transformed by the respective transforms.
      :rtype: (Any)


   .. py:method:: __len__() -> int
      :abstractmethod:


   .. py:method:: __repr__() -> str

      Return repr(self).


   .. py:method:: extra_repr() -> str


   .. py:method:: __add__(other: Dataset[T_co]) -> ConcatDataset[T_co]


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



