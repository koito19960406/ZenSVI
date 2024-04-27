:py:mod:`dinov2.data.datasets.image_net`
========================================

.. py:module:: dinov2.data.datasets.image_net


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.datasets.image_net.ImageNet




Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.data.datasets.image_net.logger


.. py:data:: logger

   

.. py:class:: ImageNet(*, split: ImageNet, root: str, extra: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None)


   Bases: :py:obj:`dinov2.data.datasets.extended.ExtendedVisionDataset`

   .. py:property:: split
      :type: ImageNet


   .. py:attribute:: Target

      

   .. py:attribute:: Split

      

   .. py:method:: find_class_id(class_index: int) -> str


   .. py:method:: find_class_name(class_index: int) -> str


   .. py:method:: get_image_data(index: int) -> bytes


   .. py:method:: get_target(index: int) -> Optional[Target]


   .. py:method:: get_targets() -> Optional[numpy.ndarray]


   .. py:method:: get_class_id(index: int) -> Optional[str]


   .. py:method:: get_class_name(index: int) -> Optional[str]


   .. py:method:: __len__() -> int


   .. py:method:: dump_extra() -> None



