:py:mod:`dinov2.data.datasets.decoders`
=======================================

.. py:module:: dinov2.data.datasets.decoders


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.data.datasets.decoders.Decoder
   dinov2.data.datasets.decoders.ImageDataDecoder
   dinov2.data.datasets.decoders.TargetDecoder




.. py:class:: Decoder


   .. py:method:: decode() -> Any
      :abstractmethod:



.. py:class:: ImageDataDecoder(image_data: bytes)


   Bases: :py:obj:`Decoder`

   .. py:method:: decode() -> PIL.Image



.. py:class:: TargetDecoder(target: Any)


   Bases: :py:obj:`Decoder`

   .. py:method:: decode() -> Any



