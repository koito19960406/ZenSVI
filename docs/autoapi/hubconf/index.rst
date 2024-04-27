:py:mod:`hubconf`
=================

.. py:module:: hubconf


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   hubconf.Weights



Functions
~~~~~~~~~

.. autoapisummary::

   hubconf.dinov2_vits14
   hubconf.dinov2_vitb14
   hubconf.dinov2_vitl14
   hubconf.dinov2_vitg14
   hubconf.dinov2_vits14_reg
   hubconf.dinov2_vitb14_reg
   hubconf.dinov2_vitl14_reg
   hubconf.dinov2_vitg14_reg



.. py:class:: Weights


   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.

   .. py:attribute:: LVD142M
      :value: 'LVD142M'

      

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



.. py:function:: dinov2_vits14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vits14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vitb14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vitl14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.


.. py:function:: dinov2_vitg14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs)

   DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.


