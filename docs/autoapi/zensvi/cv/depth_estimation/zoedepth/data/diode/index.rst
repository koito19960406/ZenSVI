:py:mod:`zensvi.cv.depth_estimation.zoedepth.data.diode`
========================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.data.diode


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.diode.ToTensor
   zensvi.cv.depth_estimation.zoedepth.data.diode.DIODE



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.diode.get_diode_loader



.. py:class:: ToTensor


   Bases: :py:obj:`object`

   .. py:method:: __call__(sample)


   .. py:method:: to_tensor(pic)



.. py:class:: DIODE(data_dir_root)


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

   .. py:method:: __getitem__(idx)


   .. py:method:: __len__()


   .. py:method:: __add__(other: Dataset[T_co]) -> ConcatDataset[T_co]


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



.. py:function:: get_diode_loader(data_dir_root, batch_size=1, **kwargs)


