:py:mod:`zensvi.cv.depth_estimation.zoedepth.data.data_mono`
============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.data.data_mono


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.data_mono.DepthDataLoader
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.RepetitiveRoundRobinDataLoader
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.MixedNYUKITTI
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.CachedReader
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.ImReader
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.DataLoadPreprocess
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.ToTensor



Functions
~~~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.data.data_mono.preprocessing_transforms
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.repetitive_roundrobin
   zensvi.cv.depth_estimation.zoedepth.data.data_mono.remove_leading_slash



.. py:function:: preprocessing_transforms(mode, **kwargs)


.. py:class:: DepthDataLoader(config, mode, device='cpu', transform=None, **kwargs)


   Bases: :py:obj:`object`


.. py:function:: repetitive_roundrobin(*iterables)

   cycles through iterables but sample wise
   first yield first sample from first iterable then first sample from second iterable and so on
   then second sample from first iterable then second sample from second iterable and so on

   If one iterable is shorter than the others, it is repeated until all iterables are exhausted
   repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E


.. py:class:: RepetitiveRoundRobinDataLoader(*dataloaders)


   Bases: :py:obj:`object`

   .. py:method:: __iter__()


   .. py:method:: __len__()



.. py:class:: MixedNYUKITTI(config, mode, device='cpu', **kwargs)


   Bases: :py:obj:`object`


.. py:function:: remove_leading_slash(s)


.. py:class:: CachedReader(shared_dict=None)


   .. py:method:: open(fpath)



.. py:class:: ImReader


   .. py:method:: open(fpath)



.. py:class:: DataLoadPreprocess(config, mode, transform=None, is_for_online_eval=False, **kwargs)


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

   .. py:method:: postprocess(sample)


   .. py:method:: __getitem__(idx)


   .. py:method:: rotate_image(image, angle, flag=Image.BILINEAR)


   .. py:method:: random_crop(img, depth, height, width)


   .. py:method:: random_translate(img, depth, max_t=20)


   .. py:method:: train_preprocess(image, depth_gt)


   .. py:method:: augment_image(image)


   .. py:method:: __len__()


   .. py:method:: __add__(other: Dataset[T_co]) -> ConcatDataset[T_co]


   .. py:method:: __class_getitem__(params)
      :classmethod:


   .. py:method:: __init_subclass__(*args, **kwargs)
      :classmethod:



.. py:class:: ToTensor(mode, do_normalize=False, size=None)


   Bases: :py:obj:`object`

   .. py:method:: __call__(sample)


   .. py:method:: to_tensor(pic)



