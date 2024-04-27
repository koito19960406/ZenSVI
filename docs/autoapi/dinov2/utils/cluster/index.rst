:py:mod:`dinov2.utils.cluster`
==============================

.. py:module:: dinov2.utils.cluster


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.utils.cluster.ClusterType



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.utils.cluster.get_cluster_type
   dinov2.utils.cluster.get_checkpoint_path
   dinov2.utils.cluster.get_user_checkpoint_path
   dinov2.utils.cluster.get_slurm_partition
   dinov2.utils.cluster.get_slurm_executor_parameters



.. py:class:: ClusterType


   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.

   .. py:attribute:: AWS
      :value: 'aws'

      

   .. py:attribute:: FAIR
      :value: 'fair'

      

   .. py:attribute:: RSC
      :value: 'rsc'

      

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



.. py:function:: get_cluster_type(cluster_type: Optional[ClusterType] = None) -> Optional[ClusterType]


.. py:function:: get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[pathlib.Path]


.. py:function:: get_user_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[pathlib.Path]


.. py:function:: get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]


.. py:function:: get_slurm_executor_parameters(nodes: int, num_gpus_per_node: int, cluster_type: Optional[ClusterType] = None, **kwargs) -> Dict[str, Any]


