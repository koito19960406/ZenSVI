:py:mod:`dinov2.eval.metrics`
=============================

.. py:module:: dinov2.eval.metrics


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   dinov2.eval.metrics.MetricType
   dinov2.eval.metrics.AccuracyAveraging
   dinov2.eval.metrics.ImageNetReaLAccuracy



Functions
~~~~~~~~~

.. autoapisummary::

   dinov2.eval.metrics.build_metric
   dinov2.eval.metrics.build_topk_accuracy_metric
   dinov2.eval.metrics.build_topk_imagenet_real_accuracy_metric



Attributes
~~~~~~~~~~

.. autoapisummary::

   dinov2.eval.metrics.logger


.. py:data:: logger

   

.. py:class:: MetricType


   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.

   .. py:property:: accuracy_averaging


   .. py:attribute:: MEAN_ACCURACY
      :value: 'mean_accuracy'

      

   .. py:attribute:: MEAN_PER_CLASS_ACCURACY
      :value: 'mean_per_class_accuracy'

      

   .. py:attribute:: PER_CLASS_ACCURACY
      :value: 'per_class_accuracy'

      

   .. py:attribute:: IMAGENET_REAL_ACCURACY
      :value: 'imagenet_real_accuracy'

      

   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


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



.. py:class:: AccuracyAveraging


   Bases: :py:obj:`enum.Enum`

   Generic enumeration.

   Derive from this class to define new enumerations.

   .. py:attribute:: MEAN_ACCURACY
      :value: 'micro'

      

   .. py:attribute:: MEAN_PER_CLASS_ACCURACY
      :value: 'macro'

      

   .. py:attribute:: PER_CLASS_ACCURACY
      :value: 'none'

      

   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


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



.. py:function:: build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None)


.. py:function:: build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5))


.. py:function:: build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5))


.. py:class:: ImageNetReaLAccuracy(num_classes: int, top_k: int = 1, **kwargs: Any)


   Bases: :py:obj:`torchmetrics.Metric`

   Base class for all metrics present in the Metrics API.

   This class is inherited by all metrics and implements the following functionality:
   1. Handles the transfer of metric states to correct device
   2. Handles the synchronization of metric states across processes

   The three core methods of the base class are
   * ``add_state()``
   * ``forward()``
   * ``reset()``

   which should almost never be overwritten by child classes. Instead, the following methods should be overwritten
   * ``update()``
   * ``compute()``


   :param kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

                  - compute_on_cpu: If metric state should be stored on CPU during computations. Only works for list states.
                  - dist_sync_on_step: If metric state should synchronize on ``forward()``. Default is ``False``
                  - process_group: The process group on which the synchronization is called. Default is the world.
                  - dist_sync_fn: Function that performs the allgather option on the metric state. Default is an custom
                    implementation that calls ``torch.distributed.all_gather`` internally.
                  - distributed_available_fn: Function that checks if the distributed backend is available. Defaults to a
                    check of ``torch.distributed.is_available()`` and ``torch.distributed.is_initialized()``.
                  - sync_on_compute: If metric state should synchronize when ``compute`` is called. Default is ``True``
                  - compute_with_cache: If results from ``compute`` should be cached. Default is ``False``

   .. py:property:: update_called
      :type: bool

      Returns `True` if `update` or `forward` has been called initialization or last `reset`.

   .. py:property:: update_count
      :type: int

      Get the number of times `update` and/or `forward` has been called since initialization or last `reset`.

   .. py:property:: metric_state
      :type: Dict[str, Union[List[torch.Tensor], torch.Tensor]]

      Get the current state of the metric.

   .. py:property:: device
      :type: torch.device

      Return the device of the metric.

   .. py:property:: dtype
      :type: torch.dtype

      Return the default dtype of the metric.

   .. py:attribute:: is_differentiable
      :type: bool
      :value: False

      

   .. py:attribute:: higher_is_better
      :type: Optional[bool]

      

   .. py:attribute:: full_state_update
      :type: bool
      :value: False

      

   .. py:attribute:: __jit_ignored_attributes__
      :type: ClassVar[List[str]]
      :value: ['device']

      

   .. py:attribute:: __jit_unused_properties__
      :type: ClassVar[List[str]]
      :value: ['is_differentiable', 'higher_is_better', 'plot_lower_bound', 'plot_upper_bound',...

      

   .. py:attribute:: plot_lower_bound
      :type: Optional[float]

      

   .. py:attribute:: plot_upper_bound
      :type: Optional[float]

      

   .. py:attribute:: plot_legend_name
      :type: Optional[str]

      

   .. py:attribute:: __iter__

      

   .. py:attribute:: dump_patches
      :type: bool
      :value: False

      

   .. py:attribute:: training
      :type: bool

      

   .. py:attribute:: call_super_init
      :type: bool
      :value: False

      

   .. py:attribute:: __call__
      :type: Callable[Ellipsis, Any]

      

   .. py:attribute:: T_destination

      

   .. py:method:: update(preds: torch.Tensor, target: torch.Tensor) -> None

      Override this method to update the state variables of your metric class.


   .. py:method:: compute() -> torch.Tensor

      Override this method to compute the final metric value.

      This method will automatically synchronize state variables when running in distributed backend.



   .. py:method:: add_state(name: str, default: Union[list, torch.Tensor], dist_reduce_fx: Optional[Union[str, Callable]] = None, persistent: bool = False) -> None

      Add metric state variable. Only used by subclasses.

      Metric state variables are either `:class:`~torch.Tensor` or an empty list, which can be appended to by the
      metric. Each state variable must have a unique name associated with it. State variables are accessible as
      attributes of the metric i.e, if ``name`` is ``"my_state"`` then its value can be accessed from an instance
      ``metric`` as ``metric.my_state``. Metric states behave like buffers and parameters of :class:`~torch.nn.Module`
      as they are also updated when ``.to()`` is called. Unlike parameters and buffers, metric states are not by
      default saved in the modules :attr:`~torch.nn.Module.state_dict`.

      :param name: The name of the state variable. The variable will then be accessible at ``self.name``.
      :param default: Default value of the state; can either be a :class:`~torch.Tensor` or an empty list.
                      The state will be reset to this value when ``self.reset()`` is called.
      :param dist_reduce_fx: Function to reduce state across multiple processes in distributed mode.
                             If value is ``"sum"``, ``"mean"``, ``"cat"``, ``"min"`` or ``"max"`` we will use ``torch.sum``,
                             ``torch.mean``, ``torch.cat``, ``torch.min`` and ``torch.max``` respectively, each with argument
                             ``dim=0``. Note that the ``"cat"`` reduction only makes sense if the state is a list, and not
                             a tensor. The user can also pass a custom function in this parameter.
      :type dist_reduce_fx: Optional
      :param persistent: whether the state will be saved as part of the modules ``state_dict``.
                         Default is ``False``.
      :type persistent: Optional

      .. note::

         Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
         However, there won't be any reduction function applied to the synchronized metric state.
         
         The metric states would be synced as follows
         
         - If the metric state is :class:`~torch.Tensor`, the synced value will be a stacked :class:`~torch.Tensor`
           across the process dimension if the metric state was a :class:`~torch.Tensor`. The original
           :class:`~torch.Tensor` metric state retains dimension and hence the synchronized output will be of shape
           ``(num_process, ...)``.
         
         - If the metric state is a ``list``, the synced value will be a ``list`` containing the
           combined elements from all processes.

      .. note::

         When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
         the format discussed in the above note.

      :raises ValueError: If ``default`` is not a ``tensor`` or an ``empty list``.
      :raises ValueError: If ``dist_reduce_fx`` is not callable or one of ``"mean"``, ``"sum"``, ``"cat"``, ``"min"``,
          ``"max"`` or ``None``.


   .. py:method:: forward(*args: Any, **kwargs: Any) -> Any

      Aggregate and evaluate batch input directly.

      Serves the dual purpose of both computing the metric on the current batch of inputs but also add the batch
      statistics to the overall accumululating metric state. Input arguments are the exact same as corresponding
      ``update`` method. The returned output is the exact same as the output of ``compute``.

      :param args: Any arguments as required by the metric ``update`` method.
      :param kwargs: Any keyword arguments as required by the metric ``update`` method.

      :returns: The output of the ``compute`` method evaluated on the current batch.

      :raises TorchMetricsUserError: If the metric is already synced and ``forward`` is called again.


   .. py:method:: sync(dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None, should_sync: bool = True, distributed_available: Optional[Callable] = None) -> None

      Sync function for manually controlling when metrics states should be synced across processes.

      :param dist_sync_fn: Function to be used to perform states synchronization
      :param process_group: Specify the process group on which synchronization is called.
                            default: `None` (which selects the entire world)
      :param should_sync: Whether to apply to state synchronization. This will have an impact
                          only when running in a distributed setting.
      :param distributed_available: Function to determine if we are running inside a distributed setting

      :raises TorchMetricsUserError: If the metric is already synced and ``sync`` is called again.


   .. py:method:: unsync(should_unsync: bool = True) -> None

      Unsync function for manually controlling when metrics states should be reverted back to their local states.

      :param should_unsync: Whether to perform unsync


   .. py:method:: sync_context(dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None, should_sync: bool = True, should_unsync: bool = True, distributed_available: Optional[Callable] = None) -> Generator

      Context manager to synchronize states.

      This context manager is used in distributed setting and makes sure that the local cache states are restored
      after yielding the synchronized state.

      :param dist_sync_fn: Function to be used to perform states synchronization
      :param process_group: Specify the process group on which synchronization is called.
                            default: `None` (which selects the entire world)
      :param should_sync: Whether to apply to state synchronization. This will have an impact
                          only when running in a distributed setting.
      :param should_unsync: Whether to restore the cache state so that the metrics can
                            continue to be accumulated.
      :param distributed_available: Function to determine if we are running inside a distributed setting


   .. py:method:: plot(*_: Any, **__: Any) -> Any
      :abstractmethod:

      Override this method plot the metric value.


   .. py:method:: reset() -> None

      Reset metric state variables to their default value.


   .. py:method:: clone() -> Metric

      Make a copy of the metric.


   .. py:method:: __getstate__() -> Dict[str, Any]

      Get the current state, including all metric states, for the metric.

      Used for loading and saving a metric.



   .. py:method:: __setstate__(state: Dict[str, Any]) -> None

      Set the state of the metric, based on a input state.

      Used for loading and saving a metric.



   .. py:method:: __setattr__(name: str, value: Any) -> None

      Overwrite default method to prevent specific attributes from being set by user.


   .. py:method:: type(dst_type: Union[str, torch.dtype]) -> Metric

      Override default and prevent dtype casting.

      Please use :meth:`Metric.set_dtype` instead.



   .. py:method:: float() -> Metric

      Override default and prevent dtype casting.

      Please use :meth:`Metric.set_dtype` instead.



   .. py:method:: double() -> Metric

      Override default and prevent dtype casting.

      Please use :meth:`Metric.set_dtype` instead.



   .. py:method:: half() -> Metric

      Override default and prevent dtype casting.

      Please use :meth:`Metric.set_dtype` instead.



   .. py:method:: set_dtype(dst_type: Union[str, torch.dtype]) -> Metric

      Transfer all metric state to specific dtype. Special version of standard `type` method.

      :param dst_type: the desired type as string or dtype object


   .. py:method:: persistent(mode: bool = False) -> None

      Change post-init if metric states should be saved to its state_dict.


   .. py:method:: state_dict(destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) -> Dict[str, Any]

      Get the current state of metric as an dictionary.

      :param destination: Optional dictionary, that if provided, the state of module will be updated into the dict and
                          the same object is returned. Otherwise, an ``OrderedDict`` will be created and returned.
      :param prefix: optional string, a prefix added to parameter and buffer names to compose the keys in state_dict.
      :param keep_vars: by default the :class:`~torch.Tensor` returned in the state dict are detached from autograd.
                        If set to ``True``, detaching will not be performed.


   .. py:method:: __hash__() -> int

      Return an unique hash of the metric.

      The hash depends on both the class itself but also the current metric state, which therefore enforces that two
      instances of the same metrics never have the same hash even if they have been updated on the same data.



   .. py:method:: __add__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the addition operator.


   .. py:method:: __and__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the logical and operator.


   .. py:method:: __eq__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the equal operator.


   .. py:method:: __floordiv__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the floor division operator.


   .. py:method:: __ge__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the greater than or equal operator.


   .. py:method:: __gt__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the greater than operator.


   .. py:method:: __le__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the less than or equal operator.


   .. py:method:: __lt__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the less than operator.


   .. py:method:: __matmul__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the matrix multiplication operator.


   .. py:method:: __mod__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the remainder operator.


   .. py:method:: __mul__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the multiplication operator.


   .. py:method:: __ne__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the not equal operator.


   .. py:method:: __or__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the logical or operator.


   .. py:method:: __pow__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the exponential/power operator.


   .. py:method:: __radd__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the addition operator.


   .. py:method:: __rand__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the logical and operator.


   .. py:method:: __rfloordiv__(other: CompositionalMetric) -> Metric

      Construct compositional metric using the floor division operator.


   .. py:method:: __rmatmul__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the matrix multiplication operator.


   .. py:method:: __rmod__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the remainder operator.


   .. py:method:: __rmul__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the multiplication operator.


   .. py:method:: __ror__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the logical or operator.


   .. py:method:: __rpow__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the exponential/power operator.


   .. py:method:: __rsub__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the subtraction operator.


   .. py:method:: __rtruediv__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the true divide operator.


   .. py:method:: __rxor__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the logical xor operator.


   .. py:method:: __sub__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the subtraction operator.


   .. py:method:: __truediv__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the true divide operator.


   .. py:method:: __xor__(other: Union[Metric, float, torch.Tensor]) -> CompositionalMetric

      Construct compositional metric using the logical xor operator.


   .. py:method:: __abs__() -> CompositionalMetric

      Construct compositional metric using the absolute operator.


   .. py:method:: __inv__() -> CompositionalMetric

      Construct compositional metric using the not operator.


   .. py:method:: __invert__() -> CompositionalMetric

      Construct compositional metric using the not operator.


   .. py:method:: __neg__() -> CompositionalMetric

      Construct compositional metric using absolute negative operator.


   .. py:method:: __pos__() -> CompositionalMetric

      Construct compositional metric using absolute operator.


   .. py:method:: __getitem__(idx: int) -> CompositionalMetric

      Construct compositional metric using the get item operator.


   .. py:method:: __getnewargs__() -> Tuple

      Needed method for construction of new metrics __new__ method.


   .. py:method:: register_buffer(name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None

      Add a buffer to the module.

      This is typically used to register a buffer that should not to be
      considered a model parameter. For example, BatchNorm's ``running_mean``
      is not a parameter, but is part of the module's state. Buffers, by
      default, are persistent and will be saved alongside parameters. This
      behavior can be changed by setting :attr:`persistent` to ``False``. The
      only difference between a persistent buffer and a non-persistent buffer
      is that the latter will not be a part of this module's
      :attr:`state_dict`.

      Buffers can be accessed as attributes using given names.

      :param name: name of the buffer. The buffer can be accessed
                   from this module using the given name
      :type name: str
      :param tensor: buffer to be registered. If ``None``, then operations
                     that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                     the buffer is **not** included in the module's :attr:`state_dict`.
      :type tensor: Tensor or None
      :param persistent: whether the buffer is part of this module's
                         :attr:`state_dict`.
      :type persistent: bool

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> self.register_buffer('running_mean', torch.zeros(num_features))



   .. py:method:: register_parameter(name: str, param: Optional[torch.nn.parameter.Parameter]) -> None

      Add a parameter to the module.

      The parameter can be accessed as an attribute using given name.

      :param name: name of the parameter. The parameter can be accessed
                   from this module using the given name
      :type name: str
      :param param: parameter to be added to the module. If
                    ``None``, then operations that run on parameters, such as :attr:`cuda`,
                    are ignored. If ``None``, the parameter is **not** included in the
                    module's :attr:`state_dict`.
      :type param: Parameter or None


   .. py:method:: add_module(name: str, module: Optional[Module]) -> None

      Add a child module to the current module.

      The module can be accessed as an attribute using the given name.

      :param name: name of the child module. The child module can be
                   accessed from this module using the given name
      :type name: str
      :param module: child module to be added to the module.
      :type module: Module


   .. py:method:: register_module(name: str, module: Optional[Module]) -> None

      Alias for :func:`add_module`.


   .. py:method:: get_submodule(target: str) -> Module

      Return the submodule given by ``target`` if it exists, otherwise throw an error.

      For example, let's say you have an ``nn.Module`` ``A`` that
      looks like this:

      .. code-block:: text

          A(
              (net_b): Module(
                  (net_c): Module(
                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                  )
                  (linear): Linear(in_features=100, out_features=200, bias=True)
              )
          )

      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
      submodule ``net_b``, which itself has two submodules ``net_c``
      and ``linear``. ``net_c`` then has a submodule ``conv``.)

      To check whether or not we have the ``linear`` submodule, we
      would call ``get_submodule("net_b.linear")``. To check whether
      we have the ``conv`` submodule, we would call
      ``get_submodule("net_b.net_c.conv")``.

      The runtime of ``get_submodule`` is bounded by the degree
      of module nesting in ``target``. A query against
      ``named_modules`` achieves the same result, but it is O(N) in
      the number of transitive modules. So, for a simple check to see
      if some submodule exists, ``get_submodule`` should always be
      used.

      :param target: The fully-qualified string name of the submodule
                     to look for. (See above example for how to specify a
                     fully-qualified string.)

      :returns: The submodule referenced by ``target``
      :rtype: torch.nn.Module

      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not an
          ``nn.Module``


   .. py:method:: get_parameter(target: str) -> torch.nn.parameter.Parameter

      Return the parameter given by ``target`` if it exists, otherwise throw an error.

      See the docstring for ``get_submodule`` for a more detailed
      explanation of this method's functionality as well as how to
      correctly specify ``target``.

      :param target: The fully-qualified string name of the Parameter
                     to look for. (See ``get_submodule`` for how to specify a
                     fully-qualified string.)

      :returns: The Parameter referenced by ``target``
      :rtype: torch.nn.Parameter

      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not an
          ``nn.Parameter``


   .. py:method:: get_buffer(target: str) -> torch.Tensor

      Return the buffer given by ``target`` if it exists, otherwise throw an error.

      See the docstring for ``get_submodule`` for a more detailed
      explanation of this method's functionality as well as how to
      correctly specify ``target``.

      :param target: The fully-qualified string name of the buffer
                     to look for. (See ``get_submodule`` for how to specify a
                     fully-qualified string.)

      :returns: The buffer referenced by ``target``
      :rtype: torch.Tensor

      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not a
          buffer


   .. py:method:: get_extra_state() -> Any

      Return any extra state to include in the module's state_dict.

      Implement this and a corresponding :func:`set_extra_state` for your module
      if you need to store extra state. This function is called when building the
      module's `state_dict()`.

      Note that extra state should be picklable to ensure working serialization
      of the state_dict. We only provide provide backwards compatibility guarantees
      for serializing Tensors; other objects may break backwards compatibility if
      their serialized pickled form changes.

      :returns: Any extra state to store in the module's state_dict
      :rtype: object


   .. py:method:: set_extra_state(state: Any)

      Set extra state contained in the loaded `state_dict`.

      This function is called from :func:`load_state_dict` to handle any extra state
      found within the `state_dict`. Implement this function and a corresponding
      :func:`get_extra_state` for your module if you need to store extra state within its
      `state_dict`.

      :param state: Extra state from the `state_dict`
      :type state: dict


   .. py:method:: apply(fn: Callable[[Module], None]) -> T

      Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

      Typical use includes initializing the parameters of a model
      (see also :ref:`nn-init-doc`).

      :param fn: function to be applied to each submodule
      :type fn: :class:`Module` -> None

      :returns: self
      :rtype: Module

      Example::

          >>> @torch.no_grad()
          >>> def init_weights(m):
          >>>     print(m)
          >>>     if type(m) == nn.Linear:
          >>>         m.weight.fill_(1.0)
          >>>         print(m.weight)
          >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
          >>> net.apply(init_weights)
          Linear(in_features=2, out_features=2, bias=True)
          Parameter containing:
          tensor([[1., 1.],
                  [1., 1.]], requires_grad=True)
          Linear(in_features=2, out_features=2, bias=True)
          Parameter containing:
          tensor([[1., 1.],
                  [1., 1.]], requires_grad=True)
          Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          )



   .. py:method:: cuda(device: Optional[Union[int, Module.cuda.device]] = None) -> T

      Move all model parameters and buffers to the GPU.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on GPU while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: self
      :rtype: Module


   .. py:method:: ipu(device: Optional[Union[int, Module.ipu.device]] = None) -> T

      Move all model parameters and buffers to the IPU.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on IPU while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: self
      :rtype: Module


   .. py:method:: xpu(device: Optional[Union[int, Module.xpu.device]] = None) -> T

      Move all model parameters and buffers to the XPU.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on XPU while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: self
      :rtype: Module


   .. py:method:: cpu() -> T

      Move all model parameters and buffers to the CPU.

      .. note::
          This method modifies the module in-place.

      :returns: self
      :rtype: Module


   .. py:method:: bfloat16() -> T

      Casts all floating point parameters and buffers to ``bfloat16`` datatype.

      .. note::
          This method modifies the module in-place.

      :returns: self
      :rtype: Module


   .. py:method:: to_empty(*, device: Optional[torch._prims_common.DeviceLikeType], recurse: bool = True) -> T

      Move the parameters and buffers to the specified device without copying storage.

      :param device: The desired device of the parameters
                     and buffers in this module.
      :type device: :class:`torch.device`
      :param recurse: Whether parameters and buffers of submodules should
                      be recursively moved to the specified device.
      :type recurse: bool

      :returns: self
      :rtype: Module


   .. py:method:: to(device: Optional[torch._prims_common.DeviceLikeType] = ..., dtype: Optional[Union[Module.to.dtype, str]] = ..., non_blocking: bool = ...) -> typing_extensions.Self
                  to(dtype: Union[Module.to.dtype, str], non_blocking: bool = ...) -> typing_extensions.Self
                  to(tensor: torch.Tensor, non_blocking: bool = ...) -> typing_extensions.Self

      Move and/or cast the parameters and buffers.

      This can be called as

      .. function:: to(device=None, dtype=None, non_blocking=False)
         :noindex:

      .. function:: to(dtype, non_blocking=False)
         :noindex:

      .. function:: to(tensor, non_blocking=False)
         :noindex:

      .. function:: to(memory_format=torch.channels_last)
         :noindex:

      Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
      floating point or complex :attr:`dtype`\ s. In addition, this method will
      only cast the floating point or complex parameters and buffers to :attr:`dtype`
      (if given). The integral parameters and buffers will be moved
      :attr:`device`, if that is given, but with dtypes unchanged. When
      :attr:`non_blocking` is set, it tries to convert/move asynchronously
      with respect to the host if possible, e.g., moving CPU Tensors with
      pinned memory to CUDA devices.

      See below for examples.

      .. note::
          This method modifies the module in-place.

      :param device: the desired device of the parameters
                     and buffers in this module
      :type device: :class:`torch.device`
      :param dtype: the desired floating point or complex dtype of
                    the parameters and buffers in this module
      :type dtype: :class:`torch.dtype`
      :param tensor: Tensor whose dtype and device are the desired
                     dtype and device for all parameters and buffers in this module
      :type tensor: torch.Tensor
      :param memory_format: the desired memory
                            format for 4D parameters and buffers in this module (keyword
                            only argument)
      :type memory_format: :class:`torch.memory_format`

      :returns: self
      :rtype: Module

      Examples::

          >>> # xdoctest: +IGNORE_WANT("non-deterministic")
          >>> linear = nn.Linear(2, 2)
          >>> linear.weight
          Parameter containing:
          tensor([[ 0.1913, -0.3420],
                  [-0.5113, -0.2325]])
          >>> linear.to(torch.double)
          Linear(in_features=2, out_features=2, bias=True)
          >>> linear.weight
          Parameter containing:
          tensor([[ 0.1913, -0.3420],
                  [-0.5113, -0.2325]], dtype=torch.float64)
          >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
          >>> gpu1 = torch.device("cuda:1")
          >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
          Linear(in_features=2, out_features=2, bias=True)
          >>> linear.weight
          Parameter containing:
          tensor([[ 0.1914, -0.3420],
                  [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
          >>> cpu = torch.device("cpu")
          >>> linear.to(cpu)
          Linear(in_features=2, out_features=2, bias=True)
          >>> linear.weight
          Parameter containing:
          tensor([[ 0.1914, -0.3420],
                  [-0.5112, -0.2324]], dtype=torch.float16)

          >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
          >>> linear.weight
          Parameter containing:
          tensor([[ 0.3741+0.j,  0.2382+0.j],
                  [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
          >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
          tensor([[0.6122+0.j, 0.1150+0.j],
                  [0.6122+0.j, 0.1150+0.j],
                  [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)



   .. py:method:: register_full_backward_pre_hook(hook: Callable[[Module, _grad_t], Union[None, _grad_t]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle

      Register a backward pre-hook on the module.

      The hook will be called every time the gradients for the module are computed.
      The hook should have the following signature::

          hook(module, grad_output) -> tuple[Tensor] or None

      The :attr:`grad_output` is a tuple. The hook should
      not modify its arguments, but it can optionally return a new gradient with
      respect to the output that will be used in place of :attr:`grad_output` in
      subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
      all non-Tensor arguments.

      For technical reasons, when this hook is applied to a Module, its forward function will
      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
      of each Tensor returned by the Module's forward function.

      .. warning ::
          Modifying inputs inplace is not allowed when using backward hooks and
          will raise an error.

      :param hook: The user-defined hook to be registered.
      :type hook: Callable
      :param prepend: If true, the provided ``hook`` will be fired before
                      all existing ``backward_pre`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``backward_pre`` hooks
                      on this :class:`torch.nn.modules.Module`. Note that global
                      ``backward_pre`` hooks registered with
                      :func:`register_module_full_backward_pre_hook` will fire before
                      all hooks registered by this method.
      :type prepend: bool

      :returns:     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``
      :rtype: :class:`torch.utils.hooks.RemovableHandle`


   .. py:method:: register_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]]) -> torch.utils.hooks.RemovableHandle

      Register a backward hook on the module.

      This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
      the behavior of this function will change in future versions.

      :returns:     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``
      :rtype: :class:`torch.utils.hooks.RemovableHandle`


   .. py:method:: register_full_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]], prepend: bool = False) -> torch.utils.hooks.RemovableHandle

      Register a backward hook on the module.

      The hook will be called every time the gradients with respect to a module
      are computed, i.e. the hook will execute if and only if the gradients with
      respect to module outputs are computed. The hook should have the following
      signature::

          hook(module, grad_input, grad_output) -> tuple(Tensor) or None

      The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
      with respect to the inputs and outputs respectively. The hook should
      not modify its arguments, but it can optionally return a new gradient with
      respect to the input that will be used in place of :attr:`grad_input` in
      subsequent computations. :attr:`grad_input` will only correspond to the inputs given
      as positional arguments and all kwarg arguments are ignored. Entries
      in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
      arguments.

      For technical reasons, when this hook is applied to a Module, its forward function will
      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
      of each Tensor returned by the Module's forward function.

      .. warning ::
          Modifying inputs or outputs inplace is not allowed when using backward hooks and
          will raise an error.

      :param hook: The user-defined hook to be registered.
      :type hook: Callable
      :param prepend: If true, the provided ``hook`` will be fired before
                      all existing ``backward`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``backward`` hooks on
                      this :class:`torch.nn.modules.Module`. Note that global
                      ``backward`` hooks registered with
                      :func:`register_module_full_backward_hook` will fire before
                      all hooks registered by this method.
      :type prepend: bool

      :returns:     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``
      :rtype: :class:`torch.utils.hooks.RemovableHandle`


   .. py:method:: register_forward_pre_hook(hook: Union[Callable[[T, Tuple[Any, Ellipsis]], Optional[Any]], Callable[[T, Tuple[Any, Ellipsis], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]]], *, prepend: bool = False, with_kwargs: bool = False) -> torch.utils.hooks.RemovableHandle

      Register a forward pre-hook on the module.

      The hook will be called every time before :func:`forward` is invoked.


      If ``with_kwargs`` is false or not specified, the input contains only
      the positional arguments given to the module. Keyword arguments won't be
      passed to the hooks and only to the ``forward``. The hook can modify the
      input. User can either return a tuple or a single modified value in the
      hook. We will wrap the value into a tuple if a single value is returned
      (unless that value is already a tuple). The hook should have the
      following signature::

          hook(module, args) -> None or modified input

      If ``with_kwargs`` is true, the forward pre-hook will be passed the
      kwargs given to the forward function. And if the hook modifies the
      input, both the args and kwargs should be returned. The hook should have
      the following signature::

          hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

      :param hook: The user defined hook to be registered.
      :type hook: Callable
      :param prepend: If true, the provided ``hook`` will be fired before
                      all existing ``forward_pre`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``forward_pre`` hooks
                      on this :class:`torch.nn.modules.Module`. Note that global
                      ``forward_pre`` hooks registered with
                      :func:`register_module_forward_pre_hook` will fire before all
                      hooks registered by this method.
                      Default: ``False``
      :type prepend: bool
      :param with_kwargs: If true, the ``hook`` will be passed the kwargs
                          given to the forward function.
                          Default: ``False``
      :type with_kwargs: bool

      :returns:     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``
      :rtype: :class:`torch.utils.hooks.RemovableHandle`


   .. py:method:: register_forward_hook(hook: Union[Callable[[T, Tuple[Any, Ellipsis], Any], Optional[Any]], Callable[[T, Tuple[Any, Ellipsis], Dict[str, Any], Any], Optional[Any]]], *, prepend: bool = False, with_kwargs: bool = False, always_call: bool = False) -> torch.utils.hooks.RemovableHandle

      Register a forward hook on the module.

      The hook will be called every time after :func:`forward` has computed an output.

      If ``with_kwargs`` is ``False`` or not specified, the input contains only
      the positional arguments given to the module. Keyword arguments won't be
      passed to the hooks and only to the ``forward``. The hook can modify the
      output. It can modify the input inplace but it will not have effect on
      forward since this is called after :func:`forward` is called. The hook
      should have the following signature::

          hook(module, args, output) -> None or modified output

      If ``with_kwargs`` is ``True``, the forward hook will be passed the
      ``kwargs`` given to the forward function and be expected to return the
      output possibly modified. The hook should have the following signature::

          hook(module, args, kwargs, output) -> None or modified output

      :param hook: The user defined hook to be registered.
      :type hook: Callable
      :param prepend: If ``True``, the provided ``hook`` will be fired
                      before all existing ``forward`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``forward`` hooks on
                      this :class:`torch.nn.modules.Module`. Note that global
                      ``forward`` hooks registered with
                      :func:`register_module_forward_hook` will fire before all hooks
                      registered by this method.
                      Default: ``False``
      :type prepend: bool
      :param with_kwargs: If ``True``, the ``hook`` will be passed the
                          kwargs given to the forward function.
                          Default: ``False``
      :type with_kwargs: bool
      :param always_call: If ``True`` the ``hook`` will be run regardless of
                          whether an exception is raised while calling the Module.
                          Default: ``False``
      :type always_call: bool

      :returns:     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``
      :rtype: :class:`torch.utils.hooks.RemovableHandle`


   .. py:method:: __getattr__(name: str) -> Any


   .. py:method:: __delattr__(name)

      Implement delattr(self, name).


   .. py:method:: register_state_dict_pre_hook(hook)

      Register a pre-hook for the :meth:`~torch.nn.Module.load_state_dict` method.

      These hooks will be called with arguments: ``self``, ``prefix``,
      and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered
      hooks can be used to perform pre-processing before the ``state_dict``
      call is made.


   .. py:method:: register_load_state_dict_post_hook(hook)

      Register a post hook to be run after module's ``load_state_dict`` is called.

      It should have the following signature::
          hook(module, incompatible_keys) -> None

      The ``module`` argument is the current module that this hook is registered
      on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
      of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
      is a ``list`` of ``str`` containing the missing keys and
      ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

      The given incompatible_keys can be modified inplace if needed.

      Note that the checks performed when calling :func:`load_state_dict` with
      ``strict=True`` are affected by modifications the hook makes to
      ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
      set of keys will result in an error being thrown when ``strict=True``, and
      clearing out both missing and unexpected keys will avoid an error.

      :returns:     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``
      :rtype: :class:`torch.utils.hooks.RemovableHandle`


   .. py:method:: load_state_dict(state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False)

      Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

      If :attr:`strict` is ``True``, then
      the keys of :attr:`state_dict` must exactly match the keys returned
      by this module's :meth:`~torch.nn.Module.state_dict` function.

      .. warning::
          If :attr:`assign` is ``True`` the optimizer must be created after
          the call to :attr:`load_state_dict`.

      :param state_dict: a dict containing parameters and
                         persistent buffers.
      :type state_dict: dict
      :param strict: whether to strictly enforce that the keys
                     in :attr:`state_dict` match the keys returned by this module's
                     :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
      :type strict: bool, optional
      :param assign: whether to assign items in the state
                     dictionary to their corresponding keys in the module instead
                     of copying them inplace into the module's current parameters and buffers.
                     When ``False``, the properties of the tensors in the current
                     module are preserved while when ``True``, the properties of the
                     Tensors in the state dict are preserved.
                     Default: ``False``
      :type assign: bool, optional

      :returns:     * **missing_keys** is a list of str containing the missing keys
                    * **unexpected_keys** is a list of str containing the unexpected keys
      :rtype: ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields

      .. note::

         If a parameter or buffer is registered as ``None`` and its corresponding key
         exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
         ``RuntimeError``.


   .. py:method:: parameters(recurse: bool = True) -> Iterator[torch.nn.parameter.Parameter]

      Return an iterator over module parameters.

      This is typically passed to an optimizer.

      :param recurse: if True, then yields parameters of this module
                      and all submodules. Otherwise, yields only parameters that
                      are direct members of this module.
      :type recurse: bool

      :Yields: *Parameter* -- module parameter

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for param in model.parameters():
          >>>     print(type(param), param.size())
          <class 'torch.Tensor'> (20L,)
          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)



   .. py:method:: named_parameters(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]

      Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

      :param prefix: prefix to prepend to all parameter names.
      :type prefix: str
      :param recurse: if True, then yields parameters of this module
                      and all submodules. Otherwise, yields only parameters that
                      are direct members of this module.
      :type recurse: bool
      :param remove_duplicate: whether to remove the duplicated
                               parameters in the result. Defaults to True.
      :type remove_duplicate: bool, optional

      :Yields: *(str, Parameter)* -- Tuple containing the name and parameter

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for name, param in self.named_parameters():
          >>>     if name in ['bias']:
          >>>         print(param.size())



   .. py:method:: buffers(recurse: bool = True) -> Iterator[torch.Tensor]

      Return an iterator over module buffers.

      :param recurse: if True, then yields buffers of this module
                      and all submodules. Otherwise, yields only buffers that
                      are direct members of this module.
      :type recurse: bool

      :Yields: *torch.Tensor* -- module buffer

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for buf in model.buffers():
          >>>     print(type(buf), buf.size())
          <class 'torch.Tensor'> (20L,)
          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)



   .. py:method:: named_buffers(prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.Tensor]]

      Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

      :param prefix: prefix to prepend to all buffer names.
      :type prefix: str
      :param recurse: if True, then yields buffers of this module
                      and all submodules. Otherwise, yields only buffers that
                      are direct members of this module. Defaults to True.
      :type recurse: bool, optional
      :param remove_duplicate: whether to remove the duplicated buffers in the result. Defaults to True.
      :type remove_duplicate: bool, optional

      :Yields: *(str, torch.Tensor)* -- Tuple containing the name and buffer

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for name, buf in self.named_buffers():
          >>>     if name in ['running_var']:
          >>>         print(buf.size())



   .. py:method:: children() -> Iterator[Module]

      Return an iterator over immediate children modules.

      :Yields: *Module* -- a child module


   .. py:method:: named_children() -> Iterator[Tuple[str, Module]]

      Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

      :Yields: *(str, Module)* -- Tuple containing a name and child module

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for name, module in model.named_children():
          >>>     if name in ['conv4', 'conv5']:
          >>>         print(module)



   .. py:method:: modules() -> Iterator[Module]

      Return an iterator over all modules in the network.

      :Yields: *Module* -- a module in the network

      .. note::

         Duplicate modules are returned only once. In the following
         example, ``l`` will be returned only once.

      Example::

          >>> l = nn.Linear(2, 2)
          >>> net = nn.Sequential(l, l)
          >>> for idx, m in enumerate(net.modules()):
          ...     print(idx, '->', m)

          0 -> Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          )
          1 -> Linear(in_features=2, out_features=2, bias=True)



   .. py:method:: named_modules(memo: Optional[Set[Module]] = None, prefix: str = '', remove_duplicate: bool = True)

      Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

      :param memo: a memo to store the set of modules already added to the result
      :param prefix: a prefix that will be added to the name of the module
      :param remove_duplicate: whether to remove the duplicated module instances in the result
                               or not

      :Yields: *(str, Module)* -- Tuple of name and module

      .. note::

         Duplicate modules are returned only once. In the following
         example, ``l`` will be returned only once.

      Example::

          >>> l = nn.Linear(2, 2)
          >>> net = nn.Sequential(l, l)
          >>> for idx, m in enumerate(net.named_modules()):
          ...     print(idx, '->', m)

          0 -> ('', Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          ))
          1 -> ('0', Linear(in_features=2, out_features=2, bias=True))



   .. py:method:: train(mode: bool = True) -> T

      Set the module in training mode.

      This has any effect only on certain modules. See documentations of
      particular modules for details of their behaviors in training/evaluation
      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
      etc.

      :param mode: whether to set training mode (``True``) or evaluation
                   mode (``False``). Default: ``True``.
      :type mode: bool

      :returns: self
      :rtype: Module


   .. py:method:: eval() -> T

      Set the module in evaluation mode.

      This has any effect only on certain modules. See documentations of
      particular modules for details of their behaviors in training/evaluation
      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
      etc.

      This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

      See :ref:`locally-disable-grad-doc` for a comparison between
      `.eval()` and several similar mechanisms that may be confused with it.

      :returns: self
      :rtype: Module


   .. py:method:: requires_grad_(requires_grad: bool = True) -> T

      Change if autograd should record operations on parameters in this module.

      This method sets the parameters' :attr:`requires_grad` attributes
      in-place.

      This method is helpful for freezing part of the module for finetuning
      or training parts of a model individually (e.g., GAN training).

      See :ref:`locally-disable-grad-doc` for a comparison between
      `.requires_grad_()` and several similar mechanisms that may be confused with it.

      :param requires_grad: whether autograd should record operations on
                            parameters in this module. Default: ``True``.
      :type requires_grad: bool

      :returns: self
      :rtype: Module


   .. py:method:: zero_grad(set_to_none: bool = True) -> None

      Reset gradients of all model parameters.

      See similar function under :class:`torch.optim.Optimizer` for more context.

      :param set_to_none: instead of setting to zero, set the grads to None.
                          See :meth:`torch.optim.Optimizer.zero_grad` for details.
      :type set_to_none: bool


   .. py:method:: share_memory() -> T

      See :meth:`torch.Tensor.share_memory_`.


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: compile(*args, **kwargs)

      Compile this Module's forward using :func:`torch.compile`.

      This Module's `__call__` method is compiled and all arguments are passed as-is
      to :func:`torch.compile`.

      See :func:`torch.compile` for details on the arguments for this function.



