:orphan:

:py:mod:`zensvi.cv.depth_estimation.zoedepth.utils.easydict`
============================================================

.. py:module:: zensvi.cv.depth_estimation.zoedepth.utils.easydict

.. autoapi-nested-parse::

   EasyDict
   Copy/pasted from https://github.com/makinacorpus/easydict
   Original author: Mathieu Leplatre <mathieu.leplatre@makina-corpus.com>



Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   zensvi.cv.depth_estimation.zoedepth.utils.easydict.EasyDict




.. py:class:: EasyDict(d=None, **kwargs)


   Bases: :py:obj:`dict`

   Get attributes

   >>> d = EasyDict({'foo':3})
   >>> d['foo']
   3
   >>> d.foo
   3
   >>> d.bar
   Traceback (most recent call last):
   ...
   AttributeError: 'EasyDict' object has no attribute 'bar'

   Works recursively

   >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
   >>> isinstance(d.bar, dict)
   True
   >>> d.bar.x
   1

   Bullet-proof

   >>> EasyDict({})
   {}
   >>> EasyDict(d={})
   {}
   >>> EasyDict(None)
   {}
   >>> d = {'a': 1}
   >>> EasyDict(**d)
   {'a': 1}
   >>> EasyDict((('a', 1), ('b', 2)))
   {'a': 1, 'b': 2}

   Set attributes

   >>> d = EasyDict()
   >>> d.foo = 3
   >>> d.foo
   3
   >>> d.bar = {'prop': 'value'}
   >>> d.bar.prop
   'value'
   >>> d
   {'foo': 3, 'bar': {'prop': 'value'}}
   >>> d.bar.prop = 'newer'
   >>> d.bar.prop
   'newer'


   Values extraction

   >>> d = EasyDict({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
   >>> isinstance(d.bar, list)
   True
   >>> from operator import attrgetter
   >>> list(map(attrgetter('x'), d.bar))
   [1, 3]
   >>> list(map(attrgetter('y'), d.bar))
   [2, 4]
   >>> d = EasyDict()
   >>> list(d.keys())
   []
   >>> d = EasyDict(foo=3, bar=dict(x=1, y=2))
   >>> d.foo
   3
   >>> d.bar.x
   1

   Still like a dict though

   >>> o = EasyDict({'clean':True})
   >>> list(o.items())
   [('clean', True)]

   And like a class

   >>> class Flower(EasyDict):
   ...     power = 1
   ...
   >>> f = Flower()
   >>> f.power
   1
   >>> f = Flower({'height': 12})
   >>> f.height
   12
   >>> f['power']
   1
   >>> sorted(f.keys())
   ['height', 'power']

   update and pop items
   >>> d = EasyDict(a=1, b='2')
   >>> e = EasyDict(c=3.0, a=9.0)
   >>> d.update(e)
   >>> d.c
   3.0
   >>> d['c']
   3.0
   >>> d.get('c')
   3.0
   >>> d.update(a=4, b=4)
   >>> d.b
   4
   >>> d.pop('a')
   4
   >>> d.a
   Traceback (most recent call last):
   ...
   AttributeError: 'EasyDict' object has no attribute 'a'

   .. py:attribute:: __setitem__

      

   .. py:method:: __setattr__(name, value)

      Implement setattr(self, name, value).


   .. py:method:: update(e=None, **f)

      D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
      If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
      If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
      In either case, this is followed by: for k in F:  D[k] = F[k]


   .. py:method:: pop(k, d=None)

      D.pop(k[,d]) -> v, remove specified key and return the corresponding value.

      If key is not found, default is returned if given, otherwise KeyError is raised


   .. py:method:: __contains__()

      True if the dictionary has the specified key, else False.


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __delitem__()

      Delete self[key].


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __getitem__()

      x.__getitem__(y) <==> x[y]


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __ior__()

      Return self|=value.


   .. py:method:: __iter__()

      Implement iter(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __len__()

      Return len(self).


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __or__()

      Return self|value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __reversed__()

      Return a reverse iterator over the dict keys.


   .. py:method:: __ror__()

      Return value|self.


   .. py:method:: __sizeof__()

      D.__sizeof__() -> size of D in memory, in bytes


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: clear()

      D.clear() -> None.  Remove all items from D.


   .. py:method:: copy()

      D.copy() -> a shallow copy of D


   .. py:method:: get()

      Return the value for key if key is in the dictionary, else default.


   .. py:method:: items()

      D.items() -> a set-like object providing a view on D's items


   .. py:method:: keys()

      D.keys() -> a set-like object providing a view on D's keys


   .. py:method:: popitem()

      Remove and return a (key, value) pair as a 2-tuple.

      Pairs are returned in LIFO (last-in, first-out) order.
      Raises KeyError if the dict is empty.


   .. py:method:: setdefault()

      Insert key with a value of default if key is not in the dictionary.

      Return the value for key if key is in the dictionary, else default.


   .. py:method:: values()

      D.values() -> an object providing a view on D's values



