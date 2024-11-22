"""EasyDict
Copy/pasted from https://github.com/makinacorpus/easydict
Original author: Mathieu Leplatre <mathieu.leplatre@makina-corpus.com>
"""


class EasyDict(dict):
    """A dictionary that allows attribute-style access.

    This class extends the standard dictionary to allow access to its
    items as attributes. It works recursively, allowing nested
    dictionaries to be accessed in a similar manner.

    Attributes:
        d (dict, optional): An initial dictionary to populate the EasyDict.
        **kwargs: Additional key-value pairs to populate the EasyDict.

    Examples:
        >>> d = EasyDict({'foo': 3})
        >>> d['foo']
        3
        >>> d.foo
        3
        >>> d.bar
        Traceback (most recent call last):
        ...
        AttributeError: 'EasyDict' object has no attribute 'bar'

        >>> d = EasyDict({'foo': 3, 'bar': {'x': 1, 'y': 2}})
        >>> isinstance(d.bar, dict)
        True
        >>> d.bar.x
        1

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

        >>> d = EasyDict({'foo': 0, 'bar': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]})
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

        >>> o = EasyDict({'clean': True})
        >>> list(o.items())
        [('clean', True)]

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
    """

    def __init__(self, d=None, **kwargs):
        """Initializes the EasyDict.

        Args:
            d (dict, optional): An initial dictionary to populate the EasyDict.
            **kwargs: Additional key-value pairs to populate the EasyDict.
        """
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in (
                "update",
                "pop",
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        """Sets an attribute of the EasyDict.

        Args:
            name (str): The name of the attribute to set.
            value: The value to assign to the attribute.
        """
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        """Updates the EasyDict with new key-value pairs.

        Args:
            e (dict, optional): A dictionary of items to update.
            **f: Additional key-value pairs to update.

        Returns:
            None
        """
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        """Removes a key from the EasyDict.

        Args:
            k (str): The key to remove.
            d: (optional) The default value to return if the key is not found.

        Returns:
            The value associated with the removed key, or d if the key was not found.
        """
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
