"""Abstract Base Class for preprocessing"""

from abc import ABC, abstractmethod
import logging
import functools
from typing import Callable
import inspect

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


class Prepro(ABC):
    """Base class for composable preprocessing functions.

    `Prepro` are the basic building blocks of a preprocessing pipeline.
    A `Prepro` defines a function on a tf.data.Dataset.

    The basic usage of a :class:`~Prepro` is to apply it on a Dataset. For
    example:

    >>> dataset = tf.data.Dataset.from_generator(range(3))
    [0, 1, 2]
    >>> prepro_fn = Map(lambda element: element + 1)
    >>> dataset = prepro_fn(dataset)
    [1, 2, 3]

    Because some preprocessing pipelines behave differently depending
    on the mode (TRAIN, EVAL, PREDICT), an optional argument can be
    provided:

    >>> def map_func(element, mode = None):
    ...     if mode == tf.estimator.ModeKeys.PREDICT:
    ...         return 0
    ...     else:
    ...         return element
    >>> prepro_fn = Map(map_func)
    >>> dataset = prepro_fn(dataset, tf.estimator.ModeKeys.TRAIN)
    [0, 1, 2]
    >>> dataset = prepro_fn(dataset, tf.estimator.ModeKeys.PREDICT)
    [0, 0, 0]

    :class:`~Map`, :class:`~Filter`, :class:`~Shuffle` and :class:`~Repeat` have a special attribute
    `modes` that you can use to specify the modes on which the
    preprocessing should be applied. For example:

    >>> def map_func(element, mode = None):
    ...     return 0
    >>> prepro_fn = Map(map_func, modes=[tf.estimator.PREDICT])
    >>> dataset = prepro_fn(dataset, tf.estimator.ModeKeys.TRAIN)
    [0, 1, 2]
    >>> dataset = prepro_fn(dataset, tf.estimator.ModeKeys.PREDICT)
    [0, 0, 0]

    Authors of new :class:`~Prepro` subclasses typically override the `apply`
    method of the base :class:`~Prepro` class::

        def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
            return dataset

    The easiest way to define custom preprocessors is to use the
    `prepro` decorator (see documentation).
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __call__(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        """Alias for apply"""
        return self.apply(dataset, mode=mode)

    @abstractmethod
    def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        """Pre-process a dataset"""
        raise NotImplementedError()


def prepro(fn: Callable = None, lazy: bool = False):
    """Decorator that creates a prepro constructor from a function.

    There are 2 ways to use the decorator
        - From a ``apply``-like function
        - From a :class:`~Prepro` factory (a function creating a Prepro)

    From a ``apply``-like function
    ------------------------------
    In that case, the decorator returns a subclass of :class:`~Prepro`
    whose ``apply`` method is defined by the decorated function.

    For example

    >>> @prepro
    ... def AddOffset(dataset, mode, offset):
    ...     return dataset.map(lambda element: element + offset)
    >>> dataset = [0, 1, 2]
    >>> prepro_fn = AddOffset(offset=1)
    >>> dataset = prepro_fn(dataset)
    [1, 2, 3]

    The class created by the decorator is roughly equivalent to

    .. code-block:: python

        class AddOffset(Prepro):

            def __init__(self, offset)
                Prepro.__init__(self)
                self.offset = offset

            def apply(self, dataset, mode: str = None):
                return dataset.map(lambda element: element + self.offset)

    Note that you need to use either "dataset" or "dataset" and "mode"
    as the first two parameters.

    From a :class:`~Prepro` factory
    -------------------------------
    Another way of using the decorator is on functions that return
    Prepro instances. In that case, the decorator behaves differently
    depending on the ``lazy`` argument:
        - ``lazy=False`` (DEFAULT) : in that case, the decorator returns
          another :class:`~Prepro` factory, whose return type is a
          subclass of the original factory return type and whose
          ``apply`` method is the same as the instance created by the
          decorated factory.
        - ``lazy=True`` : in that case, the decorator returns a subclass
          of :class:`~Prepro` whose ``apply`` calls the factory to get
          a prepro instance, and use its ``apply`` method.

    For example, in the non-lazy case (DEFAULT)

    >>> @prepro
    ... def AddOne() -> Prepro:
    ...     print("Calling prepro")
    ...     return AddOffset(offset=1)
    >>> dataset = [0, 1, 2]
    >>> prepro_fn = AddOne()
    "Calling prepro"
    >>> dataset = prepro_fn(dataset)
    [1, 2, 3]

    For example, in the lazy case

    >>> @prepro(lazy=True)
    ... def AddOne() -> Prepro:
    ...     print("Calling prepro")
    ...     return AddOffset(offset=1)
    >>> dataset = [0, 1, 2]
    >>> prepro_fn = AddOne()
    >>> dataset = prepro_fn(dataset)
    "Calling prepro"
    [1, 2, 3]
    """
    # pylint: disable=protected-access,invalid-name
    def _prepro_constructor(fn: Callable) -> Callable:
        """Decorator that creates a Layer constructor."""
        parameters = inspect.signature(fn).parameters
        signature = inspect.Signature([param for key, param in parameters.items() if key not in {"dataset", "mode"}])

        # Check order of parameters
        if "dataset" in parameters:
            if list(parameters.keys())[0] != "dataset":
                raise TypeError(f"'dataset' should be the first parameter of {fn}")
            if "mode" in parameters:
                if list(parameters.keys())[1] != "mode":
                    raise TypeError(f"'mode' should be the second parameter of {fn}")

        # In those cases, create a new Prepro class
        if "dataset" in parameters or lazy:

            @functools.wraps(fn)
            def _init(self, *args, **kwargs):
                Prepro.__init__(self)
                signature.bind(*args, **kwargs)
                self._args = args
                self._kwargs = kwargs

            if list(parameters.keys())[:2] == ["dataset", "mode"]:

                def _apply(self, dataset, mode: str = None):
                    return fn(dataset, mode, *self._args, **self._kwargs)

            elif list(parameters.keys())[:1] == ["dataset"]:

                def _apply(self, dataset, mode: str = None):
                    # pylint: disable=unused-argument
                    return fn(dataset, *self._args, **self._kwargs)

            else:

                def _apply(self, dataset, mode: str = None):
                    _prepro = fn(*self._args, **self._kwargs)
                    return _prepro.apply(dataset, mode)

            attributes = {"__module__": fn.__module__, "__doc__": fn.__doc__, "__init__": _init, "apply": _apply}
            return type(fn.__name__, (Prepro,), attributes)

        # Constructor of a subclass wrapping the prepro created by fn
        else:

            @functools.wraps(fn)
            def _constructor(*args, **kwargs):
                _prepro = fn(*args, **kwargs)
                if not isinstance(_prepro, Prepro):
                    raise TypeError(f"Expected {Prepro} but got {type(_prepro)} from {fn}")

                def _init(self):
                    Prepro.__init__(self)
                    self._prepro = _prepro

                def _apply(self, dataset, mode: str = None):
                    return self._prepro.apply(dataset, mode)

                def _getattr(self, name):
                    return getattr(self._prepro, name)

                attributes = {
                    "__module__": fn.__module__,
                    "__doc__": fn.__doc__,
                    "__init__": _init,
                    "__getattr__": _getattr,
                    "apply": _apply,
                }
                return type(fn.__name__, (type(_prepro),), attributes)()

            return _constructor

    if fn is None:
        return _prepro_constructor
    else:
        return _prepro_constructor(fn)
