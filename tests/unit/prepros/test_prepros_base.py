# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg,redefined-outer-name
"""Tests for prepros.base"""

import pytest
import tensorflow as tf
import numpy as np

import deepr as dpr


@pytest.fixture
def dataset():

    def _gen():
        yield {"a": [0]}
        yield {"a": [0, 1]}

    return tf.data.Dataset.from_generator(_gen, {"a": tf.int32}, {"a": (None,)})


@dpr.prepros.prepro
def AddOffset(dataset, offset):
    """AddOffset"""
    return dataset.map(lambda x: {"b": x["a"] + offset})


@dpr.prepros.prepro
def AddOne():
    """AddOne"""
    return AddOffset(offset=1)


@dpr.prepros.prepro(lazy=True)
def AddOneLazy():
    """AddOneLazy"""
    return AddOffset(offset=1)


def test_prepros_decorator_from_apply(dataset):
    """Create preprocessor from an apply function"""
    # Check decorated function properties
    assert issubclass(AddOffset, dpr.prepros.Prepro)
    assert AddOffset.__name__ == "AddOffset"
    assert AddOffset.__doc__ == "AddOffset"
    assert AddOffset.__module__ == __name__

    # Check instance properties
    add_one = AddOffset(offset=1)
    reader = dpr.readers.from_dataset(add_one(dataset))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_decorator_from_constructor(dataset):
    """Create preprocessor from a Prepro constructor."""
    # Check decorated function properties
    assert not issubclass(AddOne, dpr.prepros.Prepro)
    assert AddOne.__name__ == "AddOne"
    assert AddOne.__doc__ == "AddOne"
    assert AddOne.__module__ == __name__

    # Check instance properties
    add_one = AddOne()
    assert isinstance(add_one, AddOffset)
    reader = dpr.readers.from_dataset(add_one(dataset))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_decorator_from_lazy_constructor(dataset):
    """Create preprocessor from a Prepro constructor lazily."""
    # Check decorated function properties
    assert issubclass(AddOneLazy, dpr.prepros.Prepro)
    assert AddOneLazy.__name__ == "AddOneLazy"
    assert AddOneLazy.__doc__ == "AddOneLazy"
    assert AddOneLazy.__module__ == __name__

    # Check instance properties
    add_one = AddOneLazy()
    assert isinstance(add_one, AddOneLazy)
    assert not isinstance(add_one, AddOffset)
    reader = dpr.readers.from_dataset(add_one(dataset))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_decorator_from_apply_laziness(dataset):
    """Laziness is especially useful if custom prepro use hash tables"""

    @dpr.prepros.prepro
    def RaiseApply(dataset):
        raise RuntimeError()

    prepro = RaiseApply()
    with pytest.raises(RuntimeError):
        prepro(dataset)


def test_prepros_decorator_from_lazy_constructor_laziness(dataset):
    """Laziness is especially useful if custom prepro use hash tables"""

    @dpr.prepros.prepro(lazy=True)
    def RaiseConstructor():
        raise RuntimeError

    prepro = RaiseConstructor()
    with pytest.raises(RuntimeError):
        prepro(dataset)


def test_prepros_decorator_from_apply_wrong_order():
    """Test wrong order in arguments raises error at decoration time."""
    # pylint: disable=unused-variable
    with pytest.raises(TypeError):

        @dpr.prepros.prepro
        def WrongOrder(offset, dataset):
            return dataset.map(lambda x: {"b": x["a"] + offset})


def test_prepros_decorator_from_constructor_incorrect_type():
    """Test decorator failure if constructor does not return Prepro."""
    with pytest.raises(TypeError):

        @dpr.prepros.prepro
        def NotAPreproConstructor():
            pass

        NotAPreproConstructor()


@dpr.prepros.prepro
def Identity(dataset) -> tf.data.Dataset:
    return dataset


@dpr.prepros.prepro
def Typical(dataset, foo, bar=1) -> dpr.prepros.Prepro:
    return dataset.map(lambda x: {"b": x["a"] + foo + 2 * bar})


@dpr.prepros.prepro
def TypicalFactory(foo, bar=1) -> dpr.layers.Layer:
    return Typical(foo=foo, bar=bar)


@pytest.mark.parametrize(
    "cls, args, kwargs, error",
    [
        # Simple layer with no special arguments
        (Identity, (), {}, None),
        (Identity, (), {"tensors": 1}, TypeError),
        (Identity, (1,), None, TypeError),
        (Identity, (), {"foo": 1}, TypeError),
        # Typical layers with positional and keyword arguments
        (Typical, (1,), {}, None),
        (Typical, (1, 2), {}, None),
        (Typical, (), {"foo": 1, "bar": 2}, None),
        (Typical, (1,), {"bar": 2}, None),
        (Typical, (1, 2, 3), {}, TypeError),
        (Typical, (1,), {"foo": 1}, TypeError),
        (Typical, (1,), {"baz": 1}, TypeError),
        # Typical Factory
        (TypicalFactory, (1,), {}, None),
        (TypicalFactory, (1, 2), {}, None),
        (TypicalFactory, (), {"foo": 1, "bar": 2}, None),
        (TypicalFactory, (1,), {"bar": 2}, None),
        (TypicalFactory, (1, 2, 3), {}, TypeError),
        (TypicalFactory, (1,), {"foo": 1}, TypeError),
        (TypicalFactory, (1,), {"baz": 1}, TypeError),
    ],
)
def test_prepros_decorator_instantiation(cls, args, kwargs, error):
    """Test prepro instantiation from decorator."""
    if error is not None:
        with pytest.raises(error):
            cls(*args, **kwargs)
    else:
        instance = cls(*args, **kwargs)
        assert isinstance(instance, dpr.prepros.Prepro)
