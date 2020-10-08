from traffic.globals import Globals, waitKey, __name__
from traffic.utils import Singleton
from traffic.testing import name, relative_name
from traffic.imports import raises, patch


def setup_function(function):
    Singleton._instances = dict()


@name(Globals.wait_keys.fget, "1", globals())
def _():
    g = Globals()
    with raises(AttributeError):
        g.wait_keys
    g._wait_keys = [49]
    assert g.wait_keys == ["1"]


@name(Globals.wait_keys.fset, "2", globals())
def _():
    g = Globals()
    with raises(TypeError):
        g.wait_keys = 1.1
    with raises(ValueError):
        g.wait_keys = 11
    g.wait_keys = 1
    assert g._wait_keys == [49]


@name(Globals.pressed_key.fget, "good input", globals())
@patch(relative_name(__name__, waitKey))
def _(mocked_waitKey):
    mocked_waitKey.return_value = 49
    g = Globals()
    g._wait_keys = [48, 49]
    a = g.pressed_key
    mocked_waitKey.assert_called_once_with(1)
    assert a is "1"


@name(Globals.pressed_key.fget, "bad input", globals())
@patch(relative_name(__name__, waitKey))
def _(mocked_waitKey):
    mocked_waitKey.return_value = 49
    g = Globals()
    g._wait_keys = [48, 50]
    a = g.pressed_key
    mocked_waitKey.assert_called_once_with(1)
    assert a is None
