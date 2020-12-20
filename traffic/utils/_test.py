from traffic.testing import name, absolute_name
from traffic.utils import (
    get_ssid,
    webcam_server,
    Singleton,
    virtual_proxy_property,
    SingletonByIdMeta,
    load_rgb_array,
    NNInputMaterial,
)
from traffic.consts import SSID_MONOR, SSID_USED_BY_DROIDCAM, DROIDCAM, SPACE
from traffic.imports import *
from traffic.logging import root_logger
from traffic.utils.lane_unet_test import EXAMPLE_IMG_PATH, EXAMPLE_MASK_PATH


def setup_function(function):
    Singleton._instances = dict()


def teardown_function(function):
    patch.stopall()


def is_droidcam_running():
    out = check_output(["ps", "-o", "command"]).decode("ascii")
    ret = True if DROIDCAM + SPACE in out else False
    root_logger.info(out)
    root_logger.info(ret)
    return ret


@name(load_rgb_array, "1", globals())
def _():
    for i in [EXAMPLE_IMG_PATH, EXAMPLE_MASK_PATH]:
        x = load_rgb_array(i)
        assert type(x) is ndarray
        assert x.dtype.name == "uint8"
        assert x.shape == (2710, 3384, 3)
        assert x.max() <= 255 and x.min() >= 0 and x.mean() > 1


@name(NNInputMaterial.__init__, "1", globals())
def _():
    x = object.__new__(NNInputMaterial)
    x.__init__(sentinel.a)
    assert x.path is sentinel.a


@name(NNInputMaterial.data.fget, "1", globals())
def _():
    m_load_rgb_array = patch("traffic.utils.load_rgb_array", new=MagicMock(return_value=sentinel.b)).start()
    x = object.__new__(NNInputMaterial)
    x.path = sentinel.a
    y = x.data
    m_load_rgb_array.assert_called_once_with(sentinel.a)
    assert y is sentinel.b


@name(NNInputMaterial.attributes_path.fget, "1", globals())
def _():
    x = object.__new__(NNInputMaterial)
    x.path = "a.png"
    y = x.attributes_path
    assert y == "a.png.json"


@name(NNInputMaterial.save_attributes_to_json, "1", globals())
def _():
    m_attributes_path = patch.object(
        NNInputMaterial, "attributes_path", new=PropertyMock(return_value=sentinel.a)
    ).start()
    m_open = patch("traffic.utils.open", new=mock_open()).start()
    m_dump_json = patch("traffic.utils.dump_json", new=MagicMock()).start()
    x = object.__new__(NNInputMaterial)
    y = x.save_attributes_to_json(sentinel.b)
    m_open.assert_called_once_with(sentinel.a, "w")
    m_dump_json.assert_called_once_with(sentinel.b, m_open())
    assert y is None


@name(NNInputMaterial.get_attributes_from_json, "1", globals())
def _():
    m_attributes_path = patch.object(
        NNInputMaterial, "attributes_path", new=PropertyMock(return_value=sentinel.a)
    ).start()
    m_open = patch("traffic.utils.open", new=mock_open()).start()
    m_load_json = patch("traffic.utils.load_json", new=MagicMock(return_value=sentinel.b)).start()
    x = object.__new__(NNInputMaterial)
    y = x.get_attributes_from_json()
    m_open.assert_called_once_with(sentinel.a)
    m_load_json.assert_called_once_with(m_open())
    assert y is sentinel.b


@name(NNInputMaterial.attributes.fget, "not saved previousli", globals())
def _():
    m_attributes_path = patch.object(
        NNInputMaterial, "attributes_path", new=PropertyMock(return_value=sentinel.a)
    ).start()
    m_get_calculated_attributes = patch.object(
        NNInputMaterial, "get_calculated_attributes", new=MagicMock(return_value=sentinel.a)
    ).start()
    m_save_attributes_to_json = patch.object(NNInputMaterial, "save_attributes_to_json", new=MagicMock()).start()
    m_get_attributes_from_json = patch.object(
        NNInputMaterial, "get_attributes_from_json", new=MagicMock(return_value=sentinel.b)
    ).start()
    m_exists = patch("traffic.utils.exists", new=MagicMock(return_value=False)).start()
    x = object.__new__(NNInputMaterial)
    y = x.attributes
    m_exists.assert_called_once_with(sentinel.a)
    m_get_calculated_attributes.assert_called_once_with()
    m_save_attributes_to_json.assert_called_once_with(sentinel.a)
    m_get_attributes_from_json.assert_called_once_with()
    assert y is sentinel.b


@name(NNInputMaterial.attributes.fget, "saved previousli", globals())
def _():
    m_attributes_path = patch.object(
        NNInputMaterial, "attributes_path", new=PropertyMock(return_value=sentinel.a)
    ).start()
    m_get_calculated_attributes = patch.object(
        NNInputMaterial, "get_calculated_attributes", new=MagicMock(return_value=sentinel.a)
    ).start()
    m_save_attributes_to_json = patch.object(NNInputMaterial, "save_attributes_to_json", new=MagicMock()).start()
    m_get_attributes_from_json = patch.object(
        NNInputMaterial, "get_attributes_from_json", new=MagicMock(return_value=sentinel.b)
    ).start()
    m_exists = patch("traffic.utils.exists", new=MagicMock(return_value=True)).start()
    x = object.__new__(NNInputMaterial)
    y = x.attributes
    m_exists.assert_called_once_with(sentinel.a)
    m_get_calculated_attributes.assert_not_called()
    m_save_attributes_to_json.assert_not_called()
    m_get_attributes_from_json.assert_called_once_with()
    assert y is sentinel.b


@name(SingletonByIdMeta.__init__, "1", globals())
def _():
    class Base:
        def method(self):
            ...

        @staticmethod
        def get_id(name, arg2=None):
            return name

    class A(Base, metaclass=SingletonByIdMeta):
        def __init__(self, name, arg2=None):
            self.name = name

    a = A("a")
    b = A("a", 1)
    assert type(A._instances) == dict
    assert A._instances["a"] is a
    assert b is a
    assert a.id == "a"


@name(SingletonByIdMeta.__init__, "called once", globals())
def _():
    class A(metaclass=SingletonByIdMeta):
        def __init__(self, name, name2):
            self.name = name
            self.name2 = name2

        @staticmethod
        def get_id(name, name2):
            return name

    assert len(A) == 0
    a = A("a", 1)
    b = A("a", 2)
    assert a is b
    assert a.name2 == 1


@name(SingletonByIdMeta.__len__, "1", globals())
def _():
    class A(metaclass=SingletonByIdMeta):
        def __init__(self, name):
            self.name = name

        @staticmethod
        def get_id(name):
            return name

    assert len(A) == 0
    a = A("a")
    b = A("a")
    assert len(A) == 1
    b = A("b")
    assert len(A) == 2


@name(SingletonByIdMeta.clear, "1", globals())
def _():
    class A(metaclass=SingletonByIdMeta):
        def __init__(self, name):
            self.name = name

        @staticmethod
        def get_id(name):
            return name

    assert A._instances == dict()
    a = A("a")
    b = A("a")
    assert A._instances == {"a": b}
    A.clear()
    assert A._instances == dict()


@name(SingletonByIdMeta.__iter__, "1", globals())
def _():
    class A(metaclass=SingletonByIdMeta):
        def __init__(self, name):
            self.name = name

        @staticmethod
        def get_id(name):
            return name

    a = A("a")
    b = A("b")
    for i in A:
        assert i is a or i is b


@name(SingletonByIdMeta.__getitem__, "default", globals())
def _():
    class A(metaclass=SingletonByIdMeta):
        def __init__(self, name):
            self.name = name

        @staticmethod
        def get_id(name):
            return name

    a = A("a")
    assert A[a] is a


@name(SingletonByIdMeta.__getitem__, "__eq__", globals())
def _():
    class A(metaclass=SingletonByIdMeta):
        def __init__(self, name):
            self.name = name

        @staticmethod
        def get_id(name):
            return name

        def __eq__(self, other):
            return self.name == other

    a = A("a")
    assert A["a"] is a is A[a]
    with raises(IndexError):
        A["b"]


@name(SingletonByIdMeta.__new__, "without get_id", globals())
def _():
    with raises(KeyError):

        class A(metaclass=SingletonByIdMeta):
            ...


@name(SingletonByIdMeta.__new__, "with bad signature get_id", globals())
def _():
    class Base:
        def __init__(self, a, b: int, *args: "s"):
            ...

    with raises(NameError):

        class A(Base, metaclass=SingletonByIdMeta):
            @staticmethod
            def get_id():
                ...


@name(virtual_proxy_property, "1", globals())
def _():
    class A:
        @virtual_proxy_property
        def getter(self):
            return 0

    a = A()
    with raises(AttributeError):
        a._getter
    assert a.getter == 0
    assert a._getter == 0
    a._getter = 1
    assert a.getter == 1
    assert A.getter.fget.__name__ == "getter"


@name(Iterable_abc, "abc", globals())
def _():
    assert isinstance(dict(), Iterable_abc)
    assert isinstance([], Iterable_abc)
    assert isinstance(set(), Iterable_abc)
    assert isinstance("", Iterable_abc)
    assert not isinstance(1, Iterable_abc)


@name(Iterable_type, "type", globals())
def _():
    assert isinstance(dict(), Iterable_type)
    assert isinstance([], Iterable_type)
    assert isinstance(set(), Iterable_type)
    assert isinstance("", Iterable_type)
    assert not isinstance(1, Iterable_type)


@name(Singleton.__new__, "work with self", globals())
def _():
    assert Singleton._instances == dict()
    a = Singleton()
    b = Singleton()
    assert a is b
    assert len(Singleton._instances) == 1
    assert Singleton._instances[Singleton] is a


@name(Singleton.__new__, "work new class", globals())
def _():
    class A(Singleton):
        ...

    assert Singleton._instances == dict()
    a = A()
    b = A()
    assert a is b
    assert len(Singleton._instances) == 1
    assert Singleton._instances[A] is a


@name(Singleton.__new__, "work new classes", globals())
def _():
    class A(Singleton):
        ...

    class B(Singleton):
        ...

    assert Singleton._instances == dict()
    a = A()
    b = B()
    assert a is not b
    assert len(Singleton._instances) == 2
    assert Singleton._instances[A] is a
    assert Singleton._instances[B] is b


@name(get_ssid, "1", globals())
def _():
    ssids = [None, SSID_MONOR, SSID_USED_BY_DROIDCAM]
    assert get_ssid() in ssids


@name(webcam_server, "None", globals())
@patch(absolute_name(get_ssid))
def _(mock_get_ssid):
    mock_get_ssid.return_value = None
    import traffic.utils

    assert traffic.utils.get_ssid() == None
    assert is_droidcam_running() == False
    with webcam_server():
        assert is_droidcam_running() == False


@name(webcam_server, SSID_MONOR, globals())
@patch(absolute_name(get_ssid))
def _(mock_get_ssid):
    mock_get_ssid.return_value = SSID_MONOR
    import traffic.utils

    assert traffic.utils.get_ssid() == SSID_MONOR
    assert is_droidcam_running() == False
    with webcam_server():
        assert is_droidcam_running() == False


@name(webcam_server, SSID_USED_BY_DROIDCAM, globals())
@patch(absolute_name(get_ssid))
def _(mock_get_ssid):
    mock_get_ssid.return_value = SSID_USED_BY_DROIDCAM
    import traffic.utils

    assert traffic.utils.get_ssid() == SSID_USED_BY_DROIDCAM
    assert is_droidcam_running() == False
    with webcam_server():
        assert is_droidcam_running() == True
    # assert is_droidcam_running() == False #todo not working
