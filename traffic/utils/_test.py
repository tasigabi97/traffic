from traffic.testing import name, absolute_name
from traffic.utils import get_ssid, webcam_server, Singleton, virtual_proxy_property, SingletonByIdMeta
from traffic.consts import SSID_MONOR, SSID_VODAFONE, DROIDCAM, SPACE
from traffic.imports import patch, check_output, Iterable_abc, Iterable_type, raises
from traffic.logging import root_logger
from traffic.utils.lane_mrcnn import LaneDataset, LaneConfig


def setup_function(function):
    Singleton._instances = dict()


def is_droidcam_running():
    out = check_output(["ps", "-o", "command"]).decode("ascii")
    ret = True if DROIDCAM + SPACE in out else False
    root_logger.info(out)
    root_logger.info(ret)
    return ret


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


@name(LaneDataset.get_min_instance_size, "1", globals())
@patch("traffic.utils.lane_mrcnn.LaneConfig.MIN_INSTANCE_SIZE", 1)
@patch("traffic.utils.lane_mrcnn.LaneConfig.MAX_GT_INSTANCES", 1)
def _():
    assert LaneDataset.get_min_instance_size([1000, 1, 2, 3, 4]) == 4


@name(LaneDataset.get_min_instance_size, "2", globals())
@patch("traffic.utils.lane_mrcnn.LaneConfig.MIN_INSTANCE_SIZE", 5)
@patch("traffic.utils.lane_mrcnn.LaneConfig.MAX_GT_INSTANCES", 1)
def _():
    assert LaneDataset.get_min_instance_size([1000, 1, 2, 3, 4]) == 5


@name(LaneDataset.get_min_instance_size, "3", globals())
@patch("traffic.utils.lane_mrcnn.LaneConfig.MIN_INSTANCE_SIZE", 1)
@patch("traffic.utils.lane_mrcnn.LaneConfig.MAX_GT_INSTANCES", 4)
def _():
    assert LaneDataset.get_min_instance_size([1000, 1, 2, 3, 4]) == 1


@name(LaneDataset.get_min_instance_size, "4", globals())
@patch("traffic.utils.lane_mrcnn.LaneConfig.MIN_INSTANCE_SIZE", 1)
@patch("traffic.utils.lane_mrcnn.LaneConfig.MAX_GT_INSTANCES", 2)
def _():
    assert LaneDataset.get_min_instance_size([1000, 1, 2, 3, 4]) == 3


@name(LaneDataset.get_min_instance_size, "5", globals())
@patch("traffic.utils.lane_mrcnn.LaneConfig.MIN_INSTANCE_SIZE", 1)
@patch("traffic.utils.lane_mrcnn.LaneConfig.MAX_GT_INSTANCES", 1)
def _():
    assert LaneDataset.get_min_instance_size([1000, 1, 2, 3, 3]) == 4


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
    ssids = [None, SSID_MONOR, SSID_VODAFONE]
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


@name(webcam_server, SSID_VODAFONE, globals())
@patch(absolute_name(get_ssid))
def _(mock_get_ssid):
    mock_get_ssid.return_value = SSID_VODAFONE
    import traffic.utils

    assert traffic.utils.get_ssid() == SSID_VODAFONE
    assert is_droidcam_running() == False
    with webcam_server():
        assert is_droidcam_running() == True
    assert is_droidcam_running() == False
