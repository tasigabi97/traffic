from traffic.testing import name, absolute_name
from traffic.utils import get_ssid, webcam_server, Singleton
from traffic.consts import SSID_MONOR, SSID_VODAFONE, DROIDCAM, SPACE
from traffic.imports import patch, check_output, Iterable_abc, Iterable_type
from traffic.logging import root_logger
from traffic.utils.lane import LaneDataset,LaneConfig


def setup_function(function):
    Singleton._instances = dict()

def is_droidcam_running():
    out = check_output(["ps", "-o", "command"]).decode("ascii")
    ret = True if DROIDCAM + SPACE in out else False
    root_logger.info(out)
    root_logger.info(ret)
    return ret

@name(LaneDataset.get_min_instance_size, "1", globals())
@patch("traffic.utils.lane.LaneConfig.MIN_INSTANCE_SIZE",1)
@patch("traffic.utils.lane.LaneConfig.MAX_GT_INSTANCES",1)
def _():
    assert LaneDataset.get_min_instance_size([1000,1,2,3,4]) == 4

@name(LaneDataset.get_min_instance_size, "2", globals())
@patch("traffic.utils.lane.LaneConfig.MIN_INSTANCE_SIZE",5)
@patch("traffic.utils.lane.LaneConfig.MAX_GT_INSTANCES",1)
def _():
    assert LaneDataset.get_min_instance_size([1000,1,2,3,4]) == 5

@name(LaneDataset.get_min_instance_size, "3", globals())
@patch("traffic.utils.lane.LaneConfig.MIN_INSTANCE_SIZE",1)
@patch("traffic.utils.lane.LaneConfig.MAX_GT_INSTANCES",4)
def _():
    assert LaneDataset.get_min_instance_size([1000,1,2,3,4]) == 1

@name(LaneDataset.get_min_instance_size, "4", globals())
@patch("traffic.utils.lane.LaneConfig.MIN_INSTANCE_SIZE",1)
@patch("traffic.utils.lane.LaneConfig.MAX_GT_INSTANCES",2)
def _():
    assert LaneDataset.get_min_instance_size([1000,1,2,3,4]) == 3

@name(LaneDataset.get_min_instance_size, "5", globals())
@patch("traffic.utils.lane.LaneConfig.MIN_INSTANCE_SIZE",1)
@patch("traffic.utils.lane.LaneConfig.MAX_GT_INSTANCES",1)
def _():
    assert LaneDataset.get_min_instance_size([1000,1,2,3,3]) == 4



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
