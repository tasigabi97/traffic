from traffic.testing import name, absolute_name
from traffic.utils import get_ssid, webcam_server
from traffic.consts import SSID_MONOR, SSID_VODAFONE, DROIDCAM, SPACE
from traffic.imports import patch, check_output, CalledProcessError
from traffic.logging import root_logger


def is_droidcam_running():
    out = check_output(["ps", "-o", "command"]).decode("ascii")
    ret = True if DROIDCAM + SPACE in out else False
    root_logger.info(out)
    root_logger.info(ret)
    return ret


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
