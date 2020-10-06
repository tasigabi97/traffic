from traffic.testing import name
from traffic.utils import get_ssid,webcam_server
from traffic.consts import SSID_MONOR, SSID_VODAFONE
from traffic.imports import patch


@name(get_ssid, "1", globals())
def _():
    ssids = [None, SSID_MONOR, SSID_VODAFONE]
    assert get_ssid() in ssids

@name(webcam_server, "1", globals())
@patch("traffic.utils.get_ssid")
def _(mock_get_ssid):
    mock_get_ssid.return_value=2
    assert mock_get_ssid()==2
