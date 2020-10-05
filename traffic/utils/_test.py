from traffic.testing import name
from traffic.utils import get_ssid
from traffic.consts import SSID_MONOR, SSID_VODAFONE


@name(get_ssid, "1", globals())
def _():
    ssids = [None, SSID_MONOR, SSID_VODAFONE]
    assert get_ssid() in ssids
