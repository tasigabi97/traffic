from traffic.imports import check_output, CalledProcessError, contextmanager, Popen
from traffic.logging import root_logger
from traffic.consts import SSID_VODAFONE, IP_VODAFONE, DROIDCAM, DROIDCAM_PORT


def get_ssid():
    try:
        return str(check_output(["iwgetid"])).split('"')[1]
    except CalledProcessError:
        root_logger.warning("This PC is not connected to any Wifi network.")


@contextmanager
def webcam_server():
    ssid = get_ssid()
    if ssid == SSID_VODAFONE:
        ip = IP_VODAFONE
    else:
        root_logger.warning("Droidcam is not working with ssid ({}).".format(ssid))
        yield
        return
    try:
        p = Popen([DROIDCAM, "-v", ip, DROIDCAM_PORT])
    except FileNotFoundError as e:
        root_logger.warning(e)
        raise FileNotFoundError("Restart the computer and install droidcam again.")
    else:
        yield
        p.kill()


class Singleton(object):
    _instances = dict()

    def __new__(this_cls, *args, **kwargs):
        for a_singleton_cls, old_instance in Singleton._instances.items():
            if this_cls is a_singleton_cls:
                return old_instance
        new_instance = super().__new__(this_cls)
        Singleton._instances[this_cls] = new_instance
        return new_instance
