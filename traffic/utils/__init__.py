from traffic.utils.time import *
from traffic.imports import *
from traffic.logging import root_logger
from traffic.consts import SSID_VODAFONE, IP_VODAFONE, DROIDCAM, DROIDCAM_PORT, PRIVATE
from mrcnn.visualize import random_colors, apply_mask


def load_rgb_array(path: str) -> ndarray:
    try:
        img = imread_skimage(path)
    except Exception as e:
        root_logger.warning(path)
        raise e
    img = array_np(img, dtype=uint8)
    if img.shape[-1] == 3:
        return img
    elif img.shape[-1] == 4:
        return img[..., :3]
    else:
        raise IndexError(img.shape)


def show_array(array: ndarray):
    imshow_mat(array)
    show()


def virtual_proxy_property(func: Callable) -> property:
    @property
    @wraps(func)
    def new_func(self):
        ATTR_NAME = PRIVATE + func.__name__
        if hasattr(self, ATTR_NAME):
            return getattr(self, ATTR_NAME)
        ret = func(self)
        setattr(self, ATTR_NAME, ret)
        return ret

    return new_func


def save_dict_to_json(path: str, d: dict):
    with open(path, "w") as outfile:
        dump_json(d, outfile)


def get_dict_from_json(path: str) -> dict:
    with open(path) as json_file:
        d = load_json(json_file)
    return d


class NNInputSource:
    def __init__(self, path: str):
        self.path = path

    @property
    def data(self) -> ndarray:
        return load_rgb_array(self.path)

    def visualize_data(self):
        show_array(self.data)

    def get_an_input(self, *args, **kwargs) -> ndarray:
        raise NotImplemented()

    def visualize_an_input(self, *args, **kwargs):
        show_array(self.get_an_input(*args, **kwargs))

    @property
    def attributes_path(self) -> str:
        return self.path + ".json"

    def get_calculated_attributes(self):
        raise NotImplemented()

    def save_attributes_to_json(self, attributes: dict):
        save_dict_to_json(self.attributes_path, attributes)

    def get_attributes_from_json(self) -> dict:
        return get_dict_from_json(self.attributes_path)

    @property
    def attributes(self) -> dict:
        if not exists(self.attributes_path):
            attributes = self.get_calculated_attributes()
            self.save_attributes_to_json(attributes)
        return self.get_attributes_from_json()


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


class SingletonByIdMeta(type):
    _get_id_name = "get_id"
    _id_name = "id"

    def __new__(
        cls: "SingletonByIdMeta",
        new_cls_name: str,
        new_cls_bases: tuple,
        class_definition_dict: dict,
    ):
        new_cls = super().__new__(cls, new_cls_name, new_cls_bases, class_definition_dict)
        if not hasattr(new_cls, cls._get_id_name):
            raise KeyError("Please define a {} as a staticmethod".format(cls._get_id_name))
        init_params = [par for par in signature(getattr(new_cls, "__init__")).parameters.values()]
        init_params = init_params[1:]
        get_id_params = [par for par in signature(getattr(new_cls, cls._get_id_name)).parameters.values()]
        if init_params != get_id_params:
            root_logger.error(init_params)
            root_logger.error(get_id_params)
            raise NameError("Please define {} with the same signature as __init__".format(cls._get_id_name))
        return new_cls  # __init__()

    # called before __new__ return automatically
    def __init__(new_cls: type, new_cls_name: str, new_cls_bases: tuple, new_cls_dict: dict):
        super().__init__(new_cls_name, new_cls_bases, new_cls_dict)
        new_cls._instances = dict()
        original_init = new_cls.__init__

        def __new__(cls, *args, **kwargs):
            id = getattr(cls, SingletonByIdMeta._get_id_name)(*args, **kwargs)
            if id in cls._instances.keys():
                return cls._instances[id]
            new_instance = super(cls, cls).__new__(cls)
            cls._instances[id] = new_instance
            return new_instance

        def __init__(self, *args, **kwargs):
            if SingletonByIdMeta._id_name in self.__dict__.keys():
                return
            id = getattr(self, SingletonByIdMeta._get_id_name)(*args, **kwargs)
            setattr(self, SingletonByIdMeta._id_name, id)
            original_init(self, *args, **kwargs)

        new_cls.__new__, new_cls.__init__ = __new__, __init__

    def __iter__(new_cls) -> Iterable_type:
        return iter(new_cls._instances.values())

    def clear(new_cls):
        new_cls._instances.clear()

    def __getitem__(new_cls, item):
        for i in new_cls:
            if i == item:
                return i
        raise IndexError("object with index ({}) not found".format(item))

    def __len__(new_cls):
        return len(new_cls._instances)
