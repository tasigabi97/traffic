from traffic.utils.time import *
from traffic.imports import *
from traffic.logging import root_logger
from traffic.consts import SSID_USED_BY_DROIDCAM, IP_USED_BY_DROIDCAM, DROIDCAM, DEFAULT_DROIDCAM_PORT, PRIVATE
from mrcnn.visualize import random_colors, apply_mask


def load_rgb_array(path: str) -> ndarray:
    """
    Betölt egy képet egy numpy array-ként (sorokszáma,oszlopokszáma,3).
    """
    try:
        img = imread_skimage(path)
    except Exception as e:
        root_logger.warning(path)  # A 100 000-es Lane adatbázisnál volt ami hibás.
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
    """
    Olyan adattag, ami csak a legelső lekérdezéskor készül el.
    Minden további lekérdezéskor az előzőleg eltárolt visszatérési értéket adja vissza.
    Ha nem kérdezik le, nem foglal le erőforrást.
    """

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


class NNInputMaterial:
    """
    A lane_unet háló betanításánál ez felel meg 1-1 jpg/png képnek.
    Azért "material" mert nem közvetlenül ezekből a képekből tanul be,
     viszont ezekből készülnek el az egyes példák (kicsinyítés/forgatás/kivágás stb segítségével).
    """

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
        """
        Ha kell, akkor tárolhatunk valamilyen gyorsítóadatot az egyes képekhez.
        Pl eltároljuk a képek hisztogramját. Ez azért kellhet, hogy később gyorsabban sorba tudjuk rendezni a képeket
        valamilyen tulajdonságuk alapján.
        """
        if not exists(self.attributes_path):
            attributes = self.get_calculated_attributes()
            self.save_attributes_to_json(attributes)
        return self.get_attributes_from_json()


def get_ssid():
    """
    Ez csak arra kell, hogy könnyen lehessen használni a droidcam-ot.
    Visszaadja a wifi nevét vagy None-t.
    """
    try:
        return str(check_output(["iwgetid"])).split('"')[1]
    except CalledProcessError:
        root_logger.warning("This PC is not connected to any Wifi network.")


@contextmanager
def webcam_server():
    """
    Megpróbál csatlakozni a telefonon futó droidcam-hoz.
    Itt át kell írni az SSID_USED_BY_DROIDCAM és IP_USED_BY_DROIDCAM változókat.
    """
    ssid = get_ssid()
    if ssid == SSID_USED_BY_DROIDCAM:
        ip = IP_USED_BY_DROIDCAM
    else:
        root_logger.warning("Droidcam is not working with ssid ({}).".format(ssid))
        # Nekem pl valamiért nem működött az egyik routeremmel.
        yield
        return
    try:
        p = Popen([DROIDCAM, "-v", ip, DEFAULT_DROIDCAM_PORT])
    except FileNotFoundError as e:
        root_logger.warning(e)
        raise FileNotFoundError("Restart the computer and install droidcam again.")
    else:
        yield
        p.kill()


class Singleton(object):
    """
    Ezzel garantáljuk, hogy az ilyen típusú objektumokból mindíg csak max 1 legyen.
    Pl mindig ugyanaz az objektum figyelje a billentyűzet leütést a cv2-es ablakoknál.
    (Kissé túlzás, mert szekvenciális a program.)
    """

    _instances = dict()

    def __new__(this_cls, *args, **kwargs):
        for a_singleton_cls, old_instance in Singleton._instances.items():
            if this_cls is a_singleton_cls:
                return old_instance
        new_instance = super().__new__(this_cls)
        Singleton._instances[this_cls] = new_instance
        return new_instance


class SingletonByIdMeta(type):
    """
    Ezzel garantáljuk, hogy az ilyen típusú objektumokból mindíg csak max 1 legyen egy adott tulajdonsággal.
    Olvashatóbbá teszi a kódot, mivel el lehet vele kerülni a számmal indexelést, és azt, hogy mindig paraméternek
    kelljen adni az ilyen objektumokat.
    Pl ilyen objektumok a képadatbázis kategóriái.
    """

    _get_id_name = "get_id"
    _id_name = "id"

    def __new__(
        cls: "SingletonByIdMeta",
        new_cls_name: str,
        new_cls_bases: tuple,
        class_definition_dict: dict,
    ):
        """
        Már az objektum létrehozása előtt tudni kell, hogy készült-e már ugyanolyan (ezért staticmethod kell legyen).
        Legyen ugyanolyan a get_id és __init__ fejléce.
        """
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

    # automatically called before __new__ return
    def __init__(new_cls: type, new_cls_name: str, new_cls_bases: tuple, new_cls_dict: dict):
        super().__init__(new_cls_name, new_cls_bases, new_cls_dict)
        new_cls._instances = dict()
        original_init = new_cls.__init__

        def __new__(cls, *args, **kwargs):
            """
            Konstruálás előtt megnézzük készült-e már objektum ilyen paraméterrel.
            Mivel mindig visszaad egy új/régi objektumot (nem None-t) ezért garantáltan meghívódik az __init__.
            """
            id = getattr(cls, SingletonByIdMeta._get_id_name)(*args, **kwargs)
            if id in cls._instances.keys():
                return cls._instances[id]
            new_instance = super(cls, cls).__new__(cls)
            cls._instances[id] = new_instance
            return new_instance

        def __init__(self, *args, **kwargs):
            """
            Mindig meghívódik konstruáláskor, ezért le kell ellenőrizni meg volt-e már hívva.
            """
            if SingletonByIdMeta._id_name in self.__dict__.keys():
                return
            id = getattr(self, SingletonByIdMeta._get_id_name)(*args, **kwargs)
            setattr(self, SingletonByIdMeta._id_name, id)
            original_init(self, *args, **kwargs)

        new_cls.__new__, new_cls.__init__ = __new__, __init__

    def __iter__(new_cls) -> Iterable_type:
        """
        Pl végig lehet haladni az összes kategórián.
        """
        return iter(new_cls._instances.values())

    def clear(new_cls):
        """
        Újra lehet készíteni az objektumokat.
        """
        new_cls._instances.clear()

    def __getitem__(new_cls, item):
        """
        Olvashatóbb indexelésért.
        """
        for i in new_cls:
            if i == item:
                return i
        raise IndexError("object with index ({}) not found".format(item))

    def __len__(new_cls):
        """
        Pl le lehet kérdezni hány darab kategória van.
        """
        return len(new_cls._instances)
