from traffic.imports import (
    VideoCapture,
    cvtColor,
    Tuple,
    List,
    contextmanager,
    ndarray,
    cycle,
    imshow_cv2,
    destroyAllWindows,
    CAP_PROP_BUFFERSIZE,
)
from traffic.logging import root_logger
from traffic.utils import webcam_server
from traffic.cv2_input import cv2_input


class Camera:
    """
    Ez reprezentál egy kamera fájlt, pl /dev/video0 .
    Attól még, hogy létezik a fájl, nem biztos, hogy van hozzá fizikai kamera.
    """

    cameras = set()

    def __new__(cls, id, *args, **kwargs):
        """
        Ez azért van, hogy garantáltan egy objektum tartozzon egy fájlhoz.
        """
        for old_instance in cls.cameras:
            if old_instance.id == id:
                return old_instance
        new_instance = super().__new__(cls)
        cls.cameras.add(new_instance)
        return new_instance

    def __init__(self, id: int, code: int = None):
        """
        Parameters
        ----------
        id : pl 0 esetén a /dev/video0 fájl kerül megnyitásra.
        code : Ez dönti el, hogy milyen legyen a kamera színe.
            Alap esetben nincs rajta filter, de lehetne fekete-fehér stb...
        """
        self._id, self._code, self._video_capture = id, code, None

    def __enter__(self) -> "Camera":
        """
        Megnyitja a kamera fájlt, és megpróbálja használni a vele reprezentált
        fizikai kamerát. Ha le tud kérdezni egy képet, akkor a fájl egy működő fizikai kamerát reprezentál.
        """
        if self._video_capture is None:
            self._video_capture = VideoCapture(self.id)
            self._video_capture.set(CAP_PROP_BUFFERSIZE, 1)  # Kisebbre állítjuk a buffert,
            # hogy ne legyen nagy a késleltetés. Jobb lenne, ha egyáltalán nem lenne buffer, de nem sikerült
            # rájönnöm, hogyan lehetne kikapcsolni.
        if not self._video_capture.isOpened():  # Ne nyissuk meg újra a fájlt, ha már meg volt nyitva.
            self._video_capture.open(self.id)
        try:
            self.img
        except:
            self.__exit__(None, None, None)
            raise ConnectionError("Can't connect to camera ({})".format(self.id))
        else:
            root_logger.info("Connected to camera ({})".format(self.id))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Bezárja a kamera fájlt, ha meg volt nyitva.
        """
        if self._video_capture.isOpened():
            self._video_capture.release()

    @property
    def img(self) -> ndarray:
        """
        Visszaad 1db kameraképet. Nem feltétlenül aktuális a buffer miatt.
        """
        return cvtColor(self._video_capture.read()[1], self._code)

    @property
    def matplotlib_img(self) -> ndarray:
        """
        Felcseréli a színcsatornákat RGB-re.
        """
        return self.img[:, :, 2::-1]

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return "Camera: {}".format(self.id)

    @property
    def width(self) -> int:
        """
        pl 640
        """
        return int(self._video_capture.get(3))

    @property
    def height(self) -> int:
        """
        pl 480
        """
        return int(self._video_capture.get(4))

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        pl (640,480)
        """
        return self.width, self.height

    def downscale(self, factor: int):
        """
        Lejjeb veszi a kamera felbontását. Elég korlátolt a függvény,
         nem lehet sokkal kisebb a kép, vagy kicsivel kiseb sem, de fele/negyed akkora még jó.
          Igazából nem kellett használni,
          mivel alapból max (640,480)-as képeket ad vissza nagyobb fizikai felbontás esetén is.
        (640,480)-nál kisebb képekkel, meg amúgy sem lenne érdemes dolgozni.

        Parameters
        ----------
        factor: pl 2-nél (640,480)->(320,240)
        """
        self._video_capture.set(3, self.width // factor)
        self._video_capture.set(4, self.height // factor)


@contextmanager
def get_cameras(code: int = None) -> List[Camera]:
    """
    Megpróbálja elérni sorban az első N kamera képét, és visszaadja azokat a kamerákat,
    amik működtek.
    """
    with webcam_server():
        N = 4
        cameras = []
        for id in range(N):
            camera = Camera(id, code)
            try:
                camera.__enter__()
            except ConnectionError as e:
                root_logger.warning(e)
            else:
                cameras.append(camera)
        yield cameras
        for camera in reversed(cameras):
            camera.__exit__(None, None, None)


# todo ne is lehessen többször megnyitni egy kamerát
@contextmanager
def choose_camera(code: int = None) -> Camera:
    """
    Feldob minden egyes működő kamerához egy ablakot.
    Visszaadja azt az egy kamerát, amit kiválasztottunk egy számleütés segítségével.
    Nem a konzolba kell beírni a számot enterrel.
    """
    with get_cameras(code) as cameras:
        cv2_input.wait_keys = [camera.id for camera in cameras]
        root_logger.info("Waitkeys={}".format(cv2_input.wait_keys))
        for camera in cycle(cameras):
            imshow_cv2("{}-> ({})".format(choose_camera.__name__, camera.name), camera.img)
            key = cv2_input.pressed_key
            if key is not None:
                destroyAllWindows()
                yield cameras[cv2_input.wait_keys.index(key)]
                destroyAllWindows()
                return
