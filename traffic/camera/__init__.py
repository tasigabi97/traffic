from traffic.imports import VideoCapture, cvtColor, Tuple, List, contextmanager, ndarray, cycle, imshow_cv2, destroyAllWindows, CAP_PROP_BUFFERSIZE
from traffic.logging import root_logger
from traffic.utils import webcam_server
from traffic.globals import g


class Camera(object):
    cameras = set()

    def __new__(cls, id, *args, **kwargs):
        for old_instance in cls.cameras:
            if old_instance.id == id:
                return old_instance
        new_instance = super().__new__(cls)
        cls.cameras.add(new_instance)
        return new_instance

    def __init__(self, id: int, code: int = None):
        self._id, self._code, self._video_capture = id, code, None

    def __enter__(self) -> "Camera":
        if self._video_capture is None:
            self._video_capture = VideoCapture(self.id)
            self._video_capture.set(CAP_PROP_BUFFERSIZE, 1)
        if not self._video_capture.isOpened():
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
        if self._video_capture.isOpened():
            self._video_capture.release()

    @property
    def img(self) -> ndarray:
        return cvtColor(self._video_capture.read()[1], self._code)

    @property
    def matplotlib_img(self) -> ndarray:
        return self.img[:, :, 2::-1]

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return "Camera: {}".format(self.id)

    @property
    def width(self) -> int:
        return int(self._video_capture.get(3))

    @property
    def height(self) -> int:
        return int(self._video_capture.get(4))

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def downscale(self, factor: int):
        self._video_capture.set(3, self.width // factor)
        self._video_capture.set(4, self.height // factor)


@contextmanager
def get_cameras(code: int = None) -> List[Camera]:
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
    with get_cameras(code) as cameras:
        g.wait_keys = [camera.id for camera in cameras]
        root_logger.info("Waitkeys={}".format(g.wait_keys))
        for camera in cycle(cameras):
            imshow_cv2("{}-> ({})".format(choose_camera.__name__, camera.name), camera.img)
            key = g.pressed_key
            if key is not None:
                destroyAllWindows()
                yield cameras[g.wait_keys.index(key)]
                destroyAllWindows()
                return
