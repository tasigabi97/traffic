from cv2 import VideoCapture, cvtColor, imshow, waitKey, destroyAllWindows
from typing import Tuple, List
from contextlib import contextmanager
from numpy import ndarray


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
        if not self._video_capture.isOpened():
            self._video_capture.open(self.id)
        if self.resolution == (0, 0):
            self.__exit__(None, None, None)
            raise ConnectionError(f"Can't connect to camera ({self.id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._video_capture.isOpened():
            self._video_capture.release()

    @property
    def img(self) -> ndarray:
        return cvtColor(self._video_capture.read()[1], self._code)

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return f"Camera: {self.id}"

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
    N = 4
    cameras = []
    for id in range(N):
        camera = Camera(id, code)
        try:
            camera.__enter__()
        except ConnectionError as e:
            print(e)
        else:
            cameras.append(camera)
    yield cameras
    for camera in reversed(cameras):
        camera.__exit__(None, None, None)
