from cv2 import VideoCapture, cvtColor, imshow, waitKey, destroyAllWindows
from typing import Tuple


class Camera(object):
    def __init__(self, id: int, code=None):
        self._id, self._code = id, code
        self._video_capture = VideoCapture(self.id)
        if self.resolution == (0, 0):
            raise ConnectionError(f"Can't connect to camera ({self.id})")

    @property
    def img(self):
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
