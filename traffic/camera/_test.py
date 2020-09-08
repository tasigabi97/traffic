from larning.testing import name
from traffic.camera import Camera
from cv2 import VideoCapture, COLOR_RGB2LUV
from pytest import raises


@name(Camera.__init__, 1, globals())
def _():
    with raises(ConnectionError):
        Camera(100)
    a = Camera(0)
    assert a._id == 0 and a._code is None
    assert type(a._video_capture) == VideoCapture
    # Camera(0,COLOR_RGB2LUV)
    # assert a._id == 0 and a._code == COLOR_RGB2LUV
    assert type(a._video_capture) == VideoCapture


@name(Camera.id.fget, 1, globals())
def _():
    assert Camera(0).id == 0


@name(Camera.width.fget, 1, globals())
def _():
    assert Camera(0).width == 640


@name(Camera.height.fget, 1, globals())
def _():
    assert Camera(0).height == 480


@name(Camera.resolution.fget, 1, globals())
def _():
    assert Camera(0).resolution == (640, 480)


@name(Camera.downscale, 2, globals())
def _():
    a = Camera(0)
    assert a.resolution == (640, 480)
    a.downscale(2)
    assert a.resolution == (320, 240)
    a.downscale(2)
    assert a.resolution == (160, 120)
    a.downscale(2)
    assert a.resolution == (160, 120)
