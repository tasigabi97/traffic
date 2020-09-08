from larning.testing import name
from traffic.camera import Camera, get_cameras
from cv2 import VideoCapture, COLOR_RGB2LUV, COLOR_RGB2GRAY
from pytest import raises
from unittest.mock import MagicMock, patch
from numpy import ndarray


def teardown_function(function):
    Camera.cameras = set()


@name(Camera.__new__, 1, globals())
def _():
    assert Camera.cameras == set()
    a = Camera(1)
    assert Camera.cameras == {a}
    b = Camera(1)
    assert Camera.cameras == {a}
    b = Camera(1, 1)
    assert Camera.cameras == {a}
    c = Camera(2)
    assert Camera.cameras == {a, c}


@name(Camera.__init__, 1, globals())
def _():
    a = Camera(0)
    assert a._id == 0 and a._code is None and a._video_capture is None
    a = Camera(1, COLOR_RGB2LUV)
    assert a._id == 1 and a._code == COLOR_RGB2LUV and a._video_capture is None


@name(Camera.__enter__, 1, globals())
def _():
    with raises(ConnectionError):
        with Camera(100):
            ...
    a = Camera(0)
    with a as b:
        assert a is b and type(a._video_capture) is VideoCapture and a._video_capture.isOpened()
    assert not a._video_capture.isOpened()
    with a as b:
        assert a._video_capture.isOpened()

    with a as b:
        assert a._video_capture.isOpened()
        with a as b:
            assert a._video_capture.isOpened()
        assert not a._video_capture.isOpened()
    assert not a._video_capture.isOpened()


@name(Camera.id.fget, 1, globals())
def _():
    assert Camera(0).id == 0


@name(Camera.width.fget, 1, globals())
def _():
    with Camera(0) as a:
        assert a.width == 640


@name(Camera.height.fget, 1, globals())
def _():
    with Camera(0) as a:
        assert a.height == 480


@name(Camera.resolution.fget, 1, globals())
def _():
    with Camera(0) as a:
        assert a.resolution == (640, 480)


@name(Camera.downscale, 2, globals())
def _():
    with Camera(0) as a:
        assert a.resolution == (640, 480)
        a.downscale(2)
        assert a.resolution == (320, 240)
        a.downscale(2)
        assert a.resolution == (160, 120)
        a.downscale(2)
        assert a.resolution == (160, 120)


@name(Camera.img.fget, 1, globals())
def _():
    with Camera(0) as a:
        assert type(a.img) == ndarray and a.img.shape == (480, 640, 4)
        a.downscale(2)
        assert a.img.shape == (240, 320, 4)


@name(Camera.img.fget, 2, globals())
def _():
    with Camera(0, COLOR_RGB2GRAY) as a:
        assert type(a.img) == ndarray and a.img.shape == (480, 640)


@name(get_cameras, 1, globals())
def _():
    with get_cameras() as cameras:
        assert len(cameras) >= 1
        assert cameras[0].img.shape == (480, 640, 4)
    assert not cameras[0]._video_capture.isOpened()
