from traffic.testing import name, relative_name
from traffic.camera import Camera, get_cameras, __name__, webcam_server
from traffic.imports import (
    VideoCapture,
    COLOR_RGB2LUV,
    COLOR_RGB2GRAY,
    ndarray,
    raises,
    cvtColor,
    patch,
)
from traffic.logging import root_logger


def setup_function(function):
    Camera.cameras = set()


GOOD_CAMERA_INDEX, BAD_CAMERA_INDEX = None, 9
for i in range(4):
    try:
        cvtColor(VideoCapture(i).read()[1], None)
    except Exception as e:
        root_logger.warning(e)
    else:
        GOOD_CAMERA_INDEX = i
        break
root_logger.info(GOOD_CAMERA_INDEX)


@name(Camera.__new__, "Singleton by id", globals())
def _():
    assert Camera.cameras == set()
    a = Camera(1)
    assert Camera.cameras == {a}
    Camera(1)
    assert Camera.cameras == {a}
    Camera(1, 1)
    assert Camera.cameras == {a}
    c = Camera(2)
    assert Camera.cameras == {a, c}


@name(Camera.__init__, "saves args", globals())
def _():
    a = Camera(0)
    assert a._id == 0 and a._code is None and a._video_capture is None
    a = Camera(1, COLOR_RGB2LUV)
    assert a._id == 1 and a._code == COLOR_RGB2LUV and a._video_capture is None


@name(Camera.__enter__, "error when bad", globals())
def _():
    with raises(ConnectionError):
        with Camera(BAD_CAMERA_INDEX):
            ...


@name(Camera.__enter__, "return self", globals())
def _():
    a = Camera(GOOD_CAMERA_INDEX)
    with a as b:
        assert a is b


@name(Camera.__enter__, "typecheck", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert type(a._video_capture) is VideoCapture and a._video_capture.isOpened()


@name(Camera.__enter__, "check open", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert a._video_capture.isOpened()
    assert not a._video_capture.isOpened()


@name(Camera.__enter__, "check open 2", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert a._video_capture.isOpened()
        with a:
            assert a._video_capture.isOpened()
        assert not a._video_capture.isOpened()
    assert not a._video_capture.isOpened()


@name(Camera.id.fget, "1", globals())
def _():
    assert Camera(0).id == 0


@name(Camera.width.fget, "1", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert a.width == 640


@name(Camera.height.fget, "1", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert a.height == 480


@name(Camera.resolution.fget, "1", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert a.resolution == (640, 480)


@name(Camera.downscale, "1", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert a.resolution == (640, 480)
        a.downscale(2)
        assert a.resolution == (320, 240)
        a.downscale(2)
        assert a.resolution == (160, 120)
        a.downscale(2)
        assert a.resolution == (160, 120)


@name(Camera.img.fget, "basic", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX) as a:
        assert type(a.img) == ndarray and a.img.shape == (480, 640, 4)


@name(Camera.img.fget, "grayscale", globals())
def _():
    with Camera(GOOD_CAMERA_INDEX, COLOR_RGB2GRAY) as a:
        assert type(a.img) == ndarray and a.img.shape == (480, 640)


@name(get_cameras, "basic", globals())
def _():
    with get_cameras() as cameras:
        assert len(cameras) >= 1
        assert cameras[0].img.shape == (480, 640, 4)
        assert cameras[0]._video_capture.isOpened()
    assert not cameras[0]._video_capture.isOpened()


@name(get_cameras, "grayscale", globals())
def _():
    with get_cameras(COLOR_RGB2GRAY) as cameras:
        assert cameras[0].img.shape == (480, 640)


@name(get_cameras, "uses droidcam", globals())
@patch(relative_name(__name__, webcam_server))
def _(mock_webcam_server):
    mock_webcam_server.assert_not_called()
    with get_cameras():
        mock_webcam_server.assert_called_once_with()
