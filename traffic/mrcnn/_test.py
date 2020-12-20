from traffic.testing import *
from traffic.utils import *
from traffic.consts import *
from traffic.imports import *
from traffic.logging import *
from traffic.mrcnn import *


def setup_function(function):
    MrcnnCategory.clear()


def teardown_function(function):
    patch.stopall()


@name(MrcnnCategory.__new__, "1", globals())
def _():
    assert len(MrcnnCategory) == 0
    d = {"name": "x", "rgb_tuple": (1, 1, 1), "id": 1}
    m_get_id = patch.object(MrcnnCategory, "get_id", new=MagicMock(return_value=sentinel.id)).start()
    x = MrcnnCategory.__new__(MrcnnCategory)
    m_get_id.assert_called_once_with()
    assert len(MrcnnCategory) == 1 and MrcnnCategory._instances[sentinel.id] is x
    x2 = MrcnnCategory.__new__(MrcnnCategory)
    assert len(MrcnnCategory) == 1 and MrcnnCategory._instances[sentinel.id] is x and x is x2


@name(MrcnnCategory.__init__, "1", globals())
def _():
    x = object.__new__(MrcnnCategory)
    x.__init__(sentinel.name, sentinel.index, sentinel.min_confidence, sentinel.rgb_tuple)
    assert x.name == sentinel.name
    assert x.index == sentinel.index
    assert x.rgb_tuple == sentinel.rgb_tuple
    assert x.min_confidence == sentinel.min_confidence


@name(MrcnnCategory.__eq__, "1", globals())
def _():
    x = object.__new__(MrcnnCategory)
    x.name = sentinel.name
    x.index = sentinel.index
    assert x != 1
    assert x != sentinel.name2
    assert x == sentinel.name
    assert x == sentinel.index


@name(MrcnnCategory.important.fget, "1", globals())
def _():
    x = object.__new__(MrcnnCategory)
    x.name = "Alma"
    assert x.important
    x.name = "alma"
    assert not x.important


@name(DetectedObject.__init__, "1", globals())
def _():
    x = object.__new__(DetectedObject)
    x.__init__(
        sentinel.category,
        sentinel.confidence,
        sentinel.y1,
        sentinel.x1,
        sentinel.y2,
        sentinel.x2,
        sentinel.mask_boolean,
    )
    assert x.category is sentinel.category
    assert x.x1 is sentinel.x1
    assert x.x2 is sentinel.x2
    assert x.y1 is sentinel.y1
    assert x.y2 is sentinel.y2
    assert x.mask_boolean is sentinel.mask_boolean
    assert x.confidence is sentinel.confidence


@name(DetectedObject.name.fget, "1", globals())
def _():
    x = object.__new__(DetectedObject)
    m_category = MagicMock()
    m_category.name = sentinel.name
    x.category = m_category
    assert x.name is sentinel.name


@name(DetectedObject.rgb_tuple.fget, "1", globals())
def _():
    x = object.__new__(DetectedObject)
    m_rgb_tuple = MagicMock()
    m_rgb_tuple.name = sentinel.rgb_tuple
    x.category = m_rgb_tuple
    assert x.name is sentinel.rgb_tuple


@name(DetectedObject.width.fget, "1", globals())
def _():
    x = object.__new__(DetectedObject)
    x.x2 = 100
    x.x1 = 1
    assert x.width == 99


@name(DetectedObject.height.fget, "1", globals())
def _():
    x = object.__new__(DetectedObject)
    x.y2 = 100
    x.y1 = 1
    assert x.height == 99


@name(Mrcnn.model.fget, "1", globals())
def _():
    m_load_weights = patch.object(MaskRCNN, "load_weights", new=MagicMock()).start()
    x = object.__new__(Mrcnn)
    y = x.model
    assert type(y) is MaskRCNN
    assert y.mode == "inference"
    assert y.model_dir == ""
    assert y.config.IMAGE_MAX_DIM == 640
    assert y.config.IMAGE_MIN_DIM == 480
    assert y.config.BATCH_SIZE == 1
    assert list(y.config.IMAGE_SHAPE) == [512, 640, 3]
    m_load_weights.assert_called_once_with("/traffic/mrcnn/mask_rcnn_coco.h5", by_name=True)


@name(Mrcnn.get_prediction, "1", globals())
def _():
    d = dict()
    d["class_ids"] = zeros((3,), dtype=int32)
    d["scores"] = zeros((3,), dtype=float32)
    d["rois"] = zeros((3, 4), dtype=int32)
    d["masks"] = zeros((480, 640, 3), dtype=bool_np)
    m_MrcnnCategory = patch(
        "traffic.mrcnn.MrcnnCategory", new=MagicMock(__getitem__=MagicMock(return_value=sentinel.category))
    ).start()
    m_DetectedObject = patch(
        "traffic.mrcnn.DetectedObject", new=MagicMock(return_value=sentinel.detected_object)
    ).start()
    m_detect = MagicMock(return_value=(d, sentinel.second))
    m_model = patch.object(Mrcnn, "model", new=PropertyMock(return_value=MagicMock(detect=m_detect))).start()
    x = object.__new__(Mrcnn)
    y = x.get_prediction(sentinel.rgb_array)
    assert y == [sentinel.detected_object, sentinel.detected_object, sentinel.detected_object]
    first_call = m_DetectedObject.mock_calls[0]
    name, args, kwargs = first_call
    assert args[0] is sentinel.category
    assert args[1].shape == () and args[1].dtype.name == "float32"
    for i in range(2, 6):
        assert args[i].shape == () and args[i].dtype.name == "int32"
    assert args[6].shape == (480, 640) and args[6].dtype.name == "bool"
    first_call = m_MrcnnCategory.mock_calls[0]
    name, args, kwargs = first_call
    assert name == "__getitem__" and args == (0,)
    m_detect.assert_called_once_with([sentinel.rgb_array], verbose=0)
