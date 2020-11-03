from traffic.testing import *
from traffic.imports import *
from traffic.utils.lane_unet import *

EXAMPLE_MASK_PATH = "/traffic/lane/Labels_road02/Label/Record028/Camera 5/170927_071400339_Camera_5_bin.png"
EXAMPLE_IMG_PATH = "/traffic/lane/ColorImage_road02/ColorImage/Record028/Camera 5/170927_071400339_Camera_5.jpg"
IMG = array_np([[[0, 0, 0], [0, 1, 2]]], dtype=uint8)
assert IMG.shape == (1, 2, 3)


def setup_function(function):
    Singleton._instances = dict()
    Color.clear()
    Category.clear()
    assert len(Color) == 0 and len(Category) == 0


def teardown_function(function):
    patch.stopall()


# module test
assert len(Category) == 36
assert sum(len(category) for category in Category) == 38 == len(Color)
# module test
if False:
    [print(str(c)) for c in Unet.test_DB.one_hot_coder.categories]
    input()
    categories_path = "/traffic/categories"
    try:
        mkdir(categories_path)
    except Exception as e:
        root_logger.warning(e)
    for category in Category:
        category.path = join_path(categories_path, category.name)
        category.mask_ids = []
        try:
            mkdir(category.path)
        except Exception as e:
            root_logger.warning(e)
    mask_ids = list(range(len(Unet.test_DB._mask_paths)))
    shuffle(mask_ids)
    s = zeros((480 * 3, 640, 3), dtype=float64)
    for mask_id in mask_ids:
        print(mask_id)
        img, one_hot = Unet.test_DB.get_train_input(mask_id)
        for cat_i, cat in enumerate(Unet.test_DB.one_hot_coder.categories):
            if sum_np(one_hot[:, :, cat_i]) >= 150:
                cat.mask_ids.append(mask_id)
                if len(cat.mask_ids) <= 29:
                    save_path = join_path(cat.path, str(mask_id) + ".png")
                    print(save_path)
                    print(cat.mask_ids)
                    s[:480, :, 0] = one_hot[:, :, cat_i]
                    s[480:960, :, :] = img
                    s[960:, :, :] = img
                    s[480:960, :, 0] = one_hot[:, :, cat_i]
                    imwrite_ima(save_path, s)
                    if False:
                        imshow_mat(s)
                        show()


@name(lambda: None, "1", globals())
def _():
    class A:
        def __init__(self, a):
            return

        def method(self):
            return 5

    m_init = patch.object(A, "__init__", new=MagicMock(return_value=None)).start()
    o = A()


@name(lambda: None, "2", globals())
def _():
    class A:
        def __init__(self):
            raise Exception

        def method(self):
            return 5

    with raises(Exception):
        a = A()
    a = A.__new__(A)
    assert a.method() == 5
    with raises(Exception):
        a.__init__()


@name(Color.__new__, "1", globals())
def _():
    d = {"name": "x", "rgb_tuple": (1, 1, 1), "id": 1}
    m_get_id = patch.object(Color, "get_id", new=MagicMock(return_value=d["id"])).start()
    x = Color.__new__(Color)
    m_get_id.assert_called_once_with()
    assert len(Color) == 1 and Color._instances[d["id"]] is x
    x2 = Color.__new__(Color)
    assert len(Color) == 1 and Color._instances[d["id"]] is x and x is x2


@name(Color.__init__, "1", globals())
def _():
    d = {"name": "x", "rgb_tuple": (1, 1, 1), "id": 1}
    m_get_id = patch.object(Color, "get_id", new=MagicMock(return_value=d["id"])).start()
    x = Color.__new__(Color)
    for k in d.keys():
        assert not hasattr(x, k)
    x.__init__(d["name"], d["rgb_tuple"])
    assert m_get_id.mock_calls == [call_mock(), call_mock(d["name"], d["rgb_tuple"])]
    for k, v in d.items():
        assert getattr(x, k) == v


@name(Color.__eq__, "1", globals())
def _():
    d = {"name": "x", "rgb_tuple": (1, 1, 1), "id": 1}
    m_get_id = patch.object(Color, "get_id", new=MagicMock(return_value=d["id"])).start()
    x = Color.__new__(Color)
    x.name = d["name"]
    x.rgb_tuple = d["rgb_tuple"]
    assert x == d["name"] and x == d["rgb_tuple"]


@name(Category.__init__, "11", globals())
def _():
    d = {"name": "x", "colors": 2, "id": 1}
    m_get_id = patch.object(Category, "get_id", new=MagicMock(return_value=d["id"])).start()
    x = Category.__new__(Category)
    for k in d.keys():
        assert not hasattr(x, k)
    x.__init__(d["name"], d["colors"])
    assert m_get_id.mock_calls == [call_mock(), call_mock(d["name"], d["colors"])]
    assert getattr(x, "name") == d["name"]
    assert getattr(x, "id") == d["id"]
    assert d["colors"] in getattr(x, "colors")


@name(Category.__eq__, "1", globals())
def _():
    m_get_id = patch.object(Category, "get_id", new=MagicMock(return_value=1)).start()
    x = Category.__new__(Category)
    m_name = MagicMock()
    m_color1 = MagicMock()
    m_color2 = MagicMock()
    m_colors = MagicMock()
    m_name.__eq__.return_value = False
    m_color1.__eq__.return_value = False
    m_color2.__eq__.return_value = False
    m_colors.__iter__.return_value = [m_color1, m_color2]
    x.name = m_name
    x.colors = m_colors
    assert x != None
    m_name.__eq__.return_value = True
    assert x == None
    m_name.__eq__.return_value = False
    m_color1.__eq__.return_value = True
    assert x == None
    m_color1.__eq__.return_value = False
    m_color2.__eq__.return_value = True
    assert x == None
    m_color2.__eq__.return_value = False
    assert x != None


@name(OneHot.get_rgb_id, "1", globals())
def _():
    assert OneHot.get_rgb_id((0, 0, 0)) == 0
    assert OneHot.get_rgb_id((1, 1, 1)) == 65793


@name(OneHot.get_rgb_tuple, "1", globals())
def _():
    assert OneHot.get_rgb_tuple(65793) == (1, 1, 1)
    assert OneHot.get_rgb_tuple(0) == (0, 0, 0)
    assert OneHot.get_rgb_tuple(1) == (0, 0, 1)
    assert OneHot.get_rgb_tuple(256) == (0, 1, 0)


@name(OneHot.get_rgb_tuple, "inverse", globals())
def _():
    color_ids = [256 * 256 * 256, 0, 1, 255, 256, 999]
    for color_id in color_ids:
        assert OneHot.get_rgb_id(OneHot.get_rgb_tuple(color_id)) == color_id


@name(OneHot.__init__, "1", globals())
def _():
    cat1 = MagicMock()
    cat2 = MagicMock()
    color1 = MagicMock()
    color2 = MagicMock()
    color3 = MagicMock()
    cat1.name = "cat1"
    cat2.name = "cat2"
    cat1.colors.__iter__.return_value = [color1, color2]
    cat2.colors.__iter__.return_value = [color3]
    color1.rgb_tuple = (0, 0, 1)
    color2.rgb_tuple = (0, 0, 0)
    color3.rgb_tuple = (1, 0, 1)
    o = OneHot(sentinel.a, sentinel.b, (cat2, cat1))
    assert o.row_number is sentinel.a and o.col_number is sentinel.b
    assert type(o.colors) is list is type(o.categories)
    assert o.categories[0] is cat1 and o.categories[1] is cat2
    assert o.colors[0] is color2
    assert o.colors[1] is color1
    assert o.colors[2] is color3


@name(OneHot.uint32_img_container.fget, "", globals())
def _():
    x = object.__new__(OneHot)
    x.row_number = 1
    x.col_number = 1
    assert x.uint32_img_container.dtype.name == "uint32"
    assert x.uint32_img_container.shape == (1, 1, 3)
    with raises(AttributeError):
        x.uint32_img_container = None


@name(OneHot.rgb_id_container.fget, "", globals())
def _():
    x = object.__new__(OneHot)
    x.row_number = 1
    x.col_number = 1
    m_colors = MagicMock()
    m_colors.__len__.return_value = 4
    x.colors = m_colors
    assert x.rgb_id_container.dtype.name == "uint32"
    assert x.rgb_id_container.shape == (1, 1, 4)
    with raises(AttributeError):
        x.rgb_id_container = None


@name(OneHot.RGB_ID_SUBTRAHEND.fget, "", globals())
def _():
    m_get_rgb_id = patch.object(OneHot, "get_rgb_id", new=MagicMock(return_value=9)).start()
    x = object.__new__(OneHot)
    x.row_number = 1
    x.col_number = 1
    m_colors = MagicMock()
    m_colors.__len__.return_value = 2
    m_colors.__iter__.return_value = [MagicMock(rgb_tuple=sentinel.a), MagicMock(rgb_tuple=sentinel.b)]
    x.colors = m_colors
    assert x.RGB_ID_SUBTRAHEND.dtype.name == "uint32"
    assert x.RGB_ID_SUBTRAHEND.shape == (1, 1, 2)
    assert unique_np(x.RGB_ID_SUBTRAHEND) == [9]
    with raises(AttributeError):
        x.RGB_ID_SUBTRAHEND = None
    assert m_get_rgb_id.mock_calls == [call_mock(sentinel.a), call_mock(sentinel.b)]


@name(OneHot.bool_container.fget, "", globals())
def _():
    x = object.__new__(OneHot)
    x.row_number = 1
    x.col_number = 1
    m_colors = MagicMock()
    m_colors.__len__.return_value = 4
    x.colors = m_colors
    assert x.bool_container.dtype.name == "uint8"
    assert x.bool_container.shape == (1, 1, 4)
    with raises(AttributeError):
        x.bool_container = None


@name(OneHot.one_hot_container.fget, "", globals())
def _():
    x = object.__new__(OneHot)
    x.row_number = 1
    x.col_number = 1
    m_categories = MagicMock()
    m_categories.__len__.return_value = 4
    x.categories = m_categories
    assert x.one_hot_container.dtype.name == "float64"
    assert x.one_hot_container.shape == (1, 1, 4)
    with raises(AttributeError):
        x.one_hot_container = None


@name(OneHot.get_encoded, "", globals())
def _():
    img = array_np([[[0, 0, 0], [255, 1, 1]]], dtype=uint8)
    assert img.shape == (1, 2, 3)
    uint32_img_container = ones_np(img.shape, dtype=uint32) * 5
    one_hot_container = ones_np((1, 2, 2), dtype=float64) * 6
    RGB_ID_SUBTRAHEND = ones_np((1, 2, 4), dtype=uint32)
    RGB_ID_SUBTRAHEND[:, :, 0] *= 0
    RGB_ID_SUBTRAHEND[:, :, 3] *= 255 * 256 * 256 + 256 + 1
    rgb_id_container = ones_np((1, 2, 4), dtype=uint32) * 7
    bool_container = ones_np((1, 2, 4), dtype=uint8) * 2
    root_logger.info(uint32_img_container)
    root_logger.info(one_hot_container)
    root_logger.info(RGB_ID_SUBTRAHEND)
    root_logger.info(rgb_id_container)
    root_logger.info(bool_container)
    m_uint32_img_container = patch.object(OneHot, "uint32_img_container", new=PropertyMock(return_value=uint32_img_container)).start()
    m_one_hot_container = patch.object(OneHot, "one_hot_container", new=PropertyMock(return_value=one_hot_container)).start()
    m_RGB_ID_SUBTRAHEND = patch.object(OneHot, "RGB_ID_SUBTRAHEND", new=PropertyMock(return_value=RGB_ID_SUBTRAHEND)).start()
    m_rgb_id_container = patch.object(OneHot, "rgb_id_container", new=PropertyMock(return_value=rgb_id_container)).start()
    m_bool_container = patch.object(OneHot, "bool_container", new=PropertyMock(return_value=bool_container)).start()
    m_colors = MagicMock()
    m_categories = MagicMock()
    m_color1 = MagicMock(i=1)
    m_color2 = MagicMock(i=2)
    m_color3 = MagicMock(i=3)
    m_color4 = MagicMock(i=4)
    m_category1 = MagicMock(i=1)
    m_category2 = MagicMock(i=2)
    m_colors.__iter__.return_value = [m_color1, m_color2, m_color3, m_color4]
    m_categories.__iter__.return_value = [m_category1, m_category2]
    x = object.__new__(OneHot)
    x.colors = m_colors
    x.categories = m_categories
    m_category1.__eq__ = lambda self, other: other.i in [1]
    m_category2.__eq__ = lambda self, other: other.i in [2]
    assert x.get_encoded(img).tolist() == [[[1, 0], [0, 0]]]
    m_category1.__eq__ = lambda self, other: False
    m_category2.__eq__ = lambda self, other: False
    assert x.get_encoded(img).tolist() == [[[0, 0], [0, 0]]]
    m_category1.__eq__ = lambda self, other: other.i in [1]
    m_category2.__eq__ = lambda self, other: other.i in [1]
    assert x.get_encoded(img).tolist() == [[[1, 1], [0, 0]]]
    m_category1.__eq__ = lambda self, other: other.i in [1, 4]
    m_category2.__eq__ = lambda self, other: other.i in [1, 4]
    assert x.get_encoded(img).tolist() == [[[1, 1], [1, 1]]]
    m_category1.__eq__ = lambda self, other: other.i in [1]
    m_category2.__eq__ = lambda self, other: other.i in [4]
    assert x.get_encoded(img).tolist() == [[[1, 0], [0, 1]]]
    m_category1.__eq__ = lambda self, other: other.i in [2, 3]
    m_category2.__eq__ = lambda self, other: other.i in [2, 3]
    assert x.get_encoded(img).tolist() == [[[0, 0], [0, 0]]]
    assert x.get_encoded(img).dtype.name == "float64"


@name(LaneDB.create_categories, "colors", globals())
def _():
    LaneDB.create_categories()
    assert len(Color) == 38
    assert Color["void"] is Color[(0, 0, 0)]
    assert Color("void2", (0, 0, 0)) is Color["void"]
    assert Color("void2", (0, 0, 0)).name == "void"
    assert len(Color) == 38


@name(LaneDB.create_categories, "categories", globals())
def _():
    LaneDB.create_categories()
    assert len(Category) == 36
    assert Category["Hatter"] is Category[(0, 0, 0)] is Category["void"] is Category["ignored"]


@name(LaneDB._get_all_path, "", globals())
def _():
    img_paths, mask_paths = LaneDB._get_all_path()
    assert len(img_paths) == 113653 == len(mask_paths)
    assert EXAMPLE_IMG_PATH in img_paths and EXAMPLE_MASK_PATH in mask_paths


@name(LaneDB._get_train_val_test_paths, "1", globals())
def _():
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = LaneDB._get_train_val_test_paths()
    assert (len(train_img_paths) + len(val_img_paths) + len(test_img_paths)) == 113653 == (len(train_mask_paths) + len(val_mask_paths) + len(test_mask_paths))


@name(LaneDB._random_example_id.fget, "1", globals())
def _():
    m_randrange = patch("traffic.utils.lane_unet.randrange", new=MagicMock(return_value=0)).start()
    x = object.__new__(LaneDB)
    x._img_paths = [None, None]
    assert x._random_example_id == 0
    m_randrange.assert_called_once_with(2)


@name(LaneDB._get_example_paths, "1", globals())
def _():
    x = object.__new__(LaneDB)
    x._img_paths = [1, 2]
    x._mask_paths = [3, 4]
    assert x._get_example_paths(0) == (1, 3)


@name(LaneDB._load_example, "1", globals())
def _():
    x = object.__new__(LaneDB)
    m__get_example_paths = patch.object(LaneDB, "_get_example_paths", new=MagicMock(return_value=[EXAMPLE_IMG_PATH, EXAMPLE_MASK_PATH])).start()
    img, mask = x._load_example(0)
    m__get_example_paths.assert_called_once_with(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.dtype.name == mask.dtype.name == "uint8"
    assert img.shape == mask.shape == (2710, 3384, 3)
    assert img.max() <= 255 and img.min() >= 0 and img.mean() > 1
    assert mask.max() <= 255 and mask.min() >= 0 and mask.mean() > 1


@name(LaneDB._get_small_example, "1", globals())
def _():
    x = object.__new__(LaneDB)
    m__load_example = patch.object(
        LaneDB, "_load_example", new=MagicMock(return_value=[ones_np((2710, 3384, 3), dtype=uint8) * 9, ones_np((2710, 3384, 3), dtype=uint8) * 9])
    ).start()
    img, mask = x._get_small_example(0)
    m__load_example.assert_called_once_with(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.shape == mask.shape == (513, 640, 3)
    assert img.max() <= 255 and img.min() >= 0 and img.mean() > 1
    assert mask.max() <= 255 and mask.min() >= 0 and mask.mean() > 1
    assert img.dtype.name == mask.dtype.name == "uint8"


@name(LaneDB._get_small_cropped_example, "1", globals())
def _():
    x = object.__new__(LaneDB)
    m__get_small_example = patch.object(
        LaneDB, "_get_small_example", new=MagicMock(return_value=[ones_np((513, 640, 3), dtype=uint8) * 9, ones_np((513, 640, 3), dtype=uint8) * 9])
    ).start()
    img, mask = x._get_small_cropped_example(0)
    m__get_small_example.assert_called_once_with(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.shape == mask.shape == (480, 640, 3)
    assert img.max() <= 255 and img.min() >= 0 and img.mean() > 1
    assert mask.max() <= 255 and mask.min() >= 0 and mask.mean() > 1
    assert img.dtype.name == mask.dtype.name == "uint8"


@name(LaneDB.get_train_input, "1", globals())
def _():
    x = object.__new__(LaneDB)
    m__get_small_cropped_example = patch.object(
        LaneDB, "_get_small_cropped_example", new=MagicMock(return_value=[ones_np((480, 640, 3), dtype=uint8) * 9, sentinel.a])
    ).start()
    m_one_hot_coder = patch.object(LaneDB, "one_hot_coder", new=MagicMock()).start()
    m_one_hot_coder.get_encoded.return_value = sentinel.b
    img, mask = x.get_train_input(0)
    m__get_small_cropped_example.assert_called_once_with(0)
    m_one_hot_coder.get_encoded.assert_called_once_with(sentinel.a)
    assert type(img) is ndarray and mask is sentinel.b
    assert img.shape == (480, 640, 3)
    assert img.max() <= 1 and img.min() >= 0 and (0 < img.mean() < 1)
    assert img.dtype.name == "float64"


@name(Unet.model.fget, "1", globals())
def _():
    u = Unet.__new__(Unet)
    u.hdf5_path = "/traffic/Unet.hdf5"
    assert type(u.model) == Model_ke


@name(Unet.get_prediction, "1", globals())
def _():
    u = Unet.__new__(Unet)
    u.hdf5_path = "/traffic/Unet.hdf5"
    grayscale_array = ones_np((480, 640, 3), dtype=uint8) * 111
    mask = u.get_prediction(grayscale_array)
    summed_mask = sum_np(mask, axis=2)
    assert summed_mask.shape == (480, 640)
    assert mask.shape[0] == 480 and mask.shape[1] == 640 and len(mask.shape) == 3
    assert mask.shape[2] > 3
    assert_almost_equal_np(summed_mask, ones_np((480, 640), dtype=uint8), decimal=6)
    # imshow_mat(mask[:, :, :3])
    # show()


# @name(Unet.train, "1", globals())
# def _():
#     u = Unet()
#     u.train(batch_size=1,steps_per_epoch=1,validation_steps=1,early_stopping_min_delta=0,RLR_min_delta=0,early_stopping_patience=0,RLR_patience=0,RLRFactor=0.1)
# @name(Unet.structure.fget, "proxy", globals())
# def _():
#     u = Unet(1)
#     u._structure = 1
#     structure = u.structure
#     assert structure == 1
#
#
# @name(Unet.structure.fget, "1", globals())
# def _():
#     u = Unet(1)
#     model = u.structure
#     assert type(model) is Model_ke
#
#

#
#
#
