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


# << module test
assert len(Color) == 38
assert len(MaskSource.source_encoder.categories) == len(Color)
assert sum(len(category) for category in Category) == len(Color)
DBs = LaneDB.get_train_val_test_DB()
for DB in DBs:
    for image_and_mask_source_pair in DB.image_and_mask_source_pairs:
        mask_source = image_and_mask_source_pair.mask_source
        assert_almost_equal_np(sum(mask_source.attributes.values()), 1)
        assert_almost_equal_np(sum(mask_source.category_probabilities.values()), 1)
    assert_almost_equal_np(sum(DB.category_probabilities.values()), 1)
# module test >>
if False:
    for DB in DBs:
        for img_source, mask_source in zip(DB.img_sources, DB.mask_sources):
            try:
                img_source.data
                mask_source.data
                root_logger.info(img_source.path)
                root_logger.info(mask_source.path)
            except Exception as e:
                root_logger.warning(img_source.path)
                root_logger.warning(mask_source.path)
                remove_os(img_source.path)
                remove_os(mask_source.path)


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


@name(categorical_crossentropy_ke, "1d same", globals())
def _():
    arr1 = array_np([0, 1])
    arr2 = arr1
    assert arr1.shape == (2,)
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x = categorical_crossentropy_ke(y_true, y_pred)
    x = eval_ke(x)
    assert x.shape == ()
    assert_almost_equal_np(x, 0)


@name(categorical_crossentropy_ke, "1d commutative", globals())
def _():
    arr1 = array_np([0, 1])
    arr2 = array_np([1, 0])
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x1 = categorical_crossentropy_ke(y_true, y_pred)
    x1 = eval_ke(x1)
    #
    arr1 = array_np([1, 0])
    arr2 = array_np([0, 1])
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x2 = categorical_crossentropy_ke(y_true, y_pred)
    x2 = eval_ke(x2)
    #
    assert_almost_equal_np(x1, x2)


@name(categorical_crossentropy_ke, "2d same", globals())
def _():
    arr1 = array_np([[0, 1, 0], [1, 0, 0]])
    arr2 = arr1
    assert arr1.shape == (2, 3)
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x = categorical_crossentropy_ke(y_true, y_pred)
    x = eval_ke(x)
    assert x.shape == (2,)
    assert_almost_equal_np(x, [0, 0])


@name(categorical_crossentropy_ke, "3d same", globals())
def _():
    arr1 = array_np([[[0, 1, 0], [1, 0, 0]]])
    arr2 = arr1
    assert arr1.shape == (1, 2, 3)
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x = categorical_crossentropy_ke(y_true, y_pred)
    x = eval_ke(x)
    assert x.shape == (1, 2)
    assert_almost_equal_np(x, [[0, 0]])


@name(binary_crossentropy_ke, "1d same", globals())
def _():
    arr1 = array_np([0])
    arr2 = arr1
    assert arr1.shape == (1,)
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x = binary_crossentropy_ke(y_true, y_pred)
    x = eval_ke(x)
    assert x.shape == ()
    assert_almost_equal_np(x, 0)


@name(binary_crossentropy_ke, "2d same", globals())
def _():
    arr1 = array_np([[0, 1, 0]])
    arr2 = arr1
    assert arr1.shape == (1, 3)
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x = binary_crossentropy_ke(y_true, y_pred)
    x = eval_ke(x)
    assert x.shape == (1,)
    assert_almost_equal_np(x, 0)


@name(binary_crossentropy_tf, "tf 2d same", globals())
def _():
    arr1 = array_np([[0, 1, 0]])
    arr2 = arr1
    assert arr1.shape == (1, 3)
    y_true = variable_ke(arr1)
    y_pred = variable_ke(arr2)
    x = binary_crossentropy_tf(y_true, y_pred)
    x = eval_ke(x)
    assert x.shape == (1, 3)
    assert_almost_equal_np(x, 0)


@name(get_summed_dict, "1", globals())
def _():
    assert get_summed_dict([{"1": 1, "2": 2}, {"1": 1, "2": 3}]) == {"1": 2, "2": 5}
    assert get_summed_dict([{"1": 1, "2": 2}]) == {"1": 1, "2": 2}


@name(get_probabilities, "1", globals())
def _():
    assert get_probabilities({"a": 100, "b": 200}) == {"a": 100 / 300, "b": 200 / 300}


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
    assert x.one_hot_container.dtype.name == "float32"
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


@name(OneHot.get_category_histogram, "", globals())
def _():
    one_hot = array_np([[[0.0, 0.0, 1.1], [0.0, 1.0, 1.0]]], dtype=float64)
    m_get_encoded = patch.object(OneHot, "get_encoded", new=MagicMock(return_value=one_hot)).start()
    m_categories = MagicMock()
    m_category1 = MagicMock()
    m_category2 = MagicMock()
    m_category3 = MagicMock()
    m_category1.configure_mock(name="1")
    m_category2.configure_mock(name="2")
    m_category3.configure_mock(name="3")
    m_categories.__iter__.return_value = [m_category1, m_category2, m_category3]
    x = object.__new__(OneHot)
    x.categories = m_categories
    y = x.get_category_histogram(sentinel.a)
    m_get_encoded.assert_called_once_with(sentinel.a)
    assert y == {"1": 0, "2": 1, "3": 2}


@name(ImgSource.get_hres_to_cropmat_rescale_factor, "", globals())
def _():
    m_crop_material_size_factor = patch("traffic.utils.lane_unet.HyperP.crop_material_size_factor", new=1).start()
    assert_almost_equal_np(ImgSource.get_hres_to_cropmat_rescale_factor(480, 640), 1)
    assert_almost_equal_np(ImgSource.get_hres_to_cropmat_rescale_factor(4800, 640), 1)
    assert_almost_equal_np(ImgSource.get_hres_to_cropmat_rescale_factor(480, 6400), 1)
    assert_almost_equal_np(ImgSource.get_hres_to_cropmat_rescale_factor(48000, 6400), 0.1)


@name(ImgSource.get_degree, "", globals())
def _():
    assert ImgSource.get_degree(0.5, True) == -90
    assert ImgSource.get_degree(0.5, False) == 90
    assert ImgSource.get_degree(0, False) == 0
    assert ImgSource.get_degree(1, False) == 180


@name(ImgSource.get_normalized_img, "", globals())
def _():
    x = ImgSource.get_normalized_img(array_np([255, 0]))
    assert_almost_equal_np(x, array_np([1, 0]))


@name(ImgSource.get_cropped_img, "", globals())
def _():
    y = zeros((500, 640, 3))
    y[10] = 1
    x = ImgSource.get_cropped_img(y, 0, True, 0, True)
    assert x[0, 0, 0] == 1


@name(ImgSource.get_an_input, "", globals())
def _():
    m_data = patch.object(ImgSource, "data", new=PropertyMock(return_value=MagicMock(shape=(sentinel.nrows, sentinel.ncols, sentinel.ch)))).start()
    m_get_hres_to_cropmat_rescale_factor = patch.object(
        ImgSource, "get_hres_to_cropmat_rescale_factor", new=MagicMock(return_value=sentinel.rescale_factor)
    ).start()
    m_rescale_skimage = patch("traffic.utils.lane_unet.rescale_skimage", new=MagicMock(return_value=sentinel.rescaled_img)).start()
    m_array_np = patch("traffic.utils.lane_unet.array_np", new=MagicMock(return_value=MagicMock(shape=(666, 832, 3)))).start()
    m_get_cropped_img = patch.object(ImgSource, "get_cropped_img", new=MagicMock(return_value=sentinel.cropped_img)).start()
    m_imrotate_sci = patch("traffic.utils.lane_unet.imrotate_sci", new=MagicMock(return_value=sentinel.rotated_img)).start()
    m_get_degree = patch.object(ImgSource, "get_degree", new=MagicMock(return_value=sentinel.degree)).start()
    m_get_normalized_img = patch.object(ImgSource, "get_normalized_img", new=MagicMock()).start()
    x = object.__new__(ImgSource)
    y = x.get_an_input(sentinel.rotation_hardness, sentinel.clockwise, sentinel.row_hardness, sentinel.row_up, sentinel.col_hardness, sentinel.col_left, 0.5)
    m_data.assert_called_once_with()
    m_get_hres_to_cropmat_rescale_factor.assert_called_once_with(sentinel.nrows, sentinel.ncols)
    m_rescale_skimage.assert_called_once_with(m_data(), sentinel.rescale_factor, anti_aliasing=False, preserve_range=True)
    m_array_np.assert_called_once_with(sentinel.rescaled_img, dtype=uint8)
    m_get_cropped_img.assert_called_once_with(m_array_np(), sentinel.row_hardness, sentinel.row_up, sentinel.col_hardness, sentinel.col_left)
    m_get_normalized_img.assert_called_once_with(sentinel.rotated_img)
    m_get_degree.assert_called_once_with(sentinel.rotation_hardness, sentinel.clockwise)
    m_imrotate_sci.assert_called_once_with(m_get_cropped_img(), sentinel.degree, interp="bicubic")
    assert m_get_normalized_img().__setitem__.called
    assert y is m_get_normalized_img()


@name(MaskSource.get_resized_shape, "", globals())
def _():
    m_get_hres_to_cropmat_rescale_factor = patch.object(ImgSource, "get_hres_to_cropmat_rescale_factor", new=MagicMock(return_value=0.5)).start()
    x = object.__new__(MaskSource)
    y = x.get_resized_shape(100, 101)
    m_get_hres_to_cropmat_rescale_factor.assert_called_once_with(100, 101)
    assert y == (50, 50, 3)


@name(MaskSource.get_an_input, "", globals())
def _():
    m_data = patch.object(MaskSource, "data", new=PropertyMock(return_value=MagicMock(shape=(sentinel.nrows, sentinel.ncols, sentinel.ch)))).start()
    m_get_resized_shape = patch.object(MaskSource, "get_resized_shape", new=MagicMock(return_value=sentinel.resized_shape)).start()
    m_imresize_scipy = patch("traffic.utils.lane_unet.imresize_scipy", new=MagicMock(return_value=MagicMock(shape=(666, 832, 3)))).start()
    m_imrotate_sci = patch("traffic.utils.lane_unet.imrotate_sci", new=MagicMock(return_value=sentinel.rotated_img)).start()
    m_get_cropped_img = patch.object(ImgSource, "get_cropped_img", new=MagicMock(return_value=sentinel.cropped_img)).start()
    m_get_degree = patch.object(ImgSource, "get_degree", new=MagicMock(return_value=sentinel.degree)).start()
    x = object.__new__(MaskSource)
    m_one_hot = MagicMock()
    m_one_hot.get_encoded.return_value = zeros((480, 640, 30), dtype=float32)
    y = x.get_an_input(
        m_one_hot, sentinel.rotation_hardness, sentinel.clockwise, sentinel.row_hardness, sentinel.row_up, sentinel.col_hardness, sentinel.col_left
    )
    m_get_resized_shape.assert_called_once_with(sentinel.nrows, sentinel.ncols)
    m_imresize_scipy.assert_called_once_with(m_data(), sentinel.resized_shape, interp="nearest")
    m_get_cropped_img.assert_called_once_with(m_imresize_scipy(), sentinel.row_hardness, sentinel.row_up, sentinel.col_hardness, sentinel.col_left)
    m_get_degree.assert_called_once_with(sentinel.rotation_hardness, sentinel.clockwise)
    m_imrotate_sci.assert_called_once_with(m_get_cropped_img(), sentinel.degree, interp="nearest")
    m_one_hot.get_encoded.assert_called_once_with(sentinel.rotated_img)
    assert y.shape == (480 * 640, 30) and y.dtype.name == "float32"


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
    assert len(Category) == 2
    assert Category["Hatter"] is Category[(0, 0, 0)] is Category["void"] is Category["ignored"]


@name(LaneDB._get_all_path, "", globals())
def _():
    img_paths, mask_paths = LaneDB._get_all_path()
    assert len(img_paths) == 113652 == len(mask_paths)
    assert EXAMPLE_IMG_PATH in img_paths and EXAMPLE_MASK_PATH in mask_paths


@name(LaneDB._get_train_val_test_paths, "1", globals())
def _():
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = LaneDB._get_train_val_test_paths()
    if False:
        assert (
            (len(train_img_paths) + len(val_img_paths) + len(test_img_paths)) == 113652 == (len(train_mask_paths) + len(val_mask_paths) + len(test_mask_paths))
        )


@name(LaneDB.__init__, "", globals())
def _():
    m_ImgSource = patch("traffic.utils.lane_unet.ImgSource", new=MagicMock()).start()
    m_mask_source1 = MagicMock(category_probabilities={"Hatter": 2})
    m_mask_source2 = MagicMock(category_probabilities={"Hatter": 1})
    m_MaskSource = patch("traffic.utils.lane_unet.MaskSource", new=MagicMock(side_effect=[m_mask_source1, m_mask_source2])).start()
    m_IMSourcePair1 = MagicMock(mask_source=m_mask_source1)
    m_IMSourcePair2 = MagicMock(mask_source=m_mask_source2)
    m_IMSourcePair = patch("traffic.utils.lane_unet.IMSourcePair", new=MagicMock(side_effect=[m_IMSourcePair1, m_IMSourcePair2])).start()
    x = LaneDB([sentinel.img_path1, sentinel.img_path2], [sentinel.mask_path1, sentinel.mask_path2])
    assert m_ImgSource.mock_calls == [call_mock(sentinel.img_path1), call_mock(sentinel.img_path2)]
    assert m_MaskSource.mock_calls == [call_mock(sentinel.mask_path1), call_mock(sentinel.mask_path2)]
    assert m_IMSourcePair.mock_calls == [call_mock(m_ImgSource(), m_mask_source1), call_mock(m_ImgSource(), m_mask_source2)]
    assert x.image_and_mask_source_pairs[1] == m_IMSourcePair1


@name(LaneDB.orders.fget, "", globals())
def _():
    m_category_1 = MagicMock()
    m_category_2 = MagicMock()
    m_category_3 = MagicMock()
    m_category_1.name = "1"
    m_category_2.name = "2"
    m_category_3.name = "3"
    m_Category = patch("traffic.utils.lane_unet.Category", new=MagicMock()).start()
    m_Category.__iter__.return_value = [m_category_1, m_category_2, m_category_3]
    m_image_and_mask_source_pair_1 = MagicMock(mask_source=MagicMock(category_probabilities={"1": 1, "2": 2, "3": 3}))
    m_image_and_mask_source_pair_2 = MagicMock(mask_source=MagicMock(category_probabilities={"1": 3, "2": 2, "3": 1}))
    m_image_and_mask_source_pair_3 = MagicMock(mask_source=MagicMock(category_probabilities={"1": 0, "2": 2, "3": 3.1}))
    x = object.__new__(LaneDB)
    x.image_and_mask_source_pairs = [m_image_and_mask_source_pair_1, m_image_and_mask_source_pair_2, m_image_and_mask_source_pair_3]
    assert set(x.orders.keys()) == {"1", "2", "3"}
    assert x.orders["1"] == [1, 0]
    assert x.orders["2"] == [0, 1, 2]
    assert x.orders["3"] == [2, 0, 1]


@name(LaneDB.get_sources_by_category, "", globals())
def _():
    m_orders = patch.object(LaneDB, "orders", new=PropertyMock(return_value={"1": [1, 0, 2]})).start()
    m_image_and_mask_source_pair_1 = MagicMock(mask_source=sentinel.mask_source_1, img_source=sentinel.img_source_1)
    m_image_and_mask_source_pair_2 = MagicMock(mask_source=sentinel.mask_source_2, img_source=sentinel.img_source_2)
    m_image_and_mask_source_pair_3 = MagicMock(mask_source=sentinel.mask_source_3, img_source=sentinel.img_source_3)
    x = object.__new__(LaneDB)
    x.image_and_mask_source_pairs = [m_image_and_mask_source_pair_1, m_image_and_mask_source_pair_2, m_image_and_mask_source_pair_3]
    assert x.get_sources_by_category("1", 0) == (sentinel.img_source_2, sentinel.mask_source_2)
    assert x.get_sources_by_category("1", 0.3) == (sentinel.img_source_2, sentinel.mask_source_2)
    assert x.get_sources_by_category("1", 0.4) == (sentinel.img_source_1, sentinel.mask_source_1)
    assert x.get_sources_by_category("1", 1) == (sentinel.img_source_3, sentinel.mask_source_3)


@name(Unet.epoch_i_setter.fget, "1", globals())
def _():
    x = object.__new__(Unet)
    assert not hasattr(x, "epoch_i")
    x.epoch_i_setter.on_epoch_begin(sentinel.epoch_i)
    assert x.epoch_i is sentinel.epoch_i


@name(LaneUtil, "1", globals())
def _():
    x = object.__new__(Unet)
    y = next(LaneUtil.dot_cloud_coordinate_cycle)
    assert y != (0, 0)
    for _ in range((480 * 640) - 1):
        assert y != next(LaneUtil.dot_cloud_coordinate_cycle)
    assert y == next(LaneUtil.dot_cloud_coordinate_cycle)


@name(Unet.get_an_important_category_cycle, "1", globals())
def _():
    m_category_1 = MagicMock()
    m_category_2 = MagicMock()
    m_category_3 = MagicMock()
    m_category_1.name = "1"
    m_category_2.name = "Hatter"
    m_category_3.name = "3"
    m_Category = patch("traffic.utils.lane_unet.Category", new=MagicMock()).start()
    m_Category.__iter__.return_value = [m_category_1, m_category_2, m_category_3]
    x = object.__new__(Unet)
    y = x.get_an_important_category_cycle()
    assert next(y) is m_category_1
    assert next(y) is m_category_3
    assert next(y) is m_category_1
    y = x.get_an_important_category_cycle()
    assert next(y) is m_category_1


@name(Unet.max_hardness.fget, "1", globals())
def _():
    x = object.__new__(Unet)
    x.min_epochs = 3
    x.epoch_i = 0
    assert x.max_hardness == 1 / 3
    x.epoch_i = 1
    assert x.max_hardness == 1 / 3 + 1 / 3
    x.epoch_i = 2
    assert x.max_hardness == 1
    x.epoch_i = 3
    assert x.max_hardness == 1
    x.epoch_i = 4
    assert x.max_hardness == 1


@name(Unet.train, "1", globals())
def _():
    x = object.__new__(Unet)
    return
    x.train(
        batch_size=1,
        steps_per_epoch=1,
        validation_steps=1,
        early_stopping_min_delta=0,
        RLR_min_delta=0,
        early_stopping_patience=0,
        RLR_patience=0,
        RLRFactor=0.1,
    )


@name(Unet.model.fget, "1", globals())
def _():
    x = object.__new__(Unet)
    x.hdf5_path = "/traffic/Unet.hdf5"
    assert type(x.model) == Model_ke


@name(Unet.get_prediction, "1", globals())
def _():
    x = object.__new__(Unet)
    x.hdf5_path = "/traffic/Unet.hdf5"
    img = ones_np((480, 640, 3), dtype=uint8) * 111
    y = x.get_prediction(img)
    assert y.shape == (480, 640, 2)
    assert y.dtype.name == "float32"
    assert max_np(y) <= 1
