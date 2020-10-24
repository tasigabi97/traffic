from traffic.testing import *
from traffic.imports import *
from traffic.utils.lane_unet import *

EXAMPLE_MASK_PATH = "/traffic/lane/Labels_road02/Label/Record028/Camera 5/170927_071400339_Camera_5_bin.png"
EXAMPLE_IMG_PATH = "/traffic/lane/ColorImage_road02/ColorImage/Record028/Camera 5/170927_071400339_Camera_5.jpg"


@name(Color, "1", globals())
def _():
    assert len(Color.colors) == 38
    assert len({color.color for color in Color.colors}) == 38
    assert len({color.name for color in Color.colors}) == 38


@name(Category.__init__, "1", globals())
def _():
    cat = Category("alma", 1, (1, 1, 1), (2, 2, 2))
    assert cat.name == "alma" and cat.id == 1
    assert len(cat.colors) == 2 and (1, 1, 1) in cat.colors and (2, 2, 2) in cat.colors


@name(Category.add_color, "1", globals())
def _():
    cat = Category("alma", 1, (1, 1, 1), (2, 2, 2))
    cat.add_color((3, 3, 3))
    assert len(cat.colors) == 3 and (3, 3, 3) in cat.colors


@name(Categories.__len__, "1", globals())
def _():
    assert len(CATEGORIES) == 36 == len({category.id for category in CATEGORIES._categories})


@name(Categories.__getitem__, "Error", globals())
def _():
    with raises(ValueError):
        CATEGORIES["void"]
    with raises(ValueError):
        CATEGORIES[37]
    with raises(TypeError):
        CATEGORIES[(0, 0, 1)]
    with raises(TypeError):
        CATEGORIES[[]]


@name(Categories.__getitem__, "ret", globals())
def _():
    assert CATEGORIES["Background"] is CATEGORIES[0]
    assert len(CATEGORIES["Background"].colors) == 3


@name(LaneDB._get_all_path, "1", globals())
def _():
    img_paths, mask_paths = LaneDB._get_all_path()
    assert len(img_paths) == 113653 == len(mask_paths)
    assert EXAMPLE_IMG_PATH in img_paths and EXAMPLE_MASK_PATH in mask_paths


@name(LaneDB._get_train_val_test_paths, "1", globals())
def _():
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = LaneDB._get_train_val_test_paths()
    assert (len(train_img_paths) + len(val_img_paths) + len(test_img_paths)) == 113653 == (len(train_mask_paths) + len(val_mask_paths) + len(test_mask_paths))


@name(LaneDB._random_example_id.fget, "1", globals())
@patch("traffic.utils.lane_unet.randrange")
@patch("traffic.utils.lane_unet.train_DB._img_paths", [None, None])
def _(mock_randrange):
    mock_randrange.return_value = 0
    id = train_DB._random_example_id
    mock_randrange.assert_called_once_with(2)
    assert id == 0


@name(LaneDB._get_example_paths, "1", globals())
def _():
    img_path, mask_path = train_DB._get_example_paths(0)
    assert img_path in train_DB._img_paths and mask_path in train_DB._mask_paths
    assert train_DB._img_paths.index(img_path) == train_DB._mask_paths.index(mask_path)


@name(LaneDB._get_example, "1", globals())
def _():
    img, mask = train_DB._get_example(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.dtype.name == mask.dtype.name == "uint8"
    assert img.shape == mask.shape == (2710, 3384, 3)
    assert img.max() <= 255 and img.min() >= 0 and img.mean() > 1
    assert mask.max() <= 255 and mask.min() >= 0 and mask.mean() > 1


@name(LaneDB._get_small_example, "1", globals())
def _():
    img, mask = train_DB._get_small_example(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.shape == mask.shape == (513, 640, 3)
    assert img.max() <= 255 and img.min() >= 0 and img.mean() > 1
    assert mask.max() <= 255 and mask.min() >= 0 and mask.mean() > 1
    assert img.dtype.name == mask.dtype.name == "uint8"


@name(LaneDB._get_small_cropped_example, "1", globals())
def _():
    img, mask = train_DB._get_small_cropped_example(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.shape == mask.shape == (480, 640, 3)
    assert img.max() <= 255 and img.min() >= 0 and img.mean() > 1
    assert mask.max() <= 255 and mask.min() >= 0 and mask.mean() > 1
    assert img.dtype.name == mask.dtype.name == "uint8"


@name(LaneDB.get_input, "1", globals())
def _():
    img, mask = train_DB.get_input(0)
    assert type(img) is ndarray and type(mask) is ndarray
    assert img.shape == (480, 640)
    assert mask.shape == (480, 640, 36)
    assert img.max() <= 1 and img.min() >= 0 and (0 < img.mean() < 1)
    a = unique(mask)
    assert len(a) == 2 and 0 in a and 1 in a
    assert img.dtype.name == mask.dtype.name == "float64"


@name(Categories, "1", globals())
def _():
    return
    for category in CATEGORIES._categories:
        for i in range(999999):
            print(i)
            img, mask = train_DB.get_input(i)
            if max_np(mask[:, :, category.id]) == 1:
                print(category.name)
                print(train_DB._get_example_paths(i))
                mask[:, :, 0] = mask[:, :, category.id]
                mask[:, :, 2] = img
                mask[:, :, 1] = img
                imshow_mat(mask[:, :, :3])
                show()
                break
