from traffic.testing import *
from traffic.imports import *
from traffic.utils.lane_unet import *

EXAMPLE_MASK_PATH = "/traffic/lane/Labels_road02/Label/Record028/Camera 5/170927_071400339_Camera_5_bin.png"
EXAMPLE_IMG_PATH = "/traffic/lane/ColorImage_road02/ColorImage/Record028/Camera 5/170927_071400339_Camera_5.jpg"
IMG=array_np([[[0,     0 ,0],  [    0   ,  1,   2]]],dtype=uint8)
assert IMG.shape == (1,2,3)

def setup_function(function):
    Singleton._instances = dict()
    Color.clear()
    Category.clear()

@name(LaneDB.create_categories, "colors", globals())
def _():
    LaneDB.create_categories()
    assert len(Color) == 38
    assert Color["void"] is Color[(0,0,0)]
    assert Color("void2",(0,0,0)) is Color["void"]
    assert Color("void2",(0,0,0)).name == "void"
    assert len(Color) == 38

@name(LaneDB.create_categories, "categories", globals())
def _():
    LaneDB.create_categories()
    assert len(Category) == 36
    assert Category["Háttér"] is Category[(0,0,0)] is Category["void"] is Category["ignored"]

@name(OneHot.__init__, "1", globals())
def _():
    cat1=Category("cat1",Color("color1",(0,0,1)),Color("color2",(0,0,0)))
    cat2=Category("cat2",Color("color3",(1,0,1)))
    o=OneHot(1,2,(cat2,cat1))
    assert o.row_number==1 and o.col_number==2
    assert type(o.colors) is list is type(o.categories)
    assert o.categories[0] is cat1 and o.categories[1] is cat2
    assert o.colors[0].name == "color2"
    assert o.colors[1].name == "color1"
    assert o.colors[2].name == "color3"

@name(OneHot.one_hot_container.fget, "1", globals())
def _():
    cat1=Category("cat1",Color("color1",(0,0,1)),Color("color2",(0,0,0)))
    cat2=Category("cat2",Color("color3",(1,0,1)))
    o=OneHot(1,2,(cat2,cat1))
    assert type(o.one_hot_container) is ndarray
    assert o.one_hot_container.dtype.name == "float64"
    assert o.one_hot_container.shape == (1, 2, 2)

@name(OneHot.one_hot_container.fget, "2", globals())
def _():
    cat1=Category("cat1",Color("color1",(0,0,1)))
    o=OneHot(1,1,(cat1,))
    assert type(o.one_hot_container) is ndarray
    assert o.one_hot_container.dtype.name == "float64"
    assert o.one_hot_container.shape == (1, 1, 1)
    a=array_np([[[9]]],dtype=uint8)
    assert o.one_hot_container.tolist() != a.tolist()
    o.one_hot_container[:,:,:]=a
    assert o.one_hot_container.dtype.name == "float64"
    assert o.one_hot_container.tolist() == a.tolist()


@name(OneHot.color_id_container.fget, "1", globals())
def _():
    cat1=Category("cat1",Color("color1",(0,0,1)),Color("color2",(0,0,0)))
    cat2=Category("cat2",Color("color3",(1,0,1)))
    o=OneHot(1,2,(cat2,cat1))
    assert type(o.color_id_container) is ndarray
    assert o.color_id_container.dtype.name == "uint32"
    assert o.color_id_container.shape == (1, 2, 3)
@name(OneHot.color_id_container.fget, "2", globals())
def _():
    cat1=Category("cat1",Color("color1",(0,0,1)))
    o=OneHot(1,1,(cat1,))
    assert type(o.color_id_container) is ndarray
    assert o.color_id_container.dtype.name == "uint32"
    assert o.color_id_container.shape == (1, 1, 1)
    a=array_np([[[9]]],dtype=uint8)
    assert o.color_id_container.tolist() != a.tolist()
    o.color_id_container[:,:,:]=a
    assert o.color_id_container.dtype.name == "uint32"
    assert o.color_id_container.tolist() == a.tolist()

@name(OneHot.uint32_img_container.fset, "fset", globals())
def _():
    o=OneHot(IMG.shape[0],IMG.shape[1],[])
    assert o.uint32_img_container.dtype.name == "uint32"
    assert o.uint32_img_container.tolist() != IMG.tolist()
    o.uint32_img_container=IMG
    assert o.uint32_img_container.dtype.name == "uint32"
    assert o.uint32_img_container.tolist() == IMG.tolist()

@name(OneHot.get_color_id, "1", globals())
def _():
    assert OneHot.get_color_id((0,0,0)) == 0
    assert OneHot.get_color_id((1,1,1)) == 65793

@name(OneHot.get_color, "1", globals())
def _():
    assert OneHot.get_color(65793) == (1,1,1)
    assert OneHot.get_color(0) == (0,0,0)
    assert OneHot.get_color(1) == (0,0,1)
    assert OneHot.get_color(256) == (0,1,0)

@name(OneHot.get_color, "inverse", globals())
def _():
    color_ids=[256*256*256,0,1,255,256,999]
    for color_id in color_ids:
        assert OneHot.get_color_id(OneHot.get_color(color_id))==color_id

@name(OneHot.color_id_array.fget, "1", globals())
def _():
    cat1=Category("cat1",Color("color1",(0,0,1)),Color("color2",(0,0,0)))
    cat2=Category("cat2",Color("color3",(1,0,1)),Color("color4",(0,1,0)))
    o=OneHot(1,2,(cat2,cat1))
    assert type(o.color_id_array) is ndarray
    assert o.color_id_array.dtype.name == "uint32"
    assert o.color_id_array.shape == (1, 2, 4)
    assert o.color_id_array.tolist() == [[[0,     1 ,  256, 65537],  [    0   ,  1,   256, 65537]]]


@name(OneHot.get_one_hot, "1", globals())
def _():
    img, mask = train_DB._get_small_cropped_example(0)
    LaneDB.create_categories()

    cat1=Category("cat1",Color("color1",(0,0,1)),Color("color2",(0,0,0)))
    cat2=Category("cat2",Color("color3",(1,0,1)),Color("color4",(0,1,0)))
    o=OneHot(1,2,(cat2,cat1))
    o.get_one_hot(IMG)



@name(Unet.model.fget, "1", globals())
def _():
    input("end")
    u = Unet()
    assert type(u.model) ==Model_ke

@name(Unet.predict, "1", globals())
def _():
    u = Unet()
    mask=u.predict(train_DB.random_train_input[0])
    imshow_mat(mask[:, :, :3])
    show()
    input(1111111111111111111111111)

@name(Unet.train, "1", globals())
def _():
    u = Unet()
    u.train(batch_size=1,steps_per_epoch=1,validation_steps=1,early_stopping_min_delta=0,RLR_min_delta=0,early_stopping_patience=0,RLR_patience=0,RLRFactor=0.1)
@name(Unet.structure.fget, "proxy", globals())
def _():
    u = Unet(1)
    u._structure = 1
    structure = u.structure
    assert structure == 1


@name(Unet.structure.fget, "1", globals())
def _():
    u = Unet(1)
    model = u.structure
    assert type(model) is Model_ke





@name(OldCategory.__init__, "1", globals())
def _():
    cat = OldCategory("alma", 1, (1, 1, 1), (2, 2, 2))
    assert cat.name == "alma" and cat.id == 1
    assert len(cat.colors) == 2 and (1, 1, 1) in cat.colors and (2, 2, 2) in cat.colors


@name(OldCategory.add_color, "1", globals())
def _():
    cat = OldCategory("alma", 1, (1, 1, 1), (2, 2, 2))
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


@name(LaneDB.get_train_input, "1", globals())
def _():
    img, mask = train_DB.get_train_input(0)
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
            img, mask = train_DB.get_train_input(i)
            if max_np(mask[:, :, category.id]) == 1:
                print(category.name)
                print(train_DB._get_example_paths(i))
                mask[:, :, 0] = mask[:, :, category.id]
                mask[:, :, 2] = img
                mask[:, :, 1] = img
                imshow_mat(mask[:, :, :3])
                show()
                break
