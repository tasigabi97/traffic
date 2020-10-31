from traffic.consts import *
from traffic.imports import *
from traffic.utils import *
from traffic.utils.lane_helper import labels as labels_helper
from traffic.logging import root_logger


class Color(metaclass=SingletonByIdMeta):
    def __init__(self, name: str, color: Tuple[int, int, int]):
        self.name, self.color = name, color

    @staticmethod
    def get_id(name: str, color: Tuple[int, int,int ]):
        return color

    def __hash__(self):
        return hash(self.color)

    def __eq__(self, other):
        return self.name == other or self.color == other

    def __str__(self):
        return self.__class__.__name__+"("+self.name+","+str(self.color)+")"


class Category(metaclass=SingletonByIdMeta):
    def __init__(self, name: str, *colors:Color):
        self.name, self.colors = name, set(colors)

    @staticmethod
    def get_id( name: str, *colors:Color):
        return name

    def __eq__(self, other):
        return self.name == other or any(color==other for color in self.colors)

    def __str__(self):
        ret= self.__class__.__name__+"("+self.name
        for color in self.colors:
            ret+=","+str(color)
        ret+=")"
        return ret

class OneHot(object):
    @staticmethod
    def get_color_id(color: Tuple[int, int, int])->int:
        return (color[0]*256*256)+(color[1]*256)+color[2]

    @staticmethod
    def get_color(color_id: int)->Tuple[int, int, int]:
        red=color_id//(256*256)
        color_id-=(256*256*red)
        green=color_id//256
        color_id-=(256*green)
        return (red,green,color_id)

    def __init__(self,row_number,col_number,categories:Iterable_type[Category]):
        self.row_number,self.col_number=row_number,col_number
        self.categories=list(categories)
        self.colors=[color for category in self.categories for color in category.colors]
        self.categories.sort(key=lambda category:category.name)
        self.colors.sort(key=lambda color:color.color)
        self._one_hot_container= zeros((self.row_number, self.col_number, len(self.categories)), dtype=float64)
        self._color_id_container= zeros((self.row_number, self.col_number, len(self.colors)), dtype=uint32)
        self._bool_container= zeros((self.row_number, self.col_number, len(self.colors)), dtype=uint8)
        self._uint32_img_container= zeros((self.row_number, self.col_number, 3), dtype=uint32)

    @property
    def uint32_img_container(self)->ndarray:
        return self._uint32_img_container

    @property
    def bool_container(self)->ndarray:
        return self._bool_container

    @property
    def one_hot_container(self)->ndarray:
        return self._one_hot_container
    @property
    def color_id_container(self)->ndarray:
        return self._color_id_container

    @uint32_img_container.setter
    def uint32_img_container(self, img:ndarray):
        self._uint32_img_container[:,:,:]=img

    @virtual_proxy_property
    def color_id_array(self)->ndarray:
        container= ones_np((self.row_number,self.col_number,len(self.colors)),dtype=uint32)
        for i, color in enumerate(self.colors):
            container[:,:,i]*=self.get_color_id(color.color)
        return container

    def get_one_hot(self,img:ndarray)->ndarray:
        def p(i):
            root_logger.debug(i)
            root_logger.info("self.uint32_img_container")
            root_logger.warning(self.uint32_img_container)
            root_logger.info("self.color_id_array")
            root_logger.warning(self.color_id_array)
            root_logger.info("self.color_id_container")
            root_logger.warning(self.color_id_container)
            root_logger.info("self.bool_container")
            root_logger.warning(self.bool_container)
            root_logger.info("self.one_hot_container")
            root_logger.warning(self.one_hot_container)
        self.uint32_img_container=img
        self.one_hot_container[:,:,:]=0
        p(0)
        self.uint32_img_container[:, :, 0]*=(256 * 256)
        self.uint32_img_container[:, :, 1]*=256
        self.uint32_img_container[:, :, 0]+= self.uint32_img_container[:, :, 1]
        self.uint32_img_container[:, :, 0]+= self.uint32_img_container[:, :, 2]
        for i in range(len(self.colors)):
            self.color_id_container[:,:,i]=self.uint32_img_container[:, :, 0]
        p(1)
        self.color_id_container[:,:,:]-=self.color_id_array
        p(2)
        equal_np(self.color_id_container, 0, out=self.bool_container)
        p(3)
        for color_i,color in enumerate(self.colors):
            for category_i,category in enumerate(self.categories):
                if category == color:
                    self.one_hot_container[:,:,category_i]+=self.bool_container[:,:,color_i]
        return self.one_hot_container.copy()

class OldCategory(object):
    def __init__(self, name: str, id: int, *colors: Tuple[int, int, int]):
        self.name, self.id = name, id
        self.colors = {color for color in colors}

    def add_color(self, color: Tuple[int, int, int]):
        self.colors.add(color)


color_to_id = dict()


class Categories(Singleton):
    def __init__(self):
        self._categories = []
        # create used categories
        for label in labels_helper:
            if label.category is not "ignored":
                self._categories.append(OldCategory(label.name, label.trainId, label.color))
        # add unused categories color
        for label in labels_helper:
            if label.category is "ignored":
                self[0].add_color(label.color)
        # rename categories
        self["void"].name = "Background"
        # coloroid
        for category in self._categories:
            for color in category.colors:
                color_to_id[color] = category.id

    def __getitem__(self, item) -> OldCategory:
        if isinstance(item, int):
            for category in self._categories:
                if category.id == item:
                    return category
            raise ValueError(item)
        elif isinstance(item, str):
            for category in self._categories:
                if category.name == item:
                    return category
            raise ValueError(item)
        raise TypeError(item)

    def __len__(self):
        return len(self._categories)


CATEGORIES = Categories()


class LaneDB(object):
    @staticmethod
    def create_categories():
        [Color(label.name, label.color) for label in labels_helper]
        Category("Háttér", Color["noise"], Color["ignored"], Color["void"])
        special_names = ["noise", "ignored", "void"]
        for color in Color:
            if color.name in special_names:
                continue
            Category(color.name, color)

    @staticmethod
    def _get_all_path():
        img_paths, mask_paths = [], []
        for img_dirname, mask_dirname in zip(
            ["ColorImage_road02", "ColorImage_road03", "ColorImage_road04"], ["Labels_road02", "Labels_road03", "Labels_road04"]
        ):
            img_dir, mask_dir = join_path(CONTAINER_LANE_PATH, img_dirname), join_path(CONTAINER_LANE_PATH, mask_dirname)
            img_dir, mask_dir = join_path(img_dir, "ColorImage"), join_path(mask_dir, "Label")
            for record_name in listdir(img_dir):
                img_rec_dir, mask_rec_dir = join_path(img_dir, record_name), join_path(mask_dir, record_name)
                for camera_name in listdir(img_rec_dir):
                    img_cam_dir, mask_cam_dir = join_path(img_rec_dir, camera_name), join_path(mask_rec_dir, camera_name)
                    img_names, mask_names = listdir(img_cam_dir), listdir(mask_cam_dir)
                    img_names.sort()
                    mask_names.sort()
                    for img_name, mask_name in zip(img_names, mask_names):
                        img_path, mask_path = join_path(img_cam_dir, img_name), join_path(mask_cam_dir, mask_name)
                        img_paths.append(img_path)
                        mask_paths.append(mask_path)
        return img_paths, mask_paths

    @staticmethod
    def _get_train_val_test_paths():
        img_paths, mask_paths = LaneDB._get_all_path()
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = [], [], [], [], [], []
        for i in range(len(img_paths)):
            if i  == 0:
                train_img_paths.append(img_paths[i])
                train_mask_paths.append(mask_paths[i])
            elif i == 1:
                val_img_paths.append(img_paths[i])
                val_mask_paths.append(mask_paths[i])
            else:
                test_img_paths.append(img_paths[i])
                test_mask_paths.append(mask_paths[i])
        return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths

    @staticmethod
    def _get_train_val_test_DB():
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = LaneDB._get_train_val_test_paths()
        train_DB, val_DB, test_DB = LaneDB(train_img_paths, train_mask_paths), LaneDB(val_img_paths, val_mask_paths), LaneDB(test_img_paths, test_mask_paths)
        return train_DB, val_DB, test_DB

    def __init__(self, img_paths, mask_paths):
        self._img_paths, self._mask_paths = img_paths, mask_paths

    @property
    def _random_example_id(self):
        return randrange(len(self._img_paths))

    def _get_example_paths(self, id: int):
        return self._img_paths[id], self._mask_paths[id]

    def _get_example(self, id: int) -> Tuple[ndarray, ndarray]:
        img_path, mask_path = self._get_example_paths(id)
        img, mask = imread_skimage(img_path), imread_skimage(mask_path)
        img, mask = array_np(img, dtype=uint8), array_np(mask, dtype=uint8)
        mask = mask[..., :3]
        return img, mask

    def _get_small_example(self, id: int) -> Tuple[ndarray, ndarray]:
        img, mask = self._get_example(id)
        img = rescale_skimage(img, max(CAMERA_ROWS / img.shape[0], CAMERA_COLS / img.shape[1]), anti_aliasing=False, preserve_range=True)
        img = array_np(img, dtype=uint8)
        mask = imresize_scipy(mask, img.shape, interp="nearest")
        return img, mask

    def _get_small_cropped_example(self, id: int) -> Tuple[ndarray, ndarray]:
        img, mask = self._get_small_example(id)
        assert img.shape == mask.shape == (513, 640, 3)
        half_delta = (img.shape[0] - CAMERA_ROWS) // 2
        img, mask = img[half_delta : half_delta + CAMERA_ROWS], mask[half_delta : half_delta + CAMERA_ROWS]
        assert img.shape == mask.shape == (480, 640, 3)
        return img, mask

    def get_train_input(self, id: int) -> Tuple[ndarray, ndarray]:
        img, mask = self._get_small_cropped_example(id)
        normalized_grayscale_img = mean_np(img, axis=2) / 255
        one_hot = zeros((CAMERA_ROWS, CAMERA_COLS, len(CATEGORIES)), dtype=uint8)
        for row_id in range(CAMERA_ROWS):
            for col_id in range(CAMERA_COLS):
                pixels_category_color = tuple(mask[row_id, col_id, :])
                pixels_category_id = color_to_id[pixels_category_color]
                one_hot[row_id, col_id, pixels_category_id] = 1
        one_hot = array_np(one_hot, dtype=float64)
        return normalized_grayscale_img, one_hot

    # rgb to int
    @property
    def random_train_input(self) -> Tuple[ndarray, ndarray]:
        return self.get_train_input(self._random_example_id)


train_DB, val_DB, test_DB = LaneDB._get_train_val_test_DB()


class Unet(Singleton):
    save_directory_path=CONTAINER_ROOT_PATH
    def __init__(self,name:str=None):
        self.name=Unet.__name__ if name is None else name
        self.hdf5_path=join_path(self.save_directory_path,"{}.hdf5".format(self.name))
        self.png_path=join_path(self.save_directory_path,"{}.structure.png".format(self.name))
        self.category_num = len(CATEGORIES)


    @virtual_proxy_property
    def model(self) -> Model_ke:
        return load_model_ke(filepath=self.hdf5_path)

    def predict(self,img:ndarray)->ndarray:
        input_batch=img[None,:,:,None]
        output_batch=self.model.predict_on_batch(input_batch)
        ret=output_batch[0]
        return ret

    def train(self, batch_size: int,
              steps_per_epoch:int,
              validation_steps: int,
              early_stopping_min_delta:float,
              RLR_min_delta: float,
              early_stopping_patience:int,
              RLR_patience:int,
              RLRFactor:float):
        self.RLR_min_delta, self.early_stopping_min_delta = RLR_min_delta, early_stopping_min_delta# 0.0001
        self.RLR_patience, self.early_stopping_patience = RLR_patience, early_stopping_patience # 4
        self.RLRFactor = RLRFactor  # 0.2
        self.batch_size,self.steps_per_epoch,self.validation_steps = batch_size,steps_per_epoch,validation_steps
        self.max_epochs=999
        #
        plot_model_ke(self.structure, show_shapes=True, to_file=self.png_path)
        self.structure.compile(optimizer=Adam_ke(),loss="categorical_crossentropy",metrics=["acc"])
        history=self.structure.fit_generator(generator=self.train_data,
                                             steps_per_epoch=self.steps_per_epoch,
                                             verbose=1,
                                             callbacks=self.callbacks,
                                             epochs=self.max_epochs,
                                             validation_data=self.validation_data,
                                             validation_steps=self.validation_steps)
        return history

    @virtual_proxy_property
    def train_data(self):
        return self.batch_generator(train_DB)

    @virtual_proxy_property
    def validation_data(self):
        return self.batch_generator(val_DB)

    def batch_generator(self,database:LaneDB):
        img_batch, one_hot_batch=zeros((self.batch_size,CAMERA_ROWS, CAMERA_COLS, 1),dtype=float64),zeros((self.batch_size,CAMERA_ROWS, CAMERA_COLS, self.category_num),dtype=float64)
        while True:
            for i in range(self.batch_size):
                img, one_hot = database.random_train_input
                img_batch[i]=img[:,:,None]#expand dims
                one_hot_batch[i]=one_hot
            yield img_batch,one_hot_batch




    @virtual_proxy_property
    def structure(self) -> Model_ke:
        filters=[64]
        kernel_size=[3]
        activation="relu"
        input_l = Input_ke(shape=(CAMERA_ROWS, CAMERA_COLS, 1))
        conv_0_l = Conv2D_ke(filters=filters[0], kernel_size=kernel_size[0], activation=activation)(input_l)
        conv_t_0_l = Conv2DTranspose_ke(filters=filters[0],
                                      kernel_size=kernel_size[0],
                                      activation=activation,
                                      )(conv_0_l)

        output_l = Conv2D_ke(filters=self.category_num, kernel_size=3,padding="same", activation="softmax",)(conv_t_0_l)
        model = Model_ke(input_l, output_l)
        return model

    @virtual_proxy_property
    def early_stopper(self):
        return EarlyStopping_ke(monitor="acc",min_delta=self.early_stopping_min_delta,patience=self.early_stopping_patience,verbose=1)

    @virtual_proxy_property
    def saver(self):
        return ModelCheckpoint_ke(filepath=self.hdf5_path,monitor="acc",save_best_only=True)

    @virtual_proxy_property
    def learning_rate_reducer(self):
        return ReduceLROnPlateau_ke(monitor='acc',
                                    factor=self.RLRFactor,
                                    verbose=1,
                                    epsilon=self.RLR_min_delta,
                                    patience=self.RLR_patience)

    @virtual_proxy_property
    def callbacks(self):
        return [self.early_stopper,self.saver,self.learning_rate_reducer]