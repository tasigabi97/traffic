from traffic.consts import *
from traffic.imports import *
from traffic.utils import *
from traffic.utils.lane_helper import labels as labels_helper
from traffic.logging import root_logger


def get_probabilities(histogram: dict):
    sum_ = sum(histogram.values())
    return {k: (v / sum_) for k, v in histogram.items()}


class Color(metaclass=SingletonByIdMeta):
    def __init__(self, name: str, rgb_tuple: Tuple[int, int, int]):
        self.name, self.rgb_tuple = name, rgb_tuple

    @staticmethod
    def get_id(name: str, rgb_tuple: Tuple[int, int, int]):
        return rgb_tuple

    def __hash__(self):
        return hash(self.rgb_tuple)

    def __eq__(self, other):
        return self.name == other or self.rgb_tuple == other

    def __str__(self):
        return self.__class__.__name__ + "(" + self.name + "," + str(self.rgb_tuple) + ")"


class Category(metaclass=SingletonByIdMeta):
    def __init__(self, name: str, *colors: Color):
        self.name, self.colors = name, set(colors)

    @staticmethod
    def get_id(name: str, *colors: Color):
        return name

    def __eq__(self, other):
        return self.name == other or any(color == other for color in self.colors)

    def __str__(self):
        ret = self.__class__.__name__ + "(" + self.name
        for color in self.colors:
            ret += "," + str(color)
        ret += ")"
        return ret

    def __len__(self):
        return len(self.colors)


class OneHot:
    @staticmethod
    def get_rgb_id(rgb_tuple: Tuple[int, int, int]) -> int:
        return (rgb_tuple[0] * 256 * 256) + (rgb_tuple[1] * 256) + rgb_tuple[2]

    @staticmethod
    def get_rgb_tuple(color_id: int) -> Tuple[int, int, int]:
        red = color_id // (256 * 256)
        color_id -= 256 * 256 * red
        green = color_id // 256
        color_id -= 256 * green
        return (red, green, color_id)

    def __init__(self, row_number, col_number, categories: Iterable_type[Category]):
        self.row_number, self.col_number = row_number, col_number
        self.categories = list(categories)
        self.colors = [color for category in self.categories for color in category.colors]
        self.categories.sort(key=lambda category: category.name.lower())
        self.colors.sort(key=lambda color: color.rgb_tuple)

    @virtual_proxy_property
    def uint32_img_container(self) -> ndarray:
        return zeros((self.row_number, self.col_number, 3), dtype=uint32)

    @virtual_proxy_property
    def rgb_id_container(self) -> ndarray:
        return zeros((self.row_number, self.col_number, len(self.colors)), dtype=uint32)

    @virtual_proxy_property
    def RGB_ID_SUBTRAHEND(self) -> ndarray:
        container = ones_np((self.row_number, self.col_number, len(self.colors)), dtype=uint32)
        for i, color in enumerate(self.colors):
            container[:, :, i] *= self.get_rgb_id(color.rgb_tuple)
        return container

    @virtual_proxy_property
    def bool_container(self) -> ndarray:  # todo
        return zeros((self.row_number, self.col_number, len(self.colors)), dtype=uint8)

    @virtual_proxy_property
    def one_hot_container(self) -> ndarray:
        return zeros((self.row_number, self.col_number, len(self.categories)), dtype=float32)

    def get_encoded(self, rgb_array: ndarray) -> ndarray:
        self.uint32_img_container[:, :, :] = rgb_array
        self.one_hot_container[:, :, :] = 0
        self.uint32_img_container[:, :, 0] *= 256 * 256
        self.uint32_img_container[:, :, 1] *= 256
        self.uint32_img_container[:, :, 0] += self.uint32_img_container[:, :, 1]
        self.uint32_img_container[:, :, 0] += self.uint32_img_container[:, :, 2]
        self.rgb_id_container[:, :, :] = self.uint32_img_container[:, :, 0, None]
        self.rgb_id_container[:, :, :] -= self.RGB_ID_SUBTRAHEND
        equal_np(self.rgb_id_container, 0, out=self.bool_container)
        for color_i, color in enumerate(self.colors):
            for category_i, category in enumerate(self.categories):
                if category == color:
                    self.one_hot_container[:, :, category_i] += self.bool_container[:, :, color_i]
        return self.one_hot_container.copy()

    def get_category_histogram(self, rgb_array: ndarray) -> dict:
        one_hot = self.get_encoded(rgb_array)
        one_hot = array_np(one_hot, dtype=uint8)
        return {category.name: sum_np(one_hot[:, :, category_i]) for category_i, category in enumerate(self.categories)}


class ImgSource(NNInputSource):
    @staticmethod
    def get_rescale_factor(nrows: int, ncols: int) -> float:
        return max(CAMERA_ROWS / nrows, CAMERA_COLS / ncols)

    @staticmethod
    def get_normalized_img(img: ndarray) -> ndarray:
        return img / 255

    @staticmethod
    def get_cropped_img(img: ndarray) -> ndarray:
        crop_start = (img.shape[0] - CAMERA_ROWS) // 2
        cropped_img = img[crop_start : crop_start + CAMERA_ROWS]
        assert cropped_img.shape == (480, 640, 3)
        return cropped_img

    def get_an_input(self) -> ndarray:
        highres_img = self.data
        nrows, ncols, _ = highres_img.shape
        rescale_factor = ImgSource.get_rescale_factor(nrows, ncols)
        rescaled_img = rescale_skimage(highres_img, rescale_factor, anti_aliasing=False, preserve_range=True)
        rescaled_img = array_np(rescaled_img, dtype=uint8)
        assert rescaled_img.shape == (513, 640, 3)
        cropped_img = ImgSource.get_cropped_img(rescaled_img)
        normalized_img = ImgSource.get_normalized_img(cropped_img)
        return normalized_img


class MaskSource(NNInputSource):
    source_encoder = OneHot(row_number=452, col_number=564, categories=[Category(label.name, Color(label.name, label.color)) for label in labels_helper])
    Category.clear()

    @staticmethod
    def get_resized_shape(nrows: int, ncols: int) -> Tuple[int, int, int]:
        rescale_factor = ImgSource.get_rescale_factor(nrows, ncols)
        return (round(nrows * rescale_factor), round(ncols * rescale_factor), 3)

    def get_calculated_attributes(self):
        return get_probabilities(self.source_encoder.get_category_histogram(self.data[::6, ::6, :]))

    @property
    def category_probabilities(self) -> dict:
        category_probabilities = {category.name: 0 for category in Category}
        attributes = self.attributes
        for color_name, color_probability in attributes.items():
            for category_name in category_probabilities.keys():
                if Category[category_name] == Color[color_name]:
                    category_probabilities[category_name] += attributes[color_name]
        return category_probabilities

    def get_an_input(self, one_hot_coder: OneHot) -> ndarray:
        highres_img = self.data
        nrows, ncols, _ = highres_img.shape
        resized_shape = MaskSource.get_resized_shape(nrows, ncols)
        resized_img = imresize_scipy(highres_img, resized_shape, interp="nearest")
        assert resized_img.shape == (513, 640, 3)
        cropped_img = ImgSource.get_cropped_img(resized_img)
        one_hot = one_hot_coder.get_encoded(cropped_img)
        return one_hot

    def visualize_an_input(self, one_hot_coder: OneHot):
        an_input = self.get_an_input(one_hot_coder)
        root_logger.info(an_input.shape)
        for i in range(an_input.shape[-1]):
            show_array(an_input[:, :, i])


class IMSourcePair:
    def __init__(self, img_source: ImgSource, mask_source: MaskSource):
        self.img_source, self.mask_source = img_source, mask_source


class LaneDB:
    @staticmethod
    def create_categories():
        [Color(label.name, label.color) for label in labels_helper]
        Category(
            "Hatter",
            Color["noise"],
            Color["ignored"],
            Color["void"],
            Color["a_n_lu"],
            Color["a_y_t"],
            Color["db_w_g"],
            Color["db_w_s"],
            Color["db_y_g"],
            Color["ds_w_dn"],
            Color["ds_w_s"],
            Color["s_n_p"],
            Color["a_w_l"],
            Color["a_w_r"],
            Color["a_w_t"],
            Color["a_w_u"],
            Color["a_w_tl"],
            Color["a_w_tr"],
            Color["a_w_tu"],
            Color["a_w_tlr"],
            Color["a_w_lr"],
            Color["a_w_m"],
            Color["c_wy_z"],
            Color["b_n_sr"],
            Color["r_wy_np"],
            Color["s_w_s"],
            Color["s_w_p"],
            Color["d_wy_za"],
            Color["vom_wy_n"],
        )
        Category(
            "Zarovonal",
            Color["s_w_d"],
            Color["s_y_d"],
            Color["b_w_g"],
            Color["b_y_g"],
            Color["ds_y_dn"],
            Color["om_n_n"],
            Color["sb_w_do"],
            Color["sb_y_do"],
            Color["s_w_c"],
            Color["s_y_c"],
        )

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
                    mask_names = [name for name in mask_names if name[-3:] == "png"]
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
            if i < 500:
                train_img_paths.append(img_paths[i])
                train_mask_paths.append(mask_paths[i])
            elif i < 600:
                val_img_paths.append(img_paths[i])
                val_mask_paths.append(mask_paths[i])
            elif i < 700:
                test_img_paths.append(img_paths[i])
                test_mask_paths.append(mask_paths[i])
        return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths

    @staticmethod
    def get_train_val_test_DB():
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = LaneDB._get_train_val_test_paths()
        train_DB, val_DB, test_DB = LaneDB(train_img_paths, train_mask_paths), LaneDB(val_img_paths, val_mask_paths), LaneDB(test_img_paths, test_mask_paths)
        return train_DB, val_DB, test_DB

    def __init__(self, img_paths, mask_paths):
        self.image_and_mask_source_pairs = [IMSourcePair(ImgSource(img_path), MaskSource(mask_path)) for img_path, mask_path in zip(img_paths, mask_paths)]
        self.image_and_mask_source_pairs.sort(key=lambda image_and_mask_source_pair: image_and_mask_source_pair.mask_source.category_probabilities["Hatter"])
        # todo sorted indexes by categories

    @virtual_proxy_property  # todo elvileg nem kell
    def category_probabilities(self) -> dict:
        sum_dict = {category.name: 0 for category in Category}
        for image_and_mask_source_pair in self.image_and_mask_source_pairs:
            mask_source = image_and_mask_source_pair.mask_source
            for category_name, category_probability in mask_source.category_probabilities.items():
                sum_dict[category_name] += category_probability
        probabilities = get_probabilities(sum_dict)
        return probabilities

    @virtual_proxy_property
    def one_hot_coder(self) -> OneHot:
        return OneHot(CAMERA_ROWS, CAMERA_COLS, Category)

    @property  # todo elvileg nem kell
    def _random_source_id(self):
        return randrange(len(self.image_and_mask_source_pairs))

    # todo floattal és categoryval
    def get_sources(self, id: int) -> Tuple[ImgSource, MaskSource]:
        return self.image_and_mask_source_pairs[id].img_source, self.image_and_mask_source_pairs[id].mask_source

    # todo elvileg nem kell
    @property
    def random_sources(self):
        return self.get_sources(self._random_source_id)


class Unet(Singleton):
    save_directory_path = CONTAINER_ROOT_PATH
    LaneDB.create_categories()
    train_DB, val_DB, test_DB = LaneDB.get_train_val_test_DB()

    def __init__(self, name: str = None):
        self.name = Unet.__name__ if name is None else name
        self.hdf5_path = join_path(self.save_directory_path, "{}.hdf5".format(self.name))
        self.png_path = join_path(self.save_directory_path, "{}.structure.png".format(self.name))

    def get_prediction(self, rgb_array: ndarray) -> ndarray:
        normalized_img = ImgSource.get_normalized_img(rgb_array)
        input_batch = normalized_img[None, :, :, :]
        predicted_batch = self.model.predict_on_batch(input_batch)
        distribution_matrix = predicted_batch[0]
        return distribution_matrix

    def visualize_prediction(self):
        canvas = zeros((960, 640, 3))
        img_source, mask_source = self.train_DB.get_sources(0)
        expected_one_hot = mask_source.get_an_input(self.train_DB.one_hot_coder)
        input_img = img_source.get_an_input()
        predicted_distribution_matrix = self.get_prediction(input_img * 255)
        canvas[:480] = input_img
        canvas[480:] = input_img
        root_logger.info(predicted_distribution_matrix.shape)
        assert expected_one_hot.shape == predicted_distribution_matrix.shape == (480, 640, len(Category))
        for category_i in range(predicted_distribution_matrix.shape[2]):
            expected_mask = expected_one_hot[:, :, category_i]
            predicted_mask = predicted_distribution_matrix[:, :, category_i]
            canvas[:480, :, 0] = expected_mask
            canvas[480:, :, 0] = predicted_mask
            show_array(canvas)

    def train(
        self,
        batch_size: int,
        steps_per_epoch: int,
        validation_steps: int,
        early_stopping_min_delta: float,
        RLR_min_delta: float,
        early_stopping_patience: int,
        RLR_patience: int,
        RLRFactor: float,
    ):
        self.RLR_min_delta, self.early_stopping_min_delta = RLR_min_delta, early_stopping_min_delta  # 0.0001
        self.RLR_patience, self.early_stopping_patience = RLR_patience, early_stopping_patience  # 4
        self.RLRFactor = RLRFactor  # 0.2
        self.batch_size, self.steps_per_epoch, self.validation_steps = batch_size, steps_per_epoch, validation_steps
        self.max_epochs = 999
        self.metrics = ["categorical_accuracy"]
        self.monitor = "loss"
        self.verbose = 1
        self.structure.compile(optimizer=Adam_ke(), loss=self.loss, metrics=self.metrics)
        history = self.structure.fit_generator(
            generator=self.train_data,
            steps_per_epoch=self.steps_per_epoch,
            verbose=self.verbose,
            callbacks=self.callbacks,
            epochs=self.max_epochs,
            validation_data=self.validation_data,
            validation_steps=self.validation_steps,
        )
        return history

    @staticmethod
    def loss(y_true, y_pred):
        DEBUG = False

        def p(tensor, msg: str):
            if DEBUG:
                tensor_shape = shape_tf(tensor)
                return Print_tf(tensor, [tensor_shape, tensor], msg, summarize=20)
            else:
                return tensor

        y_true = p(y_true, "y_true 1.")
        y_pred = p(y_pred, "y_pred 1.")
        cce = categorical_crossentropy_ke(y_true, y_pred)
        cce = p(cce, "cce 1.")
        cce_mask = MaxPool2D_ke(pool_size=8, strides=1, padding="same")(y_true)
        for category_i, category in enumerate(Unet.train_DB.one_hot_coder.categories):
            if category.name != "Hatter":
                cce_mask = cce_mask[:, :, :, category_i]
                break
        cce_mask = p(cce_mask, "cce_mask 1.")
        relevant_cce = multiply_tf(cce, cce_mask)
        relevant_cce = p(relevant_cce, "relevant_cce 1.")
        sum_cce = sum_ke(relevant_cce)
        sum_cce = p(sum_cce, "sum_cce 1.")
        return sum_cce

    @classmethod
    def calculate_weight(cls, probability: float) -> float:
        return 1 if probability < 0.5 else 0

    def batch_generator(self, DB: LaneDB):  # this generetor should instentiate onehot coders
        img_array_container = zeros((self.batch_size, CAMERA_ROWS, CAMERA_COLS, 3), dtype=float32)
        one_hot_container = zeros((self.batch_size, CAMERA_ROWS, CAMERA_COLS, len(Category)), dtype=float32)
        while True:
            for sample_i in range(self.batch_size):
                img_source, mask_source = DB.get_sources(sample_i)
                img_array_container[sample_i] = img_source.get_an_input()
                one_hot_container[sample_i] = mask_source.get_an_input(DB.one_hot_coder)
            yield img_array_container.copy(), one_hot_container.copy()

    def visualize_batches(self):
        canvas = zeros((480, 640, 3))
        for image_batch, one_hot_batch in self.train_data:
            root_logger.info(image_batch.shape)
            root_logger.info(one_hot_batch.shape)
            for sample_i in range(image_batch.shape[0]):
                image_sample = image_batch[sample_i]
                one_hot_sample = one_hot_batch[sample_i]
                root_logger.info(image_sample.shape)
                root_logger.info(one_hot_sample.shape)
                for category_i in range(one_hot_sample.shape[2]):
                    canvas[:, :, 0] = one_hot_sample[:, :, category_i]
                    canvas[:, :, 1:3] = image_sample[:, :, 1:3]
                    show_array(canvas)
            break

    def get_output_layer(self, input_layer: int):
        def c(layers, filters, kernel_size, activation="relu"):
            layer = concatenate_ke(layers) if type(layers) is list else layers
            return Conv2D_ke(filters=filters, kernel_size=kernel_size, strides=1, activation=activation)(layer)

        def t(layers, filters, kernel_size, activation="relu", strides=(1, 1)):
            layer = concatenate_ke(layers) if type(layers) is list else layers
            return Conv2DTranspose_ke(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation)(layer)

        def p(layer):
            return MaxPool2D_ke((2, 2))(layer)

        first_filters = 16
        c_478_638 = c(input_layer, first_filters, 3)
        c_476_636 = c(c_478_638, first_filters, 3)
        c_474_634 = c(c_476_636, first_filters, 3)
        p_237_317 = p(c_474_634)
        c_235_315 = c(p_237_317, first_filters * 2, 3)
        c_233_313 = c(c_235_315, first_filters * 2, 3)
        c_230_310 = c(c_233_313, first_filters * 2, 4)
        p_115_155 = p(c_230_310)
        c_113_153 = c(p_115_155, first_filters * 4, 3)
        c_111_151 = c(c_113_153, first_filters * 4, 3)
        c_108_148 = c(c_111_151, first_filters * 4, 4)
        p_54_74 = p(c_108_148)
        c_52_72 = c(p_54_74, first_filters * 8, 3)
        c_50_70 = c(c_52_72, first_filters * 8, 3)
        c_48_68 = c(c_50_70, first_filters * 8, 3)
        p_24_34 = p(c_48_68)
        c_22_32 = c(p_24_34, first_filters * 16, 2)
        t_48_68 = t(c_22_32, first_filters * 8, 4, strides=(2, 2))
        t_50_70 = t([t_48_68, c_48_68], first_filters * 8, 3)
        t_52_72 = t([t_50_70, c_50_70], first_filters * 8, 3)
        t_54_74 = t([t_52_72, c_52_72], first_filters * 8, 3)
        t_108_148 = t([t_54_74, p_54_74], first_filters * 4, 2, strides=(2, 2))
        t_111_151 = t([t_108_148, c_108_148], first_filters * 4, 4)
        t_113_153 = t([t_111_151, c_111_151], first_filters * 4, 3)
        t_115_155 = t([t_113_153, c_113_153], first_filters * 4, 3)
        t_230_310 = t([t_115_155, p_115_155], first_filters * 2, 2, strides=(2, 2))
        t_233_313 = t([t_230_310, c_230_310], first_filters * 2, 4)
        t_235_315 = t([t_233_313, c_233_313], first_filters * 2, 3)
        t_237_317 = t([t_235_315, c_235_315], first_filters * 2, 3)
        t_474_634 = t([t_237_317, p_237_317], first_filters, 2, strides=(2, 2))
        t_476_636 = t([t_474_634, c_474_634], first_filters, 3)
        t_478_638 = t([t_476_636, c_476_636], first_filters, 3)
        t_480_640 = t([t_478_638, c_478_638], first_filters, 3)
        c_480_640 = c(t_480_640, len(Category), 1, "softmax")
        if False:
            model = Model_ke(input_layer, c_480_640)
            plot_model_ke(model, show_shapes=True, to_file=self.png_path)
            input("Check")
        return c_480_640

    @virtual_proxy_property
    def structure(self) -> Model_ke:
        input_l = Input_ke(shape=(CAMERA_ROWS, CAMERA_COLS, 3))
        output_l = self.get_output_layer(input_l)
        model = Model_ke(input_l, output_l)
        plot_model_ke(model, show_shapes=True, to_file=self.png_path)
        return model

    @virtual_proxy_property
    def early_stopper(self):
        return EarlyStopping_ke(monitor=self.monitor, min_delta=self.early_stopping_min_delta, patience=self.early_stopping_patience, verbose=self.verbose)

    @virtual_proxy_property
    def saver(self):
        return ModelCheckpoint_ke(filepath=self.hdf5_path, monitor=self.monitor, save_best_only=True, verbose=self.verbose)

    @virtual_proxy_property
    def learning_rate_reducer(self):
        return ReduceLROnPlateau_ke(monitor=self.monitor, factor=self.RLRFactor, verbose=self.verbose, epsilon=self.RLR_min_delta, patience=self.RLR_patience)

    @virtual_proxy_property
    def tensorboard(self):
        return TensorBoard_ke(log_dir="./logs")

    @virtual_proxy_property
    def callbacks(self):
        return [self.early_stopper, self.saver, self.learning_rate_reducer]  # ,self.tensorboard]

    @virtual_proxy_property
    def train_data(self):
        return self.batch_generator(self.train_DB)

    @virtual_proxy_property
    def validation_data(self):
        return self.batch_generator(self.val_DB)

    @virtual_proxy_property
    def model(self) -> Model_ke:
        return load_model_ke(filepath=self.hdf5_path)


import keras.losses

keras.losses.loss = Unet.loss
