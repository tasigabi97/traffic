"""
Ez a fájl segít neurális hálót írni az
http://apolloscape.auto/lane_segmentation.html
adatbázisra.
"""
from traffic.consts import *
from traffic.imports import *
from traffic.utils import *
from traffic.utils.lane_helper import labels as labels_helper
from traffic.logging import root_logger


def get_summed_dict(dicts: List[dict]) -> dict:
    """Több megegyező kulcsokkal rendelkező szótárból csinál 1-et ugyanolyan kulcsokkal.
    Összeadást végez kulcsonként.
    """
    return {k: sum([d[k] for d in dicts]) for k in dicts[0].keys()}


def get_probabilities(histogram: dict) -> dict:
    """
    Pl {"a":20,"b":80} -> {"a":0.2,"b":0.8}
    """
    sum_ = sum(histogram.values())
    return {k: (v / sum_) for k, v in histogram.items()}


class HyperParameters:
    """
    A háló Categorical Cross-Entropy-t használ betanuláskor. Ha minden pixel ugyanannyira számítana,
     akkor simán háttérnek jelölne mindent, ezért súlyozni kell a pixeleket.
      Alapból 0 lesz a súlya (nem számít bele) egy háttér pixelnek, és 1 egy nem háttérnek
      (lásd a calculate_weight függvénynél).
    """

    crop_material_size_factor = 1.3  # Ez dönti el mekkora képekből vágjunk ki inputot a lane unet számára.
    max_noise_pixels = int(0.05 * CAMERA_ROWS * CAMERA_COLS)  # Ez dönti el mennyi zajt
    # adhatunk hozzá az inputokhoz.
    min_weight_contour_ratio = 1.1  # Ez dönti el mennyi háttér pixelnek legyen 1 a súlya a bejelölések körül.
    weight_dot_cloud_ratio = 0.45  # Ez dönti el mennyi véletlenszerűen kiválasztott
    # háttér pixelnek legyen 1 a súlya.
    first_filters = 32  # Ez dönti el a háló méretét.


class LaneUtil:
    _coordinates = [(row_i, col_i) for row_i in range(CAMERA_ROWS) for col_i in range(CAMERA_COLS)]
    shuffle(_coordinates)
    dot_cloud_coordinate_cycle = cycle(_coordinates)  # Ez arra kell,
    # hogy ki tudjuk választani a kép 1-1 pixelét véletlenszerűen.
    _noises = [array_np(noise, dtype=float32) for noise in [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    noise_cycle = cycle(_noises)  # Ez a só-bors-r-g-b zajhoz kell.


class LaneDBMaskColor(metaclass=SingletonByIdMeta):
    """
    A lane adatbázis maszkjainál külön jelentése van minden színnek. Pl minden fekete (0,0,0) pixel háttér,
     minden fehér (255,255,255) pixel pedig annak az autónak a motorháztetője,
      aminek a tetején volt a kamera (minden képen rajta van, akármerre megy az autó).
    """

    def __init__(self, name: str, rgb_tuple: Tuple[int, int, int]):
        self.name, self.rgb_tuple = name, rgb_tuple

    @staticmethod
    def get_id(name: str, rgb_tuple: Tuple[int, int, int]):
        """
        Minden szín egyértelműen meghatározható egy számhármassal.
        """
        return rgb_tuple

    def __hash__(self):
        return hash(self.rgb_tuple)

    def __eq__(self, other):
        return self.name == other or self.rgb_tuple == other

    def __str__(self):
        return self.__class__.__name__ + "(" + self.name + "," + str(self.rgb_tuple) + ")"


class LaneCategory(metaclass=SingletonByIdMeta):
    """
    Pl az előző példából a "háttér" és "motorháztető" szín is a Háttér kategóriába tartozik.
    """

    def __init__(self, name: str, *colors: LaneDBMaskColor):
        self.name, self.colors = name, set(colors)

    @staticmethod
    def get_id(name: str, *colors: LaneDBMaskColor):
        """
        Minden kategória egyértelműen meghatározható egy névvel.
        """
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


class LaneOneHotCoder:
    """
    Elkészíti a lane unet bemeneteinek a one hot kódolását a kategóriák alapján.
    """

    @staticmethod
    def get_rgb_id(rgb_tuple: Tuple[int, int, int]) -> int:
        """
        Számhármasból csinál számot. Pl (0,1,10)->265
        """
        # todo lehet gyorsabb lenne, ha át sem kódolnánk a képet rgb_id-ra,
        # és azt néznénk kivonás után, hogy (r==0 and g==0 and b==0),
        # és nem azt, hogy rgb_id==0
        return (rgb_tuple[0] * 256 * 256) + (rgb_tuple[1] * 256) + rgb_tuple[2]

    @staticmethod
    def get_rgb_tuple(color_id: int) -> Tuple[int, int, int]:
        """
        Ez előző inverze. (dekódoláshoz kéne)
        """
        red = color_id // (256 * 256)
        color_id -= 256 * 256 * red
        green = color_id // 256
        color_id -= 256 * green
        return (red, green, color_id)

    def __init__(self, row_number, col_number, categories: Iterable_type[LaneCategory]):
        self.row_number, self.col_number = row_number, col_number
        self.categories = list(categories)
        self.colors = [color for category in self.categories for color in category.colors]
        self.categories.sort(key=lambda category: category.name.lower())
        self.colors.sort(key=lambda color: color.rgb_tuple)

    @virtual_proxy_property
    def uint32_img_container(self) -> ndarray:
        """
        Itt lesz tárolva a maszk. Azért nem 8 bites, mert 1 szám fog meghatározni 1 színt (nem pedig 3),
         és ez a szám többnyire nem fér el 8 biten.
        """
        return zeros((self.row_number, self.col_number, 3), dtype=uint32)

    @virtual_proxy_property
    def rgb_id_container(self) -> ndarray:
        """
        Itt szintén a maszk lesz eltárolva (nem rgb) de annyiszor ahány fajta szín lehet.
        """
        return zeros((self.row_number, self.col_number, len(self.colors)), dtype=uint32)

    @virtual_proxy_property
    def RGB_ID_SUBTRAHEND(self) -> ndarray:
        """
        Itt eltárolunk 1-1 egyszínű (nem rgb) képet minden egyes színhez.
        """
        container = ones_np((self.row_number, self.col_number, len(self.colors)), dtype=uint32)
        for i, color in enumerate(self.colors):
            container[:, :, i] *= self.get_rgb_id(color.rgb_tuple)
        return container

    @virtual_proxy_property
    def bool_container(self) -> ndarray:
        """
        Itt tároljuk el azt, hogy hol egyezett meg az előző két képtömb.
        """
        return zeros((self.row_number, self.col_number, len(self.colors)), dtype=uint8)

    @virtual_proxy_property
    def one_hot_container(self) -> ndarray:
        """
        Itt tároljuk a végeredményt. A gradiens végül floattal lesz számolva.
        """
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
        """
        Hány db pixel volt 1-1 kategóriából egy képen.
        """
        one_hot = self.get_encoded(rgb_array)
        one_hot = array_np(one_hot, dtype=uint8)
        return {category.name: sum_np(one_hot[:, :, category_i]) for category_i, category in enumerate(self.categories)}


class ImgMaterial(NNInputMaterial):
    """
    Ez felel meg 1-1 közúti felvételnek.
    """

    @staticmethod
    def get_highresolution_to_cropmaterial_rescale_factor(nrows: int, ncols: int) -> float:
        """
        Nem a nagyfelbontású képből vágunk ki egy 640x480-as ablakot, mert akkor túlságosan beleközelítenénk,
        és lehet nem is lenne rajta úttest.
        """
        return max(CAMERA_ROWS / nrows, CAMERA_COLS / ncols) * HyperParameters.crop_material_size_factor

    @staticmethod
    def get_normalized_img(img: ndarray) -> ndarray:
        return img / 255

    @staticmethod
    def get_degree(hardness: float, clockwise: bool) -> int:
        """
        Parameters
        ----------
        hardness: pl 0-nál 0, 1-nél fejjel lefelé azaz 180 fok.
        """
        degree = int(180 * hardness)
        if clockwise:
            degree *= -1
        return degree

    @staticmethod
    def get_cropped_img(
        img: ndarray, row_hardness: float, row_up: bool, col_hardness: float, col_left: bool
    ) -> ndarray:
        """
        Parameters
        ----------
        row_hardness: pl 0-nál a kép közepéből vágunk függőlegesen,
         1-nél pedig a kép tetejéről/aljáról (row_up-tól függően).
        col_hardness: pl 0-nál a kép közepéből vágunk vízszintesen,
         1-nél pedig a kép bal/jobb széléről (col_left-től függően).
        """
        img_rows, img_cols = img.shape[0], img.shape[1]
        row_middle_start = (img_rows - CAMERA_ROWS) // 2
        col_middle_start = (img_cols - CAMERA_COLS) // 2
        row_offset = round(row_middle_start * row_hardness)
        col_offset = round(col_middle_start * col_hardness)
        if row_up:
            row_offset *= -1
        if col_left:
            col_offset *= -1
        row_start = row_middle_start + row_offset
        col_start = col_middle_start + col_offset
        cropped_img = img[row_start : row_start + CAMERA_ROWS, col_start : col_start + CAMERA_COLS]
        assert cropped_img.shape == (480, 640, 3)
        return cropped_img

    def get_an_input(
        self,
        rotation_hardness: float,
        clockwise: bool,
        row_hardness: float,
        row_up: bool,
        col_hardness: float,
        col_left: bool,
        noise_hardness: float,
    ) -> ndarray:
        """
        Visszaad egy [0,1] intervallumban mozgó képet az augmentációs paramétereknek megfelelően.
        """
        highres_img = self.data
        nrows, ncols, _ = highres_img.shape
        rescale_factor = ImgMaterial.get_highresolution_to_cropmaterial_rescale_factor(nrows, ncols)
        rescaled_img = rescale_skimage(highres_img, rescale_factor, anti_aliasing=False, preserve_range=True)
        rescaled_img = array_np(rescaled_img, dtype=uint8)
        assert rescaled_img.shape == (666, 832, 3)
        cropped_img = ImgMaterial.get_cropped_img(rescaled_img, row_hardness, row_up, col_hardness, col_left)
        rotated_img = imrotate_sci(cropped_img, ImgMaterial.get_degree(rotation_hardness, clockwise), interp="bicubic")
        normalized_img = ImgMaterial.get_normalized_img(rotated_img)
        for _ in range(int(HyperParameters.max_noise_pixels * noise_hardness)):
            coordinate = next(LaneUtil.dot_cloud_coordinate_cycle)
            normalized_img[coordinate] = next(LaneUtil.noise_cycle)

        return normalized_img


class MaskMaterial(NNInputMaterial):
    """
    Ez felel meg 1-1 közúti felvételhez tartozó masznak.
    """

    _encoder = LaneOneHotCoder(
        row_number=452,
        col_number=564,
        categories=[LaneCategory(label.name, LaneDBMaskColor(label.name, label.color)) for label in labels_helper],
    )
    LaneCategory.clear()  # Ez a gyorsítóadatok számolásához kell.

    @staticmethod
    def get_resized_shape(nrows: int, ncols: int) -> Tuple[int, int, int]:
        rescale_factor = ImgMaterial.get_highresolution_to_cropmaterial_rescale_factor(nrows, ncols)
        return (round(nrows * rescale_factor), round(ncols * rescale_factor), 3)

    def get_calculated_attributes(self):
        """
        Minden maszkhoz eltároljuk a színeloszlását. Ez azért jó,
         hogy később sorba lehessen rendezni a közúti felvételeket.
         Pl kérek egy olyan felvételt, amin egy nagy zebra/parkolósáv látható.
         Azért nem a kategóriaeloszlás van kimentve,
          mert a kategóriákat megváltoztathatjuk, de a színeket nem.
        """
        return get_probabilities(self._encoder.get_category_histogram(self.data[::6, ::6, :]))

    @property
    def category_probabilities(self) -> dict:
        category_probabilities = {category.name: 0 for category in LaneCategory}
        attributes = self.attributes
        for color_name, color_probability in attributes.items():
            for category_name in category_probabilities.keys():
                if LaneCategory[category_name] == LaneDBMaskColor[color_name]:
                    category_probabilities[category_name] += attributes[color_name]
        return category_probabilities

    def get_an_input(
        self,
        one_hot_coder: LaneOneHotCoder,
        rotation_hardness: float,
        clockwise: bool,
        row_hardness: float,
        row_up: bool,
        col_hardness: float,
        col_left: bool,
    ) -> ndarray:
        """
        Hasonlít az előbbi get_an_input-hoz, csak itt fontos az, hogy megmaradjon minden szín, nem lehet interpoláció.
        Azért van kiterítve a one hot kódolás, mert nem működik magasabb dimenzióra a Keras-os
         sample_weight_mode="temporal".
        """
        highres_img = self.data
        nrows, ncols, _ = highres_img.shape
        resized_shape = MaskMaterial.get_resized_shape(nrows, ncols)
        resized_img = imresize_scipy(highres_img, resized_shape, interp="nearest")
        assert resized_img.shape == (666, 832, 3)
        cropped_img = ImgMaterial.get_cropped_img(resized_img, row_hardness, row_up, col_hardness, col_left)
        rotated_img = imrotate_sci(cropped_img, ImgMaterial.get_degree(rotation_hardness, clockwise), interp="nearest")
        one_hot = one_hot_coder.get_encoded(rotated_img)
        one_hot = reshape_np(one_hot, (CAMERA_ROWS * CAMERA_COLS, -1))
        return one_hot

    def visualize_an_input(self, *args, **kwargs):
        an_input = self.get_an_input(*args, **kwargs)
        reshaped_input = reshape_np(an_input, (CAMERA_ROWS, CAMERA_COLS, -1))
        root_logger.info(an_input.shape)
        root_logger.info(reshaped_input.shape)
        for i in range(reshaped_input.shape[-1]):
            show_array(reshaped_input[:, :, i])


class MaterialPair:
    """
    Csak arra kell, hogy ne keveredjenek össze a képek és a hozzájuk tartozó maszkok.
    """

    def __init__(self, img_material: ImgMaterial, mask_material: MaskMaterial):
        self.img_material, self.mask_material = img_material, mask_material


class LaneDB:
    @staticmethod
    def create_categories():
        [LaneDBMaskColor(label.name, label.color) for label in labels_helper]  # A színeket
        # az adatbázis készítők döntötték el.
        LaneCategory(
            "Hatter",
            LaneDBMaskColor["noise"],
            LaneDBMaskColor["ignored"],
            LaneDBMaskColor["void"],
            LaneDBMaskColor["a_n_lu"],
            LaneDBMaskColor["a_y_t"],
            LaneDBMaskColor["db_w_g"],
            LaneDBMaskColor["db_w_s"],
            LaneDBMaskColor["db_y_g"],
            LaneDBMaskColor["ds_w_dn"],
            LaneDBMaskColor["ds_w_s"],
            LaneDBMaskColor["s_n_p"],
            LaneDBMaskColor["vom_wy_n"],
        )
        LaneCategory(
            "Nyil",
            LaneDBMaskColor["a_w_l"],
            LaneDBMaskColor["a_w_r"],
            LaneDBMaskColor["a_w_t"],
            LaneDBMaskColor["a_w_u"],
            LaneDBMaskColor["a_w_tl"],
            LaneDBMaskColor["a_w_tr"],
            LaneDBMaskColor["a_w_tu"],
            LaneDBMaskColor["a_w_tlr"],
            LaneDBMaskColor["a_w_lr"],
            LaneDBMaskColor["a_w_m"],
            LaneDBMaskColor["d_wy_za"],  # rombusz
        )
        LaneCategory("Zebra", LaneDBMaskColor["c_wy_z"])
        LaneCategory("Zebra_elott", LaneDBMaskColor["r_wy_np"], LaneDBMaskColor["s_w_s"], LaneDBMaskColor["b_n_sr"])
        LaneCategory(
            "Teli_savelvalaszto", LaneDBMaskColor["s_w_d"], LaneDBMaskColor["s_y_d"], LaneDBMaskColor["ds_y_dn"]
        )
        LaneCategory("Szaggatott_savelvalaszto", LaneDBMaskColor["b_w_g"], LaneDBMaskColor["b_y_g"])
        LaneCategory(
            "Egyeb_savelvalaszto",
            LaneDBMaskColor["om_n_n"],
            LaneDBMaskColor["sb_w_do"],
            LaneDBMaskColor["sb_y_do"],
            LaneDBMaskColor["s_w_c"],
            LaneDBMaskColor["s_y_c"],
            LaneDBMaskColor["s_w_p"],
        )

    @staticmethod
    def _get_all_path():
        """
        http://apolloscape.auto/lane_segmentation.html
        innen kell letölteni az adatbázis részeit. Ki kell csomagolni az egyes részeket ugyanabba a mappába.
        Ezután át kell írni a HOST_LANE_PATH változót erre a mappára.
        """
        img_paths, mask_paths = [], []
        for img_dirname, mask_dirname in zip(
            ["ColorImage_road02", "ColorImage_road03", "ColorImage_road04"],
            ["Labels_road02", "Labels_road03", "Labels_road04"],
        ):
            img_dir, mask_dir = join_path(CONTAINER_LANE_PATH, img_dirname), join_path(
                CONTAINER_LANE_PATH, mask_dirname
            )
            img_dir, mask_dir = join_path(img_dir, "ColorImage"), join_path(mask_dir, "Label")
            for record_name in listdir(img_dir):
                img_rec_dir, mask_rec_dir = join_path(img_dir, record_name), join_path(mask_dir, record_name)
                for camera_name in listdir(img_rec_dir):
                    img_cam_dir, mask_cam_dir = join_path(img_rec_dir, camera_name), join_path(
                        mask_rec_dir, camera_name
                    )
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
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(len(img_paths)):
            if i % 30 == 0:
                val_img_paths.append(img_paths[i])
                val_mask_paths.append(mask_paths[i])
            else:
                train_img_paths.append(img_paths[i])
                train_mask_paths.append(mask_paths[i])

        return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths, test_img_paths, test_mask_paths

    @staticmethod
    def get_train_val_test_DB():
        (
            train_img_paths,
            train_mask_paths,
            val_img_paths,
            val_mask_paths,
            test_img_paths,
            test_mask_paths,
        ) = LaneDB._get_train_val_test_paths()
        train_DB, val_DB, test_DB = (
            LaneDB(train_img_paths, train_mask_paths),
            LaneDB(val_img_paths, val_mask_paths),
            LaneDB(test_img_paths, test_mask_paths),
        )
        return train_DB, val_DB, test_DB

    def __init__(self, img_paths, mask_paths):
        self.image_and_mask_material_pairs = [
            MaterialPair(ImgMaterial(img_path), MaskMaterial(mask_path))
            for img_path, mask_path in zip(img_paths, mask_paths)
        ]

    @virtual_proxy_property
    def orders(self) -> dict:
        """
        Sorbarendezi a képeket minden kategória területe szerint.
         Ha egy képen nincs a kategóriából, akkor az nem számít az aktuális sorrendbe.
         Csak az indexek vannak eltárolva a sorrendekben.
        """
        orders = {
            c.name: [
                pair_i
                for pair_i, pair in enumerate(self.image_and_mask_material_pairs)
                if pair.mask_material.category_probabilities[c.name] > 0
            ]
            for c in LaneCategory
        }
        for category_name, order in orders.items():
            order.sort(
                reverse=True,
                key=lambda pair_i: self.image_and_mask_material_pairs[pair_i].mask_material.category_probabilities[
                    category_name
                ],
            )
        return orders

    def get_materials_by_category(self, category_name, hardness: float) -> Tuple[ImgMaterial, MaskMaterial]:
        """
        Pl (category_name="Zebra",hardness=0)-nál visszaadja azt a példát, amin a legnagyobb zebra van.
          (category_name="Zebra",hardness=1)-nál visszaadja azt a példát, amin a legkisebb zebra van,
           de van legalább 1 pixelnyi zebra.
        """
        order = self.orders[category_name]
        STEP_SIZE = 1 / len(order)
        if (1 - STEP_SIZE) <= hardness <= 1:
            order_index = len(order) - 1
        else:
            for i in range(len(order)):
                if (STEP_SIZE * i) <= hardness <= (STEP_SIZE * (i + 1)):
                    order_index = i
        image_and_mask_source_pair_index = order[order_index]
        image_and_mask_source_pair = self.image_and_mask_material_pairs[image_and_mask_source_pair_index]
        return image_and_mask_source_pair.img_material, image_and_mask_source_pair.mask_material

    @virtual_proxy_property
    def one_hot_coder(self) -> LaneOneHotCoder:
        return LaneOneHotCoder(CAMERA_ROWS, CAMERA_COLS, LaneCategory)

    @virtual_proxy_property
    def category_probabilities(self) -> dict:
        """
        Az egész adatbázis eloszlása.
        """
        summed_category_probabilities = get_summed_dict(
            [i_a_m_m_p.mask_material.category_probabilities for i_a_m_m_p in self.image_and_mask_material_pairs]
        )
        return get_probabilities(summed_category_probabilities)


class Unet(Singleton):
    save_directory_path = CONTAINER_ROOT_PATH
    LaneDB.create_categories()

    @virtual_proxy_property
    def train_val_test_DB(self):
        """
        Lehet nincs meg valakinek az adatbázis, ezért nem az __init__-ben készül.
        """
        return LaneDB.get_train_val_test_DB()

    @property
    def train_DB(self):
        return self.train_val_test_DB[0]

    @property
    def val_DB(self):
        return self.train_val_test_DB[1]

    def __init__(self, name: str = None):
        self.name = Unet.__name__ if name is None else name
        self.hdf5_path = join_path(self.save_directory_path, "{}.hdf5".format(self.name))
        self.png_path = join_path(self.save_directory_path, "{}.structure.png".format(self.name))

    def get_prediction(self, rgb_array: ndarray) -> ndarray:
        """
        Parameters
        ----------
        rgb_array: [0,255]/(480,640,3)
        Returns
        -------
            [0,1]/(480,640,7)
        """
        normalized_img = ImgMaterial.get_normalized_img(rgb_array)
        input_batch = normalized_img[None, :, :, :]
        predicted_batch = self.model.predict_on_batch(input_batch)
        distribution_list = predicted_batch[0]
        distribution_matrix = reshape_np(distribution_list, (CAMERA_ROWS, CAMERA_COLS, -1))
        return distribution_matrix

    def set_axis(self, *, axis: Axes, probability_matrix: ndarray, threshold: float, title: str):
        """
        Hozzáad egy küszöbölt [0,1]/(480,640) képet a megjelenítéshez.
        """
        probability_matrix[probability_matrix < threshold] = 0
        axis.axis("off")
        axis.set_title(title)
        axis.imshow(probability_matrix)

    def visualize_prediction(self):
        canvas = zeros((960, 640, 3))
        hardness = int(input("Hardness[0-100]")) / 100.0
        common_input_params = {
            "rotation_hardness": hardness,
            "clockwise": True,
            "row_hardness": hardness,
            "row_up": False,
            "col_hardness": hardness,
            "col_left": False,
        }
        root_logger.info(hardness)
        img_material, mask_material = self.train_DB.get_materials_by_category("Zebra", hardness)
        expected_one_hot = mask_material.get_an_input(self.train_DB.one_hot_coder, **common_input_params)
        reshaped_expected_one_hot = reshape_np(expected_one_hot, (CAMERA_ROWS, CAMERA_COLS, -1))
        input_img = img_material.get_an_input(**common_input_params, noise_hardness=hardness)
        predicted_distribution_matrix = self.get_prediction(input_img * 255)
        canvas[:480] = input_img
        canvas[480:] = input_img
        root_logger.info(predicted_distribution_matrix.shape)
        assert reshaped_expected_one_hot.shape == predicted_distribution_matrix.shape == (480, 640, len(LaneCategory))
        for category_i in range(predicted_distribution_matrix.shape[2]):
            expected_mask = reshaped_expected_one_hot[:, :, category_i]
            predicted_mask = predicted_distribution_matrix[:, :, category_i]
            canvas[:480, :, 0] = expected_mask
            canvas[480:, :, 0] = predicted_mask
            show_array(canvas)

    def train(
        self,
        min_epochs: int,
        batch_size: int,
        steps_per_epoch: int,
        validation_steps: int,
        early_stopping_min_delta: float,
        reduce_learning_rate_min_delta: float,
        early_stopping_patience: int,
        reduce_learning_rate_patience: int,
        reduce_learning_rate_factor: float,
    ):
        """
        Parameters
        ----------
        min_epochs: -csak ezután kapcsol be az early stopping
                    -újraindul a mentés ha eléri
                    -itt éri el a legnehezebb fokozatot a háló betanulása
        """
        self.RLR_min_delta, self.early_stopping_min_delta = (
            reduce_learning_rate_min_delta,
            early_stopping_min_delta,
        )  # 0.0001
        self.RLR_patience, self.early_stopping_patience = reduce_learning_rate_patience, early_stopping_patience  # 4
        self.RLRFactor = reduce_learning_rate_factor  # 0.2
        self.batch_size, self.validation_steps = batch_size, validation_steps
        self.steps_per_epoch = (
            int(len(self.train_DB.image_and_mask_material_pairs) / self.batch_size)
            if steps_per_epoch is None
            else steps_per_epoch
        )
        self.min_epochs = min_epochs
        self.max_epochs = 99999
        self.metrics = ["categorical_accuracy"]
        self.monitor = "loss"
        self.verbose = 1
        self.structure.compile(optimizer=Adam_ke(), loss=self.loss, metrics=self.metrics, sample_weight_mode="temporal")
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
    def custom_monitor(y_true, y_pred):
        ...

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
        return cce

    @property
    def max_hardness(self) -> float:
        """
        Ez valósítja meg a curriculum learning-ot. A háló egyre nehezebb példákat kap a betanulás során.
        """
        if (self.epoch_i + 1) >= self.min_epochs:
            return 1
        return (self.epoch_i + 1) * (1 / self.min_epochs)

    def get_a_random_hardness(self) -> float:
        return uniform(0, self.max_hardness)

    def get_a_random_boolean(self) -> bool:
        if random_r() > 0.5:
            return True
        return False

    @classmethod
    def calculate_weight(cls, probability: float) -> float:
        """
        Ez dönti el az egyes kategóriák alap súlyát.
        """
        return 1 if probability < 0.5 else 0

    def get_an_important_category_cycle(self) -> Iterable_type[LaneCategory]:
        """
        Sorban 1-1 kategóriára koncentrál a háló. Pl a betanulás elején:
         1. kell egy példa, ahol nagy a zebra
         2. kell egy példa, ahol sok a nyíl
        """
        return cycle([category for category in LaneCategory if category.name != "Hatter"])

    def batch_generator(self, DB: LaneDB):
        i_cat_cycle = self.get_an_important_category_cycle()
        img_array_container = zeros((self.batch_size, CAMERA_ROWS, CAMERA_COLS, 3), dtype=float32)
        one_hot_container = zeros((self.batch_size, CAMERA_ROWS * CAMERA_COLS, len(LaneCategory)), dtype=float32)
        weight_container = zeros((self.batch_size, CAMERA_ROWS * CAMERA_COLS), dtype=float32)
        dilated_weight_container = zeros((CAMERA_ROWS, CAMERA_COLS), dtype=float32)  # itt még nincs kiterítve a kép,
        # mert 2D-ben dilatálunk.
        category_index_container = zeros((CAMERA_ROWS * CAMERA_COLS,), dtype=uint8)  # 1D
        category_weight_container = zeros((len(LaneCategory),), dtype=float32)  # pl [1,0,1,1,1,1,1,1]
        for category_i, category in enumerate(DB.one_hot_coder.categories):
            category_weight_container[category_i] = self.calculate_weight(DB.category_probabilities[category.name])
        while True:
            # collect sources
            img_materials, mask_materials = [], []
            for sample_i in range(self.batch_size):
                img_material, mask_material = DB.get_materials_by_category(
                    next(i_cat_cycle).name, self.get_a_random_hardness()
                )
                img_materials.append(img_material)
                mask_materials.append(mask_material)
            # save inputs
            for sample_i, (img_material, mask_material) in enumerate(zip(img_materials, mask_materials)):
                common_input_params = {
                    "rotation_hardness": self.get_a_random_hardness(),
                    "clockwise": self.get_a_random_boolean(),
                    "row_hardness": self.get_a_random_hardness(),
                    "row_up": self.get_a_random_boolean(),
                    "col_hardness": self.get_a_random_hardness(),
                    "col_left": self.get_a_random_boolean(),
                }
                img_array_container[sample_i] = img_material.get_an_input(
                    **common_input_params, noise_hardness=self.get_a_random_hardness()
                )
                one_hot_container[sample_i] = mask_material.get_an_input(DB.one_hot_coder, **common_input_params)
            for sample_i in range(self.batch_size):
                argmax_np(one_hot_container[sample_i], axis=-1, out=category_index_container)
                original_weight_matrix = reshape_np(
                    category_weight_container[category_index_container], (CAMERA_ROWS, CAMERA_COLS)
                )  # ez már 2D
                the_samples_original_weight_sum = sum_np(original_weight_matrix)
                dilation_sum_threshold = the_samples_original_weight_sum * (
                    1 + HyperParameters.min_weight_contour_ratio
                )
                unsetted_weight_sum = the_samples_original_weight_sum * HyperParameters.weight_dot_cloud_ratio
                dilation_size_int = 2
                while True:
                    # Fokozatosan felhízlaljuk a súlyképet.
                    grey_dilation(
                        input=original_weight_matrix,
                        output=dilated_weight_container,
                        size=(dilation_size_int, dilation_size_int),
                        mode="constant",
                    )
                    if sum_np(dilated_weight_container) >= dilation_sum_threshold:
                        break
                    dilation_size_int += 1
                while True:
                    # Egyesével beleszámítuk eg újabb pixelt a loss-ba.
                    coordinate = next(LaneUtil.dot_cloud_coordinate_cycle)
                    if dilated_weight_container[coordinate] > 0:
                        continue
                    dilated_weight_container[coordinate] = 1
                    unsetted_weight_sum -= 1
                    if unsetted_weight_sum <= 0:
                        break
                weight_container[sample_i] = reshape_np(dilated_weight_container, (CAMERA_ROWS * CAMERA_COLS))  # vissza
                # kell teríteni 1D-be a súlyképet
            yield img_array_container, one_hot_container, weight_container

    def visualize_batches(self):
        canvas = zeros((480, 640, 3))
        for image_batch, one_hot_batch, weight_batch in self.train_data:
            root_logger.info(image_batch.shape)
            root_logger.info(one_hot_batch.shape)
            root_logger.info(weight_batch.shape)
            for sample_i in range(image_batch.shape[0]):
                image_sample = image_batch[sample_i]
                one_hot_sample = one_hot_batch[sample_i]
                weight_sample = weight_batch[sample_i]
                root_logger.info(image_sample.shape)
                root_logger.info(one_hot_sample.shape)
                root_logger.info(weight_sample.shape)
                reshaped_one_hot_sample = reshape_np(one_hot_sample, (CAMERA_ROWS, CAMERA_COLS, -1))
                reshaped_weight_sample = reshape_np(weight_sample, (CAMERA_ROWS, CAMERA_COLS))
                root_logger.info(reshaped_one_hot_sample.shape)
                root_logger.info(reshaped_weight_sample.shape)
                for category_i in range(reshaped_one_hot_sample.shape[2]):
                    canvas[:, :, 1:3] = image_sample[:, :, 1:3]
                    canvas[:, :, 0] = reshaped_one_hot_sample[:, :, category_i]
                    show_array(canvas)
                show_array(reshaped_weight_sample)
            break

    def get_output_layer(self, input_layer: int):
        """
        Ez készíti el a Unet struktúrát.
        """

        def c(layers, filters, kernel_size, activation="relu"):
            layer = concatenate_ke(layers) if type(layers) is list else layers
            return Conv2D_ke(filters=filters, kernel_size=kernel_size, strides=1, activation=activation)(layer)

        def t(layers, filters, kernel_size, activation="relu", strides=(1, 1)):
            layer = concatenate_ke(layers) if type(layers) is list else layers
            return Conv2DTranspose_ke(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation)(
                layer
            )

        def p(layer):
            return MaxPool2D_ke((2, 2))(layer)

        first_filters = HyperParameters.first_filters
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
        c_480_640 = c(t_480_640, len(LaneCategory), 1, "softmax")
        return c_480_640

    @virtual_proxy_property
    def structure(self) -> Model_ke:
        """
        Kiegészti a Unet struktúrát azzal, hogy kiteríti az utolsó réteget.
        Elmenti képként a szerkezetet.
        """
        input_l = Input_ke(shape=(CAMERA_ROWS, CAMERA_COLS, 3))
        output_l = self.get_output_layer(input_l)
        output_l = Reshape_ke((CAMERA_ROWS * CAMERA_COLS, -1))(output_l)
        model = Model_ke(input_l, output_l)
        plot_model_ke(model, show_shapes=True, to_file=self.png_path)
        return model

    class EarlyStopper(EarlyStopping_ke):
        """
        Csak azért kell, mert növekedhet a loss a curriculum learning végéig,
        még akkor is, ha valójában pontosabb a háló.
        """

        def __init__(self, unet: "Unet", *args, **kwargs):
            self.unet = unet
            super().__init__(*args, **kwargs)

        def on_epoch_end(self, epoch, logs=None):
            if epoch >= (self.unet.min_epochs - 1):
                super().on_epoch_end(epoch, logs)
            else:
                root_logger.info("Early stopper is inactivated")

    @virtual_proxy_property
    def early_stopper(self):
        return self.EarlyStopper(
            self,
            monitor=self.monitor,
            min_delta=self.early_stopping_min_delta,
            patience=self.early_stopping_patience,
            verbose=self.verbose,
        )

    class Saver(ModelCheckpoint_ke):
        """
        Csak azért kell, mert növekedhet a loss a curriculum learning végéig,
        még akkor is, ha valójában pontosabb a háló.
        """

        def __init__(self, unet: "Unet", *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.unet = unet
            super().__init__(*self.args, **self.kwargs)

        def on_epoch_end(self, epoch, logs=None):
            if epoch == (self.unet.min_epochs - 1):
                root_logger.info("The saver had been reinitialized")
                super().__init__(*self.args, **self.kwargs)
            super().on_epoch_end(epoch, logs)

    @virtual_proxy_property
    def saver(self):
        return self.Saver(
            self, filepath=self.hdf5_path, monitor=self.monitor, save_best_only=True, verbose=self.verbose
        )

    @virtual_proxy_property
    def learning_rate_reducer(self):
        """
        Nem tudom, hogy végülis használja-e a leaning ratet az Adam.
        """
        return ReduceLROnPlateau_ke(
            monitor=self.monitor,
            factor=self.RLRFactor,
            verbose=self.verbose,
            epsilon=self.RLR_min_delta,
            patience=self.RLR_patience,
        )

    @virtual_proxy_property
    def tensorboard(self):
        return TensorBoard_ke(log_dir="./logs")

    class EpochISetter(Callback_ke):
        """
        Ez csak arra kell, hogy tudjuk hol jár a curriculum learning.
        """

        def __init__(self, unet: "Unet"):
            super().__init__()
            self.unet = unet

        def on_epoch_begin(self, epoch, logs=None):
            self.unet.epoch_i = epoch
            root_logger.info(self.unet.epoch_i)

        def on_epoch_end(self, epoch, logs=None):
            root_logger.info(logs)

    @virtual_proxy_property
    def epoch_i_setter(self):
        return self.EpochISetter(self)

    @virtual_proxy_property
    def callbacks(self):
        return [self.early_stopper, self.saver, self.learning_rate_reducer, self.epoch_i_setter]  # ,self.tensorboard]

    @virtual_proxy_property
    def train_data(self):
        return self.batch_generator(self.train_DB)

    @virtual_proxy_property
    def validation_data(self):
        return self.batch_generator(self.val_DB)

    @virtual_proxy_property
    def model(self) -> Model_ke:
        return load_model_ke(filepath=self.hdf5_path)


# Ez azért kell, hogy egyszerűen el lehessen menteni a hdf5 fájlt.
import keras.losses

keras.losses.loss = Unet.loss
