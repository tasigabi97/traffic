from traffic.consts import *
from traffic.imports import *
from traffic.utils import *
from traffic.utils.lane_helper import labels as labels_helper
from traffic.logging import root_logger


class Category(object):
    def __init__(self, name: str, id: int, *colors: Tuple[int, int, int]):
        self.name, self.id = name, id
        self.colors = {color for color in colors}

    def add_color(self, color: Tuple[int, int, int]):
        self.colors.add(color)

    def is_me(self, color: Tuple[int, int, int]):
        for c in self.colors:
            if c == color:
                return True
        return False
        return color in self.colors


class Categories(Singleton):
    def __init__(self):
        self._categories = []
        # create used categories
        for label in labels_helper:
            if label.category is not "ignored":
                self._categories.append(Category(label.name, label.trainId, label.color))
        # add unused categories color
        for label in labels_helper:
            if label.category is "ignored":
                self[0].add_color(label.color)
        # rename categories
        self["void"].name = "Background"

    def __getitem__(self, item) -> Category:
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
        elif isinstance(item, tuple):
            for category in self._categories:
                if category.is_me(item):
                    return category
            raise ValueError(item)
        raise TypeError(item)

    def __len__(self):
        return len(self._categories)


CATEGORIES = Categories()


class LaneDB(object):
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
            if i % 3 == 0:
                train_img_paths.append(img_paths[i])
                train_mask_paths.append(mask_paths[i])
            elif i % 3 == 1:
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

    @property
    def _random_example_paths(self):
        random_i = self._random_example_id
        random_img_path, random_mask_path = self._img_paths[random_i], self._mask_paths[random_i]
        return random_img_path, random_mask_path

    @property
    def random_example(self) -> Tuple[ndarray, ndarray]:
        img_path, mask_path = self._random_example_paths
        img, mask = imread_skimage(img_path), imread_skimage(mask_path)
        img, mask = array_np(img, dtype=uint8), array_np(mask, dtype=uint8)
        mask = mask[..., :3]
        return img, mask

    @property
    def random_small_example(self) -> Tuple[ndarray, ndarray]:
        img, mask = self.random_example
        img = rescale_skimage(img, max(CAMERA_ROWS / img.shape[0], CAMERA_COLS / img.shape[1]), anti_aliasing=False, preserve_range=True)
        img = array_np(img, dtype=uint8)
        mask = imresize_scipy(mask, img.shape, interp="nearest")
        return img, mask

    @property
    def random_small_cropped_example(self) -> Tuple[ndarray, ndarray]:
        img, mask = self.random_small_example
        assert img.shape == mask.shape == (513, 640, 3)
        half_delta = (img.shape[0] - CAMERA_ROWS) // 2
        img, mask = img[half_delta : half_delta + CAMERA_ROWS], mask[half_delta : half_delta + CAMERA_ROWS]
        assert img.shape == mask.shape == (480, 640, 3)
        return img, mask

    @property
    def random_input(self) -> Tuple[ndarray, ndarray]:
        img, mask = self.random_small_cropped_example
        normalized_grayscale_img = mean_np(img, axis=2) / 255
        one_hot = zeros((CAMERA_ROWS, CAMERA_COLS, len(CATEGORIES)), dtype=uint8)
        for row_id in range(CAMERA_ROWS):
            for col_id in range(CAMERA_COLS):
                pixels_category_color = tuple(mask[row_id][col_id])
                pixels_category_id = CATEGORIES[pixels_category_color].id
                one_hot[row_id][col_id][pixels_category_id] = 1
        one_hot = array_np(one_hot, dtype=float64)
        return normalized_grayscale_img, one_hot


train_DB, val_DB, test_DB = LaneDB._get_train_val_test_DB()
