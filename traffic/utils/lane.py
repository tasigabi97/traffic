from mrcnn.config import Config
from mrcnn.utils import Dataset
from traffic.consts import LANE
from traffic.consts import CONTAINER_LANE_PATH
from traffic.imports import (
    listdir,
    join_path,
    imread,
    count_nonzero,
    unique,
    array,
    int32,
    bool_np,
    reshape,
    zeros,
    uint8,
    apply_along_axis,
    resize,
    imresize,
    label,
    histogram,
    max_np,
)
from traffic.utils.lane_helper import labels, color_to_train_id_dict
from traffic.utils.time import Timer
from traffic.logging import root_logger


class LaneConfig(Config):
    NAME = LANE
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 3  # todo
    IMAGE_MIN_DIM = IMAGE_MAX_DIM = 448
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.25, 0.5, 1, 2, 4]
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    MIN_INSTANCE_SIZE = 200


class LaneDataset(Dataset):
    @staticmethod
    def get_lane_paths():
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

    def __init__(self):
        super().__init__()
        for label in labels:
            if label.category in ["void", "ignored"]:
                continue
            self.add_class(LaneConfig.NAME, label.trainId, label.name)
        self.img_paths, self.mask_paths = self.get_lane_paths()
        for i, img_path in enumerate(self.img_paths):
            self.add_image(LaneConfig.NAME, i, img_path)
        self.prepare()

    @staticmethod
    def get_square_img(path):
        img = imread(path)
        img = img[..., :3] if img.shape[-1] == 4 else img
        big_i, small_i = (0, 1) if img.shape[0] > img.shape[1] else (1, 0)
        # kivágás középről
        half_delta = (img.shape[big_i] - img.shape[small_i]) // 2
        return img[tuple(slice(half_delta, half_delta + img.shape[small_i]) if i == big_i else slice(None) for i in range(len(img.shape)))]

    @staticmethod
    def get_resized(img):
        return resize(img, (LaneConfig.IMAGE_MAX_DIM, LaneConfig.IMAGE_MIN_DIM), anti_aliasing=True) * 255

    @staticmethod
    def get_nearest_neighbour(img):
        img = imresize(img, (LaneConfig.IMAGE_MAX_DIM, LaneConfig.IMAGE_MIN_DIM), interp="nearest")
        return img

    def load_image(self, image_id):
        img = self.get_square_img(self.img_paths[image_id])
        return self.get_resized(img)

    def load_mask(self, image_id):
        def get_train_id(a):
            a = tuple(a)
            # void or noise
            if a == (0, 0, 0) or a == (255, 255, 255) or a == (0, 153, 153):
                return 0
            return color_to_train_id_dict[a]

        root_logger.debug("image_id={}".format(image_id))
        root_logger.debug("self.mask_paths[image_id]={}".format(self.mask_paths[image_id]))
        original_squared_mask = self.get_square_img(self.mask_paths[image_id])
        original_squared_resized_mask = self.get_nearest_neighbour(original_squared_mask)
        root_logger.debug("original_squared_resized_mask.shape={}".format(original_squared_resized_mask.shape))
        # creae a Integer mask instead of RGB
        # also delete unnecessary categories
        train_id_mask = apply_along_axis(get_train_id, 2, original_squared_resized_mask)
        # create Boolean object masks instead of category masks
        component_id_mask, max_component_id = label(train_id_mask.copy(), background=0, return_num=True, connectivity=2)
        root_logger.debug("max_component_id={}".format(max_component_id))
        root_logger.debug("max_np(component_id_mask)={}".format(max_np(component_id_mask)))
        # delete little objecs because we dont want too much Boolean masks
        component_sizes, _ = histogram(component_id_mask, bins=(max_component_id + 1))
        root_logger.debug("component_sizes={}".format(component_sizes))
        root_logger.debug("len(component_sizes)={}".format(len(component_sizes)))
        root_logger.debug("sum(component_sizes)={}".format(sum(component_sizes)))
        root_logger.debug("train_id_mask.shape[0]*train_id_mask.shape[1]={}".format(train_id_mask.shape[0] * train_id_mask.shape[1]))
        for i in range(train_id_mask.shape[0]):
            for j in range(train_id_mask.shape[1]):
                component_id = component_id_mask[i][j]
                component_size = component_sizes[component_id]
                if component_size < LaneConfig.MIN_INSTANCE_SIZE:
                    train_id_mask[i][j] = 0
        component_id_mask, max_component_id = label(train_id_mask, background=0, return_num=True, connectivity=2)
        root_logger.debug("max_component_id={}".format(max_component_id))
        root_logger.debug("max_np(component_id_mask)={}".format(max_np(component_id_mask)))
        root_logger.debug("unique(train_id_mask)={}".format(unique(train_id_mask)))
        # background cavity is not an object so we need max_component_id * Boolean mask
        # get component ids which need masks
        ret_masks = zeros((component_id_mask.shape[0], component_id_mask.shape[1], max_component_id), dtype=uint8)
        class_ids = [None for _ in range(max_component_id)]
        for i in range(train_id_mask.shape[0]):
            for j in range(train_id_mask.shape[1]):
                component_id = component_id_mask[i][j]
                if component_id == 0:
                    continue
                ret_masks[i][j][component_id - 1] = 1
                train_id = train_id_mask[i][j]
                class_ids[component_id - 1] = train_id
        root_logger.debug("class_ids={}".format(class_ids))
        root_logger.debug("ret_masks.shape={}".format(ret_masks.shape))
        root_logger.debug("count_nonzero(ret_masks)={}".format(count_nonzero(ret_masks)))
        root_logger.debug("count_nonzero(train_id_mask)={}".format(count_nonzero(train_id_mask)))
        root_logger.debug("count_nonzero(component_id_mask)={}".format(count_nonzero(component_id_mask)))
        ret_masks, class_ids = ret_masks.astype(bool_np), array(class_ids, dtype=int32)
        return ret_masks, class_ids

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == "square":
            cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array(
                [
                    [
                        (x, y - s),
                        (x - s / math.sin(math.radians(60)), y + s),
                        (x + s / math.sin(math.radians(60)), y + s),
                    ]
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
