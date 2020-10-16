from mrcnn.config import Config
from mrcnn.utils import Dataset
from traffic.consts import LANE
from traffic.consts import CONTAINER_LANE_PATH
from traffic.imports import listdir, join_path, imread, unique, reshape, zeros, uint8, apply_along_axis, resize, imresize, label, histogram
from traffic.utils.lane_helper import labels, color2trainId
from traffic.utils.time import Timer


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


class LaneDataset(Dataset):
    @staticmethod
    def get_lane_paths():
        img_paths, mask_paths = [], []
        for img_dirname, mask_dirname in zip(["ColorImage_road02"], ["Labels_road02"]):
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
        def get_trainId(a):
            if a[0] == 0 == a[1] == a[2]:
                return 0
            return color2trainId[tuple(a)]

        img = self.get_square_img(self.mask_paths[image_id])
        img = self.get_nearest_neighbour(img)
        img = apply_along_axis(get_trainId, 2, img)
        trainId_img = img.copy()
        component_img, num = label(img, background=0, return_num=True, connectivity=2)
        hist, _ = histogram(component_img, bins=num)

        return trainId_img

        def count_area(a):
            if a[0] == 0 == a[1] == a[2]:
                return 0

        img = apply_along_axis(count_area, 2, img)

        print(num)

        return img
        # todo

        original_mask = self.get_square_img(self.mask_paths[image_id])

        with Timer("d"):
            id_mask = apply_along_axis(get_trainId, 2, original_mask)
        with Timer("lambda"):
            id_mask = apply_along_axis(lambda a: color2trainId[tuple(a)], 2, original_mask)

        print(id_mask.shape)
        print(id_mask)

        id_mask = zeros(original_mask.shape[0:2], dtype=uint8)
        for i in range(original_mask.shape[0]):
            for j in range(original_mask.shape[1]):
                id_mask[i, j] = color2trainId[tuple(original_mask[i, j, :])]

        print(original_mask.shape)
        print(id_mask.shape)
        print(id_mask)
        input()

        return original_mask

        info = self.image_info[image_id]
        shapes = info["shapes"]
        count = len(shapes)
        mask = np.zeros([info["height"], info["width"], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info["shapes"]):
            mask[:, :, i : i + 1] = self.draw_shape(mask[:, :, i : i + 1].copy(), shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

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
