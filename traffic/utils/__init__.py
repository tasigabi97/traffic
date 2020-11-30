from traffic.utils.time import *
from traffic.imports import *
from traffic.logging import root_logger
from traffic.consts import SSID_VODAFONE, IP_VODAFONE, DROIDCAM, DROIDCAM_PORT, PRIVATE
from mrcnn.visualize import random_colors, apply_mask


def load_rgb_array(path: str) -> ndarray:
    try:
        img = imread_skimage(path)
    except Exception as e:
        root_logger.warning(path)
        raise e
    img = array_np(img, dtype=uint8)
    if img.shape[-1] == 3:
        return img
    elif img.shape[-1] == 4:
        return img[..., :3]
    else:
        raise IndexError(img.shape)


def show_array(array: ndarray):
    imshow_mat(array)
    show()


def virtual_proxy_property(func: Callable) -> property:
    @property
    @wraps(func)
    def new_func(self):
        ATTR_NAME = PRIVATE + func.__name__
        if hasattr(self, ATTR_NAME):
            return getattr(self, ATTR_NAME)
        ret = func(self)
        setattr(self, ATTR_NAME, ret)
        return ret

    return new_func


class NNInputSource:
    def __init__(self, path: str):
        self.path = path

    @property
    def data(self) -> ndarray:
        return load_rgb_array(self.path)

    def visualize_data(self):
        show_array(self.data)

    def get_an_input(self, *args, **kwargs) -> ndarray:
        raise NotImplemented()

    def visualize_an_input(self, *args, **kwargs):
        show_array(self.get_an_input(*args, **kwargs))

    @property
    def attributes_path(self) -> str:
        return self.path + ".json"

    def get_calculated_attributes(self):
        raise NotImplemented()

    def save_attributes_to_json(self, attributes: dict):
        with open(self.attributes_path, "w") as outfile:
            dump_json(attributes, outfile)
        # root_logger.info('Saved {} to {}'.format(attributes,self.attributes_path))

    def get_attributes_from_json(self) -> dict:
        with open(self.attributes_path) as json_file:
            attributes = load_json(json_file)
        # root_logger.info('Loaded {} from {}'.format(attributes,self.attributes_path))
        return attributes

    @property
    def attributes(self) -> dict:
        if not exists(self.attributes_path):
            attributes = self.get_calculated_attributes()
            self.save_attributes_to_json(attributes)
        return self.get_attributes_from_json()


def set_axes(
    *,
    axis: Axes,
    canvas_uint8: ndarray,
    input_img_uint8: ndarray,
    lane_visualization_uint8: ndarray,
    instance_boxes: ndarray,
    instance_masks_boolean: ndarray,
    instance_class_ids: ndarray,
    instance_colors: List[Tuple[int, int, int]] = None,
    instance_captions: List[str] = None,
    all_class_names: List[str],
    all_class_colors: List[Tuple[int, int, int]] = None,
    all_class_min_confidences: List[float] = None,
    instance_scores: ndarray = None,
    title: str = None,
    show_mask=True,
    show_mask_contour=True,
    show_bbox=True,
    show_only_upper=False,
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    INSTANCE_NUMBER = instance_boxes.shape[0]
    if not INSTANCE_NUMBER:
        root_logger.warning("No instances to display!")
    else:
        assert INSTANCE_NUMBER == instance_masks_boolean.shape[-1] == instance_class_ids.shape[0]
    title = title or "{}->Number of instances: {}".format(set_axes.__name__, INSTANCE_NUMBER)
    if instance_colors is not None:
        pass
    elif all_class_colors is not None:
        instance_colors = [all_class_colors[class_id] for class_id in instance_class_ids]
    else:
        instance_colors = random_colors(INSTANCE_NUMBER)
    if True:
        root_logger.info("type(image):{}".format(type(input_img_uint8)))
        root_logger.info("max_np(image):{}".format(max_np(input_img_uint8)))
        root_logger.info("type(boxes):{}".format(type(instance_boxes)))
        root_logger.info("str(boxes):{}".format(str(instance_boxes)))
        root_logger.info("type(masks):{}".format(type(instance_masks_boolean)))
        root_logger.info("str(masks):{}".format(str(instance_masks_boolean)))
        root_logger.info("type(class_ids){}:".format(type(instance_class_ids)))
        root_logger.info("str(class_ids):{}".format(str(instance_class_ids)))
        root_logger.info("type(class_names){}:".format(type(all_class_names)))
        root_logger.info("str(class_names):{}".format(str(all_class_names)))
        root_logger.info("type(scores){}:".format(type(instance_scores)))
        root_logger.info("str(scores):{}".format(str(instance_scores)))
        root_logger.info("type(colors):{}".format(type(instance_colors)))
        root_logger.info("str(colors):{}".format(str(instance_colors)))
    # átlátszó fehér keret a kép körül
    axis.cla()
    axis.axis("off")
    axis.set_title(title)
    for i, (color, class_id) in enumerate(zip(instance_colors, instance_class_ids)):
        label = all_class_names[class_id]
        if show_only_upper and label[0].islower():
            continue
        y1, x1, y2, x2 = instance_boxes[i]
        if not (x1 or x2 or y1 or y2):
            continue  # Skip this instance. Has no bbox. Likely lost in image cropping.
        score = instance_scores[i] if instance_scores is not None else None
        if all_class_min_confidences and score and (score < all_class_min_confidences[class_id]):
            continue
        mask = instance_masks_boolean[:, :, i]
        caption = instance_captions[i] if instance_captions else "{} {:.3f}".format(label, score) if score else label
        axis.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")
        if show_bbox:
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=0.7,
                    linestyle="dashed",
                    edgecolor=color,
                    facecolor="none",
                )
            )
        if show_mask:
            input_img_uint8 = apply_mask(input_img_uint8, mask, color)
        if show_mask_contour:
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = fliplr(verts) - 1
                axis.add_patch(Polygon(verts, facecolor="none", edgecolor=color))
    canvas_uint8[:480] = input_img_uint8
    canvas_uint8[480:, :, 1] = lane_visualization_uint8
    axis.imshow(canvas_uint8)


def get_ssid():
    try:
        return str(check_output(["iwgetid"])).split('"')[1]
    except CalledProcessError:
        root_logger.warning("This PC is not connected to any Wifi network.")


@contextmanager
def webcam_server():
    ssid = get_ssid()
    if ssid == SSID_VODAFONE:
        ip = IP_VODAFONE
    else:
        root_logger.warning("Droidcam is not working with ssid ({}).".format(ssid))
        yield
        return
    try:
        p = Popen([DROIDCAM, "-v", ip, DROIDCAM_PORT])
    except FileNotFoundError as e:
        root_logger.warning(e)
        raise FileNotFoundError("Restart the computer and install droidcam again.")
    else:
        yield
        p.kill()


class Singleton(object):
    _instances = dict()

    def __new__(this_cls, *args, **kwargs):
        for a_singleton_cls, old_instance in Singleton._instances.items():
            if this_cls is a_singleton_cls:
                return old_instance
        new_instance = super().__new__(this_cls)
        Singleton._instances[this_cls] = new_instance
        return new_instance


class SingletonByIdMeta(type):
    _get_id_name = "get_id"
    _id_name = "id"

    def __new__(
        cls: "SingletonByIdMeta",
        new_cls_name: str,
        new_cls_bases: tuple,
        class_definition_dict: dict,
    ):
        new_cls = super().__new__(cls, new_cls_name, new_cls_bases, class_definition_dict)
        if not hasattr(new_cls, cls._get_id_name):
            raise KeyError("Please define a {} as a staticmethod".format(cls._get_id_name))
        init_params = [par for par in signature(getattr(new_cls, "__init__")).parameters.values()]
        init_params = init_params[1:]
        get_id_params = [par for par in signature(getattr(new_cls, cls._get_id_name)).parameters.values()]
        if init_params != get_id_params:
            root_logger.error(init_params)
            root_logger.error(get_id_params)
            raise NameError("Please define {} with the same signature as __init__".format(cls._get_id_name))
        return new_cls  # __init__()

    # called before __new__ return automatically
    def __init__(new_cls: type, new_cls_name: str, new_cls_bases: tuple, new_cls_dict: dict):
        super().__init__(new_cls_name, new_cls_bases, new_cls_dict)
        new_cls._instances = dict()
        original_init = new_cls.__init__

        def __new__(cls, *args, **kwargs):
            id = getattr(cls, SingletonByIdMeta._get_id_name)(*args, **kwargs)
            if id in cls._instances.keys():
                return cls._instances[id]
            new_instance = super(cls, cls).__new__(cls)
            cls._instances[id] = new_instance
            return new_instance

        def __init__(self, *args, **kwargs):
            if SingletonByIdMeta._id_name in self.__dict__.keys():
                return
            id = getattr(self, SingletonByIdMeta._get_id_name)(*args, **kwargs)
            setattr(self, SingletonByIdMeta._id_name, id)
            original_init(self, *args, **kwargs)

        new_cls.__new__, new_cls.__init__ = __new__, __init__

    def __iter__(new_cls) -> Iterable_type:
        return iter(new_cls._instances.values())

    def clear(new_cls):
        new_cls._instances.clear()

    def __getitem__(new_cls, item):
        for i in new_cls:
            if i == item:
                return i
        raise IndexError("object with index ({}) not found".format(item))

    def __len__(new_cls):
        return len(new_cls._instances)
