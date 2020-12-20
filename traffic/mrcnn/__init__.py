"""
Ez egy segédfájl ami átláthatóbbá teszi az Mrcnn háló (
https://github.com/matterport/Mask_RCNN/tree/master/mrcnn
) hasznlatát.
"""
from traffic.utils import *
from traffic.consts import *
from mrcnn.model import MaskRCNN
from mrcnn.utils import download_trained_weights
from coco import CocoConfig


class MrcnnCategory(metaclass=SingletonByIdMeta):
    def __init__(self, name: str, index: int, min_confidence: float, rgb_tuple: Tuple[int, int, int]):
        """
        Parameters
        ----------
        name: pl Ember
        index: Hanyadik kategriának felel meg. pl 0.
        min_confidence: [0,1] közötti érték, ami azt jelzi, hogy mennyitől releváns egy detektálás.
        Pl kicsi 0.25-s értéknél gyakran két embert is bejelöl egy helyett.
        vagy nagy 0.9-es értéknél nem jelöli be azokat az autókat amik kicsik a képen.
        rgb_tuple: pl (255,255,255)-nél fehérrel jelöli be az adott kategóriába tartozó objektumokat.
        """
        self.name, self.index, self.min_confidence, self.rgb_tuple = name, index, min_confidence, rgb_tuple

    @staticmethod
    def get_id(name: str, index: int, min_confidence: float, rgb_tuple: Tuple[int, int, int]):
        """
        Egy indexhez csak 1 kategória objektum tartozhat.
        """
        return index

    def __eq__(self, other):
        return self.name == other or self.index == other

    @property
    def important(self) -> bool:
        """
        Pl a "spoon" kategória nem fontos de az "Ember" igen.
        """
        return self.name[0].isupper()


class DetectedObject:
    @staticmethod
    def get_picked_detected_objects(
        detected_objects: List["DetectedObject"], show_only_important: bool, show_only_confident: bool
    ):
        """
        Kb 80 ketegóriára van betantva a háló amiből kb 5 releváns a program szempontjából.
        Ha valamit csak épphogy detektált a háló, az lehet hogy nem releváns.
        Ha valaminek rosszul adja meg a befoglaló téglalapját, azt elrontotta és ki kell szűrni.
        """
        detected_objects = [d for d in detected_objects if d.has_bbox]
        detected_objects = [d for d in detected_objects if d.important] if show_only_important else detected_objects
        detected_objects = [d for d in detected_objects if d.confident] if show_only_confident else detected_objects
        return detected_objects

    def __init__(
        self, category: MrcnnCategory, confidence: float, y1: int, x1: int, y2: int, x2: int, mask_boolean: ndarray
    ):
        self.category, self.confidence = category, confidence
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.mask_boolean = mask_boolean

    @property
    def name(self) -> str:
        return self.category.name

    @property
    def rgb_tuple(self) -> Tuple[int, int, int]:
        return self.category.rgb_tuple

    @property
    def normalized_rgb_tuple(self) -> Tuple[float, float, float]:
        r, g, b = self.rgb_tuple
        return (r / 255, g / 255, b / 255)

    @property
    def important(self) -> bool:
        return self.category.important

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def confident(self) -> bool:
        """
        Megbízhatónak találjuk-e a háló bejelölését.
        """
        return self.category.min_confidence <= self.confidence

    @property
    def has_bbox(self) -> bool:
        return bool(self.x1) or bool(self.x2) or bool(self.y1) or (self.y2)


class Mrcnn:
    MIN_CONFIDENCE = 0.25
    NMS_THRESHOLD = 0.3  # nem maximumok elnyomása
    MODEL_PATH = "/traffic/mrcnn/mask_rcnn_coco.h5"
    _category_names = [
        "BG",
        EMBER,
        BICIKLI,
        AUTO,
        MOTOR,
        "airplane",
        BUSZ,
        VONAT,
        KAMION,
        "boat",
        JELZOLAMPA,
        "fire hydrant",
        STOPTABLA,
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    @staticmethod
    def create_categories():
        [MrcnnCategory(name, i, Mrcnn.MIN_CONFIDENCE, (255, 255, 255)) for i, name in enumerate(Mrcnn._category_names)]
        MrcnnCategory[EMBER].min_confidence = 0.8
        MrcnnCategory[EMBER].rgb_tuple = (0, 0, 255)
        MrcnnCategory[AUTO].rgb_tuple = (0, 255, 0)
        MrcnnCategory[BUSZ].rgb_tuple = (255, 0, 0)
        MrcnnCategory[BICIKLI].rgb_tuple = (255, 255, 0)
        MrcnnCategory[MOTOR].rgb_tuple = (255, 0, 255)

    @virtual_proxy_property
    def model(self) -> MaskRCNN:
        self.create_categories()
        if not exists(self.MODEL_PATH):
            download_trained_weights(self.MODEL_PATH)

        class InferenceConfig(CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1  # szekvenciálisan értékelünk ki minden képkockát
            DETECTION_MIN_CONFIDENCE = self.MIN_CONFIDENCE
            DETECTION_NMS_THRESHOLD = self.NMS_THRESHOLD
            IMAGE_MAX_DIM = CAMERA_COLS
            IMAGE_MIN_DIM = CAMERA_ROWS

        config = InferenceConfig()
        config.IMAGE_SHAPE = array_np([512, CAMERA_COLS, config.IMAGE_CHANNEL_COUNT])  # Nem jó a 480, mert
        # nem lehet 6-szor elosztani 2-vel.
        config.display()
        model = MaskRCNN(mode="inference", model_dir="", config=config)
        model.load_weights(self.MODEL_PATH, by_name=True)
        return model

    def get_prediction(self, rgb_array: ndarray) -> List[DetectedObject]:
        res = self.model.detect([rgb_array], verbose=0)[0]

        return [
            DetectedObject(MrcnnCategory[category_id], confidence, *bbox, mask)
            for category_id, confidence, bbox, mask in zip(
                res["class_ids"], res["scores"], res["rois"], moveaxis_np(res["masks"], -1, 0)
            )
        ]

    def set_axis(
        self,
        *,
        axis: Axes,
        rgb_array: ndarray,
        detected_objects: List[DetectedObject],
        title: str,
        show_mask: bool,
        show_mask_contour: bool,
        show_bbox: bool,
        show_caption: bool
    ):
        """
        Hozzáadja az objektumok befoglaló téglalapját/körvonalát/kategóriáját/magabiztosságát
         a megjelenítéshez. Beállítja háttérnek a detektálandó képet.
          A maszkok helyénél maga a kép intenzitásai vannak átállítva.
        """
        rgb_array = rgb_array.copy()
        if not len(detected_objects):
            root_logger.warning("No instances to display!")
        title = title or "{}->Number of instances: {}".format(self.set_axis.__name__, len(detected_objects))
        axis.cla()
        axis.axis("off")
        axis.set_title(title)
        for d in detected_objects:
            if show_caption:
                axis.text(
                    d.x1, d.y1 + 8, "{} {:.3f}".format(d.name, d.confidence), color="w", size=11, backgroundcolor="none"
                )
            if show_bbox:
                axis.add_patch(
                    Rectangle(
                        (d.x1, d.y1),
                        d.width,
                        d.height,
                        linewidth=2,
                        alpha=0.7,
                        linestyle="dashed",
                        edgecolor=d.normalized_rgb_tuple,
                        facecolor="none",
                    )
                )
            if show_mask:
                rgb_array = apply_mask(rgb_array, d.mask_boolean, d.normalized_rgb_tuple)
            if show_mask_contour:
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = zeros((d.mask_boolean.shape[0] + 2, d.mask_boolean.shape[1] + 2), dtype=uint8)
                padded_mask[1:-1, 1:-1] = d.mask_boolean
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = fliplr(verts) - 1
                    axis.add_patch(Polygon_mat(verts, facecolor="none", edgecolor=d.normalized_rgb_tuple))
        axis.imshow(rgb_array)
