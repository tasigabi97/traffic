def main():
    from traffic.imports import listdir, imread, join_path, choice, exists
    from mrcnn import visualize
    from mrcnn.model import MaskRCNN
    from mrcnn.utils import download_trained_weights
    from coco import CocoConfig

    ROOT_DIR = "/traffic/mrcnn"
    MODEL_DIR = join_path(ROOT_DIR, "logs")
    COCO_MODEL_PATH = join_path(ROOT_DIR, "mask_rcnn_coco.h5")
    IMAGE_DIR = join_path(ROOT_DIR, "images")
    if not exists(COCO_MODEL_PATH):
        download_trained_weights(COCO_MODEL_PATH)
    assert ROOT_DIR == "/traffic/mrcnn"
    assert MODEL_DIR == "/traffic/mrcnn/logs"
    assert IMAGE_DIR == "/traffic/mrcnn/images"
    assert COCO_MODEL_PATH == "/traffic/mrcnn/mask_rcnn_coco.h5"

    class InferenceConfig(CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = [
        "BG",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
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
    image = imread(join_path(IMAGE_DIR, choice(listdir(IMAGE_DIR))))
    results = model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])


if __name__ == "__main__":
    main()
