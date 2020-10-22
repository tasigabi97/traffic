def main():
    from traffic.imports import listdir, imread_skimage, join_path, choice, exists, subplots, show, ion, sleep
    from traffic.utils import set_axes, Timer
    from mrcnn.model import MaskRCNN
    from mrcnn.utils import download_trained_weights
    from coco import CocoConfig

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

    ion()
    fig, ax = subplots()
    show()
    input("Wait for figure.")

    timers = []
    for batchsize in range(1, 5):
        timer = Timer(str(batchsize))
        timers.append(timer)
        InferenceConfig.IMAGES_PER_GPU = batchsize
        config = InferenceConfig()
        config.display()
        model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        for _ in range(2):
            images = [imread_skimage(join_path(IMAGE_DIR, choice(listdir(IMAGE_DIR)))) for _ in range(batchsize)]
            with timer:
                result = model.detect(images, verbose=0)[0]
            set_axes(
                ax,
                images[0],
                result["rois"],
                result["masks"],
                result["class_ids"],
                class_names,
                result["scores"],
                title="{}->{}".format(timer.name, timer.last_block_time),
            )
            fig.canvas.draw()
    [print(timer.last_block_time / float(timer.name)) for timer in timers]


if __name__ == "__main__":
    main()
