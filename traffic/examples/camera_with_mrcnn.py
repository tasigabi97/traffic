def main():
    from traffic.camera import choose_camera
    from traffic.imports import join_path, exists, subplots, show, ion, zeros, uint8, Thread, array_np
    from traffic.consts import CAMERA_COLS, CAMERA_ROWS
    from traffic.utils import set_axes, Timer
    from mrcnn.model import MaskRCNN
    from mrcnn.utils import download_trained_weights
    from coco import CocoConfig
    from traffic.utils.lane_unet import Unet
    from traffic.logging import root_logger, WARNING, INFO
    from mrcnn.visualize import random_colors

    EMBER = "Ember"
    BICIKLI = "Bicikli"
    AUTO = "Auto"
    MOTOR = "Motor"
    BUSZ = "Busz"
    VONAT = "Vonat"
    KAMION = "Kamion"
    JELZOLAMPA = "Jelzolampa"
    STOPTABLA = "Stoptabla"

    class_names = [
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
    class_colors = random_colors(len(class_names))
    ROOT_DIR = "/traffic/mrcnn"
    MODEL_DIR = join_path(ROOT_DIR, "logs")
    COCO_MODEL_PATH = join_path(ROOT_DIR, "mask_rcnn_coco.h5")
    if not exists(COCO_MODEL_PATH):
        download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.25
        DETECTION_NMS_THRESHOLD = 0.3
        IMAGE_MAX_DIM = CAMERA_COLS
        IMAGE_MIN_DIM = CAMERA_ROWS

    config = InferenceConfig()
    config.IMAGE_SHAPE = array_np([512, CAMERA_COLS, config.IMAGE_CHANNEL_COUNT])
    config.display()
    with choose_camera() as camera:
        ion()
        fig, ax = subplots()
        show()
        root_logger.warning("Wait for figure.")
        root_logger.warning("Press ENTER")
        input()
        model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        x = Unet()
        canvas = zeros((480 * 2, 640, 3), uint8)
        root_logger.setLevel(WARNING)
        in_str = None
        detection_min_confidences = [0.9 for _ in class_names]
        detection_min_confidences[class_names.index(AUTO)] = 0
        detection_min_confidences[class_names.index(BUSZ)] = 0
        detection_min_confidences[class_names.index(KAMION)] = 0
        threshold = 0.5
        show_only_upper = True
        show_bbox = True
        show_mask = True
        show_mask_contour = True
        EXIT = "q"
        WAIT = "w"
        SHOW_ONLY_UPPER = "u"
        SHOW_BBOX = "b"
        SHOW_MASK = "m"
        SHOW_MASK_CONTOUR = "c"
        T_STRS = ["t" + str(i) for i in range(10)]

        def in_str_setter():
            nonlocal in_str
            while True:
                in_str = input()
                if in_str == EXIT:
                    break

        thread = Thread(target=in_str_setter)
        thread.start()
        while True:
            img = camera.matplotlib_img
            mrcnn_result = model.detect([img], verbose=0)[0]
            set_axes(
                axis=ax,
                canvas_uint8=canvas,
                input_img_uint8=img,
                lane_visualization_uint8=x.get_lane_visualization(img, threshold),
                instance_boxes=mrcnn_result["rois"],
                instance_masks_boolean=mrcnn_result["masks"],
                instance_class_ids=mrcnn_result["class_ids"],
                instance_scores=mrcnn_result["scores"],
                all_class_names=class_names,
                all_class_colors=class_colors,
                all_class_min_confidences=detection_min_confidences,
                show_only_upper=show_only_upper,
                show_bbox=show_bbox,
                show_mask=show_mask,
                show_mask_contour=show_mask_contour,
            )
            fig.canvas.draw()
            if in_str == EXIT:
                break
            elif in_str == SHOW_ONLY_UPPER:
                show_only_upper = not show_only_upper
                root_logger.warning(show_only_upper)
            elif in_str == SHOW_MASK:
                show_mask = not show_mask
                root_logger.warning(show_mask)
            elif in_str == SHOW_MASK_CONTOUR:
                show_mask_contour = not show_mask_contour
                root_logger.warning(show_mask_contour)
            elif in_str == SHOW_BBOX:
                show_bbox = not show_bbox
                root_logger.warning(show_bbox)
            elif in_str == WAIT:
                input()
            elif in_str in T_STRS:
                threshold = int(in_str[1]) / 10
                root_logger.warning(threshold)
            in_str = None

        thread.join()


if __name__ == "__main__":
    main()
