def main():
    from traffic.imports import subplots, listdir, join_path, exists, choice, expand_dims, choice_np, mean_np
    from mrcnn.visualize import display_images, display_top_masks, display_instances
    from mrcnn.model import MaskRCNN, load_image_gt, log, mold_image
    from mrcnn.utils import download_trained_weights, compute_ap
    from traffic.utils.lane import LaneConfig, LaneDataset

    MODEL_DIR = "/traffic/mrcnn/logs"
    COCO_MODEL_PATH = "/traffic/mrcnn/mask_rcnn_coco.h5"
    if not exists(COCO_MODEL_PATH):
        download_trained_weights(COCO_MODEL_PATH)
    config = LaneConfig()
    config.display()
    dataset_train = LaneDataset()
    dataset_val = dataset_train  # todo
    model = MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers="heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=2, layers="all")

    class InferenceConfig(LaneConfig):
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()
    model = MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    model_path = model.find_last()
    print("Loading weights from ", model_path)
    # Test on a random image
    image_id = choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))
    results = model.detect([original_image], verbose=1)

    r = results[0]

    def get_ax(rows=1, cols=1, size=8):
        _, ax = subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    display_instances(original_image, r["rois"], r["masks"], r["class_ids"], dataset_val.class_names, r["scores"], ax=get_ax())
    image_ids = choice_np(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
        molded_images = expand_dims(mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"])
        APs.append(AP)

    print("mAP: ", mean_np(APs))
    input("END")

    model.load_weights(model_path, by_name=True)
    for i, path in enumerate(dataset_train.mask_paths):
        masks, class_ids = dataset_train.load_mask(i)
        image = dataset_train.load_image(i)
        display_top_masks(image, masks, class_ids, dataset_train.class_names)


if __name__ == "__main__":
    main()
