def main():
    from traffic.consts import CONTAINER_LANE_PATH
    from traffic.imports import subplots, listdir, join_path
    from mrcnn.visualize import display_images, display_top_masks

    from traffic.utils.lane import LaneConfig, LaneDataset

    def get_ax(rows=1, cols=1, size=8):
        _, ax = subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    config = LaneConfig()
    config.display()
    dataset_train = LaneDataset()
    image_ids = [1000, 2000]
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        display_top_masks(image, mask, class_ids, dataset_train.class_names)


if __name__ == "__main__":
    main()
