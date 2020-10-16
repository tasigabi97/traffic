def main():
    from traffic.consts import CONTAINER_LANE_PATH
    from traffic.imports import subplots, listdir, join_path
    from mrcnn.visualize import display_images

    from traffic.utils.lane import LaneConfig, LaneDataset

    def get_ax(rows=1, cols=1, size=8):
        _, ax = subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

    config = LaneConfig()
    config.display()
    dataset_train = LaneDataset()
    img1 = dataset_train.load_image(1000)
    img2 = dataset_train.load_image(2000)
    mask1 = dataset_train.load_mask(1000)
    mask2 = dataset_train.load_mask(2000)
    print(img1.shape)
    print(mask1.shape)

    display_images([img1, img2, mask1, mask2])


if __name__ == "__main__":
    main()
