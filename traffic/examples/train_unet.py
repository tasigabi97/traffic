def main():
    from traffic.utils.lane_unet import Unet, LaneDB
    from traffic.utils import show_array
    from traffic.logging import root_logger, INFO
    from traffic.imports import show, imshow_mat, zeros, reshape_np

    root_logger.setLevel(INFO)
    x = Unet()
    if input("bath") == "b":
        x.batch_size = 1
        for i in x.train_data:
            ...
    if input("train") == "t":
        x.train(
            batch_size=1,
            steps_per_epoch=10,
            validation_steps=1,
            early_stopping_min_delta=0,
            RLR_min_delta=0,
            early_stopping_patience=10,
            RLR_patience=1,
            RLRFactor=0.5,
        )
    while True:
        img_source, mask_source = x.train_DB.get_sources(0)
        expected_one_hot = mask_source.get_an_input(x.train_DB.one_hot_coder)
        expected_one_hot = reshape_np(expected_one_hot, (480, 640, -1))
        target_i = None
        for category_i, category in enumerate(x.train_DB.one_hot_coder.categories):
            if category.name != "Hatter":
                target_i = category_i
                break
        expected_mask = expected_one_hot[:, :, target_i]
        input_img = img_source.get_an_input()
        predicted_mask = x.get_prediction(input_img)
        assert predicted_mask.shape == expected_mask.shape == (480, 640)
        view_arr = zeros((960, 640, 3))
        view_arr[:480] = input_img
        view_arr[480:] = input_img
        view_arr[:, :, 0] = 0
        for category_i, category in enumerate(x.train_DB.one_hot_coder.categories):
            if category.name != "Hatter":
                view_arr[:480, :, 0] += expected_mask
                view_arr[480:, :, 0] += predicted_mask
        show_array(view_arr)
    input("end train unet")


if __name__ == "__main__":
    main()
