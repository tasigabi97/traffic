def main():
    from traffic.utils.lane_unet import Unet
    from traffic.logging import root_logger, INFO
    from traffic.imports import show, imshow_mat, zeros

    root_logger.setLevel(INFO)
    x = Unet()
    if input("train") == "t":
        x.train(
            batch_size=1,
            steps_per_epoch=100,
            validation_steps=1,
            early_stopping_min_delta=0,
            RLR_min_delta=0,
            early_stopping_patience=10,
            RLR_patience=1,
            RLRFactor=0.5,
        )
    while True:
        input_img, expected_one_hot = x.train_DB.random_train_input
        real_one_hot = x.get_prediction(input_img)
        s = zeros((960, 640, 3))
        s[:480] = input_img
        s[480:] = input_img
        s[:, :, 0] = 0
        for category_i, category in enumerate(x.train_DB.one_hot_coder.categories):
            if category.name != "Hatter":
                s[:480, :, 0] += expected_one_hot[:, :, category_i]
                s[480:, :, 0] += real_one_hot[:, :, category_i]
        imshow_mat(s)
        show()
    input("end train unet")


if __name__ == "__main__":
    main()
