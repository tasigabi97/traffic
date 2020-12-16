def main():
    from traffic.utils.lane_unet import Unet
    from traffic.logging import root_logger, INFO

    root_logger.setLevel(INFO)
    batch_size = 2
    min_epochs = 250
    x = Unet()
    while 1:
        char = input("p/b/t/s")
        if char == "p":
            x.visualize_prediction()
        elif char == "b":
            x.batch_size = batch_size
            x.epoch_i = 60
            x.min_epochs = min_epochs
            x.visualize_batches()
        elif char == "t":
            x.train(
                min_epochs=min_epochs,
                batch_size=batch_size,
                steps_per_epoch=250,
                validation_steps=1,
                early_stopping_min_delta=0,
                RLR_min_delta=0,
                early_stopping_patience=20,
                RLR_patience=90000,
                RLRFactor=0.5,
            )
        elif char == "s":
            img_source, mask_source = Unet.train_DB.get_sources_by_category("Zarovonal", 0)
            common_input_params = {"rotation_hardness": 0, "clockwise": True, "row_hardness": 0, "row_up": False, "col_hardness": 1, "col_left": False}
            mask_source.visualize_an_input(Unet.train_DB.one_hot_coder, **common_input_params)
            img_source.visualize_an_input(**common_input_params, noise_hardness=1)
            img_source.visualize_data()
        else:
            break

    input("end train unet")


if __name__ == "__main__":
    main()
