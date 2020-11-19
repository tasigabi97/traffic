def main():
    from traffic.utils.lane_unet import Unet, LaneDB
    from traffic.utils import show_array
    from traffic.logging import root_logger, INFO
    from traffic.imports import show, imshow_mat, zeros, reshape_np

    root_logger.setLevel(INFO)
    batch_size = 1
    x = Unet()
    while 1:
        char = input("p/b/t")
        if char == "p":
            x.visualize_prediction()
        elif char == "b":
            x.batch_size = batch_size
            x.visualize_batches()
        elif char == "t":
            x.train(
                batch_size=batch_size,
                steps_per_epoch=10,
                validation_steps=1,
                early_stopping_min_delta=0,
                RLR_min_delta=0,
                early_stopping_patience=2,
                RLR_patience=1,
                RLRFactor=0.5,
            )
        else:
            break

    input("end train unet")


if __name__ == "__main__":
    main()
