if __name__ == "__main__":
    from traffic.examples.camera_with_mrcnn import main as camera_with_mrcnn_main

    camera_with_mrcnn_main()
    from traffic.examples.train_unet import main as train_unet_main

    train_unet_main()
