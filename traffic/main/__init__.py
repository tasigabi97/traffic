if __name__ == "__main__":
    from traffic.examples.train_unet import main as train_unet_main

    train_unet_main()
    input()

    from traffic.examples.train_mrcnn import main as train_mrcnn_main

    train_mrcnn_main()
    input()
    from traffic.examples.camera_with_mrcnn import main as camera_with_mrcnn_main

    camera_with_mrcnn_main()
    input()

    from traffic.examples.camera import main as camera_main

    camera_main()
    input()

    from traffic.examples.camera_batchsize_test import main as camera_batchsize_test_main

    camera_batchsize_test_main()
    input()

    from traffic.examples.tensorflow_1 import main as tensorflow_1_main

    tensorflow_1_main()
    input()

    from traffic.examples.original_camera import main as original_camera_main

    original_camera_main()  # nem működik legtöbbször mert bele van égetve a kamera index
