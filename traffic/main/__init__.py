if __name__ == "__main__":
    from traffic.examples.new import main as new_main

    new_main()
    input()

    from traffic.examples.camera import main as camera_main

    camera_main()
    input()

    from traffic.examples.mrcnn import main as mrcnn_main

    mrcnn_main()
    input()

    from traffic.examples.tensorflow_1 import main as tensorflow_1_main

    tensorflow_1_main()
    input()

    from traffic.examples.original_camera import main as original_camera_main

    original_camera_main()  # nem működik legtöbbször mert bele van égetve a kamera index
