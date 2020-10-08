from traffic.examples.original_camera import main as original_camera_main
from traffic.examples.tensorflow_1 import main as tensorflow_1_main
from traffic.examples.camera import main as camera_main


if __name__ == "__main__":
    camera_main()
    tensorflow_1_main()
    # original_camera_main() #nem működik legtöbbször mert bele van égetve a kamera index
