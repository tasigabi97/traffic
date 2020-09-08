import numpy as np
from traffic.camera import Camera
from cv2 import VideoCapture, cvtColor, imshow, waitKey, destroyAllWindows, COLOR_RGB2GRAY
from time import sleep

if __name__ == "__main__":
    cam_1 = Camera(0)
    # cam_2 = Camera(2)

    while True:
        # sleep(0.2)
        imshow(cam_1.name, cam_1.img)
        # imshow(str(cam_2.id), cam_2.img)

        if waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cam_1.release()
    # cam_2.release()
    destroyAllWindows()
