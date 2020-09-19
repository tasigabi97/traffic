#! /usr/bin/env python
from sys import path,version_info
assert version_info.major==3 and version_info.minor >= 5
from os.path import dirname
path.append(dirname(dirname(dirname(__file__))))
######################################################################################################
from traffic.camera import get_cameras, webcam_server
from cv2 import imshow, waitKey, destroyAllWindows
from itertools import cycle

if __name__ == "__main__":
    with webcam_server():
        with get_cameras() as cameras:
            for camera in cycle(cameras):
                imshow(camera.name, camera.img)
                if waitKey(1) & 0xFF == ord("q"):
                    break
        destroyAllWindows()
