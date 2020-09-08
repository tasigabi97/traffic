from traffic.camera import get_cameras
from cv2 import imshow, waitKey, destroyAllWindows
from itertools import cycle

if __name__ == "__main__":
    with get_cameras() as cameras:
        for camera in cycle(cameras):
            imshow(camera.name, camera.img)
            if waitKey(1) & 0xFF == ord("q"):
                break
    destroyAllWindows()
