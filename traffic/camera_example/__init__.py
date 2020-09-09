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
