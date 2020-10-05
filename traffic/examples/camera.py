def main():
    from traffic.camera import get_cameras, webcam_server
    from traffic.imports import imshow, waitKey, destroyAllWindows, cycle

    with webcam_server():
        with get_cameras() as cameras:
            for camera in cycle(cameras):
                imshow(camera.name, camera.img)
                if waitKey(1) & 0xFF == ord("q"):
                    break
        destroyAllWindows()


if __name__ == "__main__":
    main()
