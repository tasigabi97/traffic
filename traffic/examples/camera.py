def main():
    from traffic.camera import choose_camera
    from traffic.imports import imshow, waitKey, destroyAllWindows

    with choose_camera() as camera:
        while True:
            imshow("{}-> ({})".format("Chosen", camera.name), camera.img)
            key = waitKey(1) & 0xFF
            if key == ord("q"):
                break


if __name__ == "__main__":
    main()
