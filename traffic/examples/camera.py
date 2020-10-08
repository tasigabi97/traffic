def main():
    from traffic.camera import choose_camera
    from traffic.imports import imshow
    from traffic.globals import g

    with choose_camera() as camera:
        g.wait_keys = "q"
        while True:
            imshow("Chosen-> ({})".format(camera.name), camera.img)
            if g.pressed_key is not None:
                break


if __name__ == "__main__":
    main()
