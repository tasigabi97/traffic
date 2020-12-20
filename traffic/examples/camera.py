"""
Egy példa arra, hogyan lehet használni a cv2-es kamera ablakokat.
"""


def main():
    from traffic.camera import choose_camera
    from traffic.imports import imshow_cv2
    from traffic.cv2_input import cv2_input

    with choose_camera() as camera:  # Feldobja az elérhető kamerákat, amikből egyet kell kiválasztani.
        cv2_input.wait_keys = "q"
        while True:
            imshow_cv2("Chosen-> ({})".format(camera.name), camera.img)
            if cv2_input.pressed_key is not None:  # A q betű lenyomásval be lehet zárni a kiválasztott kamera képét.
                break


if __name__ == "__main__":
    main()
