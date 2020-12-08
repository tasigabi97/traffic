from traffic.camera import *
from traffic.imports import *
from traffic.utils.lane_unet import Unet
from traffic.logging import *
from traffic.mrcnn import Mrcnn
from traffic.gui import *


def main():
    with choose_camera() as camera:
        window = Window(fullscreen=False)
        unet, mrcnn = Unet(), Mrcnn()
        root_logger.setLevel(INFO)
        while window.draw():
            img = camera.matplotlib_img
            img = camera.matplotlib_img
            detected_objects = mrcnn.get_prediction(img)
            unet_prediction = unet.get_prediction(img)
            mrcnn.set_axis(
                axis=window.mrcnn_axis,
                rgb_array=img,
                detected_objects=detected_objects,
                title=None,
                show_mask=window.show_mask,
                show_mask_contour=window.show_mask_contour,
                show_bbox=window.show_bbox,
                show_caption=True,
                show_only_important=window.show_only_upper,
                show_only_confident=window.show_only_confident,
            )
            unet.set_axis(axis=window.unet_lane_axis, probability_matrix=unet_prediction[:, :, 1], threshold=window.unet_lane_threshold, title="unet_lane_axis")
            unet.set_axis(axis=window.unet_bg_axis, probability_matrix=unet_prediction[:, :, 0], threshold=window.unet_bg_threshold, title="unet_bg_axis")
            window.set_axis(
                rgb_array=img,
                detected_objects=detected_objects,
                title=None,
                show_only_important=window.show_only_upper,
                show_only_confident=window.show_only_confident,
            )


if __name__ == "__main__":
    main()
