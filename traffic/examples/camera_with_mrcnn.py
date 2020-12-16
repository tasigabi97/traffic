from traffic.camera import *
from traffic.imports import *
from traffic.utils.lane_unet import Unet
from traffic.logging import *
from traffic.mrcnn import Mrcnn
from traffic.gui import *


def main():
    with choose_camera() as camera:
        window = Window(fullscreen=True)
        category_names = [category.name for category in OneHot(None, None, Category).categories]
        unet, mrcnn = Unet(), Mrcnn()
        root_logger.setLevel(INFO)
        while not window.exit:
            with window:
                img = camera.matplotlib_img
                img = camera.matplotlib_img
                detected_objects = mrcnn.get_prediction(img)
                detected_objects = DetectedObject.get_picked_detected_objects(
                    detected_objects, show_only_important=window.show_only_important, show_only_confident=window.show_only_confident
                )
                unet_prediction = unet.get_prediction(img)
                mrcnn.set_axis(
                    axis=window.mrcnn_and_area_axis,
                    rgb_array=img,
                    detected_objects=detected_objects,
                    title=None,
                    show_mask=window.show_mask,
                    show_mask_contour=window.show_contour,
                    show_bbox=window.show_bbox,
                    show_caption=window.show_caption,
                )
                unet.set_axis(
                    axis=window.unet_axis,
                    probability_matrix=unet_prediction[:, :, window.unet_category_i],
                    threshold=window.unet_threshold,
                    title=category_names[window.unet_category_i],
                )
                window.set_axis(
                    detected_objects=detected_objects,
                    title=None,
                )


if __name__ == "__main__":
    main()
