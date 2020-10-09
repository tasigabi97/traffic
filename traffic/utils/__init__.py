from traffic.imports import (
    check_output,
    CalledProcessError,
    contextmanager,
    Popen,
    subplots,
    any_np,
    Rectangle,
    Polygon,
    uint8,
    uint32,
    zeros,
    find_contours,
    fliplr,
    show,
)
from traffic.logging import root_logger
from traffic.consts import SSID_VODAFONE, IP_VODAFONE, DROIDCAM, DROIDCAM_PORT
from mrcnn.visualize import random_colors, apply_mask


def display_instances(
    image,
    boxes,
    masks,
    class_ids,
    class_names,
    scores=None,
    title="",
    figsize=(16, 16),
    ax=None,
    show_mask=True,
    show_bbox=True,
    colors=None,
    captions=None,
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not any_np(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(uint8))
    if auto_show:
        show()


def get_ssid():
    try:
        return str(check_output(["iwgetid"])).split('"')[1]
    except CalledProcessError:
        root_logger.warning("This PC is not connected to any Wifi network.")


@contextmanager
def webcam_server():
    ssid = get_ssid()
    if ssid == SSID_VODAFONE:
        ip = IP_VODAFONE
    else:
        root_logger.warning("Droidcam is not working with ssid ({}).".format(ssid))
        yield
        return
    try:
        p = Popen([DROIDCAM, "-v", ip, DROIDCAM_PORT])
    except FileNotFoundError as e:
        root_logger.warning(e)
        raise FileNotFoundError("Restart the computer and install droidcam again.")
    else:
        yield
        p.kill()


class Singleton(object):
    _instances = dict()

    def __new__(this_cls, *args, **kwargs):
        for a_singleton_cls, old_instance in Singleton._instances.items():
            if this_cls is a_singleton_cls:
                return old_instance
        new_instance = super().__new__(this_cls)
        Singleton._instances[this_cls] = new_instance
        return new_instance
