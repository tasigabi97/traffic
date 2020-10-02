from traffic.bash_scripts import *
from traffic import __name__ as PROJECT_NAME
from traffic import __file__ as TRAFFIC_FILE
from traffic.independent import dirname, join_path

CONTAINER_ROOT_PATH = join_path("/", PROJECT_NAME)
PROJECT_ROOT_PATH = dirname(dirname(TRAFFIC_FILE))
####################################################
ALL = "--all"
BASH = "bash"
COMMIT = "commit"
CONTAINER = "container"
CONTAINER_NAME = "container_name"
DOCKER = "sudo docker"
DOUBLE_QUOTE = '"'
ENABLE_DISPLAY_HOST = "xhost +local:"
FORCE = "-f"
IMAGE = "image"
IMAGES = "images"
INTERACTIVE = "-i"
INTERPRET = "-c"
LIST = "list"
PRUNE = "prune"
PYTHON = "python"
REMOVE = "rm"
RUN = "run"
SHARE_HARDWARE = "--privileged"
SHARE_IPC = "--ipc=host"
SHARE_NET = "--net=host"
SPACE = " "
TENSORFLOW_VERSION = "1.3.0"
TTY = "-t"
USE_GPU = "--gpus all"
X11_PATH = "/tmp/.X11-unix"
###############################################################################################################
CUSTOM_IMAGE_NAME = f"{PROJECT_NAME}/{PROJECT_NAME}:{TENSORFLOW_VERSION}"
ENABLE_DISPLAY_CONTAINER = f"--env DISPLAY=$DISPLAY --mount type=bind,source={X11_PATH},destination={X11_PATH},readonly"
MOUNT_PROJECT = (
    f"--mount type=bind,source={PROJECT_ROOT_PATH},destination={CONTAINER_ROOT_PATH}"
)
NAME_CONTAINER = f"--name {CONTAINER_NAME}"
TENSORFLOW_IMAGE_NAME = f"tensorflow/tensorflow:{TENSORFLOW_VERSION}-gpu-py3"
