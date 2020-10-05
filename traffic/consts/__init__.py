from traffic.consts.i import *
from traffic.bash_scripts import *
from traffic.imports.builtins import dirname, join_path
from traffic import __name__ as PROJECT_NAME

PROJECT_NAME = PROJECT_NAME
from traffic import __file__ as TRAFFIC_PATH

TRAFFIC_PATH = TRAFFIC_PATH
from traffic.main import __file__ as MAIN_PATH

MAIN_PATH = MAIN_PATH

CONTAINER_ROOT_PATH = join_path("/", PROJECT_NAME)
HOST_ROOT_PATH = dirname(dirname(TRAFFIC_PATH))
CUSTOM_IMAGE_NAME = f"{PROJECT_NAME}/{PROJECT_NAME}:{TENSORFLOW_VERSION}"
ENABLE_DISPLAY_CONTAINER = f"--env DISPLAY=$DISPLAY --mount type=bind,source={X11_PATH},destination={X11_PATH},readonly"
MOUNT_PROJECT = f"--mount type=bind,source={HOST_ROOT_PATH},destination={CONTAINER_ROOT_PATH}"
NAME_CONTAINER = f"--name {CONTAINER_NAME}"
TENSORFLOW_IMAGE_NAME = f"tensorflow/tensorflow:{TENSORFLOW_VERSION}-gpu-py3"
IMAGE_WORKDIR = f'--change "WORKDIR {CONTAINER_ROOT_PATH}"'
BUILD_PATH = join_path(HOST_ROOT_PATH, "build")
EGG_PATH = join_path(HOST_ROOT_PATH, PROJECT_NAME + ".egg-info")
DIST_PATH = join_path(HOST_ROOT_PATH, "dist")
DOCS_PATH = join_path(HOST_ROOT_PATH, "docs")
PYTEST_CACHE_PATH = join_path(HOST_ROOT_PATH, ".pytest_cache")
VENV_PATH = join_path(HOST_ROOT_PATH, "venv")
DOCS_BUILD_PATH = join_path(DOCS_PATH, "_build")
