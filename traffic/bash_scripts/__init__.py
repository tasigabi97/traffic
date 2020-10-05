from traffic.imports.builtins import dirname, join_path

_SCRIPTS_DIR = dirname(__file__)
INSTALL_DROIDCAM_PATH = join_path(_SCRIPTS_DIR, "install_droidcam.sh")
INSTALL_NVIDIA_PATH = join_path(_SCRIPTS_DIR, "install_nvidia_ubuntu_20.04_tensorflow_2.sh")
RUN_DROIDCAM_PATH = join_path(_SCRIPTS_DIR, "run_droidcam.sh")
CHANGE_CAMERA_PATH = join_path(_SCRIPTS_DIR, "change_camera.sh")
INSTALL_NVIDIA_DOCKER_PATH = join_path(_SCRIPTS_DIR, "install_nvidia_docker.sh")
INIT_CONTAINER_PATH = join_path(_SCRIPTS_DIR, "init_container.sh")
