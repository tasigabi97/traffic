from traffic.imports.builtins import dirname, join_path

_SCRIPTS_DIR = dirname(__file__)
INSTALL_DROIDCAM_HOST_PATH = join_path(_SCRIPTS_DIR, "install_droidcam_host.sh")
INSTALL_PYTHON_HOST_PATH = join_path(_SCRIPTS_DIR, "install_python_host.sh")
INSTALL_NVIDIA_DOCKER_PATH = join_path(_SCRIPTS_DIR, "install_nvidia_docker.sh")
INIT_CONTAINER_PATH = join_path(_SCRIPTS_DIR, "init_container.sh")
