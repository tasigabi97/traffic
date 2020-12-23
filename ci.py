#! /usr/bin/env python3.8
from subprocess import run

run(["pip3", "install", "larning==0.0.10"], capture_output=True)
from larning.ci import ci_manager, rmdirs, mkdirs, cpdirs
from traffic.consts import *
from traffic.imports.builtins import join_path, normpath
from traffic.strings import concat


def path_in_container(path_in_host: str) -> str:
    return join_path(CONTAINER_ROOT_PATH, normpath(path_in_host)[len(HOST_ROOT_PATH) + 1 :])


def bash_proc(*strings: str) -> list:
    return [
        HOST_ROOT_PATH,
        BASH,
        INTERPRET,
        concat([*strings], SPACE),
    ]


def container_proc(*strings: str) -> list:
    return bash_proc(
        DOCKER,
        RUN,
        INTERACTIVE,
        TTY,
        SHARE_HARDWARE,
        SHARE_NET,
        SHARE_IPC,
        USE_GPU,
        ENABLE_DISPLAY_CONTAINER,
        MOUNT_PROJECT,
        MOUNT_LANE,
        NAME_CONTAINER,
        *strings,
    )


def interactive_bash_command(*strings: str) -> str:
    return concat([BASH, INTERACTIVE, INTERPRET, DOUBLE_QUOTE, *strings, DOUBLE_QUOTE], SPACE)


with ci_manager() as (iF, tF, pF, sF):
    tF.delete_before = [
        rmdirs,
        [BUILD_PATH, EGG_PATH, DIST_PATH, DOCS_BUILD_PATH, PYTEST_CACHE_PATH],
    ]
    tF.delete_after = [rmdirs, [BUILD_PATH, EGG_PATH, DIST_PATH, PYTEST_CACHE_PATH]]
    tF.save = [
        cpdirs,
        ["/home/gabi/Desktop/save/traffic", [HOST_ROOT_PATH], ["venv", "venv3.5"]],
    ]
    tF.create_docs_dir = [mkdirs, [DOCS_PATH]]
    pF.chmod = [HOST_ROOT_PATH, SUDO, CHMOD, ALL_PERMISSION, RECURSIVE, CURR_DIR]
    pF.install_make = [HOST_ROOT_PATH, SUDO, "apt", "install", "make"]
    pF.install_nvidia_docker = [HOST_ROOT_PATH, BASH, INSTALL_NVIDIA_DOCKER_PATH]
    pF.install_droidcam_host = [HOST_ROOT_PATH, BASH, INSTALL_DROIDCAM_HOST_PATH]
    pF.init_docs = [DOCS_PATH, "sphinx-quickstart"]
    pF.apidoc = [
        HOST_ROOT_PATH,
        "sphinx-apidoc",
        "-f",
        "-e",
        "-M",
        "-o",
        "./docs",
        f"./{PROJECT_NAME}",
    ]
    pF.latexpdf = [
        HOST_ROOT_PATH,
        SPHINX_BUILD,
        BUILDERNAME,
        LATEXPDF,
        DOCS_PATH,
        DOCS_BUILD_PATH,
    ]
    pF.html = [
        HOST_ROOT_PATH,
        SPHINX_BUILD,
        BUILDERNAME,
        HTML,
        DOCS_PATH,
        DOCS_BUILD_PATH,
    ]
    pF.black = [HOST_ROOT_PATH, BLACK, CURR_DIR, "-t", "py35", "-l", "120"]
    pF.git_status = [HOST_ROOT_PATH, GIT, "status"]
    pF.git_add_all = [HOST_ROOT_PATH, GIT, "add", "."]
    pF.git_commit = [HOST_ROOT_PATH, GIT, "commit", "-m", iF.commit_message]
    pF.git_push = [HOST_ROOT_PATH, GIT, "push"]
    pF.setup_install = [HOST_ROOT_PATH, "./setup.py", "install"]
    pF.sdist = [HOST_ROOT_PATH, "./setup.py", "sdist", "bdist_wheel"]
    pF.twine_check = [HOST_ROOT_PATH, "twine", "check", "dist/*"]
    pF.twine_upload = [
        HOST_ROOT_PATH,
        "twine",
        "upload",
        "-u",
        "tasigabi97",
        "dist/*",
    ]
    pF.create_container = container_proc(
        TENSORFLOW_IMAGE_NAME,
        BASH,
        path_in_container(INIT_CONTAINER_PATH),
    )
    pF.container_bash = container_proc(AUTO_REMOVE, CUSTOM_IMAGE_NAME, BASH)
    pF.container_main = container_proc(
        AUTO_REMOVE,
        CUSTOM_IMAGE_NAME,
        interactive_bash_command(PYTHON, path_in_container(MAIN_PATH)),
    )
    pF.pytest = container_proc(
        AUTO_REMOVE,
        CUSTOM_IMAGE_NAME,
        interactive_bash_command(PYTEST, "./traffic", DONT_CAPTURE_OUTPUT),
    )
    pF.delete_stopped_containers = bash_proc(DOCKER, CONTAINER, PRUNE, FORCE)
    pF.delete_custom_image = bash_proc(DOCKER, IMAGE, REMOVE, FORCE, CUSTOM_IMAGE_NAME)
    pF.commit_container = bash_proc(DOCKER, COMMIT, IMAGE_WORKDIR, CONTAINER_NAME, CUSTOM_IMAGE_NAME)
    pF.list_containers = bash_proc(DOCKER, CONTAINER, LIST, ALL)
    pF.list_images = bash_proc(DOCKER, IMAGES)
    pF.enable_display = bash_proc(ENABLE_DISPLAY_HOST)
    ######################################################################################
    sF.ci = [
        ("", pF.black),
        ("", pF.pytest),
        ("", pF.chmod),
        ("", tF.delete_before),
        ("", pF.git_status),
        pF.git_add_all,
        pF.git_commit,
        pF.git_push,
        ("", tF.delete_after),
        ("", tF.save),
    ]
    sF.init_docs = [
        tF.create_docs_dir,
        pF.init_docs,
    ]
    sF.setup = [
        ("", pF.setup_install),
        ("", pF.install_make),
        ("", pF.install_nvidia_docker),
        ("", pF.delete_stopped_containers),
        ("", pF.delete_custom_image),
        ("", pF.create_container),
        ("", pF.commit_container),
        ("", pF.delete_stopped_containers),
        ("", pF.install_droidcam_host),
    ]
    sF.list = [("", pF.list_containers), ("", pF.list_images)]
    sF.recreate = [
        (None, pF.delete_stopped_containers),
        (None, pF.delete_custom_image),
        (None, pF.create_container),
        (None, pF.commit_container),
        (None, pF.delete_stopped_containers),
    ]
    sF.main = [
        ("", pF.enable_display),
        ("", pF.container_main),
    ]
    sF.bash = [
        ("", pF.enable_display),
        ("", pF.container_bash),
    ]
