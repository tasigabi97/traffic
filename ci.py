#! /usr/bin/env python
from subprocess import run

run(["pip", "install", "larning"], capture_output=True)
from larning.ci import ci_manager, rmdirs, mkdirs, cpdirs
from traffic.consts import *
from traffic.independent import join_path, normpath
from traffic.strings import concatenate_with_separation
from traffic.main import __file__ as MAIN_FILE


def path_in_container(path_in_host: str) -> str:
    path_in_host = normpath(path_in_host)
    return join_path(CONTAINER_ROOT_PATH, path_in_host[len(PROJECT_ROOT_PATH) + 1 :])


def bash_proc(*strings: str) -> list:
    return [
        PROJECT_ROOT_PATH,
        BASH,
        INTERPRET,
        concatenate_with_separation([*strings], SPACE),
    ]


def subcommand(*strings: str) -> str:
    return "$({})".format(concatenate_with_separation([*strings], " "))


def interactive_bash_command(*strings: str) -> str:
    return concatenate_with_separation(
        [BASH, INTERACTIVE, INTERPRET, DOUBLE_QUOTE, *strings, DOUBLE_QUOTE], SPACE
    )


###############################################################################################################
with ci_manager() as (iF, tF, pF, sF):
    BUILD, EGG, DIST, DOCS, PYTEST, PROJ, VENV = (
        join_path(PROJECT_ROOT_PATH, "build"),
        join_path(PROJECT_ROOT_PATH, PROJECT_NAME + ".egg-info"),
        join_path(PROJECT_ROOT_PATH, "dist"),
        join_path(PROJECT_ROOT_PATH, "docs"),
        join_path(PROJECT_ROOT_PATH, ".pytest_cache"),
        join_path(PROJECT_ROOT_PATH, PROJECT_NAME),
        join_path(PROJECT_ROOT_PATH, "venv"),
    )
    _BUILD = join_path(DOCS, "_build")
    tF.delete_before = [rmdirs, [BUILD, EGG, DIST, _BUILD, PYTEST]]
    tF.delete_after = [rmdirs, [BUILD, EGG, DIST, PYTEST]]
    tF.save = [
        cpdirs,
        ["/home/gabi/Desktop/save/traffic", [PROJECT_ROOT_PATH], ["venv"]],
    ]
    tF.create_docs_dir = [mkdirs, [DOCS]]
    pF.install_make = [PROJECT_ROOT_PATH, "sudo", "apt", "install", "make"]
    pF.install_nvidia_docker = [PROJECT_ROOT_PATH, "bash", INSTALL_NVIDIA_DOCKER_PATH]
    pF.init_docs = [DOCS, "sphinx-quickstart"]
    pF.apidoc = [
        PROJECT_ROOT_PATH,
        "sphinx-apidoc",
        "-f",
        "-e",
        "-M",
        "-o",
        "./docs",
        f"./{PROJECT_NAME}",
    ]
    pF.latexpdf = [
        PROJECT_ROOT_PATH,
        "sphinx-build",
        "-M",
        "latexpdf",
        "./docs",
        f"./docs/_build",
    ]
    pF.html = [
        PROJECT_ROOT_PATH,
        "sphinx-build",
        "-M",
        "html",
        "./docs",
        f"./docs/_build",
    ]
    pF.black = [PROJECT_ROOT_PATH, "black", ".", "-t", "py38", "-l", "160"]
    pF.git_status = [PROJECT_ROOT_PATH, "git", "status"]
    pF.git_add_all = [PROJECT_ROOT_PATH, "git", "add", "."]
    pF.git_commit = [PROJECT_ROOT_PATH, "git", "commit", "-m", iF.commit_message]
    pF.git_push = [PROJECT_ROOT_PATH, "git", "push"]
    pF.pytest = [PROJECT_ROOT_PATH, "pytest", "-s"]
    pF.setup_install = [PROJECT_ROOT_PATH, "./setup.py", "install"]
    pF.sdist = [PROJECT_ROOT_PATH, "./setup.py", "sdist", "bdist_wheel"]
    pF.twine_check = [PROJECT_ROOT_PATH, "twine", "check", "dist/*"]
    pF.twine_upload = [
        PROJECT_ROOT_PATH,
        "twine",
        "upload",
        "-u",
        "tasigabi97",
        "dist/*",
    ]
    pF.create_container = bash_proc(
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
        NAME_CONTAINER,
        TENSORFLOW_IMAGE_NAME,
        BASH,
        path_in_container(INIT_CONTAINER_PATH),
    )
    pF.run_container = bash_proc(
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
        NAME_CONTAINER,
        CUSTOM_IMAGE_NAME,
        interactive_bash_command(PYTHON, path_in_container(MAIN_FILE)),
    )
    pF.delete_containers = bash_proc(DOCKER, CONTAINER, PRUNE, FORCE)
    pF.delete_images = bash_proc(DOCKER, IMAGE, PRUNE, ALL)
    pF.commit_container = bash_proc(DOCKER, COMMIT, CONTAINER_NAME, CUSTOM_IMAGE_NAME)
    pF.list_containers = bash_proc(DOCKER, CONTAINER, LIST, ALL)
    pF.list_images = bash_proc(DOCKER, IMAGES)
    pF.enable_display = bash_proc(ENABLE_DISPLAY_HOST)
    ######################################################################################
    sF.ci = [
        ("", pF.pytest),
        ("", pF.black),
        ("", tF.delete_before),
        ("", pF.apidoc),
        ("", pF.latexpdf),
        ("", pF.html),
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
        ("", pF.delete_containers),
        ("", pF.delete_images),
        ("", pF.create_container),
        ("", pF.commit_container),
    ]
    sF.list = [("", pF.list_containers), ("", pF.list_images)]
    sF.delete = [("", pF.delete_containers), ("", pF.delete_images)]
    sF.run = [
        ("", pF.delete_containers),
        ("", pF.enable_display),
        ("", pF.run_container),
    ]
