#! /usr/bin/env python

from larning.ci import ci_manager, rmdirs, mkdirs, cpdirs
from os import getcwd
from os.path import join
from traffic import __name__ as PROJ_NAME

with ci_manager() as (iF, tF, pF, sF):
    WD = getcwd()
    BUILD, EGG, DIST, DOCS, PYTEST, PROJ, VENV = (
        join(WD, "build"),
        join(WD, PROJ_NAME + ".egg-info"),
        join(WD, "dist"),
        join(WD, "docs"),
        join(WD, ".pytest_cache"),
        join(WD, PROJ_NAME),
        join(WD, "venv"),
    )
    _BUILD = join(DOCS, "_build")
    tF.delete_before = [rmdirs, [BUILD, EGG, DIST, _BUILD, PYTEST]]
    tF.delete_after = [rmdirs, [BUILD, EGG, DIST, PYTEST]]
    tF.save = [cpdirs, ["/home/gabi/Desktop/save/traffic", [WD], ["venv"]]]
    tF.create_docs_dir = [mkdirs, [DOCS]]
    pF.install_make = [DOCS, "sudo", "apt", "install", "make"]
    pF.init_docs = [DOCS, "sphinx-quickstart"]
    pF.apidoc = [WD, "sphinx-apidoc", "-f", "-e", "-M", "-o", "./docs", f"./{PROJ_NAME}"]
    pF.latexpdf = [WD, "sphinx-build", "-M", "latexpdf", "./docs", f"./docs/_build"]
    pF.html = [WD, "sphinx-build", "-M", "html", "./docs", f"./docs/_build"]
    pF.black = [WD, "black", ".", "-t", "py38", "-l", "160"]
    pF.git_status = [WD, "git", "status"]
    pF.git_add_all = [WD, "git", "add", "."]
    pF.git_commit = [WD, "git", "commit", "-m", iF.commit_message]
    pF.git_push = [WD, "git", "push"]
    pF.pytest = [WD, "pytest", "-s"]
    pF.setup_install = [WD, "./setup.py", "install"]
    pF.sdist = [WD, "./setup.py", "sdist", "bdist_wheel"]
    pF.twine_check = [WD, "twine", "check", "dist/*"]
    pF.twine_upload = [
        WD,
        "twine",
        "upload",
        "-u",
        "tasigabi97",
        "dist/*",
    ]

    sF.init_docs = [
        tF.create_docs_dir,
        pF.init_docs,
    ]
    sF.setup = [pF.setup_install, pF.install_make]

    sF.a = [
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
