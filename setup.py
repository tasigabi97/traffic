#! /usr/bin/env python

from larning.setup import get_version, get_github_url, PACKAGE_NAME, PACKAGES, setup, LONG_DESCRIPTION, require_interpreter_version

# ˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇˇ
require_interpreter_version(3, 6, 0)
version = get_version(0, 0, 0)
INSTALL_REQUIRES = []
AUTHOR = "Tasnádi Gábor"
EMAIL = "tasi.gabi97@gmail.com"
URL = get_github_url("tasigabi97")
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
setup(
    name=PACKAGE_NAME,
    version=version,
    author=AUTHOR,
    author_email=EMAIL,
    description=PACKAGE_NAME,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    packages=PACKAGES,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    install_requires=INSTALL_REQUIRES,
    keywords=[
        PACKAGE_NAME,
    ],
    license="MIT",
)
