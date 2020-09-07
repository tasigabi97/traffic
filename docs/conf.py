import os
import sys
from os import chdir

ROOT = os.path.abspath("..")
chdir(ROOT)
sys.path.insert(0, ROOT)
from larning.setup import get_version

project = "Traffic"
copyright = "2020"
author = "Varga László Gábor & Tasnádi Gábor"

release = "0.0.0"

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "classic"

html_static_path = ["_static"]

autodoc_default_options = {"member-order": "bysource", "special-members": True, "private-members": True, "exclude-members": "__weakref__,__dict__"}
latex_elements = {"extraclassoptions": "openany,oneside"}
html_theme_options = {"relbarbgcolor": "black"}
