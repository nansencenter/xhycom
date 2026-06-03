import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "xhycom"
author = "Nora Loose"
copyright = "2024, Nora Loose"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

nb_execution_mode = "off"   # notebooks are pre-executed; don't re-run at build time
myst_enable_extensions = ["colon_fence"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "cftime": ("https://unidata.github.io/cftime", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "sphinx_rtd_theme"
# To use furo instead: pip install furo, then set html_theme = "furo"
html_theme_options = {
    "navigation_depth": 3,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_static_path = []
