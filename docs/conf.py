"""Configure sphinx."""
import os
import sys
from datetime import date

import sphinx_bootstrap_theme
import sphinx_gallery  # noqa: F401

import pyprep

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "pyprep")))
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))

# -- Project information -----------------------------------------------------
project = "pyprep"
_today = date.today()
copyright = f"2018-{_today.year}, pyprep developers. Last updated {_today.isoformat()}"
author = "pyprep developers"

# The short X.Y version
version = pyprep.__version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "gh_substitutions",  # custom extension, see ./sphinxext/gh_substitutions.py
    "sphinx_copybutton",
]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

master_doc = "index"
autosummary_generate = True
numpydoc_show_class_members = False  # https://stackoverflow.com/a/34604043/5201771
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_attributes_as_param_list = False
numpydoc_xref_ignore = {
    # words
    "of",
    "shape",
}

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML options (e.g., theme)
# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False

html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

templates_path = ["_templates"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "navbar_title": "pyprep",
    "bootswatch_theme": "flatly",
    "navbar_sidebarrel": False,  # no "previous / next" navigation
    "navbar_pagenav": False,  # no "Page" navigation in sidebar
    "bootstrap_version": "3",
    "navbar_links": [
        ("Examples", "auto_examples/index"),
        ("API", "api"),
        ("What's new", "whats_new"),
        ("Differences from PREP", "matlab_differences"),
        ("GitHub", "https://github.com/sappelhoff/pyprep", True),
    ],
}


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mne": ("https://mne.tools/dev", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "matplotlib": ("https://matplotlib.org", None),
}
intersphinx_timeout = 10

sphinx_gallery_conf = {
    "doc_module": "pyprep",
    "reference_url": {
        "pyprep": None,
    },
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": "^((?!sgskip).)*$",
    "backreferences_dir": "generated",
}
