"""Configure Sphinx."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import sys
from datetime import datetime
from pathlib import Path

from intersphinx_registry import get_intersphinx_mapping
from sphinx.config import is_serializable

import pyprep

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
curdir = Path(__file__).parent
sys.path.append((curdir / ".." / "eeg_positions").resolve())
sys.path.append((curdir / ".." / "sphinxext").resolve())

# -- Project information -----------------------------------------------------
project = "pyprep"
author = "PyPREP developers"
copyright = f"2018, {author}. Last updated {datetime.now().isoformat()}"  # noqa: A001

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

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# configure numpydoc
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
html_copy_source = False

html_theme = "pydata_sphinx_theme"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/sappelhoff/pyprep",
            icon="fab fa-github-square",
        ),
    ],
    "icon_links_label": "Quick Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_context = {
    "default_mode": "auto",
    "doc_path": "doc",
}

html_sidebars = {}

# When functions from other packages are mentioned, link to them
# If a package is not in the intersphinx_registry, add like:
# intersphinx_mapping["mne"] = ("https://mne.tools/dev", None)
intersphinx_mapping = get_intersphinx_mapping(
    packages={
        "python",
        "mne",
        "numpy",
        "scipy",
        "matplotlib",
    }
)
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

assert is_serializable(sphinx_gallery_conf)
