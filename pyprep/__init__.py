"""initialize pyprep."""
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
