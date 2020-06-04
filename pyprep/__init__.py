"""initialize pyprep."""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from pyprep.noisy import find_bad_epochs, Noisydata  # noqa: F401
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401
