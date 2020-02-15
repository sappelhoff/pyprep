"""initialize pyprep."""
__version__ = "0.3.dev0"

from pyprep.noisy import find_bad_epochs, Noisydata  # noqa: F401
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401
