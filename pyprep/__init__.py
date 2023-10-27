"""initialize pyprep."""
import pyprep.ransac as ransac  # noqa: F401
from pyprep.find_noisy_channels import NoisyChannels  # noqa: F401
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401
from pyprep.reference import Reference  # noqa: F401

from . import _version

__version__ = _version.get_versions()["version"]
