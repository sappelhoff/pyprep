"""Initialize PyPREP."""
import pyprep.ransac as ransac  # noqa: F401
from pyprep.find_noisy_channels import NoisyChannels  # noqa: F401
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401
from pyprep.reference import Reference  # noqa: F401

try:
    from importlib.metadata import version

    __version__ = version("pyprep")
except Exception:
    __version__ = "0.0.0"
