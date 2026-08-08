"""Initialize PyPREP."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import pyprep.ransac as ransac  # noqa: F401
from pyprep._logging import setup_logging  # noqa: F401
from pyprep.find_noisy_channels import NoisyChannels  # noqa: F401
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401
from pyprep.reference import Reference  # noqa: F401

# Loud by default, like MNE. Applications turn it down with
# setup_logging("WARNING") or take over the output with setup_logging(propagate=True).
setup_logging()

try:
    from importlib.metadata import version

    __version__ = version("pyprep")
except Exception:
    __version__ = "0.0.0"
