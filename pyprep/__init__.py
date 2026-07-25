"""Initialize PyPREP."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import pyprep.ransac as ransac  # noqa: F401
from pyprep.find_noisy_channels import NoisyChannels  # noqa: F401
from pyprep.prep_pipeline import PrepPipeline  # noqa: F401
from pyprep.reference import Reference  # noqa: F401

# Lazy import helper for optional pyprep.gpu module
try:
    import pyprep.gpu as gpu  # noqa: F401
except Exception:  # pragma: no cover
    gpu = None  # pragma: no cover

try:
    from importlib.metadata import version

    __version__ = version("pyprep")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"  # pragma: no cover
