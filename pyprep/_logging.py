"""Configure the logging of pyprep."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import logging
import sys

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOGGER_NAME = "pyprep"


class _StreamHandler(logging.StreamHandler):
    """Marker subclass identifying the handler that pyprep installed."""


def setup_logging(level="INFO", stream=None, propagate=False):
    """Configure the ``pyprep`` logger.

    This is called at import time with its defaults, so pyprep logs at ``"INFO"``
    to :data:`sys.stdout` without any setup. Call it again to change that.

    Parameters
    ----------
    level : int | str
        Level for the ``pyprep`` logger. Pass ``"WARNING"`` to keep only warnings
        and errors, or ``"DEBUG"`` for more detail. Names are case-insensitive, as
        in :func:`mne.set_log_level`. Defaults to ``"INFO"``.
    stream : file-like | None
        Destination for pyprep's own handler. ``None`` keeps the current
        destination, which is :data:`sys.stdout` on a fresh interpreter. Ignored
        when ``propagate`` is ``True``.
    propagate : bool
        ``False`` (the default) prints through pyprep's own handler and stops the
        records there, so an application that configures the root logger does not
        see them twice. ``True`` removes that handler and passes the records to
        ancestor loggers instead, letting the application's own handlers format
        and route them.

    Notes
    -----
    Only the ``pyprep`` namespace is configured; the root logger is never touched.
    MNE has its own ``mne`` logger, controlled separately through
    :func:`mne.set_log_level`. Before pyprep 0.8.0 several pyprep modules logged
    through MNE's logger, so ``mne.set_log_level`` used to silence them; use this
    function instead.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    # The stdlib only knows upper-case level names; MNE accepts any case, so do we.
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    own = [h for h in logger.handlers if isinstance(h, _StreamHandler)]
    if propagate:
        for handler in own:
            logger.removeHandler(handler)
            handler.close()
    elif own:
        if stream is not None:
            for handler in own:
                handler.setStream(stream)
    else:
        handler = _StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.propagate = propagate
