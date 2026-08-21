"""Configure the logging of pyprep."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import logging
import sys

LOGGER_NAME = "pyprep"
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class _StreamHandler(logging.StreamHandler):
    """Marker subclass identifying the handler that this package installed."""


def _as_level(level):
    """Accept a name in any case, or a stdlib constant like logging.DEBUG."""
    return level.upper() if isinstance(level, str) else level


def set_log_level(level, return_old_level=False):
    """Change the level of the ``pyprep`` logger, leaving its handlers alone.

    Use this rather than :func:`pyprep.setup_logging` to turn the package up or
    down: :func:`pyprep.setup_logging` also decides *where* the records go, so
    calling it again just to change the level would reset the stream, the format,
    and the propagation that an application had chosen.

    Parameters
    ----------
    level : int | str
        Level for the ``pyprep`` logger, either a name (e.g. ``"debug"``,
        ``"WARNING"``) or a stdlib constant such as :data:`logging.DEBUG`. Names
        are case-insensitive, as in :func:`mne.set_log_level`.
    return_old_level : bool
        Whether to return the level the logger had before this call, so that a
        caller can put it back afterwards. Defaults to ``False``.

    Returns
    -------
    old_level : int | None
        The previous level of the ``pyprep`` logger if ``return_old_level`` is
        ``True``, and ``None`` otherwise.

    Notes
    -----
    This cannot make records appear on its own. Until :func:`pyprep.setup_logging`
    has run, or the application has configured a logger that this one propagates
    to, there is no handler to write them.
    """
    logger = logging.getLogger(LOGGER_NAME)
    old_level = logger.level
    logger.setLevel(_as_level(level))
    if return_old_level:
        return old_level


def setup_logging(level="info", *, stream=None, fmt=DEFAULT_FORMAT):
    """Send pyprep's log records to a stream, at the given level.

    If your application already routes logging somewhere of its own, do not call
    this at all. The records will reach your handlers by propagation, and
    :func:`pyprep.set_log_level` is there if you want the ``"INFO"`` ones too.

    Parameters
    ----------
    level : int | str
        Level for the ``pyprep`` logger, either a name (e.g. ``"debug"``,
        ``"WARNING"``) or a stdlib constant such as :data:`logging.DEBUG`. Names
        are case-insensitive, as in :func:`mne.set_log_level`. Defaults to
        ``"info"``.
    stream : file-like | None
        Destination for pyprep's handler. ``None`` means :data:`sys.stdout`.
    fmt : str
        Format string for the handler, in the style of :class:`logging.Formatter`.
        Defaults to ``pyprep._logging.DEFAULT_FORMAT``, which prefixes each
        record with a timestamp, its level, and the name of the logger it came
        from.

    Returns
    -------
    logger : logging.Logger
        The configured ``pyprep`` logger.

    Notes
    -----
    Only the ``pyprep`` namespace is configured; the root logger is never
    touched. Records stop at pyprep's own handler, so an application that
    configures the root logger does not see them twice. To hand the output back,
    drop the handler again with
    ``logging.getLogger("pyprep").handlers.clear()``.

    MNE has its own ``mne`` logger, controlled separately through
    :func:`mne.set_log_level`. The progress bar that window-wise RANSAC draws is
    an ``mne.utils.ProgressBar``, which writes to its own stream rather than
    through either logger, so neither function silences it.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_as_level(level))

    # Repeated calls swap our handler out instead of stacking a new one on top. The
    # marker subclass is what makes ours identifiable: a plain isinstance check
    # against logging.StreamHandler would also match a FileHandler that the
    # application attached to this logger, and we would remove and close it.
    for handler in [h for h in logger.handlers if isinstance(h, _StreamHandler)]:
        logger.removeHandler(handler)
        handler.close()

    handler = _StreamHandler(sys.stdout if stream is None else stream)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
