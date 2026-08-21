"""Test the configuration of pyprep's logger."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import io
import logging
import subprocess
import sys

import pytest

import pyprep
from pyprep._logging import LOGGER_NAME, _StreamHandler


@pytest.fixture
def logger():
    """Hand out the package logger and undo whatever the test did to it."""
    logger = logging.getLogger(LOGGER_NAME)
    level, propagate, handlers = logger.level, logger.propagate, logger.handlers[:]
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _run(code):
    """Run a snippet in a clean interpreter, where logging state is untouched."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )


def _child(name=LOGGER_NAME):
    return logging.getLogger(f"{name}.a_module")


def test_import_configures_nothing():
    """Importing the package installs no handler and leaves the root logger alone."""
    done = _run(
        "import logging, pyprep;"
        "print(logging.getLogger('pyprep').handlers, logging.getLogger().handlers)"
    )
    assert done.stdout.strip() == "[] []"


def test_warning_is_visible_without_any_configuration():
    """A warning reaches the user even when setup_logging was never called.

    A NullHandler on the package logger would satisfy the handler search in
    ``logging.Logger.callHandlers``, which stops ``logging.lastResort`` from
    firing and silently drops warnings and errors instead of only hiding INFO.
    """
    done = _run(
        "import logging, pyprep;"
        "logging.getLogger('pyprep.a_module').warning('the warning')"
    )
    assert "the warning" in done.stderr


def test_setup_logging_writes_formatted_records(logger):
    """The records land on the given stream, with the level and logger name."""
    stream = io.StringIO()
    pyprep.setup_logging(stream=stream)
    _child().info("hello")
    assert "[INFO] pyprep.a_module: hello" in stream.getvalue()


def test_setup_logging_level_filters(logger):
    """A higher level keeps warnings and drops INFO."""
    stream = io.StringIO()
    pyprep.setup_logging("warning", stream=stream)
    _child().info("dropped")
    _child().warning("kept")
    assert "dropped" not in stream.getvalue()
    assert "kept" in stream.getvalue()


@pytest.mark.parametrize("level", ["warning", "WARNING", "Warning", logging.WARNING])
def test_level_spelling(logger, level):
    """Level names are accepted in any case, and stdlib constants work too."""
    pyprep.setup_logging(level, stream=io.StringIO())
    assert logger.level == logging.WARNING


def test_setup_logging_is_idempotent(logger):
    """Repeated calls swap the handler out rather than stacking another one on."""
    stream = io.StringIO()
    for _ in range(3):
        pyprep.setup_logging(stream=stream)
    assert len([h for h in logger.handlers if isinstance(h, _StreamHandler)]) == 1
    _child().info("once")
    assert stream.getvalue().count("once") == 1


def test_setup_logging_keeps_a_handler_it_did_not_install(logger):
    """An application's own handler on this logger survives, and still gets records."""
    app_stream = io.StringIO()
    app_handler = logging.StreamHandler(app_stream)
    app_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(app_handler)

    pyprep.setup_logging(stream=io.StringIO())
    _child().info("seen by both")

    assert app_handler in logger.handlers
    assert app_stream.getvalue() == "seen by both\n"


def test_set_log_level_changes_only_the_level(logger):
    """The handler, its stream and its format survive a level change."""
    stream = io.StringIO()
    pyprep.setup_logging("info", stream=stream, fmt="%(message)s")
    handler = logger.handlers[-1]

    old_level = pyprep.set_log_level("warning", return_old_level=True)
    assert old_level == logging.INFO
    assert logger.handlers[-1] is handler
    assert handler.stream is stream
    assert handler.formatter._fmt == "%(message)s"

    _child().info("dropped")
    _child().warning("kept")
    pyprep.set_log_level(old_level)
    _child().info("kept again")
    assert stream.getvalue() == "kept\nkept again\n"


def test_set_log_level_returns_nothing_by_default(logger):
    """The previous level is handed back only when it is asked for."""
    pyprep.setup_logging("info", stream=io.StringIO())
    assert pyprep.set_log_level("warning") is None
    assert logger.level == logging.WARNING
