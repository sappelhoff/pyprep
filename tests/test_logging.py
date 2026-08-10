"""Test the configuration of pyprep's logger."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import io
import logging
import subprocess
import sys

import pytest

from pyprep import setup_logging
from pyprep._logging import _StreamHandler

# Emitted from a clean interpreter, so the logger state of this test session
# cannot influence what is observed.
_FRESH = (
    "import logging, pyprep; "
    "logging.getLogger('pyprep.mod').info('hello'); "
    "print('root handlers:', logging.getLogger().handlers)"
)


@pytest.fixture
def restore_logger():
    """Restore the pyprep logger after a test reconfigured it."""
    logger = logging.getLogger("pyprep")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _run_fresh():
    """Import pyprep in a clean interpreter and return its output."""
    return subprocess.run(
        [sys.executable, "-c", _FRESH], capture_output=True, text=True, check=True
    )


def test_visible_by_default():
    """Test that importing pyprep is enough to see its log messages."""
    completed = _run_fresh()
    assert "[INFO] pyprep.mod: hello" in completed.stdout
    assert "hello" not in completed.stderr


def test_root_logger_untouched():
    """Test that pyprep does not configure the root logger."""
    completed = _run_fresh()
    assert "root handlers: []" in completed.stdout


def test_setup_logging_level(restore_logger):
    """Test that the level set by setup_logging is respected."""
    stream = io.StringIO()
    setup_logging(level="WARNING", stream=stream)
    logger = logging.getLogger("pyprep.mod")
    logger.info("invisible")
    logger.warning("visible")
    assert "invisible" not in stream.getvalue()
    assert "visible" in stream.getvalue()


@pytest.mark.parametrize("level", ["warning", "WARNING", "Warning", logging.WARNING])
def test_setup_logging_accepts_any_level_spelling(restore_logger, level):
    """Level names are case-insensitive, and plain ints work too."""
    setup_logging(level=level, stream=io.StringIO())
    assert restore_logger.level == logging.WARNING


def test_setup_logging_is_idempotent(restore_logger):
    """Test that repeated setup_logging calls do not duplicate output."""
    stream = io.StringIO()
    setup_logging(stream=stream)
    n_handlers = len(restore_logger.handlers)
    setup_logging(stream=stream)
    logging.getLogger("pyprep.mod").info("once")
    assert len(restore_logger.handlers) == n_handlers
    assert stream.getvalue().count("once") == 1


def test_setup_logging_propagate_hands_over(restore_logger, caplog):
    """Test that propagate=True lets an application handle the records."""
    setup_logging(propagate=True)
    assert not [h for h in restore_logger.handlers if isinstance(h, _StreamHandler)]
    with caplog.at_level(logging.INFO, logger="pyprep.mod"):
        logging.getLogger("pyprep.mod").info("handed over")
    assert "handed over" in caplog.text
