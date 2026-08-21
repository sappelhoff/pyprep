:orphan:

.. _api_documentation:

=================
API Documentation
=================

Here we list the Application Programming Interface (API) for pyprep.

The :class:`~pyprep.NoisyChannels` class
----------------------------------------

.. automodule:: pyprep
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyprep

.. autosummary::
   :toctree: generated/

   NoisyChannels

The :class:`~pyprep.Reference` class
------------------------------------

.. autosummary::
   :toctree: generated/

   Reference

The :class:`~pyprep.PrepPipeline` class
---------------------------------------

.. autosummary::
   :toctree: generated/

   PrepPipeline

The :mod:`~pyprep.ransac` module
================================

.. automodule:: pyprep.ransac
   :no-members:
   :no-inherited-members:

.. currentmodule:: ransac

.. autosummary::
   :toctree: generated/

   find_bad_by_ransac

Logging
=======

pyprep logs through the standard :mod:`logging` module, on a logger named
``pyprep``, and — like any well-behaved library — configures nothing on import.
It is quiet, but not silent: with no configuration at all, warnings and errors
still reach :data:`sys.stderr`, because Python falls back to
:data:`logging.lastResort` when no handler is found. ``"INFO"`` records, which
is where the pipeline reports what it decided, are hidden until you ask for
them.

There are two ways to ask. Interactively, :func:`~pyprep.setup_logging`
attaches a handler to the ``pyprep`` logger and stops those records from
propagating any further, so they are not printed twice:

.. code-block:: python

   import pyprep

   pyprep.setup_logging("info")

In an application that already routes logging somewhere of its own — a file, a
JSON aggregator, a Rich console — do not call it at all. The records reach your
handlers by propagation, and :func:`~pyprep.set_log_level` raises pyprep's
verbosity without touching the handler, the stream, the format, or the
propagation you chose:

.. code-block:: python

   pyprep.set_log_level("info")

The root logger is never configured either way. MNE has its own ``mne`` logger,
controlled separately with :func:`mne.set_log_level`; note also that the
progress bar drawn during window-wise RANSAC is an ``mne.utils.ProgressBar``,
which writes to its own stream rather than through either logger, so neither
function silences it.

.. currentmodule:: pyprep

.. autosummary::
   :toctree: generated/

   setup_logging
   set_log_level
