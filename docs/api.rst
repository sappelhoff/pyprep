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

pyprep logs at ``"INFO"`` through its own ``pyprep`` logger, which is configured
when the package is imported. Use the function below to change the level, redirect
the stream, or hand the records to your application's own handlers. The root logger
is never configured.

.. currentmodule:: pyprep

.. autosummary::
   :toctree: generated/

   setup_logging
