:orphan:

.. _api_documentation:

=================
API Documentation
=================

Here we list the Application Programming Interface (API) for pyprep.

The :mod:`find_noisy_channels` module
=====================================

.. currentmodule:: find_noisy_channels

The :class:`NoisyChannels` class
----------------------------
.. autosummary::
   :toctree: generated/

   NoisyChannels
   NoisyChannels.find_bad_by_correlation
   NoisyChannels.find_bad_by_deviation
   NoisyChannels.find_bad_by_flat
   NoisyChannels.find_bad_by_hf_noise
   NoisyChannels.find_bad_by_nan
   NoisyChannels.find_bad_by_ransac
   NoisyChannels.get_bads

The :mod:`prep_pipeline` module
===============================

.. currentmodule:: prep_pipeline

The :class:`PrepPipeline` class
-------------------------------
.. autosummary::
   :toctree: generated/

   PrepPipeline
   PrepPipeline.fit
