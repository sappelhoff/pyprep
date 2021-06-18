:orphan:

Authors
=======

People who contributed to this software across releases (in **alphabetical order (last name)**)

* `Stefan Appelhoff`_
* `Austin Hurst`_
* `Aamna Lawrence`_
* `Adam Li`_
* `Yorguin Mantilla`_
* `Christian O'Reilly`_
* `Victor Xiang`_

.. _whats_new:

What's New
==========

Here we list a changelog of pyprep.

.. contents:: Contents
   :local:
   :depth: 1

.. currentmodule:: pyprep

.. _current:

Current
-------

Changelog
~~~~~~~~~
- Created a new module named :mod:`pyprep.ransac` which contains :func:`find_bad_by_ransac <ransac.find_bad_by_ransac>`,  a standalone function mirroring the previous ransac method from the :class:`NoisyChannels` class, by `Yorguin Mantilla`_ (:gh:`51`)
- Added two attributes :attr:`PrepPipeline.noisy_channels_before_interpolation <pyprep.PrepPipeline>` and :attr:`PrepPipeline.noisy_channels_after_interpolation <pyprep.PrepPipeline>` which have the detailed output of each noisy criteria, by `Yorguin Mantilla`_ (:gh:`45`)
- Added two keys to the :attr:`PrepPipeline.noisy_channels_original <pyprep.PrepPipeline>` dictionary: ``bad_by_dropout`` and ``bad_by_SNR``, by `Yorguin Mantilla`_ (:gh:`45`)
- Changed RANSAC chunking logic to reduce max memory use and prefer equal chunk sizes where possible, by `Austin Hurst`_ (:gh:`44`)
- Changed RANSAC's random channel sampling code to produce the same results as MATLAB PREP for the same random seed, additionally changing the default RANSAC sample size from 25% of all *good* channels (e.g. 15 for a 64-channel dataset with 4 bad channels) to 25% of *all* channels (e.g. 16 for the same dataset), by `Austin Hurst`_ (:gh:`62`)
- Changed RANSAC so that "bad by high-frequency noise" channels are retained when making channel predictions (provided they aren't flagged as bad by any other metric), matching MATLAB PREP behaviour, by `Austin Hurst`_ (:gh:`64`)
- Added a new flag ``matlab_strict`` to :class:`~pyprep.PrepPipeline`, :class:`~pyprep.Reference`, :class:`~pyprep.NoisyChannels`, and :func:`~pyprep.ransac.find_bad_by_ransac` for optionally matching MATLAB PREP's internal math as closely as possible, overriding areas where PyPREP attempts to improve on the original, by `Austin Hurst`_ (:gh:`70`)
- Added a ``matlab_strict`` method for high-pass trend removal, exactly matching MATLAB PREP's values if ``matlab_strict`` is enabled, by `Austin Hurst`_ (:gh:`71`)
- Added a window-wise implementaion of RANSAC and made it the default method, reducing the typical RAM demands of robust re-referencing considerably, by `Austin Hurst`_ (:gh:`66`)
- Added `max_chunk_size` parameter for specifying the maximum chunk size to use for channel-wise RANSAC, allowing more control over PyPREP RAM usage, by `Austin Hurst`_ (:gh:`66`)
- Changed :class:`~pyprep.Reference` to exclude "bad-by-SNR" channels from initial average referencing, matching MATLAB PREP behaviour, by `Austin Hurst`_ (:gh:`78`)
- Changed :class:`~pyprep.Reference` to only flag "unusable" channels (bad by flat, NaNs, or low SNR) from the first pass of noisy detection for permanent exclusion from the reference signal, matching MATLAB PREP behaviour, by `Austin Hurst`_ (:gh:`78`)
- Added a framework for automated testing of PyPREP's components against their MATLAB PREP counterparts (using ``.mat`` and ``.set`` files generated with the `matprep_artifacts`_ script), helping verify that the two PREP implementations are numerically equivalent when `matlab_strict` is ``True``, by `Austin Hurst`_ (:gh:`79`)
- Changed :class:`~pyprep.NoisyChannels` to reuse the same random state for each run of RANSAC when ``matlab_strict`` is ``True``, matching MATLAB PREP behaviour, by `Austin Hurst`_ (:gh:`89`)

.. _matprep_artifacts: https://github.com/a-hurst/matprep_artifacts

Bug
~~~
- Fixed RANSAC to give consistent results with a fixed seed across different chunk sizes, by `Austin Hurst`_ and `Yorguin Mantilla`_ (:gh:`43`)
- Fixed "bad channel by flat" threshold in :meth:`NoisyChannels.find_bad_by_nan_flat` to be consistent with MATLAB PREP, by `Austin Hurst`_ (:gh:`60`)
- Changed "bad channel by deviation" and "bad channel by correlation" detection code in :class:`NoisyChannels` to compute IQR and quantiles in the same manner as MATLAB, thus producing identical results to MATLAB PREP, by `Austin Hurst`_ (:gh:`57`)
- Fixed a bug where EEG data was getting reshaped into RANSAC windows incorrectly (channel samples were not sequential), which was causing considerable variability and noise in RANSAC results, by `Austin Hurst`_ (:gh:`67`)
- Fixed RANSAC to avoid making unnecessary signal predictions for known-bad channels, matching MATLAB behaviour and reducing RAM requirements, by `Austin Hurst`_ (:gh:`72`)
- Fixed a bug in :meth:`NoisyChannels.find_bad_by_correlation` that prevented it from being able to handle channels with dropouts (intermittent flat regions), by `Austin Hurst`_ (:gh:`81`).
- Fixed :class:`~pyprep.NoisyChannels` so that it always runs "bad channel by NaN" and "bad channel by flat" detection, preventing these channels from causing problems with other :class:`~pyprep.NoisyChannels` methods, by `Austin Hurst`_ (:gh:`79`)
- Fixed :class:`~pyprep.Reference` so that channels are no longer excluded from final average reference calculation if they were originally bad by NaN, flat, or low SNR, by `Austin Hurst`_ (:gh:`92`)

API
~~~
- The permissible parameters for the following methods were removed and/or reordered: `ransac._ransac_correlations`, `ransac._run_ransac`, and `ransac._get_ransac_pred` methods, by `Yorguin Mantilla`_ (:gh:`51`)
- The following methods have been moved to a new module named :mod:`~pyprep.ransac` and are now private: `NoisyChannels.ransac_correlations`, `NoisyChannels.run_ransac`, and `NoisyChannels.get_ransac_pred` methods, by `Yorguin Mantilla`_ (:gh:`51`)
- The permissible parameters for the following methods were removed and/or reordered: `NoisyChannels.ransac_correlations`, `NoisyChannels.run_ransac`, and `NoisyChannels.get_ransac_pred` methods, by `Austin Hurst`_ and `Yorguin Mantilla`_ (:gh:`43`)
- Changed the meaning of the argument `channel_wise` in :meth:`~pyprep.NoisyChannels.find_bad_by_ransac` to mean 'perform RANSAC across chunks of channels instead of window-wise', from its original meaning of 'perform channel-wise RANSAC one channel at a time', by `Austin Hurst`_ (:gh:`66`)
- The arguments `fraction_bad` and `fraction_good` were renamed to `frac_bad` and `sample_prop`, respectively, for :meth:`~pyprep.NoisyChannels.find_bad_by_ransac` and :func:`~pyprep.ransac.find_bad_by_ransac`, by `Austin Hurst`_ (:gh:`88`)


.. _changes_0_3_1:

Version 0.3.1
-------------

Changelog
~~~~~~~~~
- It's now possible to pass keyword arguments to the notch filter inside :class:`PrepPipeline <pyprep.PrepPipeline>`; see the ``filter_kwargs`` parameter by `Yorguin Mantilla`_ (:gh:`40`)
- The default filter length for the spectrum_fit method will be '10s' to fix memory issues, by `Yorguin Mantilla`_ (:gh:`40`)
- Channel types  are now available from a new ``ch_types_all`` attribute, and non-EEG channel names are now available from a new ``ch_names_non_eeg`` attribute from :class:`PrepPipeline <pyprep.PrepPipeline>`, by `Yorguin Mantilla`_ (:gh:`34`)
- Renaming of ``ch_names`` attribute of :class:`PrepPipeline <pyprep.PrepPipeline>` to ``ch_names_all``, by `Yorguin Mantilla`_ (:gh:`34`)
- It's now possible to pass ``'eeg'`` to ``ref_chs`` and ``reref_chs`` keywords to the ``prep_params`` parameter of :class:`PrepPipeline <pyprep.PrepPipeline>` to select only eeg channels for referencing, by `Yorguin Mantilla`_ (:gh:`34`)
- :class:`PrepPipeline <pyprep.PrepPipeline>` will retain the non eeg channels through the ``raw`` attribute. The eeg-only and non-eeg parts will be in raw_eeg and raw_non_eeg respectively. See the ``raw`` attribute, by `Christian Oreilly`_ (:gh:`34`)
- When a ransac call needs more memory than available, pyprep will now automatically switch to a slower but less memory-consuming version of ransac, by `Yorguin Mantilla`_ (:gh:`32`)
- It's now possible to pass an empty list for the ``line_freqs`` param in :class:`PrepPipeline <pyprep.PrepPipeline>` to skip the line noise removal, by `Yorguin Mantilla`_ (:gh:`29`)
- The three main classes :class:`~pyprep.PrepPipeline`, :class:`~pyprep.NoisyChannels`, and :class:`pyprep.Reference` now have a ``random_state`` parameter to set a seed that gets passed on to all their internal methods and class calls, by `Stefan Appelhoff`_ (:gh:`31`)


Bug
~~~

- Corrected inconsistency of :class:`~pyprep.Reference` with the matlab version (:gh:`19`), by `Yorguin Mantilla`_ (:gh:`32`)
- Prevented an over detrending in :class:`~pyprep.Reference`, by `Yorguin Mantilla`_ (:gh:`32`)


API
~~~

- Remove ``noisy.py`` module from the ``pyprep`` package. Its main functionality has been migrated to the remaining modules, and the functions for FASTER have been dropped because they were out of scope, by `Stefan Appelhoff`_ (:gh:`39`)

.. _changes_0_3_0:

Version 0.3.0
-------------

Changelog
~~~~~~~~~

- Include a boolean ``do_detrend`` in :meth:`~pyprep.Reference.robust_reference` to indicate whether detrend should be done internally or not for the use with :class:`~pyprep.NoisyChannels`, by `Yorguin Mantilla`_ (:gh:`9`)
- Robust average referencing + tests, by  `Victor Xiang`_ (:gh:`6`)
- Removing trend in the EEG data by high pass filtering and local linear regression + tests, by `Aamna Lawrence`_ (:gh:`6`)
- Finding noisy channels with comparable output to Matlab + tests-including test for ransac, by `Aamna Lawrence`_ (:gh:`6`)
- Stringing all the things together for the PREP pipeline + tests, by `Victor Xiang`_ (:gh:`6`)
- Finding noisy channels with comparable output to Matlab + tests-including test for ransac, by `Aamna Lawrence`_ (:gh:`6`)
- Finding the appropriate parameters in the MNE notch filter for implementing clean line noise functionality of Eeglab, by `Aamna Lawrence`_ (:gh:`6`)
- Finding the reason for the difference between the Matlab and Pyprep’s output- Probably minor differences in the filter functions and also rounding done by functions like quantile, by `Victor Xiang`_  and `Aamna Lawrence`_ (:gh:`6`)

Bug
~~~

- Prevent an undoing of the detrending in :class:`~pyprep.NoisyChannels`, by `Yorguin Mantilla`_ (:gh:`9`)

API
~~~

- Oversaw modularization of PREP pipeline's submodules and a scikit learn style ``pyprep.fit``, by `Adam Li`_ (:gh:`6`)
- Oversaw ChangeLog by `Victor Xiang`_  and `Aamna Lawrence`_ for transitioning pyprep to Version 0.3.0, by `Adam Li`_ (:gh:`6`)

.. _changes_0_2_3:

Version 0.2.3
-------------

Doc
~~~

- update formatting and docs and fix tags and releases post-hoc, by `Stefan Appelhoff`_

.. _changes_0_2_2:

Version 0.2.2
-------------

Bug
~~~

- :class:`mne.Epochs` index start at 0, not 1, by `Stefan Appelhoff`_ (:gh:`commit/3780abb`)

.. _changes_0_2_1:

Version 0.2.1
-------------

Changelog
~~~~~~~~~
- Add ``find_bad_epochs`` based on the FASTER algorithm, by `Stefan Appelhoff`_ (:gh:`commit/0fa9c06`)

.. _Stefan Appelhoff: http://stefanappelhoff.com/
.. _Aamna Lawrence: https://github.com/AamnaLawrence
.. _Adam Li: https://github.com/adam2392/
.. _Christian O'Reilly: https://github.com/christian-oreilly
.. _Victor Xiang: https://github.com/Nick3151
.. _Yorguin Mantilla: https://github.com/yjmantilla
.. _Austin Hurst: https://github.com/a-hurst
