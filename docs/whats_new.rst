:orphan:

Authors
=======

People who contributed to this software across releases (**in alphabetical order**)

* `Aamna Lawrence`_
* `Adam Li`_
* `Christian Oreilly`_
* `Stefan Appelhoff`_
* `Victor Xiang`_
* `Yorguin Mantilla`_
* `Austin Hurst`_

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
- Added two attributes :attr:`PrepPipeline.noisy_channels_before_interpolation` and :attr:`PrepPipeline.noisy_channels_after_interpolation` which have the detailed output of each noisy criteria, by `Yorguin Mantilla`_ (:gh:`45`)
- Added two keys to the :attr:`PrepPipeline.noisy_channels_original` dictionary: ``bad_by_dropout`` and ``bad_by_SNR``, by `Yorguin Mantilla`_ (:gh:`45`)
- Changed RANSAC chunking logic to reduce max memory use and prefer equal chunk sizes where possible, by `Austin Hurst`_ (:gh:`44`)

Bug
~~~
- Fixed RANSAC to give consistent results with a fixed seed across different chunk sizes, by `Austin Hurst`_ and `Yorguin Mantilla`_ (:gh:`43`)

API
~~~
- The permissible parameters for the following methods were removed and/or reordered: :meth:`NoisyChannels.ransac_correlations`, :meth:`NoisyChannels.run_ransac`, and :meth:`NoisyChannels.get_ransac_pred` methods, by `Austin Hurst`_ and `Yorguin Mantilla`_ (:gh:`43`)

.. _changes_0_3_1:

Version 0.3.1
-------------

Changelog
~~~~~~~~~
- It's now possible to pass keyword arguments to the notch filter inside prep; see the ``filter_kwargs`` parameter by `Yorguin Mantilla`_ (:gh:`40`)
- The default filter length for the spectrum_fit method will be '10s' to fix memory issues, by `Yorguin Mantilla`_ (:gh:`40`)
- Channel types  are now available from a new ``ch_types_all`` attribute, and non-EEG channel names are now available from a new ``ch_names_non_eeg`` attribute from :class:`pyprep.PrepPipeline`, by `Yorguin Mantilla`_ (:gh:`34`)
- Renaming of ``ch_names`` attribute of ``PrepPipeline`` to ``ch_names_all``, by `Yorguin Mantilla`_ (:gh:`34`)
- It's now possible to pass ``'eeg'`` to ``ref_chs`` and ``reref_chs`` parameters to select only eeg channels for referencing, by `Yorguin Mantilla`_ (:gh:`34`)
- Prep will retain the non eeg channels through the ``raw`` attribute. The eeg-only and non-eeg parts will be in raw_eeg and raw_non_eeg respectively. See the ``raw`` attribute, by `Christian Oreilly`_ (:gh:`34`)
- When a ransac call needs more memory than available, pyprep will now automatically switch to a slower but less memory-consuming version of ransac. See the ``channel_wise`` parameter, by `Yorguin Mantilla`_ (:gh:`32`)
- It's now possible to pass an empty list for the ``line_freqs`` param in ``PrepPipeline`` to skip the line noise removal, by `Yorguin Mantilla`_ (:gh:`29`)
- The three main classes ``PrepPipeline``, ``NoisyChannels``, and ``Reference`` now have a ``random_state`` parameter to set a seed that gets passed on to all their internal methods and class calls, by `Stefan Appelhoff`_ (:gh:`31`)


Bug
~~~

- Corrected inconsistentcy of :mod:`reference` with the matlab version (:gh:`19`), by `Yorguin Mantilla`_ (:gh:`32`)
- Prevented an over detrending in :mod:`reference`, by `Yorguin Mantilla`_ (:gh:`32`)


API
~~~

- Remove ``noisy.py`` module from the ``pyprep`` package. Its main functionality has been migrated to the remaining modules, and the functions for FASTER have been dropped because they were out of scope, by `Stefan Appelhoff`_ (:gh:`39`)

.. _changes_0_3_0:

Version 0.3.0
-------------

Changelog
~~~~~~~~~

- Include a boolean ``do_detrend`` in :meth:`reference.robust_reference` to indicate whether detrend should be done internally or not for the use with :mod:`find_noisy_channels`, by `Yorguin Mantilla`_ (:gh:`9`)
- Robust average referencing + tests, by  `Victor Xiang`_ (:gh:`6`)
- Removing trend in the EEG data by high pass filtering and local linear regression + tests, by `Aamna Lawrence`_ (:gh:`6`)
- Finding noisy channels with comparable output to Matlab +tests-including test for ransac, by `Aamna Lawrence`_ (:gh:`6`)
- Stringing all the things together for the PREP pipeline +tests, by `Victor Xiang`_ (:gh:`6`)
- Finding noisy channels with comparable output to Matlab +tests-including test for ransac, by `Aamna Lawrence`_ (:gh:`6`)
- Finding the appropriate parameters in the MNE notch filter for implementing clean line noise functionality of Eeglab, by `Aamna Lawrence`_ (:gh:`6`)
- Finding the reason for the difference between the Matlab and Pyprep’s output- Probably minor differences in the filter functions and also rounding done by functions like quantile, by `Victor Xiang`_  and `Aamna Lawrence`_ (:gh:`6`)

Bug
~~~

- Prevent an undoing of the detrending in :mod:`find_noisy_channels`, by `Yorguin Mantilla`_ (:gh:`9`)

API
~~~

- Oversaw modularization of PREP pipeline's submodules and a scikit learn style :func:`pyprep.fit`, by `Adam Li`_ (:gh:`6`)
- Oversaw ChangeLog by `Victor Xiang`_  and `Aamna Lawrence`_ for transitioning pyPrep to Version 0.3.0, by `Adam Li`_ (:gh:`6`)

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

- mne epochs index start at 0, not 1, by `Stefan Appelhoff`_ (`3780abb <https://github.com/sappelhoff/pyprep/commit/3780abb922ebb790c74d3e871e2958d87d8c7e23>`_)

.. _changes_0_2_1:

Version 0.2.1
-------------

Changelog
~~~~~~~~~

- Add :func:`find_bad_epochs` based on the FASTER algorithm, by `Stefan Appelhoff`_ (`0fa9c06 <https://github.com/sappelhoff/pyprep/commit/0fa9c065481c4cbaaf83b275f92b16b8807810b5>`_)

.. _Stefan Appelhoff: http://stefanappelhoff.com/
.. _Aamna Lawrence: https://github.com/AamnaLawrence
.. _Adam Li: https://github.com/adam2392/
.. _Christian Oreilly: https://github.com/christian-oreilly
.. _Victor Xiang: https://github.com/Nick3151
.. _Yorguin Mantilla: https://github.com/yjmantilla
.. _Austin Hurst: https://github.com/a-hurst
