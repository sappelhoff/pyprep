:orphan:

.. _whats_new:

What's New
==========

Here we list a changelog of pyprep.

.. contents:: Contents
   :local:
   :depth: 1

.. currentmodule:: pyprep

Authors
-------
People who contributed to this software across releases (in alphabetical order)

* `Aamna Lawrence`_
* `Adam Li`_
* `Stefan Appelhoff`_
* `Victor Xiang`_
* `Yorguin Mantilla`_

.. _current:

Current
-------

Changelog
~~~~~~~~~
- When a ransac call needs more memory than available, pyprep will now automatically switch to a slower but less memory-consuming version of ransac. See the ``channel_wise``parameter, by `Yorguin Mantilla`_ (`#32 <https://github.com/sappelhoff/pyprep/pull/32>`_)
- It's now possible to pass an empty list for the ``line_freqs`` param in ``PrepPipeline`` to skip the line noise removal, by `Yorguin Mantilla`_ (`#29 <https://github.com/sappelhoff/pyprep/pull/29>`_)
- The three main classes ``PrepPipeline``, ``NoisyChannels``, and ``Reference`` now have a ``random_state`` parameter to set a seed that gets passed on to all their internal methods and class calls, by `Stefan Appelhoff`_ (`#31 <https://github.com/sappelhoff/pyprep/pull/31>`_)


Bug
~~~

- nothing yet

API
~~~

- nothing yet

.. _changes_0_3_0:

Version 0.3.0
-------------

Changelog
~~~~~~~~~

- Include a boolean ``do_detrend`` in :meth:`reference.robust_reference` to indicate whether detrend should be done internally or not for the use with :mod:`find_noisy_channels`, by `Yorguin Mantilla`_ (`#9 <https://github.com/sappelhoff/pyprep/pull/9>`_)
- Robust average referencing + tests, by  `Victor Xiang`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Removing trend in the EEG data by high pass filtering and local linear regression + tests, by `Aamna Lawrence`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Finding noisy channels with comparable output to Matlab +tests-including test for ransac, by `Aamna Lawrence`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Stringing all the things together for the PREP pipeline +tests, by `Victor Xiang`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Finding noisy channels with comparable output to Matlab +tests-including test for ransac, by `Aamna Lawrence`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Finding the appropriate parameters in the MNE notch filter for implementing clean line noise functionality of Eeglab, by `Aamna Lawrence`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Finding the reason for the difference between the Matlab and Pyprep’s output- Probably minor differences in the filter functions and also rounding done by functions like quantile, by `Victor Xiang`_  and `Aamna Lawrence`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)

Bug
~~~

- Prevent an undoing of the detrending in :mod:`find_noisy_channels`, by `Yorguin Mantilla`_ (`#9 <https://github.com/sappelhoff/pyprep/pull/9>`_)

API
~~~

- Oversaw modularization of PREP pipeline's submodules and a scikit learn style :func:`pyprep.fit`, by `Adam Li`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)
- Oversaw ChangeLog by `Victor Xiang`_  and `Aamna Lawrence`_ for transitioning pyPrep to Version 0.3.0, by `Adam Li`_ (`#6 <https://github.com/sappelhoff/pyprep/pull/6>`_)

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
.. _Victor Xiang: https://github.com/Nick3151
.. _Yorguin Mantilla: https://github.com/yjmantilla
