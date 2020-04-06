What's New
==========

.. currentmodule:: pyprep

.. _current:

Current
-------

Bug
~~~

- In find_noisy_channels the signal detrend was accidentally undone which destabilized the RANSAC.
- The detrend is now done over the NoisyChannels.raw_mne object to avoid this and to force that any signal in there is detrended like in matlab's prep.
- Included a boolean which indicates if detrend should be done internally or not for the use of find_noisy_channels in reference.py


Changelog
~~~~~~~~~

- Robust average referencing + tests by  `Victor Xiang`_
- Removing trend in the EEG data by high pass filtering and local linear regression + tests `Aamna Lawrence`_
- Finding noisy channels with comparable output to Matlab +tests-including test for ransac `Aamna Lawrence`_
- Stringing all the things together for the PREP pipeline +tests `Victor Xiang`_
- Finding noisy channels with comparable output to Matlab +tests-including test for ransac `Aamna Lawrence`_
- Finding the appropriate parameters in the MNE notch filter for implementing clean line noise functionality of Eeglab `Aamna Lawrence`_
- Finding the reason for the difference between the Matlab and Pyprep’s output- Probably minor differences in the filter functions and also rounding done by functions like quantile `Victor Xiang`_  and `Aamna Lawrence`_
- Overall debugging `Victor Xiang`_  and `Aamna Lawrence`_

API
~~~

- Oversaw modularization of PREP pipeline's submodules and a scikit learn style :func:`pyprep.fit` by `Adam Li`_
- Oversaw ChangeLog by `Victor Xiang`_  and `Aamna Lawrence`_ for transitioning pyPrep to Version 0.3.0 by `Adam Li`_

.. _changes_0_2_3:

Version 0.2.3
-------------

Doc
~~~

- update formatting and docs and fix tags and releases post-hoc by `Stefan Appelhoff`_

.. _changes_0_2_2:

Version 0.2.2
-------------

Bug
~~~

- mne epochs index start at 0, not 1 by `Stefan Appelhoff`_ (`3780abb <https://github.com/sappelhoff/pyprep/commit/3780abb922ebb790c74d3e871e2958d87d8c7e23>`_)

.. _changes_0_2_1:

Version 0.2.1
-------------

Changelog
~~~~~~~~~

- Add :func:`find_bad_epochs` based on the FASTER algorithm by `Stefan Appelhoff`_ (`0fa9c06 <https://github.com/sappelhoff/pyprep/commit/0fa9c065481c4cbaaf83b275f92b16b8807810b5>`_)


Authors
-------
People who contributed to this software (in alphabetical order)

* Stefan Appelhoff
* Aamna Lawrence
* Adam Li
* Victor Xiang
* Yorguin Mantilla

.. _Stefan Appelhoff: http://stefanappelhoff.com/
.. _Aamna Lawrence: https://github.com/AamnaLawrence
.. _Adam Li: https://github.com/adam2392/
.. _Victor Xiang: https://github.com/Nick3151
.. _Yorguin Mantilla: https://github.com/yjmantilla