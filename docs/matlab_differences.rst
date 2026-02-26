:orphan:

.. _matlab-diffs:

Deliberate Differences from MATLAB PREP
=======================================

Although PyPREP aims to be a faithful reimplementation of the original MATLAB
version of PREP, there are a few places where PyPREP has deliberately chosen
to use different defaults than the MATLAB PREP.

To override these differerences, you can set the ``matlab_strict`` parameter
for :class:`~pyprep.PrepPipeline`, :class:`~pyprep.Reference`, or
:class:`~pyprep.NoisyChannels` to ``True`` in order to match the original
PREP's internal math.

.. contents:: Table of Contents
    :depth: 3


Differences in ``find_bad_by_nan_flat``
---------------------------------------

Unlike in MATLAB PREP, PyPREP allows editing the threshold value for a channel
to be considered 'bad-by-flat' by modifying the argument ``flat_threshold``
in the method :meth:`~pyprep.NoisyChannels.find_bad_by_nan_flat`.
However, the default value remains the same as in MATLAB PREP: ``1e-15`` volts
(that is, 1e-9 microvolts).

Differences in Signal Detrending
--------------------------------

In the PREP pipeline, trends (i.e., slow drifts in EEG baseline signal) are
temporarily removed from the data prior to adaptive line-noise removal
as well as prior to bad channel detection via :class:`~pyprep.NoisyChannels`,
which occurs at multiple points during robust re-referencing. This is done to
improve the accuracy of both of these processes, which are sensitive to
influence from trends in the signal.

In MATLAB PREP, the default method of trend removal is to use EEGLAB's
``pop_eegfiltnew``, which creates and applies an FIR high-pass filter to the
data. MNE's :func:`mne.filter.filter_data` offers similar functionality, but
with two key differences:

1) MNE's method of FIR filter design has minor differences, resulting in
   slightly lower FIR filter orders than EEGLAB for the same input
   values (e.g. 845 instead of EEGLAB's 847).
2) EEGLAB's ``pop_eegfiltnew`` only applies the filter to the signal forwards,
   resulting in minor phase shift in the filtered signal. By contrast, MNE
   defaults to applying the filter both forwards and backwards, eliminating any
   phase shift from the filtering process.

As a result of these differences, :class:`~pyprep.NoisyChannels` values also
differ slightly (on the order of ~0.002) for RANSAC correlations between the
filtering methods.

Because MNE's filtering code is faster and technically preferable (due to
the lack of phase shift) PyPREP defaults to using :func:`mne.filter.filter_data`
for high-pass trend removal. However, for exact numeric equivalence, PyPREP
has a basic re-implementation of EEGLAB's ``pop_eegfiltnew`` in Python that
produces identical results to MATLAB PREP's ``removeTrend`` when the
``matlab_strict`` parameter is set to ``True``.


Differences in RANSAC
---------------------

During the "find-bad-by-RANSAC" step of noisy channel detection (see
:func:`~pyprep.ransac.find_bad_by_ransac`), PREP does the following steps to
identify channels that aren't well-predicted by the signals of other channels:

1) Generates a bunch of random subsets of currently-good channels from the data
   (50 samples by default, each containing 25% of the total EEG channels in the
   dataset).

2) Uses the signals and spatial locations of those channels to predict what the
   signals will be at the spatial locations of all the other channels, with each
   random subset of channels generating a different prediction for each channel
   (i.e., 50 predicted signals per channel by default).

3) For each channel, calculates the median predicted signal from the full set of
   predictions.

4) Splits the full data into small non-overlapping windows (5 seconds by
   default) and calculates the correlation between the median predicted signal
   and the actual signal for each channel within each window.

5) Compares the correlations for each channel against a threshold value (0.75
   by default), flags all windows that fall below that threshold as 'bad', and
   calculates the proportions of 'bad' windows for each channel.

6) Flags all channels with an excessively high proportion of 'bad' windows
   (minimum 0.4 by default) as being 'bad-by-RANSAC'.

With that in mind, here are the areas where PyPREP's defaults deliberately
differ from the original PREP implementation:


Use of random seeds
^^^^^^^^^^^^^^^^^^^

In MATLAB PREP, the random seed used for RANSAC is always ``435656``, which is
set just before random channel sampling occurs. This means that every run of
RANSAC will result in identical random samples of channels given the same
input, and will produce similar random samples of channels if a channel or two
are removed between iterations.

Conversely, PyPREP defaults to setting an initial random state for the whole
pipeline, meaning that RANSAC's random channel picks will differ between
consecutive runs during robust re-referencing or bad channel detection. This
approach has the benefit of better randomness, but may also lead to more
variability in PREP results between different seed values. More testing is
required to determine which approach produces better results.

Note that to match MATLAB PREP exactly when the ``matlab_strict`` parameter is
set to ``True``, the random seed ``435656`` must be used.


Calculation of median estimated signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In MATLAB PREP, the median signal in step 3 is calculated by sorting the
different predictions for each EEG sample/channel from low to high and then
taking the value at the middle index for each. The relevant lines of MATLAB
PREP's ``findNoisyChannels.m`` are reproduced below:

.. code-block:: matlab

   function rX = calculateRansacWindow(XX, P, n, m, p)
       YY = sort(reshape(XX*P, n, m, p),3);
       YY = YY(:, :, round(end/2));
       rX = sum(XX.*YY)./(sqrt(sum(XX.^2)).*sqrt(sum(YY.^2)));

The first line of the function generates the full set of predicted signals for
each RANSAC sample, and then sorts the predicted values for each channel /
timepoint from low to high. The second line calculates the index of the middle
value (``round(end/2)``) and then uses it to take the middle slice of the
sorted array to get the median predicted signal for each channel.

Because this logic only returns the correct result for odd numbers of samples,
the current function will instead return the true median signal across
predictions unless strict MATLAB equivalence is requested.


Correlation of predicted vs. actual signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In MATLAB PREP, RANSAC channel predictions are correlated with actual data
in step 4 using a non-standard method: essentially, it uses the standard Pearson
correlation formula but without subtracting the channel means from each channel
before calculating sums of squares. This is done in the last line of the
``calculateRansacWindow`` function reproduced above:

.. code-block:: matlab

   rX = sum(XX.*YY)./(sqrt(sum(XX.^2)).*sqrt(sum(YY.^2)));

For readability, here's the same formula written in Python code::

   SSxx = np.sum(xx ** 2)
   SSyy = np.sum(yy ** 2)
   rX = np.sum(xx * yy) / (np.sqrt(SSxx) * np.sqrt(SSyy))

Because the EEG data will have already been filtered to remove slow drifts in
baseline before RANSAC, the signals correlated by this method will already be
roughly mean-centered. and will thus produce similar values to normal Pearson
correlation. However, to avoid making any assumptions about the signal for any
given channel / window, PyPREP defaults to normal Pearson correlation unless
strict MATLAB equivalence is requested.


Differences in Robust Referencing
---------------------------------

During the robust referencing part of the pipeline, PREP tries to estimate a
"clean" average reference signal for the dataset, excluding any channels
flagged as noisy from contaminating the reference. The robust referencing
process is performed using the following logic:

1) First, an initial pass of noisy channel detection is performed to identify
   channels bad by NaN values, flat signal, or low SNR: the data is then
   average-referenced excluding these channels. These channels are subsequently
   marked as "unusable" and are excluded from any future average referencing.

2) Noisy channel detection is performed on a copy of the re-referenced signal,
   and any newly detected bad channels are added to the full set of channels
   to be excluded from the reference signal.

3) After noisy channel detection, all bad channels detected so far are
   interpolated, and a new estimate of the robust average reference is
   calculated using the mean signal of all good channels and all interpolated
   bad channels (except those flagged as "unusable" during the first step).

4) A fresh copy of the re-referenced signal from Step 1 is re-referenced using
   the new reference signal calculated in Step 3.

5) Steps 2 through 4 are repeated until either two iterations have passed and
   no new noisy channels have been detected since the previous iteration, or
   the maximum number of reference iterations has been exceeded (default: 4).


Exclusion of dropout channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In MATLAB PREP, dropout channels (i.e., channels that have intermittent periods
of flat signal) are detected on each iteration of the reference loop, but are
currently not factored into the full set of "bad" channels to be interpolated.
By contrast, PyPREP will detect and interpolate any bad-by-dropout channels
detected during robust referencing.


Bad channel interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^

MATLAB PREP uses EEGLAB's internal ``eeg_interp`` method of spherical spline
interpolation for interpolating identified bad channels during robust reference
estimation and (if enabled) immediately after the robust reference signal is
applied in order to remove any remaining detected bad channels once referencing
is complete.

However, ``eeg_interp``'s method of spherical interpolations differs quite a bit
numerically from MNE's implementation as well as the interpolation method used
by MATLAB PREP for RANSAC predictions, both of which are numerically identical
and based directly on the formulas in Perrin et al. (1989) [1]_. ``eeg_interp``
seems to use a modified variation of the Perrin et al. method, but diverges in
a number of ways that are not clearly documented or cited in the code.

To keep with the more established method of spherical interpolation and stay
consistent with the interpolation code used in RANSAC, PyPREP defaults to using
MNE's :meth:`~mne.io.Raw.interpolate_bads` method for interpolation during and
following robust referencing. However, for full numeric equivalence with
MATLAB PREP, PyPREP will use a Python reimplementation of ``eeg_interp`` instead
when the ``matlab_strict`` parameter is set to ``True``.


PyPREP-Only Features
--------------------

The following features are available in PyPREP but are not present in the
original MATLAB PREP implementation.


Bad channel detection by PSD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`~pyprep.NoisyChannels.find_bad_by_PSD` method detects channels with
abnormally high or low power spectral density (PSD) compared to other channels.
This method is not part of the original MATLAB PREP pipeline, but can be
considered a refinement of the ``bad_by_hfnoise`` detection in MATLAB PREP,
which flags channels based on the ratio of high-frequency power (>50 Hz) to
total power.

A channel is considered "bad-by-PSD" if its total PSD (computed using Welch's
method over a configurable frequency range, defaulting to 1-45 Hz to exclude
line noise) deviates considerably from the median channel PSD. The deviation
is calculated using robust Z-scoring based on the median absolute deviation
(MAD).

This method is called by :meth:`~pyprep.NoisyChannels.find_all_bads` by default,
but is skipped when ``matlab_strict=True`` to maintain equivalence with the
original MATLAB PREP pipeline.


Annotation-Based Segment Rejection
----------------------------------

PyPREP supports the ``reject_by_annotation`` parameter in
:class:`~pyprep.PrepPipeline`, :class:`~pyprep.Reference`, and
:class:`~pyprep.NoisyChannels`, which allows excluding BAD-annotated time
segments from channel quality assessment. BAD segments are any MNE annotations
with descriptions starting with "BAD" or "bad" (see
:ref:`mne:tut-reject-data-spans` for details). This is useful when recordings
contain breaks, movement artifacts, or other periods that shouldn't influence
channel rejection decisions.

MATLAB PREP does not have this feature. In fact, MATLAB PREP explicitly warns
against using PREP on data with discontinuities (such as boundary markers from
paused/resumed recordings). However, the ``reject_by_annotation`` feature in
PyPREP is designed for a different use case: temporarily excluding known-bad
segments (e.g., participant movement during breaks) from *statistical analysis*
while preserving the original continuous data structure in the output.

When ``reject_by_annotation`` is set to ``'omit'``, MNE's
:meth:`~mne.io.Raw.get_data` is used to concatenate non-BAD segments for
computing channel quality metrics. The final processed output retains the
original continuous structure with all time points intact.

.. note::

   This feature is intended for excluding a small number of longer segments
   (e.g., recording breaks). Using it with many short BAD segments (e.g., from
   automated muscle artifact detection via
   :func:`mne.preprocessing.annotate_muscle_zscore`) may introduce edge effects
   at concatenation boundaries, particularly for methods that apply filtering
   to the concatenated data. PyPREP will emit a warning if many small BAD
   segments are detected.

This parameter has no equivalent in MATLAB PREP. When ``matlab_strict`` is set
to ``True``, ``reject_by_annotation`` is automatically set to ``None``.


References
----------

.. [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
   Spherical splines for scalp potential and current density mapping.
   Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
