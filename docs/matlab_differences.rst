Deliberate Differences from the Original PREP
=============================================

Although PyPREP aims to be a faithful reimplementaion of the original MATLAB
version of PREP, there are a few places where PyPREP has deliberately chosen
to use different defaults than the MATLAB PREP.

To override these differerences, you can set the ``matlab_strict`` argument to
:class:`pyprep.prep_pipeline.PrepPipeline`, :class:`pyprep.reference.Reference`,
or :class:`pyprep.find_noisy_channels.NoisyChannels` as ``True`` to match the
original PREP's internal math.

Differences in RANSAC
=====================

During the "find-bad-by-RANSAC" step of noisy channel detection, PREP does the
follwing steps to identify channels that aren't well-predicted by the signals
of other channels:

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

Calculation of median estimated signal
--------------------------------------

In MATLAB PREP, the median signal in step 3 is calculated by sorting the
different predictions for each EEG sample/channel from low to high and then
taking the value at the middle index (as calculated by
``int(n_ransac_samples / 2.0)``) for each.

Because this logic only returns the correct result for odd numbers of samples,
the current function will instead return the true median signal across
predictions unless strict MATLAB equivalence is requested.

Calculation of predicted vs. actual correlations in RANSAC
----------------------------------------------------------

In MATLAB PREP, RANSAC channel predictions are correlated with actual data
in step 4 using a non-standard method: essentialy, it uses the standard Pearson
correlation formula but without subtracting the channel means from each channel
before calculating sums of squares, i.e.,::

   SSa = np.sum(a ** 2)
   SSb = np.sum(b ** 2)
   correlation = np.sum(a * b) / (np.sqrt(SSa) * np.sqrt(SSb))

Because EEG data is roughly mean-centered to begin with, this produces similar
values to normal Pearson correlation. However, to avoid making any assumptions
about the signal for any given channel/window, PyPREP defaults to normal
Pearson correlation unless strict MATLAB equivalence is requested.
