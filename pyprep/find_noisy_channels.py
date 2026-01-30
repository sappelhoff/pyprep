"""Find bad channels."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import mne
import numpy as np
from mne.utils import check_random_state, logger
from scipy import signal
from scipy.stats import median_abs_deviation

from pyprep.ransac import find_bad_by_ransac
from pyprep.removeTrend import removeTrend
from pyprep.utils import _filter_design, _mat_iqr, _mat_quantile


class NoisyChannels:
    """Detect bad channels in an EEG recording using a range of methods.

    This class provides a number of methods for detecting bad channels across a
    full-session EEG recording. Specifically, this class implements all of the
    noisy channel detection methods used in the PREP pipeline, as described in [1]_.
    The detection methods in this class can be run independently, or can be run
    all at once using the :meth:`~.find_all_bads` method.

    At present, only EEG channels are supported and any non-EEG channels in the
    provided data will be ignored.

    Parameters
    ----------
    raw : mne.io.Raw
        An MNE Raw object to check for bad EEG channels. Channels set to bad
        in ``raw.info["bads"]`` will not be used to find additional bad channels.
    do_detrend : bool
        Whether or not low-frequency (<1.0 Hz) trends should be removed from the
        EEG signal prior to bad channel detection. This should always be set to
        ``True`` unless the signal has already had low-frequency trends removed.
        Defaults to ``True``.
    random_state : {int, None, np.random.RandomState} | None
        The seed to use for random number generation within RANSAC. This can be
        ``None``, an integer, or a :class:`~numpy.random.RandomState` object.
        If ``None``, a random seed will be obtained from the operating system.
        Defaults to ``None``.
    matlab_strict : bool
        Whether or not PyPREP should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code
        (see :ref:`matlab-diffs` for more details). Defaults to ``False``.
    ransac : bool
        Whether RANSAC should be used for bad channel detection, in addition
        to other methods. RANSAC can detect bad channels that other
        methods are unable to catch, but also slows down noisy channel
        detection considerably. Defaults to ``True``.
    correlation : bool
        Whether correlation should be used for bad channel detection, in addition
        to other methods. Defaults to ``True``.
    bad_by_manual : list of str | None
        List of channels that are bad. These channels will be excluded when
        trying to find additional bad channels. Note that the union of these channels
        and those declared in ``raw.info["bads"]`` will be used. Defaults to ``None``.
    reject_by_annotation : {None, 'omit'} | None
        How to handle BAD-annotated time segments (annotations starting with
        "BAD" or "bad") during channel quality assessment. If ``'omit'``,
        annotated segments are excluded from analysis (clean segments are
        concatenated). If ``None`` (default), annotations are ignored and the
        full recording is used. This is useful when recordings contain breaks
        or movement artifacts that shouldn't influence channel rejection
        decisions.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       EEG analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(
        self,
        raw,
        do_detrend=True,
        random_state=None,
        matlab_strict=False,
        *,
        ransac=True,
        correlation=True,
        bad_by_manual=None,
        reject_by_annotation=None,
    ):
        # Make sure that we got an MNE object
        assert isinstance(raw, mne.io.BaseRaw)

        raw.load_data()
        self.raw_mne = raw.copy()
        bad_by_manual = bad_by_manual if bad_by_manual else []
        self.bad_by_manual = list(set(bad_by_manual + raw.info["bads"]))
        self.raw_mne.pick("eeg")  # excludes bads
        self.sample_rate = raw.info["sfreq"]
        if do_detrend:
            self.raw_mne._data = removeTrend(
                self.raw_mne.get_data(), self.sample_rate, matlab_strict=matlab_strict
            )
        self.matlab_strict = matlab_strict

        msg = f"ransac must be boolean, got: {ransac}"
        assert isinstance(ransac, bool), msg
        self.ransac = ransac

        msg = f"correlation must be boolean, got: {correlation}"
        assert isinstance(correlation, bool), msg
        self.correlation = correlation

        # Validate reject_by_annotation parameter
        if reject_by_annotation is not None and reject_by_annotation != "omit":
            raise ValueError(
                f"reject_by_annotation must be None or 'omit', "
                f"got: {reject_by_annotation}"
            )
        # reject_by_annotation is not available in MATLAB PREP
        if matlab_strict and reject_by_annotation is not None:
            logger.warning(
                "reject_by_annotation is not available in MATLAB PREP. "
                f"Setting reject_by_annotation to None (was '{reject_by_annotation}')."
            )
            reject_by_annotation = None
        self.reject_by_annotation = reject_by_annotation

        # Warn if many small BAD segments are present (potential edge effects)
        if reject_by_annotation is not None:
            bad_annots = [
                a
                for a in raw.annotations
                if a["description"].startswith(("BAD", "bad"))
            ]
            n_bad_segments = len(bad_annots)
            if n_bad_segments > 0:
                total_bad_time = sum(a["duration"] for a in bad_annots)
                recording_length = raw.times[-1]
                bad_percentage = (total_bad_time / recording_length) * 100
                mean_duration = total_bad_time / n_bad_segments
                if bad_percentage > 15 and mean_duration < 5.0:
                    logger.warning(
                        f"Found {n_bad_segments} BAD segments covering "
                        f"{bad_percentage:.1f}% of the recording with mean duration "
                        f"{mean_duration:.1f}s. Using reject_by_annotation with many "
                        "short segments may introduce edge effects from concatenation. "
                        "This feature is intended for excluding a small number of "
                        "longer segments (e.g., recording breaks)."
                    )

        # Extra data for debugging
        self._extra_info = {
            "bad_by_deviation": {},
            "bad_by_hf_noise": {},
            "bad_by_correlation": {},
            "bad_by_dropout": {},
            "bad_by_psd": {},
            "bad_by_ransac": {},
        }

        # random_state
        self.random_state = check_random_state(random_state)

        # The identified bad channels
        self.bad_by_nan = []
        self.bad_by_flat = []
        self.bad_by_deviation = []
        self.bad_by_hf_noise = []
        self.bad_by_correlation = []
        self.bad_by_SNR = []
        self.bad_by_dropout = []
        self.bad_by_psd = []
        self.bad_by_ransac = []

        # Get original EEG channel names, channel count & samples
        ch_names = np.asarray(self.raw_mne.info["ch_names"])
        self.ch_names_original = ch_names
        self.n_chans_original = len(ch_names)
        self.n_samples_original = raw.get_data().shape[1]

        # Before anything else, flag bad-by-NaNs and bad-by-flats
        self.find_bad_by_nan_flat()
        bads_by_nan_flat = self.bad_by_nan + self.bad_by_flat

        # unusable channels are also those manually marked as bad
        bads_unusable = self.bad_by_manual + bads_by_nan_flat

        # Make a subset of the data containing only usable EEG channels
        self.usable_idx = np.isin(ch_names, bads_unusable, invert=True)
        self.EEGData = self.raw_mne.get_data(
            picks=ch_names[self.usable_idx],
            reject_by_annotation=self.reject_by_annotation,
        )
        self.n_samples = self.EEGData.shape[1]
        self.EEGFiltered = None

        # Get usable EEG channel names & channel counts
        self.ch_names_new = np.asarray(ch_names[self.usable_idx])
        self.n_chans_new = len(self.ch_names_new)

    def _get_filtered_data(self):
        """Apply a [1 Hz - 50 Hz] bandpass filter to the EEG signal.

        Only applied if the sample rate is above 100 Hz to avoid violating the
        Nyquist theorem.

        """
        if self.sample_rate <= 100:
            return self.EEGData.copy()

        bandpass_filter = _filter_design(
            N_order=100,
            amp=np.array([1, 1, 0, 0]),
            freq=np.array([0, 90 / self.sample_rate, 100 / self.sample_rate, 1]),
        )
        EEG_filt = np.zeros_like(self.EEGData)
        for i in range(EEG_filt.shape[0]):
            EEG_filt[i, :] = signal.filtfilt(bandpass_filter, 1, self.EEGData[i, :])

        return EEG_filt

    def get_bads(self, verbose=False, as_dict=False):
        """Get the names of all channels currently flagged as bad.

        Note that this method does not perform any bad channel detection itself,
        and only reports channels already detected as bad by other methods.

        Parameters
        ----------
        verbose : bool | None
            If ``True``, a summary of the channels currently flagged as by bad per
            category is printed. Defaults to ``False``.
        as_dict: bool | None
            If ``True``, this method will return a dict of the channels currently
            flagged as bad by each individual bad channel type. If ``False``, this
            method will return a list of all unique bad channels detected so far.
            Defaults to ``False``.

        Returns
        -------
        bads : list or dict
            The names of all bad channels detected so far, either as a combined
            list or a dict indicating the channels flagged bad by each type.

        """
        bads = {
            "bad_by_nan": self.bad_by_nan,
            "bad_by_flat": self.bad_by_flat,
            "bad_by_deviation": self.bad_by_deviation,
            "bad_by_hf_noise": self.bad_by_hf_noise,
            "bad_by_correlation": self.bad_by_correlation,
            "bad_by_SNR": self.bad_by_SNR,
            "bad_by_dropout": self.bad_by_dropout,
            "bad_by_psd": self.bad_by_psd,
            "bad_by_ransac": self.bad_by_ransac,
            "bad_by_manual": self.bad_by_manual,
        }

        all_bads = set()
        for bad_chs in bads.values():
            all_bads.update(bad_chs)

        name_map = {
            "nan": "NaN",
            "hf_noise": "HF noise",
            "psd": "PSD",
            "ransac": "RANSAC",
        }
        if verbose:
            out = f"Found {len(all_bads)} uniquely bad channels:\n"
            for bad_type, bad_chs in bads.items():
                bad_type = bad_type.replace("bad_by_", "")
                if bad_type in name_map.keys():
                    bad_type = name_map[bad_type]
                out += f"\n{len(bad_chs)} by {bad_type}: {bad_chs}\n"
            logger.info(out)

        if as_dict:
            bads["bad_all"] = list(all_bads)
        else:
            bads = list(all_bads)

        return bads

    def find_all_bads(
        self,
        *,
        ransac=None,
        channel_wise=False,
        max_chunk_size=None,
        correlation=None,
        reject_by_annotation=None,
    ):
        """Call all the functions to detect bad channels.

        This function calls all the bad-channel detecting functions.

        Parameters
        ----------
        ransac : bool | None
            Whether RANSAC should be used for bad channel detection, in addition
            to the other methods. RANSAC can detect bad channels that other
            methods are unable to catch, but also slows down noisy channel
            detection considerably. If ``None`` (default), then the value at
            instantiation of the ``NoisyChannels`` class is taken (defaults
            to ``True``), else the instantiation value is overwritten.
        channel_wise : bool | None
            Whether RANSAC should predict signals for chunks of channels over the
            entire signal length ("channel-wise RANSAC", see `max_chunk_size`
            parameter). If ``False``, RANSAC will instead predict signals for all
            channels at once but over a number of smaller time windows instead of
            over the entire signal length ("window-wise RANSAC"). Channel-wise
            RANSAC generally has higher RAM demands than window-wise RANSAC
            (especially if `max_chunk_size` is ``None``), but can be faster on
            systems with lots of RAM to spare. Has no effect if not using RANSAC.
            Defaults to ``False``.
        max_chunk_size : {int, None} | None
            The maximum number of channels to predict at once during
            channel-wise RANSAC. If ``None``, RANSAC will use the largest chunk
            size that will fit into the available RAM, which may slow down
            other programs on the host system. If using window-wise RANSAC
            (the default) or not using RANSAC at all, this parameter has no
            effect. Defaults to ``None``.
        correlation : bool | None
            Whether correlation should be used for bad channel detection, in addition
            to the other methods. If ``None`` (default), then the value at
            instantiation of the ``NoisyChannels`` class is taken (defaults
            to ``True``), else the instantiation value is overwritten.
        reject_by_annotation : {None, 'omit'} | None
            This parameter is accepted for compatibility but is ignored here.
            Annotation rejection is applied during ``NoisyChannels`` initialization,
            not during ``find_all_bads``. To use annotation rejection, pass
            ``reject_by_annotation`` to the ``NoisyChannels`` constructor.

        """
        # Note: reject_by_annotation is accepted but ignored here - it's applied
        # during __init__ when data is extracted. This parameter exists only for
        # compatibility with ransac_settings dict unpacking.
        del reject_by_annotation  # unused, applied in __init__
        if ransac is not None and ransac != self.ransac:
            msg = f"ransac must be boolean, got: {ransac}"
            assert isinstance(ransac, bool), msg
            logger.warning(
                "Overwriting `ransac` value. "
                f"Was `{self.ransac}` at instantiation "
                f"of NoisyChannels. Now setting to `{ransac}`."
            )
            self.ransac = ransac

        if correlation is not None and correlation != self.correlation:
            msg = f"correlation must be boolean, got: {correlation}"
            assert isinstance(correlation, bool), msg
            logger.warning(
                "Overwriting `correlation` value. "
                f"Was `{self.correlation}` at instantiation "
                f"of NoisyChannels. Now setting to `{correlation}`."
            )
            self.correlation = correlation

        # NOTE: Bad-by-NaN/flat is already run during init, no need to re-run here
        self.find_bad_by_deviation()
        self.find_bad_by_hfnoise()
        if self.correlation:
            self.find_bad_by_correlation()
        self.find_bad_by_SNR()
        if not self.matlab_strict:
            self.find_bad_by_PSD()
        if self.ransac:
            self.find_bad_by_ransac(
                channel_wise=channel_wise, max_chunk_size=max_chunk_size
            )

    def find_bad_by_nan_flat(self, flat_threshold=1e-15):
        """Detect channels than contain NaN values or have near-flat signals.

        A channel is considered flat if its standard deviation or its median
        absolute deviation from the median (MAD) are below the provided flat
        threshold (default: ``1e-15`` volts).

        This method is run automatically when a ``NoisyChannels`` object is
        initialized, preventing flat or NaN-containing channels from interfering
        with the detection of other types of bad channels.

        Parameters
        ----------
        flat_threshold : float | None
            The lowest standard deviation or MAD value for a channel to be
            considered bad-by-flat. Defaults to ``1e-15`` volts (corresponds to
            10e-10 µV in MATLAB PREP).
        """
        # Get all EEG channels from original copy of data
        EEGData = self.raw_mne.get_data()

        # Detect channels containing any NaN values
        nan_channel_mask = np.isnan(np.sum(EEGData, axis=1))
        nan_channels = self.ch_names_original[nan_channel_mask]

        # Detect channels with flat or extremely weak signals
        flat_by_mad = median_abs_deviation(EEGData, axis=1) < flat_threshold
        flat_by_stdev = np.std(EEGData, axis=1) < flat_threshold
        flat_channel_mask = flat_by_mad | flat_by_stdev
        flat_channels = self.ch_names_original[flat_channel_mask]

        # Update names of bad channels by NaN or flat signal
        self.bad_by_nan = nan_channels.tolist()
        self.bad_by_flat = flat_channels.tolist()

    def find_bad_by_deviation(self, deviation_threshold=5.0):
        """Detect channels with abnormally high or low overall amplitudes.

        A channel is considered "bad-by-deviation" if its amplitude deviates
        considerably from the median channel amplitude, as calculated using a
        robust Z-scoring method and the given deviation threshold.

        Amplitude Z-scores are calculated using the formula
        ``(channel_amplitude - median_amplitude) / amplitude_sd``, where
        channel amplitudes are calculated using a robust outlier-resistant estimate
        of the signals' standard deviations (IQR scaled to units of SD), and the
        amplitude SD is the IQR-based SD of those amplitudes.

        Parameters
        ----------
        deviation_threshold : float | None
            The minimum absolute z-score of a channel for it to be considered
            bad-by-deviation. Defaults to ``5.0``.

        """
        IQR_TO_SD = 0.7413  # Scales units of IQR to units of SD, assuming normality
        # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/IQR.html

        # Get channel amplitudes and the median / robust SD of those amplitudes
        chan_amplitudes = _mat_iqr(self.EEGData, axis=1) * IQR_TO_SD
        amp_sd = _mat_iqr(chan_amplitudes) * IQR_TO_SD
        amp_median = np.nanmedian(chan_amplitudes)

        # Calculate robust Z-scores for the channel amplitudes
        amplitude_zscore = np.zeros(self.n_chans_original)
        amplitude_zscore[self.usable_idx] = (chan_amplitudes - amp_median) / amp_sd

        # Flag channels with amplitudes that deviate excessively from the median
        abnormal_amplitude = np.abs(amplitude_zscore) > deviation_threshold
        deviation_channel_mask = np.isnan(amplitude_zscore) | abnormal_amplitude

        # Update names of bad channels by excessive deviation & save additional info
        deviation_channels = self.ch_names_original[deviation_channel_mask]
        self.bad_by_deviation = deviation_channels.tolist()
        self._extra_info["bad_by_deviation"].update(
            {
                "median_channel_amplitude": amp_median,
                "channel_amplitude_sd": amp_sd,
                "robust_channel_deviations": amplitude_zscore,
            }
        )

    def find_bad_by_hfnoise(self, HF_zscore_threshold=5.0):
        """Detect channels with abnormally high amounts of high-frequency noise.

        The noisiness of a channel is defined as the amplitude of its
        high-frequency (>50 Hz) components divided by its overall amplitude.
        A channel is considered "bad-by-high-frequency-noise" if its noisiness
        is considerably higher than the median channel noisiness, as determined
        by a robust Z-scoring method and the given Z-score threshold.

        Due to the Nyquist theorem, this method will only attempt bad channel
        detection if the sample rate of the given signal is above 100 Hz.

        Parameters
        ----------
        HF_zscore_threshold : float | None
            The minimum noisiness z-score of a channel for it to be considered
            bad-by-high-frequency-noise. Defaults to ``5.0``.

        """
        MAD_TO_SD = 1.4826  # Scales units of MAD to units of SD, assuming normality
        # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mad.html

        if self.EEGFiltered is None:
            self.EEGFiltered = self._get_filtered_data()

        # Set default values for noise parameters
        noise_median, noise_sd = (0, 1)
        noise_zscore = np.zeros(self.n_chans_original)

        # If sample rate is high enough, calculate ratio of > 50 Hz amplitude to
        # < 50 Hz amplitude for each channel and get robust z-scores of values
        if self.sample_rate > 100:
            noisiness = np.divide(
                median_abs_deviation(self.EEGData - self.EEGFiltered, axis=1),
                median_abs_deviation(self.EEGFiltered, axis=1),
            )
            noise_median = np.nanmedian(noisiness)
            noise_sd = np.median(np.abs(noisiness - noise_median)) * MAD_TO_SD
            noise_zscore[self.usable_idx] = (noisiness - noise_median) / noise_sd

        # Flag channels with much more high-frequency noise than the median channel
        hf_mask = np.isnan(noise_zscore) | (noise_zscore > HF_zscore_threshold)
        hf_noise_channels = self.ch_names_original[hf_mask]

        # Update names of high-frequency noise channels & save additional info
        self.bad_by_hf_noise = hf_noise_channels.tolist()
        self._extra_info["bad_by_hf_noise"].update(
            {
                "median_channel_noisiness": noise_median,
                "channel_noisiness_sd": noise_sd,
                "hf_noise_zscores": noise_zscore,
            }
        )

    def find_bad_by_correlation(
        self, correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01
    ):
        """Detect channels that sometimes don't correlate with any other channels.

        Channel correlations are calculated by splitting the recording into
        non-overlapping windows of time (default: 1 second), getting the absolute
        correlations of each usable channel with every other usable channel for
        each window, and then finding the highest correlation each channel has
        with another channel for each window (by taking the 98th percentile of
        the absolute correlations).

        A correlation window is considered "bad" for a channel if its maximum
        correlation with another channel is below the provided correlation
        threshold (default: ``0.4``). A channel is considered "bad-by-correlation"
        if its fraction of bad correlation windows is above the bad fraction
        threshold (default: ``0.01``).

        This method also detects channels with intermittent dropouts (i.e.,
        regions of flat signal). A channel is considered "bad-by-dropout" if
        its fraction of correlation windows with a completely flat signal is
        above the bad fraction threshold (default: ``0.01``).

        Parameters
        ----------
        correlation_secs : float | None
            The length (in seconds) of each correlation window. Defaults to ``1.0``.
        correlation_threshold : float | None
            The lowest maximum inter-channel correlation for a channel to be
            considered "bad" within a given window. Defaults to ``0.4``.
        frac_bad : float | None
            The minimum proportion of bad windows for a channel to be considered
            "bad-by-correlation" or "bad-by-dropout". Defaults to ``0.01`` (1% of
            all windows).

        """
        IQR_TO_SD = 0.7413  # Scales units of IQR to units of SD, assuming normality
        # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/IQR.html

        if self.EEGFiltered is None:
            self.EEGFiltered = self._get_filtered_data()

        # Determine the number and size (in frames) of correlation windows
        win_size = int(correlation_secs * self.sample_rate)
        win_offsets = np.arange(1, (self.n_samples - win_size), win_size)
        win_count = len(win_offsets)

        # Initialize per-window arrays for each type of noise info calculated below
        max_correlations = np.ones((win_count, self.n_chans_original))
        dropout = np.zeros((win_count, self.n_chans_original), dtype=bool)
        noiselevels = np.zeros((win_count, self.n_chans_original))
        channel_amplitudes = np.zeros((win_count, self.n_chans_original))

        for w in range(win_count):
            # Get both filtered and unfiltered data for the current window
            start, end = (w * win_size, (w + 1) * win_size)
            eeg_filtered = self.EEGFiltered[:, start:end]
            eeg_raw = self.EEGData[:, start:end]

            # Get channel amplitude info for the window
            usable = self.usable_idx.copy()
            channel_amplitudes[w, usable] = _mat_iqr(eeg_raw, axis=1) * IQR_TO_SD

            # Check for any channel dropouts (flat signal) within the window
            eeg_amplitude = median_abs_deviation(eeg_filtered, axis=1)
            dropout[w, usable] = eeg_amplitude == 0

            # Exclude any dropout chans from further calculations (avoids div-by-zero)
            usable[usable] = eeg_amplitude > 0
            eeg_raw = eeg_raw[eeg_amplitude > 0, :]
            eeg_filtered = eeg_filtered[eeg_amplitude > 0, :]
            eeg_amplitude = eeg_amplitude[eeg_amplitude > 0]

            # Get high-frequency noise ratios for the window
            high_freq_amplitude = median_abs_deviation(eeg_raw - eeg_filtered, axis=1)
            noiselevels[w, usable] = high_freq_amplitude / eeg_amplitude

            # Get inter-channel correlations for the window
            win_correlations = np.corrcoef(eeg_filtered)
            abs_corr = np.abs(win_correlations - np.diag(np.diag(win_correlations)))
            max_correlations[w, usable] = _mat_quantile(abs_corr, 0.98, axis=0)
            max_correlations[w, dropout[w, :]] = 0  # Set dropout correlations to 0

        # Flag channels with above-threshold fractions of bad correlation windows
        thresholded_correlations = max_correlations < correlation_threshold
        fraction_bad_corr_windows = np.mean(thresholded_correlations, axis=0)
        bad_correlation_mask = fraction_bad_corr_windows > frac_bad
        bad_correlation_channels = self.ch_names_original[bad_correlation_mask]

        # Flag channels with above-threshold fractions of drop-out windows
        fraction_dropout_windows = np.mean(dropout, axis=0)
        dropout_mask = fraction_dropout_windows > frac_bad
        dropout_channels = self.ch_names_original[dropout_mask]

        # Update names of low-correlation/dropout channels & save additional info
        self.bad_by_correlation = bad_correlation_channels.tolist()
        self.bad_by_dropout = dropout_channels.tolist()
        self._extra_info["bad_by_correlation"] = {
            "max_correlations": np.transpose(max_correlations),
            "median_max_correlations": np.median(max_correlations, axis=0),
            "bad_window_fractions": fraction_bad_corr_windows,
        }
        self._extra_info["bad_by_dropout"] = {
            "dropouts": np.transpose(dropout.astype(np.int8)),
            "bad_window_fractions": fraction_dropout_windows,
        }
        self._extra_info["bad_by_deviation"]["channel_amplitudes"] = channel_amplitudes
        self._extra_info["bad_by_hf_noise"]["noise_levels"] = noiselevels

    def find_bad_by_SNR(self):
        """Detect channels that have a low signal-to-noise ratio.

        Channels are considered "bad-by-SNR" if they are bad by both high-frequency
        noise and bad by low correlation.

        """
        # Get names of bad-by-HF-noise and bad-by-correlation channels
        if not len(self._extra_info["bad_by_hf_noise"]) > 1:
            self.find_bad_by_hfnoise()
        if not len(self._extra_info["bad_by_correlation"]) and self.correlation:
            self.find_bad_by_correlation()
        bad_by_hf = set(self.bad_by_hf_noise)
        bad_by_corr = set(self.bad_by_correlation)

        # Flag channels bad by both HF noise and low correlation as bad by low SNR
        self.bad_by_SNR = list(bad_by_corr.intersection(bad_by_hf))

    def find_bad_by_PSD(self, zscore_threshold=3.0, fmin=1.0, fmax=45.0):
        """Detect channels with abnormally high or low power spectral density.

        This is a PyPREP-only method not present in the original MATLAB PREP.

        A channel is considered "bad-by-psd" if:
        1. Its power in any frequency band (low: 1-15 Hz, mid: 15-30 Hz,
           high: 30-45 Hz) deviates considerably from other channels, OR
        2. Its high-frequency band has more power than its low-frequency band
           (violating the typical 1/f spectral profile of EEG).

        PSD is computed using Welch's method over the specified frequency range.
        The default range (1-45 Hz) excludes line noise frequencies (50/60 Hz).

        Parameters
        ----------
        zscore_threshold : float, optional
            The minimum absolute z-score of a channel for it to be considered
            bad-by-psd. Defaults to ``3.0``.
        fmin : float, optional
            The lower frequency bound (in Hz) for PSD computation.
            Defaults to ``1.0``.
        fmax : float, optional
            The upper frequency bound (in Hz) for PSD computation. The default
            of ``45.0`` excludes 50/60 Hz line noise from the analysis.

        """
        MAD_TO_SD = 1.4826  # Scales units of MAD to units of SD, assuming normality
        # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mad.html

        # Define frequency bands (in Hz)
        BAND_LOW = (fmin, 15.0)  # ~ delta, theta, alpha
        BAND_MID = (15.0, 30.0)  # ~ beta
        BAND_HIGH = (30.0, fmax)  # ~ gamma

        if self.EEGFiltered is None:
            self.EEGFiltered = self._get_filtered_data()

        # Create a temporary Raw object from filtered data for PSD computation
        info = mne.create_info(
            ch_names=self.ch_names_new.tolist(),
            sfreq=self.sample_rate,
            ch_types="eeg",
        )
        raw_filtered = mne.io.RawArray(self.EEGFiltered, info, verbose=False)

        # Compute PSD using Welch method and convert to log scale (dB)
        psd = raw_filtered.compute_psd(
            method="welch", fmin=fmin, fmax=fmax, verbose=False
        )
        psd_data = psd.get_data()
        freqs = psd.freqs
        log_psd = 10 * np.log10(psd_data)

        # Get frequency indices for each band
        idx_low = (freqs >= BAND_LOW[0]) & (freqs < BAND_LOW[1])
        idx_mid = (freqs >= BAND_MID[0]) & (freqs < BAND_MID[1])
        idx_high = (freqs >= BAND_HIGH[0]) & (freqs <= BAND_HIGH[1])

        # Compute band power (sum of log PSD within each band) for each channel
        band_power_low = np.sum(log_psd[:, idx_low], axis=1)
        band_power_mid = np.sum(log_psd[:, idx_mid], axis=1)
        band_power_high = np.sum(log_psd[:, idx_high], axis=1)

        def robust_zscore(values):
            """Compute robust z-scores using MAD."""
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            sd = mad * MAD_TO_SD
            if sd > 0:
                return (values - median) / sd
            return np.zeros_like(values)

        # Criterion 1: Outlier in any single band
        zscore_low = robust_zscore(band_power_low)
        zscore_mid = robust_zscore(band_power_mid)
        zscore_high = robust_zscore(band_power_high)

        bad_by_band = (
            (np.abs(zscore_low) > zscore_threshold)
            | (np.abs(zscore_mid) > zscore_threshold)
            | (np.abs(zscore_high) > zscore_threshold)
        )

        # Criterion 2: 1/f violation (high freq band has more power than low freq band)
        # This is unusual for normal EEG and suggests muscle artifact or bad contact
        bad_by_1f_violation = band_power_high > band_power_low

        # Criterion 3: Abnormal band ratios compared to other channels
        # Use small epsilon to avoid division by zero
        eps = np.finfo(float).eps
        ratio_low_mid = band_power_low / (band_power_mid + eps)
        ratio_low_high = band_power_low / (band_power_high + eps)
        ratio_mid_high = band_power_mid / (band_power_high + eps)

        zscore_ratio_low_mid = robust_zscore(ratio_low_mid)
        zscore_ratio_low_high = robust_zscore(ratio_low_high)
        zscore_ratio_mid_high = robust_zscore(ratio_mid_high)

        bad_by_ratio = (
            (np.abs(zscore_ratio_low_mid) > zscore_threshold)
            | (np.abs(zscore_ratio_low_high) > zscore_threshold)
            | (np.abs(zscore_ratio_mid_high) > zscore_threshold)
        )

        # Combine criteria (bad if ANY criterion is met)
        # Note: bad_by_ratio is computed for diagnostics but not used in final
        # decision as it tends to be overly sensitive and theoretically debatable
        bad_by_psd_usable = bad_by_band | bad_by_1f_violation

        # Map back to original channel indices
        psd_channel_mask = np.zeros(self.n_chans_original, dtype=bool)
        psd_channel_mask[self.usable_idx] = bad_by_psd_usable
        abnormal_psd_channels = self.ch_names_original[psd_channel_mask]

        # Compute combined z-score for reporting (max absolute z-score across bands)
        psd_zscore = np.zeros(self.n_chans_original)
        max_band_zscore = np.maximum(
            np.abs(zscore_low), np.maximum(np.abs(zscore_mid), np.abs(zscore_high))
        )
        psd_zscore[self.usable_idx] = max_band_zscore

        # Update names of bad channels by abnormal PSD & save additional info
        self.bad_by_psd = abnormal_psd_channels.tolist()
        self._extra_info["bad_by_psd"].update(
            {
                "psd_zscore": psd_zscore,
                "band_power_low": band_power_low,
                "band_power_mid": band_power_mid,
                "band_power_high": band_power_high,
                "zscore_low": zscore_low,
                "zscore_mid": zscore_mid,
                "zscore_high": zscore_high,
                "bad_by_band": bad_by_band,
                "bad_by_1f_violation": bad_by_1f_violation,
                "bad_by_ratio": bad_by_ratio,
            }
        )

    def find_bad_by_ransac(
        self,
        n_samples=50,
        sample_prop=0.25,
        corr_thresh=0.75,
        frac_bad=0.4,
        corr_window_secs=5.0,
        channel_wise=False,
        max_chunk_size=None,
    ):
        """Detect channels that are predicted poorly by other channels.

        This method uses a random sample consensus approach (RANSAC, see [1]_,
        and a short discussion in [2]_) to try and predict what the signal should
        be for each channel based on the signals and spatial locations of other
        currently-good channels. RANSAC correlations are calculated by splitting
        the recording into non-overlapping windows of time (default: 5 seconds)
        and correlating each channel's RANSAC-predicted signal with its actual
        signal within each window.

        A RANSAC window is considered "bad" for a channel if its predicted signal
        vs. actual signal correlation falls below the given correlation threshold
        (default: ``0.75``). A channel is considered "bad-by-RANSAC" if its fraction
        of bad RANSAC windows is above the given threshold (default: ``0.4``).

        Due to its random sampling component, the channels identified as
        "bad-by-RANSAC" may vary slightly between calls of this method.
        Additionally, bad channels may vary between different montages given that
        RANSAC's signal predictions are based on the spatial coordinates of each
        electrode.

        This method is a wrapper for the :func:`~ransac.find_bad_by_ransac`
        function.

        .. warning:: For optimal performance, RANSAC requires that channels bad by
                     deviation, correlation, and/or dropout have already been
                     flagged. Otherwise RANSAC will attempt to use those channels
                     when making signal predictions, decreasing accuracy and thus
                     increasing the likelihood of false positives.

        Parameters
        ----------
        n_samples : int | None
            Number of random channel samples to use for RANSAC. Defaults
            to ``50``.
        sample_prop : float | None
            Proportion of total channels to use for signal prediction per RANSAC
            sample. This needs to be in the range [0, 1], where 0 would mean no
            channels would be used and 1 would mean all channels would be used
            (neither of which would be useful values). Defaults to ``0.25``
            (e.g., 16 channels per sample for a 64-channel dataset).
        corr_thresh : float | None
            The minimum predicted vs. actual signal correlation for a channel to
            be considered good within a given RANSAC window. Defaults
            to ``0.75``.
        frac_bad : float | None
            The minimum fraction of bad (i.e., below-threshold) RANSAC windows
            for a channel to be considered bad-by-RANSAC. Defaults to ``0.4``.
        corr_window_secs : float | None
            The duration (in seconds) of each RANSAC correlation window. Defaults
            to 5 seconds.
        channel_wise : bool | None
            Whether RANSAC should predict signals for chunks of channels over the
            entire signal length ("channel-wise RANSAC", see `max_chunk_size`
            parameter). If ``False``, RANSAC will instead predict signals for all
            channels at once but over a number of smaller time windows instead of
            over the entire signal length ("window-wise RANSAC"). Channel-wise
            RANSAC generally has higher RAM demands than window-wise RANSAC
            (especially if `max_chunk_size` is ``None``), but can be faster on
            systems with lots of RAM to spare. Defaults to ``False``.
        max_chunk_size : {int, None} | None
            The maximum number of channels to predict at once during
            channel-wise RANSAC. If ``None``, RANSAC will use the largest chunk
            size that will fit into the available RAM, which may slow down
            other programs on the host system. If using window-wise RANSAC
            (the default), this parameter has no effect. Defaults to ``None``.

        References
        ----------
        .. [1] Fischler, M.A., Bolles, R.C. (1981). Random sample consensus: A
            Paradigm for Model Fitting with Applications to Image Analysis and
            Automated Cartography. Communications of the ACM, 24, 381-395
        .. [2] Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
            (2017). Autoreject: Automated Artifact Rejection for MEG and EEG
            Data. NeuroImage, 159, 417-429

        """
        if self.EEGFiltered is None:
            self.EEGFiltered = self._get_filtered_data()

        exclude_from_ransac = (
            self.bad_by_correlation + self.bad_by_deviation + self.bad_by_dropout
        )

        if self.matlab_strict:
            random_state = self.random_state.get_state()
            rng = np.random.RandomState()
            rng.set_state(random_state)
        else:
            rng = self.random_state

        self.bad_by_ransac, ch_correlations_usable = find_bad_by_ransac(
            self.EEGFiltered,
            self.sample_rate,
            self.ch_names_new,
            self.raw_mne._get_channel_positions(self.raw_mne.ch_names)[
                self.usable_idx, :
            ],
            exclude_from_ransac,
            n_samples,
            sample_prop,
            corr_thresh,
            frac_bad,
            corr_window_secs,
            channel_wise,
            max_chunk_size,
            rng,
            self.matlab_strict,
        )

        # Reshape correlation matrix to match original channel count
        n_ransac_windows = ch_correlations_usable.shape[0]
        ch_correlations = np.ones((n_ransac_windows, self.n_chans_original))
        ch_correlations[:, self.usable_idx] = ch_correlations_usable

        self._extra_info["bad_by_ransac"] = {
            "ransac_correlations": ch_correlations,
            "bad_window_fractions": np.mean(ch_correlations < corr_thresh, axis=0),
        }
