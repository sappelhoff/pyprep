"""finds bad channels."""
import mne
import numpy as np
from mne.utils import check_random_state
from scipy import signal
from scipy.stats import iqr
from statsmodels import robust

from pyprep.ransac import find_bad_by_ransac
from pyprep.removeTrend import removeTrend
from pyprep.utils import filter_design


class NoisyChannels:
    """Implements the functionality of the `findNoisyChannels` function.

    It is a part of the PREP (preprocessing pipeline) for EEG data recorded using
    10-20 montage style described in [1]_.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       EEG analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(self, raw, do_detrend=True, random_state=None):
        """Initialize the class.

        Parameters
        ----------
        raw : mne.io.Raw
            The MNE raw object.
        do_detrend : bool
            Whether or not to remove a trend from the data upon initializing the
            `NoisyChannels` object. Defaults to True.
        random_state : int | None | np.random.RandomState
            The random seed at which to initialize the class. If random_state
            is an int, it will be used as a seed for RandomState.
            If None, the seed will be obtained from the operating system
            (see RandomState for details). Default is None.

        """
        # Make sure that we got an MNE object
        assert isinstance(raw, mne.io.BaseRaw)

        self.raw_mne = raw.copy()
        self.sample_rate = raw.info["sfreq"]
        if do_detrend:
            self.raw_mne._data = removeTrend(
                self.raw_mne.get_data(), sample_rate=self.sample_rate
            )

        self.EEGData = self.raw_mne.get_data(picks="eeg")
        self.EEGData_beforeFilt = self.EEGData
        self.ch_names_original = np.asarray(raw.info["ch_names"])
        self.n_chans_original = len(self.ch_names_original)
        self.n_chans_new = self.n_chans_original
        self.signal_len = len(self.raw_mne.times)
        self.original_dimensions = np.shape(self.EEGData)
        self.new_dimensions = self.original_dimensions
        self.original_channels = np.arange(self.original_dimensions[0])
        self.new_channels = self.original_channels
        self.ch_names_new = self.ch_names_original
        self.channels_interpolate = self.original_channels

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
        self.bad_by_ransac = []

    def get_bads(self, verbose=False):
        """Get a list of all bad channels.

        This function makes a list of all the bad channels and prints them if verbose
        is True.

        Parameters
        ----------
        verbose : bool
            If verbose, print a summary of bad channels.

        """
        bads = (
            self.bad_by_nan
            + self.bad_by_flat
            + self.bad_by_deviation
            + self.bad_by_hf_noise
            + self.bad_by_SNR
            + self.bad_by_correlation
            + self.bad_by_dropout
            + self.bad_by_ransac
        )
        bads = list(set(bads))

        if verbose:
            print("Found {} uniquely bad channels.".format(len(bads)))
            print("\n{} by n/a: {}".format(len(self.bad_by_nan), self.bad_by_nan))
            print("\n{} by flat: {}".format(len(self.bad_by_flat), self.bad_by_flat))
            print(
                "\n{} by deviation: {}".format(
                    len(self.bad_by_deviation), self.bad_by_deviation
                )
            )
            print(
                "\n{} by hf noise: {}".format(
                    len(self.bad_by_hf_noise), self.bad_by_hf_noise
                )
            )
            print(
                "\n{} by correl: {}".format(
                    len(self.bad_by_correlation), self.bad_by_correlation
                )
            )
            print("\n{} by SNR {}".format(len(self.bad_by_SNR), self.bad_by_SNR))
            print(
                "\n{} by dropout: {}".format(
                    len(self.bad_by_dropout), self.bad_by_dropout
                )
            )
            print(
                "\n{} by ransac: {}".format(len(self.bad_by_ransac), self.bad_by_ransac)
            )
        return bads

    def find_all_bads(self, ransac=True):
        """Call all the functions to detect bad channels.

        This function calls all the bad-channel detecting functions.

        Parameters
        ----------
        ransac : bool
            To detect channels by ransac or not.

        """
        self.find_bad_by_nan_flat()
        self.find_bad_by_deviation()
        self.find_bad_by_SNR()
        if ransac:
            self.find_bad_by_ransac()

    def find_bad_by_nan_flat(self):
        """Detect channels that appear flat or have NaN values."""
        nan_channel_mask = [False] * self.original_dimensions[0]
        no_signal_channel_mask = [False] * self.original_dimensions[0]

        FLAT_THRESHOLD = 1e-15  # corresponds to 10e-10 µV in MATLAB PREP
        for i in range(0, self.original_dimensions[0]):
            nan_channel_mask[i] = np.sum(np.isnan(self.EEGData[i, :])) > 0
        for i in range(0, self.original_dimensions[0]):
            no_signal_channel_mask[i] = (
                robust.mad(self.EEGData[i, :], c=1) < FLAT_THRESHOLD
                or np.std(self.EEGData[i, :]) < FLAT_THRESHOLD
            )
        nan_channels = self.channels_interpolate[nan_channel_mask]
        flat_channels = self.channels_interpolate[no_signal_channel_mask]

        nans_no_data_channels = np.union1d(nan_channels, flat_channels)
        self.channels_interpolate = np.setdiff1d(
            self.channels_interpolate, nans_no_data_channels
        )

        for i in range(0, len(nan_channels)):
            self.bad_by_nan.append(self.ch_names_original[nan_channels[i]])
        for i in range(0, len(flat_channels)):
            self.bad_by_flat.append(self.ch_names_original[flat_channels[i]])

        self.raw_mne.drop_channels(list(set(self.bad_by_nan + self.bad_by_flat)))
        self.EEGData = self.raw_mne.get_data(picks="eeg")
        self.ch_names_new = np.asarray(self.raw_mne.info["ch_names"])
        self.n_chans_new = len(self.ch_names_new)
        self.new_dimensions = np.shape(self.EEGData)

    def find_bad_by_deviation(self, deviation_threshold=5.0):
        """Robust z-score of the robust standard deviation for each channel is calculated.

        Channels having a z-score greater than 5 are detected as bad.

        Parameters
        ----------
        deviation_threshold : float
            z-score threshold above which channels will be labelled bad.

        """
        deviation_channel_mask = [False] * (self.new_dimensions[0])
        channel_deviation = np.zeros(self.new_dimensions[0])
        for i in range(0, self.new_dimensions[0]):
            channel_deviation[i] = 0.7413 * iqr(self.EEGData[i, :])
        channel_deviationSD = 0.7413 * iqr(channel_deviation)
        channel_deviationMedian = np.nanmedian(channel_deviation)
        robust_channel_deviation = np.divide(
            np.subtract(channel_deviation, channel_deviationMedian), channel_deviationSD
        )
        for i in range(0, self.new_dimensions[0]):
            deviation_channel_mask[i] = abs(
                robust_channel_deviation[i]
            ) > deviation_threshold or np.isnan(robust_channel_deviation[i])
        deviation_channels = self.channels_interpolate[deviation_channel_mask]
        for i in range(0, len(deviation_channels)):
            self.bad_by_deviation.append(self.ch_names_original[deviation_channels[i]])

    def find_bad_by_hfnoise(self, HF_zscore_threshold=5.0):
        """Determine noise of channel through high frequency ratio.

        Noisiness of the channel is determined by finding the ratio of the median
        absolute deviation of high frequency to low frequency components.

        Low pass 50 Hz filter is used to separate the frequency components.
        A robust z-score is then calculated relative to all the channels.

        Parameters
        ----------
        HF_zscore_threshold : float
            z-score threshold above which channels would be labelled as bad.

        """
        data_tmp = np.transpose(self.EEGData)
        dimension = np.shape(data_tmp)
        if self.sample_rate > 100:
            EEG_filt = np.zeros((dimension[0], dimension[1]))
            bandpass_filter = filter_design(
                N_order=100,
                amp=np.array([1, 1, 0, 0]),
                freq=np.array([0, 90 / self.sample_rate, 100 / self.sample_rate, 1]),
            )
            for i in range(0, dimension[1]):
                EEG_filt[:, i] = signal.filtfilt(bandpass_filter, 1, data_tmp[:, i])
            noisiness = np.divide(
                robust.mad(np.subtract(data_tmp, EEG_filt), c=1),
                robust.mad(EEG_filt, c=1),
            )
            noisiness_median = np.nanmedian(noisiness)
            noiseSD = (
                np.median(np.absolute(np.subtract(noisiness, np.median(noisiness))))
                * 1.4826
            )
            zscore_HFNoise = np.divide(
                np.subtract(noisiness, noisiness_median), noiseSD
            )
            HFnoise_channel_mask = [False] * self.new_dimensions[0]
            for i in range(0, self.new_dimensions[0]):
                HFnoise_channel_mask[i] = zscore_HFNoise[
                    i
                ] > HF_zscore_threshold or np.isnan(zscore_HFNoise[i])
            HFNoise_channels = self.channels_interpolate[HFnoise_channel_mask]
        else:
            EEG_filt = data_tmp
            noisiness_median = 0
            # noisinessSD = 1
            zscore_HFNoise = np.zeros((self.new_dimensions[0], 1))
            HFNoise_channels = []
        self.EEGData_beforeFilt = data_tmp
        self.EEGData = np.transpose(EEG_filt)
        for i in range(0, len(HFNoise_channels)):
            self.bad_by_hf_noise.append(self.ch_names_original[HFNoise_channels[i]])

    def find_bad_by_correlation(
        self, correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01
    ):
        """Find correlation between the low frequency components of the EEG below 50 Hz.

        Correlation is done using a sliding non-overlapping time window.
        The maximum absolute correlation is as the 98th percentile of the absolute
        values of the correlations with the other channels
        If the maximum correlation is less than 0.4 then the channel is designated as
        bad by correlation.

        Parameters
        ----------
        correlation_secs : float
            length of the correlation time window (default: 1 secs).
        correlation_threshold : float
            correlation threshold below which channel is marked bad.
        frac_bad : float
            percentage of data windows in which the correlation threshold was
            not surpassed and if a channel gets a value of greater than 1%, it
            is designated bad. Notice that if `correlation_secs` is high, and
            thus the number of windows is low, the default `frac_bad` may
            be too extreme. For example if only 60 windows are available
            for a dataset, only a single bad window would be needed to
            classify as bad.

        """
        self.find_bad_by_hfnoise()  # since filtering is performed there
        correlation_frames = correlation_secs * self.sample_rate
        correlation_window = np.arange(correlation_frames)
        correlation_offsets = np.arange(
            1, (self.new_dimensions[1] - correlation_frames), correlation_frames
        )
        w_correlation = len(correlation_offsets)
        maximum_correlations = np.ones((self.original_dimensions[0], w_correlation))
        drop_out = np.zeros((self.new_dimensions[0], w_correlation))
        channel_correlation = np.ones((w_correlation, self.new_dimensions[0]))
        noiselevels = np.zeros((w_correlation, self.new_dimensions[0]))
        channel_deviations = np.zeros((w_correlation, self.new_dimensions[0]))
        drop = np.zeros((w_correlation, self.new_dimensions[0]))
        len_correlation_window = len(correlation_window)
        EEGData = np.transpose(self.EEGData)
        EEG_new_win = np.reshape(
            np.transpose(EEGData[0 : len_correlation_window * w_correlation, :]),
            (self.new_dimensions[0], len_correlation_window, w_correlation),
            order="F",
        )
        data_win = np.reshape(
            np.transpose(
                self.EEGData_beforeFilt[0 : len_correlation_window * w_correlation, :]
            ),
            (self.new_dimensions[0], len_correlation_window, w_correlation),
            order="F",
        )
        for k in range(0, w_correlation):
            eeg_portion = np.transpose(np.squeeze(EEG_new_win[:, :, k]))
            data_portion = np.transpose(np.squeeze(data_win[:, :, k]))
            window_correlation = np.corrcoef(np.transpose(eeg_portion))
            abs_corr = np.abs(
                np.subtract(window_correlation, np.diag(np.diag(window_correlation)))
            )
            channel_correlation[k, :] = np.quantile(abs_corr, 0.98, axis=0)
            noiselevels[k, :] = np.divide(
                robust.mad(np.subtract(data_portion, eeg_portion), c=1),
                robust.mad(eeg_portion, c=1),
            )
            channel_deviations[k, :] = 0.7413 * iqr(data_portion, axis=0)
        for i in range(0, w_correlation):
            for j in range(0, self.new_dimensions[0]):
                drop[i, j] = np.int(
                    np.isnan(channel_correlation[i, j]) or np.isnan(noiselevels[i, j])
                )
                if drop[i, j] == 1:
                    channel_deviations[i, j] = 0
                    noiselevels[i, j] = 0
        maximum_correlations[self.channels_interpolate, :] = np.transpose(
            channel_correlation
        )
        drop_out[:] = np.transpose(drop)
        thresholded_correlations = maximum_correlations < correlation_threshold
        thresholded_correlations = thresholded_correlations.astype(int)
        fraction_BadCorrelationWindows = np.mean(thresholded_correlations, axis=1)
        fraction_BadDropOutWindows = np.mean(drop_out, axis=1)

        bad_correlation_channels_idx = np.argwhere(
            fraction_BadCorrelationWindows > frac_bad
        )
        bad_correlation_channels_name = self.ch_names_original[
            bad_correlation_channels_idx.astype(int)
        ]
        self.bad_by_correlation = [i[0] for i in bad_correlation_channels_name]

        dropout_channels_idx = np.argwhere(fraction_BadDropOutWindows > frac_bad)
        dropout_channels_name = self.ch_names_original[dropout_channels_idx.astype(int)]
        self.bad_by_dropout = [i[0] for i in dropout_channels_name]

    def find_bad_by_SNR(self):
        """Determine the channels that fail both by correlation and HF noise."""
        self.find_bad_by_correlation()
        set_hf = set(self.bad_by_hf_noise)
        set_correlation = set(self.bad_by_correlation)
        self.bad_by_SNR = list(set_correlation.intersection(set_hf))
        return None

    def find_bad_by_ransac(
        self,
        n_samples=50,
        fraction_good=0.25,
        corr_thresh=0.75,
        fraction_bad=0.4,
        corr_window_secs=5.0,
        channel_wise=False,
    ):
        """Detect channels that are not predicted well by other channels.

        This method is a wrapper of the ``find_bad_by_ransac`` function
        from the ``ransac`` module.

        Here, a ransac approach (see [1]_, and a short discussion in [2]_) is
        adopted to predict a "clean EEG" dataset. After identifying clean EEG
        channels through the other methods, the clean EEG dataset is
        constructed by repeatedly sampling a small subset of clean EEG channels
        and interpolation the complete data. The median of all those
        repetitions forms the clean EEG dataset. In a second step, the original
        and the ransac predicted data are correlated and channels, which do not
        correlate well with themselves across the two datasets are considered
        `bad_by_ransac`.

        Parameters
        ----------
        n_samples : int
            Number of samples used for computation of ransac.
        fraction_good : float
            Fraction of channels used for robust reconstruction of the signal.
            This needs to be in the range [0, 1], where obviously neither 0
            nor 1 would make sense.
        corr_thresh : float
            The minimum correlation threshold that should be attained within a
            data window.
        fraction_bad : float
            If this percentage of all data windows in which the correlation
            threshold was not surpassed is exceeded, classify a
            channel as `bad_by_ransac`.
        corr_window_secs : float
            Size of the correlation window in seconds.
        channel_wise : bool
            If True the ransac will be done 1 channel at a time, if false
            it will be done as fast as possible (more channels at a time).

        References
        ----------
        .. [1] Fischler, M.A., Bolles, R.C. (1981). Random rample consensus: A
            Paradigm for Model Fitting with Applications to Image Analysis and
            Automated Cartography. Communications of the ACM, 24, 381-395
        .. [2] Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
            (2017). Autoreject: Automated Artifact Rejection for MEG and EEG
            Data. NeuroImage, 159, 417-429
        """
        self.bad_by_ransac, _ = find_bad_by_ransac(
            self.EEGData,
            self.sample_rate,
            self.signal_len,
            self.ch_names_new,
            self.raw_mne._get_channel_positions(),
            self.get_bads(),
            n_samples,
            fraction_good,
            corr_thresh,
            fraction_bad,
            corr_window_secs,
            channel_wise,
            self.random_state,
        )
