"""finds bad channels."""
import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix
from mne.utils import check_random_state
from psutil import virtual_memory
from scipy import signal
from scipy.stats import iqr
from statsmodels import robust

from pyprep.removeTrend import removeTrend
from pyprep.utils import filter_design, split_list


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
        raw : mne.raw
            The MNE raw object.
        do_detrend : bool
            Whether or not to remove a trend from the data upon initializing the
            `NoisyChannels` object. Defaults to True.
        random_state : int | None | np.random.mtrand.RandomState
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

        for i in range(0, self.original_dimensions[0]):
            nan_channel_mask[i] = np.sum(np.isnan(self.EEGData[i, :])) > 0
        for i in range(0, self.original_dimensions[0]):
            no_signal_channel_mask[i] = (
                robust.mad(self.EEGData[i, :], c=1) < 1e-10
                or np.std(self.EEGData[i, :]) < 1e-10
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

        References
        ----------
        .. [1] Fischler, M.A., Bolles, R.C. (1981). Random rample consensus: A
           Paradigm for Model Fitting with Applications to Image Analysis and
           Automated Cartography. Communications of the ACM, 24, 381-395
        .. [2] Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
           (2017). Autoreject: Automated Artifact Rejection for MEG and EEG
           Data. NeuroImage, 159, 417-429

        """
        # First, identify all bad channels by other means:
        bads = self.get_bads()

        # Get all channel positions and the position subset of "clean channels"
        good_idx = mne.pick_channels(list(self.ch_names_new), include=[], exclude=bads)
        good_chn_labs = self.ch_names_new[good_idx]
        n_chans_good = good_idx.shape[0]
        chn_pos = self.raw_mne._get_channel_positions()
        chn_pos_good = chn_pos[good_idx, :]

        # Check if we have enough remaining channels
        # after exclusion of bad channels
        n_pred_chns = int(np.ceil(fraction_good * n_chans_good))

        if n_pred_chns <= 3:
            raise IOError(
                "Too few channels available to reliably perform"
                " ransac. Perhaps, too many channels have failed"
                " quality tests."
            )

        # Correlation windows setup
        correlation_frames = corr_window_secs * self.sample_rate
        correlation_window = np.arange(correlation_frames)
        n = correlation_window.shape[0]
        correlation_offsets = np.arange(
            0, (self.signal_len - correlation_frames), correlation_frames
        )
        w_correlation = correlation_offsets.shape[0]

        # Preallocate
        channel_correlations = np.ones((w_correlation, self.n_chans_new))
        # Notice self.EEGData.shape[0] = self.n_chans_new
        # They came from the same drop of channels

        print("Executing RANSAC\nThis may take a while, so be patient...")

        chunk_size = self.n_chans_new
        mem_error = True
        job = list(range(self.n_chans_new))

        if channel_wise:
            chunk_size = 1
        while mem_error:
            try:
                channel_chunks = split_list(job, chunk_size)
                total_chunks = len(channel_chunks)
                current = 1
                for chunk in channel_chunks:
                    channel_correlations[:, chunk] = self.ransac_correlations(
                        chunk,
                        chn_pos,
                        chn_pos_good,
                        good_chn_labs,
                        n_pred_chns,
                        self.EEGData,
                        n_samples,
                        n,
                        w_correlation,
                    )
                    if chunk == channel_chunks[0]:
                        # If it gets here, it means it is the optimal
                        print("Finding optimal chunk size :", chunk_size)
                        print("Total # of chunks:", total_chunks)
                        print("Current chunk:", end=" ", flush=True)

                    print(current, end=" ", flush=True)
                    current = current + 1

                mem_error = False  # All chunks processed, hurray!
                del current
            except MemoryError:
                chunk_size = chunk_size - 1  # // 2
                # dividing strategy may miss the most optimal chunk size
                if chunk_size == 0:
                    raise MemoryError(
                        "Not even doing 1 channel at a time the data fits in ram..."
                        "You could downsample the data or reduce the number of requ"
                        "ested samples."
                    )

        # Thresholding
        thresholded_correlations = channel_correlations < corr_thresh
        frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

        # find the corresponding channel names and return
        bad_ransac_channels_idx = np.argwhere(frac_bad_corr_windows > fraction_bad)
        bad_ransac_channels_name = self.ch_names_new[
            bad_ransac_channels_idx.astype(int)
        ]
        self.bad_by_ransac = [i[0] for i in bad_ransac_channels_name]
        print("\nRANSAC done!")

    def run_ransac(
        self, chn_pos, chn_pos_good, good_chn_labs, n_pred_chns, data, n_samples
    ):
        """Detect noisy channels apart from the ones described previously.

        It creates a random subset of the so-far good channels
        and predicts the values of the channels not in the subset.

        Parameters
        ----------
        chn_pos : ndarray
            3-D coordinates of the electrode position
        chn_pos_good : ndarray
            3-D coordinates of all the channels not detected noisy so far
        good_chn_labs : array_like
            channel labels for the ch_pos_good channels-
        n_pred_chns : int
            channel numbers used for interpolation for RANSAC
        data : ndarry
            2-D EEG data
        n_samples : int
            number of interpolations from which a median will be computed

        Returns
        -------
        ransac_eeg : ndarray
            The EEG data predicted by RANSAC

        """
        # n_chns, n_timepts = data.shape
        # 2 next lines should be equivalent but support single channel processing
        n_timepts = data.shape[1]
        n_chns = chn_pos.shape[0]

        # Before running, make sure we have enough memory
        try:
            available_gb = virtual_memory().available * 1e-9
            needed_gb = (data[:n_chns, :].nbytes * 1e-9) * n_samples
            assert available_gb > needed_gb
        except AssertionError:
            raise MemoryError(
                "For given data of shape {} and the requested"
                " number of {} samples, {} GB or memory would be"
                " needed but only {} GB are available. You"
                " could downsample the data or reduce the number"
                " of requested samples."
                "".format(data.shape, n_samples, needed_gb, available_gb)
            )

        # Memory seems to be fine ...
        # Make the predictions
        eeg_predictions = np.zeros((n_chns, n_timepts, n_samples))
        for sample in range(n_samples):
            eeg_predictions[..., sample] = self.get_ransac_pred(
                chn_pos, chn_pos_good, good_chn_labs, n_pred_chns, data
            )

        # Form median from all predictions
        ransac_eeg = np.median(eeg_predictions, axis=-1, overwrite_input=True)
        return ransac_eeg

    def get_ransac_pred(self, chn_pos, chn_pos_good, good_chn_labs, n_pred_chns, data):
        """Perform RANSAC prediction.

        Parameters
        ----------
        chn_pos : ndarray
            3-D coordinates of the electrode position
        chn_pos_good : ndarray
            3-D coordinates of all the channels not detected noisy so far
        good_chn_labs : array_like
            channel labels for the ch_pos_good channels
        n_pred_chns : int
            channel numbers used for interpolation for RANSAC
        data : ndarray
            2-D EEG data

        Returns
        -------
        ransac_pred : ndarray
            Single RANSAC prediction

        """
        rng = check_random_state(self.random_state)

        # Pick a subset of clean channels for reconstruction
        reconstr_idx = rng.choice(
            np.arange(chn_pos_good.shape[0]), size=n_pred_chns, replace=False
        )

        # Get positions and according labels
        reconstr_labels = good_chn_labs[reconstr_idx]
        reconstr_pos = chn_pos_good[reconstr_idx, :]

        # Map the labels to their indices within the complete data
        # Do not use mne.pick_channels, because it will return a sorted list.
        reconstr_picks = [
            list(self.ch_names_new).index(chn_lab) for chn_lab in reconstr_labels
        ]

        # Interpolate
        interpol_mat = _make_interpolation_matrix(reconstr_pos, chn_pos)
        ransac_pred = np.matmul(interpol_mat, data[reconstr_picks, :])
        return ransac_pred

    def ransac_correlations(
        self,
        chans_to_predict,
        chn_pos,
        chn_pos_good,
        good_chn_labs,
        n_pred_chns,
        data,
        n_samples,
        n,
        w_correlation,
    ):
        """Get correlations of channels to their ransac predicted values.

        Parameters
        ----------
        chans_to_predict: list of int
            Indexes of the channels to predict.
        chn_pos : ndarray
            3-D coordinates of the electrode positions to predict
        chn_pos_good : ndarray
            3-D coordinates of all the channels not detected noisy so far
        good_chn_labs : array_like
            channel labels for the ch_pos_good channels
        n_pred_chns : int
            channel numbers used for interpolation for RANSAC
        data : ndarray
            2-D EEG data
        n_samples : int
            Number of samples used for computation of ransac.
        n : int
            Number of frames/samples of each window.
        w_correlation: int
            Number of windows.

        Returns
        -------
        channel_correlations : ndarray
            correlations of the given channels to their ransac predicted values.

        """
        # Preallocate
        channel_correlations = np.ones((w_correlation, len(chans_to_predict)))

        # Make the ransac predictions
        ransac_eeg = self.run_ransac(
            chn_pos=chn_pos[chans_to_predict, :],
            chn_pos_good=chn_pos_good,
            good_chn_labs=good_chn_labs,
            n_pred_chns=n_pred_chns,
            data=data,
            n_samples=n_samples,
        )

        # Correlate ransac prediction and eeg data

        # For the actual data
        data_window = data[chans_to_predict, : n * w_correlation]
        data_window = data_window.reshape(len(chans_to_predict), n, w_correlation)

        # For the ransac predicted eeg
        pred_window = ransac_eeg[: len(chans_to_predict), : n * w_correlation]
        pred_window = pred_window.reshape(len(chans_to_predict), n, w_correlation)

        # Perform correlations
        for k in range(w_correlation):
            data_portion = data_window[:, :, k]
            pred_portion = pred_window[:, :, k]

            R = np.corrcoef(data_portion, pred_portion)

            # Take only correlations of data with pred
            # and use diag to extract correlation of
            # data_i with pred_i
            R = np.diag(R[0 : len(chans_to_predict), len(chans_to_predict) :])
            channel_correlations[k, :] = R

        return channel_correlations
