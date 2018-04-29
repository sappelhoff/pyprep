"""Module contains functions and classes for noisy EEG data detection."""

from psutil import virtual_memory

import mne
from mne.channels.interpolation import _make_interpolation_matrix
from mne.preprocessing import find_outliers
import numpy as np
from scipy.stats import iqr
from statsmodels.robust.scale import mad


def find_bad_epochs(epochs, picks=None, thresh=3.29053):
    """Find bad epochs based on amplitude, deviation, and variance.

    Inspired by [1], based on code by Marijn van Vliet [2]. This
    function is working on z-scores. You might want to select the
    thresholds according to how much of the data is expected to
    fall within the absolute bounds:

    95.0% --> 1.95996

    97.0% --> 2.17009

    99.0% --> 2.57583

    99.9% --> 3.29053

    Notes
    -----
    For this function to work, bad channels should have been identified
    and removed or interpolated beforehand. Additionally, baseline
    correction or highpass filtering is recommended to reduce signal
    drifts over time.

    Parameters
    ----------
    epochs : mne epochs object
        The epochs to analyze.

    picks : list of int | None
        Channels to operate on. Defaults to all clean EEG channels. Drops
        EEG channels marked as bad.

    thresh : float
        Epochs that surpass the threshold with their z-score based
        on amplitude, deviation, or variance, will be considered
        bad.

    Returns
    -------
    bads : list of int
        Indices of the bad epochs.

    References
    ----------
    .. [1] Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER:
       fully automated statistical thresholding for EEG artifact
       rejection. Journal of neuroscience methods, 192(1), 152-162.

    .. [2] https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc

    """
    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False,
                               eeg=True, exclude='bads')

    def calc_deviation(data):
        ch_mean = np.mean(data, axis=2)
        return ch_mean - np.mean(ch_mean, axis=0)

    metrics = {'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
               'deviation': lambda x: np.mean(calc_deviation(x), axis=1),
               'variance': lambda x: np.mean(np.var(x, axis=2), axis=1)}

    data = epochs.get_data()[:, picks, :]

    bads = []
    for m in metrics.keys():
        signal = metrics[m](data)
        bad_idx = find_outliers(signal, thresh)
        bads.append(bad_idx)

    # MNE starts counting epochs at 1, so adjust indices
    return np.unique(np.concatenate(bads)+1).tolist()


class Noisydata():
    """For a given raw data object, detect bad EEG channels.

    This class implements the functionality of the `findNoisyChannels` function
    as part of the PREP (preprocessing pipeline) for EEG data described in [1].

    Parameters
    ----------
    instance : raw mne object

    montage_kind : str
        Which kind of montage should be used to infer the electrode
        positions? E.g., 'standard_1020'

    low_cut : float
        Frequency low cutoff value for the highpass filter

    high_cut : float
        Frequency high cutoff value for the lowpass filter

    Attributes
    ----------
    _channel_correlations : ndarray, shape (k_windows, n_chans)
        For each k_window the correlation measure for each channel, where
        the correlation measure is an index of how well a channel correlates
        with all other channels.

    _ransac_channel_correlations : ndarray, shape (k_windows, n_chans)
        For each k_window the correlation for each channel with itself across
        the original data versus the ransac predicted data.

    _channel_deviations : ndarray, shape (n_chans,)
        The robust z-score deviation aggregates per channel.

    _channel_hf_noise : ndarray, shape (n_chans,)
        The robust z-score estimates of high frequency noise per channel.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       EEG analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(self, instance,
                 montage_kind='standard_1020',
                 low_cut=0.01, high_cut=50.):
        """Initialize the class."""
        # Make sure that we got an MNE object
        assert isinstance(instance, mne.io.BaseRaw)

        # The data that we are targeting
        # and a modifiable copy
        self.raw_mne = instance
        self.raw_copy = self.raw_mne.copy()

        # Set montage, pick data type, get data and transform to uVolts
        # We also filter all data at `low_cut` Hz highpass and obtain some data
        # bandpassed between `low_cut` and `high_cut` Hz.
        montage = mne.channels.read_montage(kind=montage_kind,
                                            ch_names=self.raw_copy.ch_names)
        self.raw_copy.set_montage(montage)
        self.raw_copy.pick_types(eeg=True, stim=False)
        self.raw_copy.filter(l_freq=low_cut, h_freq=None,
                             method='fir', fir_design='firwin', verbose=False)
        self.x = self.raw_copy.get_data() * 1e6
        self.raw_copy.filter(l_freq=None, h_freq=high_cut,
                             method='fir', fir_design='firwin', verbose=False)
        self.x_bp = self.raw_copy.get_data() * 1e6
        self.ch_names = np.asarray(self.raw_copy.ch_names)
        self.n_chans = len(self.ch_names)
        self.signal_len = len(self.raw_copy.times)
        self.sfreq = self.raw_copy.info['sfreq']
        self.chn_pos = self.raw_copy._get_channel_positions()

        # The identified bad channels
        self.bad_by_flat = []
        self.bad_by_nan = []
        self.bad_by_deviation = []
        self.bad_by_hf_noise = []
        self.bad_by_correlation = []
        self.bad_by_ransac = []

        return None

    def find_all_bads(self, ransac=True):
        """Call all functions that detect bad channels.

        Notes
        -----
            This will be using the functions default thresholds
            and settings.

        Parameters
        ----------
        ransac: boolean
            Whether or not to also fetch the bad_by_ransac channels.

        """
        self.find_bad_by_nan()
        self.find_bad_by_flat()
        self.find_bad_by_deviation()
        self.find_bad_by_hf_noise()
        self.find_bad_by_correlation()
        if ransac:
            self.find_bad_by_ransac()
        return None

    def get_bads(self, verbose=False):
        """Get a list of all bad channels.

        Parameters
        ----------
        verbose : boolean
            If verbose, print a summary of bad channels.

        """
        bads = (self.bad_by_flat + self.bad_by_nan + self.bad_by_deviation +
                self.bad_by_hf_noise + self.bad_by_correlation +
                self.bad_by_ransac)
        bads = list(set(bads))

        if verbose:
            print('Found {} uniquely bad channels.'.format(len(bads)))
            print('\n{} by n/a: {}'.format(len(self.bad_by_nan),
                                           self.bad_by_nan))
            print('\n{} by flat: {}'.format(len(self.bad_by_flat),
                                            self.bad_by_flat))
            print('\n{} by deviation: {}'.format(len(self.bad_by_deviation),
                                                 self.bad_by_deviation))
            print('\n{} by hf noise: {}'.format(len(self.bad_by_hf_noise),
                                                self.bad_by_hf_noise))
            print('\n{} by correl: {}'.format(len(self.bad_by_correlation),
                                              self.bad_by_correlation))
            print('\n{} by ransac: {}'.format(len(self.bad_by_ransac),
                                              self.bad_by_ransac))

        return bads

    def find_bad_by_nan(self):
        """Detect channels containing n/a data."""
        bad_idxs = np.argwhere(np.sum(np.isnan(self.x), axis=-1) > 0)
        bads = self.ch_names[bad_idxs.astype(int)]
        bads = [i[0] for i in bads]
        bads.sort()
        self.bad_by_nan = bads
        return None

    def find_bad_by_flat(self, flat_thresh=1, std_thresh=1):
        """Detect channels containing constant or very small values.

        Use the median absolute deviation and the standard deviation
        to find channels that have consistently low values.

        Parameters
        ----------
        flat_thresh : float
            Channels with a median absolute deviation below `flat_thresh`
            will be considered bac_by_flat.

        std_thresh : float
            Channels with a standard deviation below `std_thresh`
            will be considered bad_by_flat.

        """
        bad_by_mad = mad(self.x, c=1, axis=1) < flat_thresh
        bad_by_std = np.std(self.x, axis=1) < std_thresh
        bad_idxs = np.argwhere(np.logical_or(bad_by_mad, bad_by_std))
        bads = self.ch_names[bad_idxs.astype(int)]
        bads = [i[0] for i in bads]
        bads.sort()
        self.bad_by_flat = bads
        return None

    def find_bad_by_deviation(self, deviation_thresh=3.29053):
        """Detect channels that contain extreme amplitudes.

        This function is working on robust z-scores. You might want to
        select the thresholds according to how much of the data is expected
        to fall within the absolute bounds:

        95.0% --> 1.95996

        97.0% --> 2.17009

        99.0% --> 2.57583

        99.9% --> 3.29053

        Parameters
        ----------
        deviation_thresh : float
            Channels with a higher amplitude z-score than `deviation_thresh`
            will be considered bad_by_deviation.

        """
        # Calculate robust z-score of robust standard deviation for each chan
        chn_devi = 0.7413 * iqr(self.x, axis=1)
        chn_devi_sd = 0.7413 * iqr(chn_devi, axis=0)
        chn_devi_median = np.median(chn_devi)
        robust_chn_devi = (chn_devi - chn_devi_median) / chn_devi_sd

        # z-scores exceeding our theshold are classified as bad
        bad_idxs_bool = np.abs(robust_chn_devi) > deviation_thresh
        bad_idxs = np.argwhere(bad_idxs_bool)
        bads = self.ch_names[bad_idxs.astype(int)]
        bads = [i[0] for i in bads]
        bads.sort()
        self.bad_by_deviation = bads
        self._channel_deviations = robust_chn_devi
        return None

    def find_bad_by_hf_noise(self, hf_noise_thresh=3.29053):
        """Detect channels that contain high frequency (hf) noise.

        Use a robust estimate of the power of the high frequency components
        to the power of the low frequency components. This function depends
        on the `low_cut` and `high_cut` parameters given at initialization,
        as they determine the bandpass.

        This function is working on robust z-scores. You might want to
        select the thresholds according to how much of the data is expected
        to fall within the absolute bounds:

        95.0% --> 1.95996

        97.0% --> 2.17009

        99.0% --> 2.57583

        99.9% --> 3.29053

        Parameters
        ----------
        hf_noise_thresh : float
            The threshold for z-scores, when exceeded: classify as bad.

        """
        # Determine z-scored level of estimated signal-to-noise
        # ratio for each channel
        noisiness = (mad(self.x - self.x_bp, c=1, axis=1) /
                     mad(self.x_bp, c=1, axis=1))
        noisiness_median = np.median(noisiness)
        # robust estimate of STD
        noisiness_sd = mad(noisiness, c=1, axis=0) * 1.4826
        hf_noise_z = (noisiness - noisiness_median) / noisiness_sd
        bad_idxs_bool = hf_noise_z > hf_noise_thresh
        bad_idxs = np.argwhere(bad_idxs_bool)
        bads = self.ch_names[bad_idxs.astype(int)]
        bads = [i[0] for i in bads]
        bads.sort()
        self.bad_by_hf_noise = bads
        self._channel_hf_noise = hf_noise_z
        return None

    def find_bad_by_correlation(self, corr_thresh=0.4, fraction_bad=0.1,
                                corr_window_secs=1.):
        """Detect channels that do not correlate well with the other channels.

        Divide the whole signal into windows and compute window wise
        correlations. If a channel has more than `fraction_bad` windows that
        have correlate less than `corr_thresh` with the other channels, that
        channel is considered `bad_by_correlation`. The measure of correlation
        with other channels is defined as the 98th percentile of the absolute
        values of the correlations with the other channels in each window.

        Parameters
        ----------
        corr_thresh : float
            The minimum correlation threshold that should be attained within a
            data window.

        fraction_bad : float
            If this percentage of all data windows in which the correlation
            threshold was not surpassed is exceeded, classify a
            channel as `bad_by_correlation`.

        corr_window_secs : float
            Width of the correlation window in seconds.

        """
        # Based on the data, determine how many windows we need
        # and how large they should be
        correlation_frames = corr_window_secs * self.sfreq
        correlation_window = np.arange(0, correlation_frames)
        n = correlation_window.shape[0]
        correlation_offsets = np.arange(0, (self.signal_len -
                                            correlation_frames),
                                        correlation_frames)
        w_correlation = correlation_offsets.shape[0]

        # preallocate
        channel_correlations = np.ones((w_correlation, self.n_chans))

        # Cut the data indo windows
        x_bp_window = self.x_bp[:self.n_chans, :n*w_correlation]
        x_bp_window = x_bp_window.reshape(self.n_chans, n, w_correlation)

        # Perform Pearson correlations across channels per window
        # For each channel, take the absolute of the 98th percentile of
        # correlations with the other channels as a measure of how well
        # correlated that channel is with the others.
        for k in range(w_correlation):
            eeg_portion = x_bp_window[:, :, k]
            window_correlation = np.corrcoef(eeg_portion)
            abs_corr = np.abs((window_correlation -
                               np.diag(np.diag(window_correlation))))
            channel_correlations[k, :] = np.percentile(abs_corr, 98, axis=0)

        # Perform thresholding to see which channels correlate badly with the
        # other channels in a certain fraction of windows (bad_time_threshold)
        thresholded_correlations = channel_correlations < corr_thresh
        frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

        # find the corresponding channel names and return
        bad_idxs_bool = frac_bad_corr_windows > fraction_bad
        bad_idxs = np.argwhere(bad_idxs_bool)
        bads = self.ch_names[bad_idxs.astype(int)]
        bads = [i[0] for i in bads]
        bads.sort()
        self.bad_by_correlation = bads
        self._channel_correlations = channel_correlations
        return None

    def find_bad_by_ransac(self, n_samples=50, fraction_good=0.25,
                           corr_thresh=0.75, fraction_bad=0.4,
                           corr_window_secs=4.):
        """Detect channels that are not predicted well by other channels.

        Here, a ransac approach (see [1], and a short discussion in [2]) is
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
        self.find_all_bads(ransac=False)
        bads = self.get_bads()

        # Get all channel positions and the position subset of "clean channels"
        good_idx = mne.pick_channels(self.ch_names, include=[], exclude=bads)
        good_chn_labs = self.ch_names[good_idx]
        n_chans_good = good_idx.shape[0]
        chn_pos_good = self.chn_pos[good_idx, :]

        # Check if we have enough remaning channels
        # after exclusion of bad channels
        n_pred_chns = int(np.ceil(fraction_good * n_chans_good))

        if n_pred_chns <= 3:
            raise IOError('Too few channels available to reliably perform'
                          ' ransac. Perhaps, too many channels have failed'
                          ' quality tests. You could call `.find_all_bads`'
                          ' with the ransac=False option.')

        # Make the ransac predictions
        ransac_eeg = self._run_ransac(chn_pos=self.chn_pos,
                                      chn_pos_good=chn_pos_good,
                                      good_chn_labs=good_chn_labs,
                                      n_pred_chns=n_pred_chns,
                                      data=self.x_bp,
                                      n_samples=n_samples)

        # Correlate ransac prediction and eeg data
        correlation_frames = corr_window_secs * self.sfreq
        correlation_window = np.arange(correlation_frames)
        n = correlation_window.shape[0]
        correlation_offsets = np.arange(0, (self.signal_len -
                                            correlation_frames),
                                        correlation_frames)
        w_correlation = correlation_offsets.shape[0]

        # For the actual data
        data_window = self.x_bp[:self.n_chans, :n*w_correlation]
        data_window = data_window.reshape(self.n_chans, n, w_correlation)

        # For the ransac predicted eeg
        pred_window = ransac_eeg[:self.n_chans, :n*w_correlation]
        pred_window = pred_window.reshape(self.n_chans, n, w_correlation)

        # Preallocate
        channel_correlations = np.ones((w_correlation, self.n_chans))

        # Perform correlations
        for k in range(w_correlation):
            data_portion = data_window[:, :, k]
            pred_portion = pred_window[:, :, k]

            R = np.corrcoef(data_portion, pred_portion)

            # Take only correlations of data with pred
            # and use diag to exctract correlation of
            # data_i with pred_i
            R = np.diag(R[0:self.n_chans, self.n_chans:])
            channel_correlations[k, :] = R

        # Thresholding
        thresholded_correlations = channel_correlations < corr_thresh
        frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

        # find the corresponding channel names and return
        bad_idxs_bool = frac_bad_corr_windows > fraction_bad
        bad_idxs = np.argwhere(bad_idxs_bool)
        bads = self.ch_names[bad_idxs.astype(int)]
        bads = [i[0] for i in bads]
        bads.sort()
        self.bad_by_ransac = bads
        self._ransac_channel_correlations = channel_correlations
        return None

    def _run_ransac(self, chn_pos, chn_pos_good, good_chn_labs,
                    n_pred_chns, data, n_samples):
        """Predict the EEG timecourse of a channel using a ransac approach.

        Given the EEG data and electrode positions, form `sample_size`
        reconstructions of one channel based on spherical spline interpolation
        specified in [1]. The median of these reconstructions is used as a
        "clean prediction" of the channel.

        Parameters
        ----------
        chn_pos : ndarray, shape(n_chns, 3)
            3D coordinates of the electrodes used to collect
            the EEG data.

        chn_pos_good : ndarray, shape(n_good_chns, 3)
            3D coordinates of only the "clean" electrodes used to collect
            the EEG data.

        good_chn_labs : array_like, shape(n_good_chns,)
            The channel labels of the channels in `chn_good_pos`.

        n_pred_chns : int
            Number of channels used for each interpolation during
            ransac.

        data : ndarray, shape(n_chns, n_timepoints)
            The EEG data.

        n_samples : int
            Number of interpolations(reconstructions), from which
            a median will be formed to provide the final prediction.

        Returns
        -------
        ransac_eeg : ndarray of shape(n_chns, n_timepts)
            The EEG data as predicted by ransac.

        References
        ----------
        .. [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
           Spherical splines for scalp potential and current density mapping.
           Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.

        """
        # Before running, make sure we have enough memory
        try:
            available_gb = virtual_memory().available * 1e-9
            needed_gb = (data.nbytes * 1e-9) * n_samples
            assert available_gb > needed_gb
        except AssertionError:
            raise MemoryError('For given data of shape {} and the requested'
                              ' number of {} samples, {} GB or memory would be'
                              ' needed but only {} GB are available. You'
                              ' could downsample the data or reduce the number'
                              ' of requested samples.'
                              ''.format(data.shape, n_samples, needed_gb,
                                        available_gb))

        # Memory seems to be fine ...
        # Make the predictions
        n_chns, n_timepts = data.shape
        eeg_predictions = np.zeros((n_chns, n_timepts, n_samples))
        for sample in range(n_samples):
            eeg_predictions[..., sample] = self._get_ransac_pred(chn_pos,
                                                                 chn_pos_good,
                                                                 good_chn_labs,
                                                                 n_pred_chns,
                                                                 data)

        # Form median from all predictions
        ransac_eeg = np.median(eeg_predictions, axis=-1, overwrite_input=True)
        return ransac_eeg

    def _get_ransac_pred(self, chn_pos, chn_pos_good,
                         good_chn_labs, n_pred_chns, data):
        """Make a single ransac prediction.

        Parameters
        ----------
        chn_pos : ndarray, shape(n_chns, 3)
            3D coordinates of the electrodes used to collect
            the EEG data.

        chn_pos_good : ndarray, shape(n_good_chns, 3)
            3D coordinates of only the "clean" electrodes used to collect
            the EEG data.

        good_chn_labs : array_like, shape(n_good_chns,)
            The channel labels of the channels in `chn_good_pos`.

        n_pred_chns : int
            Number of channels used for each interpolation during
            ransac.

        data : ndarray, shape(n_chns, n_timepoints)
            The EEG data.

        Returns
        -------
        ransac_pred : ndarray of shape(n_chns, n_timepts)
            A single prediction based on ransac. Several of these
            should be averaged (e.g., median) to get `ransac_eeg`

        See Also
        --------
        _run_ransac, find_bad_by_ransac

        """
        # Pick a subset of clean channels for reconstruction
        reconstr_idx = np.random.choice(np.arange(chn_pos_good.shape[0]),
                                        size=n_pred_chns,
                                        replace=False)

        # Get positions and according labels
        reconstr_labels = good_chn_labs[reconstr_idx]
        reconstr_pos = chn_pos_good[reconstr_idx, :]

        # Map the labels to their indices within the complete data
        # Do not use mne.pick_channels, because it will return a sorted list.
        reconstr_picks = [list(self.ch_names).index(chn_lab) for chn_lab
                          in reconstr_labels]

        # Interpolate
        interpol_mat = _make_interpolation_matrix(reconstr_pos, chn_pos)
        ransac_pred = np.matmul(interpol_mat, data[reconstr_picks, :])
        return ransac_pred
