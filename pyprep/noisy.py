"""Contains functions to implement a find_noisy_channels function."""

from multiprocessing import cpu_count
from functools import partial
import psutil

import numpy as np
import mne

# https://github.com/sappelhoff/remedian
from remedian.remedian import Remedian


n_workers = cpu_count()


def help_fun(__, gen):
    """Exchange raw data with that yielded by a generator `gen`."""
    return next(gen)


def help_gen(data_mat):
    """Create a generator that yields rows of `data_mat`."""
    for i in range(data_mat.shape[0]):
        yield data_mat[i, :].squeeze()


def mad(X, axis):
    """Calculate the median absolute deviation (MAD) of an array.

    Parameters
    ----------
    X : ndarray, shape(n, t)
        Input data.

    Returns
    -------
    mad : ndarray, shape(n,)
        The median absolute deviation of the input data.

    """
    mad = np.median(np.abs(X - np.median(X, axis=axis, keepdims=True)),
                    axis=axis)
    return mad


def iqr(X, axis):
    """Calculate the interquartile range along an axis.

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    iqr : ndarray or int
        The interquartile range.

    """
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=axis))
    return iqr


def find_bad_by_nan(X, ch_names):
    """Detect channels containing NaN data.

    NOTE: Assumes EEG data to be measured in microvolts.

    Parameters
    ----------
    X : ndarray, shape(n, t)
        2D input data. NaN values will be searched for
        along the column axis.

    ch_names : list, len n
        List of channel names corresponding to rows in input data.

    Returns
    -------
    bads : list
        List of bad channel names.

    """
    assert X.ndim == 2
    bad_idxs = np.argwhere(np.sum(np.isnan(X), axis=0))
    bads = np.asarray(ch_names)[bad_idxs.astype(int)]
    bads = [i[0] for i in bads]
    return bads


def find_bad_by_flat(X, ch_names):
    """Detect channels containing constant or very small values.

    NOTE: Assumes EEG data to be measured in microvolts.

    Parameters
    ----------
    X : ndarray, shape(n, t)
        2D input data.

    ch_names : list, len n
        List of channel names corresponding to rows in input data.

    Returns
    -------
    bads : list
        List of bad channel names.

    """
    assert X.ndim == 2
    bad_by_mad = mad(X, axis=1) < 10e-10
    bad_by_std = np.std(X, axis=1) < 10e-10
    bad_idxs = np.argwhere(np.logical_or(bad_by_mad, bad_by_std))
    bads = np.asarray(ch_names)[bad_idxs.astype(int)]

    bads = [i[0] for i in bads]
    return bads


def find_bad_by_deviation(X, ch_names, robust_deviation_threshold=5.):
    """Detect channels that contain extreme amplitudes.

    NOTE: Assumes EEG data to be measured in microvolts.

    Parameters
    ----------
    X : ndarray, shape(n, t)
        2D input data.

    ch_names : list, len n
        List of channel names corresponding to rows in input data.

    robust_deviation_threshold : float
        The threshold for z-scores, when exceeded: classify as bad.

    Returns
    -------
    bads : list
        List of bad channel names.

    """
    assert X.ndim == 2
    # Calculate robust z-score of robust standard deviation for each channel
    chn_devi = 0.7413 * iqr(X, axis=1)
    chn_devi_sd = 0.7413 * iqr(chn_devi, axis=0)
    chn_devi_median = np.median(chn_devi)
    robust_chn_devi = (chn_devi - chn_devi_median) / chn_devi_sd

    # z-scores exceeding our theshold are classified as bad
    bad_idxs_bool = (np.abs(robust_chn_devi) >
                     robust_deviation_threshold)
    bad_idxs = np.argwhere(bad_idxs_bool)
    bads = np.asarray(ch_names)[bad_idxs.astype(int)]

    bads = [i[0] for i in bads]
    return bads


def find_bad_by_hf_noise(X, X_bp, ch_names, high_frequency_noise_threshold=5.):
    """Detect channels that contain high frequency (hf) noise.

    Parameters
    ----------
    X : ndarray, shape(n, t)
        2D input data.

    X_bp : ndarray, shape(n, t)
        2D input data X but band-pass filtered between 1Hz and 50Hz.

    ch_names : list, len n
        List of channel names corresponding to rows in input data.

    high_frequency_noise_threshold : float
        The threshold for z-scores, when exceeded: classify as bad.

    Returns
    -------
    bads : list
        List of bad channel names.

    """
    assert X.ndim == 2
    assert X_bp.ndim == 2
    # Determine z-scored level of estimated signal-to-noise
    # ratio for each channel
    noisiness = (mad(X - X_bp, axis=1) /
                 mad(X_bp, axis=1))
    noisiness_median = np.median(noisiness)
    noisiness_sd = mad(noisiness, axis=0) * 1.4826
    zscore_hf_noise_temp = (noisiness - noisiness_median) / noisiness_sd
    bad_idxs_bool = zscore_hf_noise_temp > high_frequency_noise_threshold
    bad_idxs = np.argwhere(bad_idxs_bool)
    bads = np.asarray(ch_names)[bad_idxs.astype(int)]

    bads = [i[0] for i in bads]
    return bads


def find_bad_by_correlation(X, X_bp, ch_names, srate,
                            correlation_threshold=0.4,
                            bad_time_threshold=0.1,
                            correlation_window_seconds=1.):
    """Detect channels that do not correlate well with the other channels.

    Parameters
    ----------
    X : ndarray, shape(n, t)
        2D input data.

    X_bp : ndarray, shape(n, t)
        2D input data X but band-pass filtered between 1Hz and 50Hz.

    ch_names : list, len n
        List of channel names corresponding to rows in input data.

    srate : float or int
        Sampling rate of X.

    correlation_threshold : float
        The minimum correlation threshold that should be attained within a
        data window.

    bad_time_threshold : float
        If this percentage of all data windows in which the correlation
        threshold was not surpassed is exceeded, classify a channel as bad.

    correlation_window_seconds : float or int
        Size of the correlation window in seconds.


    Returns
    -------
    bads : list
        List of bad channel names.

    """
    # Based on the data, determine how many windows we need
    # and how large they should be
    n_chans, signal_size = X.shape
    correlation_frames = correlation_window_seconds * srate
    correlation_window = np.arange(0, correlation_frames)
    n = correlation_window.shape[0]
    correlation_offsets = np.arange(0, (signal_size - correlation_frames),
                                    correlation_frames)
    w_correlation = correlation_offsets.shape[0]

    # preallocate
    channel_correlations = np.ones((w_correlation, n_chans))
    # noise_levels = np.zeros((w_correlation, n_chans))
    # chn_devis = np.zeros((w_correlation, n_chans))

    # Cut the data indo windows
    X_bp_window = X_bp[:n_chans, :n*w_correlation]
    X_bp_window = X_bp_window.reshape(n_chans, n, w_correlation)

    # X_window = X[:n_chans, :n*w_correlation]
    # X_window = X_window.reshape(n_chans, n, w_correlation)

    # Perform Pearson correlations across channels per window
    # For each channel, take the absolute of the 98th percentile of
    # correlations with the other channels as a measure of how well
    # correlated that channel is with the others.
    for k in range(w_correlation):
        eeg_portion = X_bp_window[:, :, k]
        # data_portion = X_window[:, :, k]
        window_correlation = np.corrcoef(eeg_portion)
        abs_corr = np.abs((window_correlation -
                           np.diag(np.diag(window_correlation))))
        channel_correlations[k, :] = np.percentile(abs_corr, 98, axis=0)

        """ # Not needed for now ...
        noise_levels[k, :] = (mad((data_portion - eeg_portion), axis=1) -
                              mad(eeg_portion, axis=1))
        chn_devis[k, :] = 0.7413 * iqr(data_portion, axis=1)
        """

    # Perform thresholding to see which channels correlate badly with the
    # other channels in a certain fraction of windows (bad_time_threshold)
    thresholded_correlations = channel_correlations < correlation_threshold
    fraction_bad_correlation_windows = np.mean(thresholded_correlations,
                                               axis=0)

    # find the corresponding channel names and return
    bad_idxs_bool = fraction_bad_correlation_windows > bad_time_threshold
    bad_idxs = np.argwhere(bad_idxs_bool)
    bads = np.asarray(ch_names)[bad_idxs.astype(int)]

    bads = [i[0] for i in bads]
    return bads


def find_bad_by_ransac(raw_mne, exclude_bads=[],
                       ransac_channel_fraction=0.25,
                       ransac_sample_size=50,
                       ransac_corr_thresh=0.75,
                       ransac_bad_time_thresh=0.4,
                       ransac_window_s=5.):
    """Detect channels that are not predicted well by the other channels.

    Parameters
    ----------
    raw_mne : mne raw object
        2D input data X, band-pass filtered between 1Hz and 50Hz. Noisy
        channels already removed.

    exclude_bads : list
        Channels previously identified to be bad. Should be excluded from
        ransac.

    ransac_channel_fraction : 0.1<=float<=0.9
        Fraction of channels used for robust reconstruction of the signal.
        Defaults to 0.25

    ransac_sample_size : int
        Number of samples used for computation of ransac. Defaults to 50.

    ransac_corr_thresh : float
        The minimum correlation threshold that should be attained within a
        data window. Defaults to 0.75

    ransac_bad_time_thresh : float
        If this percentage of all data windows in which the correlation
        threshold was not surpassed is exceeded, classify a channel as bad.
        Defaults to 0.4

    ransac_window_s : float or int
        Size of the correlation window in seconds. Defaults to 5.


    Returns
    -------
    bads : list
        List of bad channel names.

    """
    # Exclude previously identified bad channels
    raw = raw_mne.copy()
    raw.drop_channels(exclude_bads)

    # Save the true data ... later, we will interpolate and
    # then always come back to the true data
    raw.pick_types(eeg=True, stim=False)
    raw_true_data = raw.get_data()

    # Get some information
    ch_names = raw.ch_names
    n_chans, signal_size = raw.get_data().shape
    srate = raw.info['sfreq']
    ransac_subset = int(np.ceil(ransac_channel_fraction * n_chans))

    if (n_chans < ransac_subset+1) or (n_chans < 3) or (ransac_subset < 2):
        raise IOError('Too many channels have'
                      ' failed quality tests to perform ransac.')

    # Check memory: Can load all data into memory --> MEDIAN
    # else ... Initialize a remedian class: memory efficient approximation
    # of the median, see: https://github.com/sappelhoff/remedian
    available_mb = psutil.virtual_memory().available * 1e-6
    raw_size_mb = raw.get_data().nbytes() * 1e-6
    required_mb = raw_size_mb * ransac_sample_size
    if available_mb > required_mb:
        use_remedian = False
        median_eeg = np.zeros((n_chans, signal_size, ransac_sample_size))
    else:
        use_remedian = True
        n_obs = int(np.floor(available_mb / raw_size_mb))
        median_eeg = Remedian((n_chans, signal_size), n_obs,
                              ransac_sample_size)

    # Perform Ransac
    for sample in range(ransac_sample_size):
        # Assign a subset of all channels "bad" label
        # to be interpolated.
        bad_subset = np.random.choice(ch_names, (n_chans - ransac_subset),
                                      replace=False)

        raw.info['bads'] += list(bad_subset)

        # Interpolate the "bad" channels, thereby resetting their
        # status to "good"
        raw.interpolate_bads(reset_bads=True)

        # Form the median of the interpolated data across all iterations
        # Either true median or remedian approach depending on available
        # memory
        if not use_remedian:
            median_eeg[:, :, sample] = raw.get_data()
        else:  # use_remedian
            median_eeg.add_obs(raw.get_data())

        # Before starting with next iteration, reset the interpolated
        # data to the true data
        mygen = help_gen(raw_true_data)
        myfun = partial(help_fun, gen=mygen)
        raw.apply_function(myfun)

    # Get the median based on the collected data
    if not use_remedian:
        median_data = np.median(median_eeg, axis=-1, overwrite_input=True)
    else:
        median_data = median_eeg.remedian

    # Now check the predictability
    bads = find_bad_by_correlation(raw_true_data,
                                   median_data,
                                   ch_names=ch_names,
                                   srate=srate,
                                   correlation_threshold=ransac_corr_thresh,
                                   bad_time_threshold=ransac_bad_time_thresh,
                                   correlation_window_seconds=ransac_window_s)

    return bads


def find_noisy_channels(raw_mne):
    """Find noisy channels emplying several methods.

    This function tries to implement the findNoisyChannels function of the
    PREP as described in [1]. Currently, only default parameters are
    supported.

    Note: This function might need a minute to run.

    Parameters
    ----------
    raw_mne : mne python raw object

    Returns
    -------
    bads : list
        List of bad channel names.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M.,
       & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing
       for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16.
       https://doi.org/10.3389/fninf.2015.00016

    """
    raw = raw_mne.copy()

    # Add a montage
    montage = mne.channels.read_montage(kind='standard_1020',
                                        ch_names=raw.ch_names)
    raw.set_montage(montage)

    # Explicitly DO NOT set a new reference ...
    mne.set_eeg_reference(raw, ref_channels=[], verbose=False)

    # Reject the stim channel
    raw.pick_types(eeg=True, stim=False)

    # We work on 1Hz highpass filtered data ...
    # in original PREP, also line noise is removed
    raw.filter(l_freq=1.,
               h_freq=None,
               fir_design='firwin',
               n_jobs=n_workers)

    # Assuming the mne object to be measured in VOLTS
    # ... we need to convert to microvolts
    X = raw.get_data() * 10e6
    ch_names = raw.ch_names
    srate = raw.info['sfreq']

    # We also needraw a bandpass filtered version of the data:
    # Remove signal content above 50Hz
    # Below 1Hz was removed beforehand (see above)
    raw.filter(l_freq=None,
               h_freq=50.,
               fir_design='firwin',
               n_jobs=n_workers)

    # Again, convert to microvolts
    X_bp = raw.get_data() * 10e6

    # Find all bad channels emplying several methods
    all_bads = []

    bads = find_bad_by_nan(X, ch_names)
    all_bads += bads

    bads = find_bad_by_flat(X, ch_names)
    all_bads += bads

    bads = find_bad_by_deviation(X, ch_names)
    all_bads += bads

    bads = find_bad_by_hf_noise(X, X_bp, ch_names)
    all_bads += bads

    bads = find_bad_by_correlation(X, X_bp, ch_names, srate)
    all_bads += bads

    # Find bad by ransac by removing all bad channels found previously
    all_bads = list(set(all_bads))
    bads = find_bad_by_ransac(raw, exclude_bads=all_bads)
    all_bads += bads

    return all_bads
