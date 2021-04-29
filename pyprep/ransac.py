"""RANSAC bad channel identification."""
import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix
from mne.utils import check_random_state

from pyprep.utils import (
    split_list, verify_free_ram, _get_random_subset, _mat_round, _correlate_arrays
)


def find_bad_by_ransac(
    data,
    sample_rate,
    complete_chn_labs,
    chn_pos,
    exclude,
    n_samples=50,
    fraction_good=0.25,
    corr_thresh=0.75,
    fraction_bad=0.4,
    corr_window_secs=5.0,
    channel_wise=False,
    window_wise=False,
    random_state=None,
    matlab_strict=False,
):
    """Detect channels that are not predicted well by other channels.

    Here, a RANSAC approach (see [1]_, and a short discussion in [2]_) is
    adopted to predict a "clean EEG" dataset. After identifying clean EEG
    channels through the other methods, the clean EEG dataset is
    constructed by repeatedly sampling a small subset of clean EEG channels
    and interpolation the complete data. The median of all those
    repetitions forms the clean EEG dataset. In a second step, the original
    and the RANSAC-predicted data are correlated and channels, which do not
    correlate well with themselves across the two datasets are considered
    `bad_by_ransac`.

    Parameters
    ----------
    data : np.ndarray
        A 2-D array of detrended EEG data, with bad-by-flat and bad-by-NaN
        channels removed.
    sample_rate : float
        The sample rate (in Hz) of the EEG data.
    complete_chn_labs : array_like
        Labels for all channels in `data`, in the same order as they appear
        in `data`.
    chn_pos : np.ndarray
        3-D electrode coordinates for all channels in `data`, in the same order
        as they appear in `data`.
    exclude : list
        Labels of channels to exclude as signal predictors during RANSAC
        (i.e., channels already flagged as bad by metrics other than HF noise).
    n_samples : int, optional
        Number of random channel samples to use for RANSAC. Defaults to ``50``.
    sample_prop : float, optional
        Proportion of total channels to use for signal prediction per RANSAC
        sample. This needs to be in the range [0, 1], where 0 would mean no
        channels would be used and 1 would mean all channels would be used
        (neither of which would be useful values). Defaults to ``0.25`` (e.g.,
        16 channels per sample for a 64-channel dataset).
    corr_thresh : float, optional
        The minimum predicted vs. actual signal correlation for a channel to
        be considered good within a given RANSAC window. Defaults to ``0.75``.
    fraction_bad : float, optional
        The minimum fraction of bad (i.e., below-threshold) RANSAC windows for a
        channel to be considered bad-by-RANSAC. Defaults to ``0.4``.
    corr_window_secs : float, optional
        The duration (in seconds) of each RANSAC correlation window. Defaults to
        5 seconds.
    channel_wise : bool, optional
        Whether RANSAC should be performed one channel at a time (lower RAM
        demands) or in chunks of as many channels as can fit into the currently
        available RAM (faster). Defaults to ``False`` (i.e., using the faster
        method).
    random_state : {int, None, np.random.RandomState}, optional
        The random seed with which to generate random samples of channels during
        RANSAC. If random_state is an int, it will be used as a seed for RandomState.
        If ``None``, the seed will be obtained from the operating system
        (see RandomState for details). Defaults to ``None``.
    matlab_strict : bool, optional
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code
        (see :ref:`matlab-diffs` for more details). Defaults to ``False``.

    Returns
    -------
    bad_by_ransac : list
        List containing the labels of all channels flagged as bad by RANSAC.
    channel_correlations : np.ndarray
        Array of shape (windows, channels) containing the correlations of
        the channels with their predicted RANSAC values for each window.

    References
    ----------
    .. [1] Fischler, M.A., Bolles, R.C. (1981). Random rample consensus: A
        Paradigm for Model Fitting with Applications to Image Analysis and
        Automated Cartography. Communications of the ACM, 24, 381-395
    .. [2] Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
        (2017). Autoreject: Automated Artifact Rejection for MEG and EEG
        Data. NeuroImage, 159, 417-429

    """
    # First, check that the argument types are valid
    if type(n_samples) != int:
        err = "Argument 'n_samples' must be an int (got {0})"
        raise TypeError(err.format(type(n_samples).__name__))

    # Get all channel positions and the position subset of "clean channels"
    # Exclude should be the bad channels from other methods
    # That is, identify all bad channels by other means
    good_idx = mne.pick_channels(list(complete_chn_labs), include=[], exclude=exclude)
    n_chans_good = good_idx.shape[0]
    chn_pos_good = chn_pos[good_idx, :]

    # Check if we have enough remaining channels
    # after exclusion of bad channels
    n_chans = data.shape[0]
    n_pred_chns = int(np.around(fraction_good * n_chans))

    if n_pred_chns <= 3:
        sample_pct = int(fraction_good * 100)
        e = "Too few channels in the original data to reliably perform RANSAC "
        e += "(minimum {0} for a sample size of {1}%)."
        raise IOError(e.format(int(np.floor(4.0 / fraction_good)), sample_pct))
    elif n_chans_good < (n_pred_chns + 1):
        e = "Too many noisy channels in the data to reliably perform RANSAC "
        e += "(only {0} good channels remaining, need at least {1})."
        raise IOError(e.format(n_chans_good, n_pred_chns + 1))

    # Before running, make sure we have enough memory when using the
    # smallest possible chunk size
    if window_wise:
        window_size = int(sample_rate * corr_window_secs)
        verify_free_ram(data[:, :window_size], n_samples, n_chans_good)
    else:
        verify_free_ram(data, n_samples, 1)

    # Generate random channel picks for each RANSAC sample
    random_ch_picks = []
    good_chans = np.arange(chn_pos_good.shape[0])
    rng = check_random_state(random_state)
    for i in range(n_samples):
        # Pick a random subset of clean channels to use for interpolation
        picks = _get_random_subset(good_chans, n_pred_chns, rng)
        random_ch_picks.append(picks)

    # Generate interpolation matrix for each RANSAC sample
    interp_mats = _make_interpolation_matrices(random_ch_picks, chn_pos_good)

    # Calculate the size (in frames) and count of correlation windows
    correlation_frames = corr_window_secs * sample_rate
    signal_frames = data.shape[1]
    correlation_offsets = np.arange(
        0, (signal_frames - correlation_frames), correlation_frames
    )
    win_size = int(correlation_frames)
    win_count = correlation_offsets.shape[0]

    # Preallocate RANSAC correlation matrix
    n_chans_complete = len(complete_chn_labs)
    channel_correlations = np.ones((win_count, n_chans_complete))
    # Notice self.EEGData.shape[0] = self.n_chans_new
    # Is now data.shape[0] = n_chans_complete
    # They came from the same drop of channels

    print("Executing RANSAC\nThis may take a while, so be patient...")

    # If enabled, run window-wise RANSAC
    if window_wise:
        # Get correlations between actual vs predicted signals for each RANSAC window
        channel_correlations[:, good_idx] = _ransac_by_window(
            data[good_idx, :], interp_mats, win_size, win_count, matlab_strict
        )

    # Calculate smallest chunk size for each possible chunk count
    chunk_sizes = []
    chunk_count = 0
    for i in range(1, n_chans_good + 1):
        n_chunks = int(np.ceil(n_chans_good / i))
        if n_chunks != chunk_count:
            chunk_count = n_chunks
            chunk_sizes.append(i)

    chunk_size = 1 if channel_wise else chunk_sizes.pop()
    mem_error = True
    job = list(range(n_chans_good))

    # If not using window-wise RANSAC, do channel-wise RANSAC
    while mem_error and not window_wise:
        try:
            channel_chunks = split_list(job, chunk_size)
            total_chunks = len(channel_chunks)
            current = 1
            for chunk in channel_chunks:
                interp_mats_for_chunk = [mat[chunk, :] for mat in interp_mats]
                channel_correlations[:, good_idx[chunk]] = _ransac_by_channel(
                    chunk,
                    random_ch_picks,
                    interp_mats_for_chunk,
                    data[good_idx, :],
                    win_size,
                    win_count,
                    matlab_strict,
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
            if len(chunk_sizes):
                chunk_size = chunk_sizes.pop()
            else:  # pragma: no cover
                raise MemoryError(
                    "Not even doing 1 channel at a time the data fits in ram..."
                    "You could downsample the data or reduce the number of requ"
                    "ested samples."
                )

    # Calculate fractions of bad RANSAC windows for each channel
    thresholded_correlations = channel_correlations < corr_thresh
    frac_bad_corr_windows = np.mean(thresholded_correlations, axis=0)

    # find the corresponding channel names and return
    bad_ransac_channels_idx = np.argwhere(frac_bad_corr_windows > fraction_bad)
    bad_ransac_channels_name = complete_chn_labs[bad_ransac_channels_idx.astype(int)]
    bad_by_ransac = [i[0] for i in bad_ransac_channels_name]
    print("\nRANSAC done!")

    return bad_by_ransac, channel_correlations


def _make_interpolation_matrices(random_ch_picks, chn_pos_good):
    """Create an interpolation matrix for each RANSAC sample of channels.

    This function takes the spatial coordinates of random subsets of currently-good
    channels and uses them to predict what the signal will be at the spatial
    coordinates of all other currently-good channels. The results of this process are
    returned as matrices that can be multiplied with EEG data to generate predicted
    signals.

    Parameters
    ----------
    random_ch_picks : list of list of int
        A list containing multiple random subsets of currently-good channels.
    chn_pos_good : np.ndarray
        3-D spatial coordinates of all currently-good channels.

    Returns
    -------
    interpolation_mats : list of np.ndarray
        A list of interpolation matrices, one for each random subset of channels.
        Each matrix has the shape `[num_good_channels, num_good_channels]`, with the
        number of good channels being inferred from the size of `ch_pos_good`.

    Notes
    -----
    This function currently makes use of a private MNE function,
    ``mne.channels.interpolation._make_interpolation_matrix``, to generate matrices.

    """
    n_chans_good = chn_pos_good.shape[0]

    interpolation_mats = []
    for sample in random_ch_picks:
        mat = np.zeros((n_chans_good, n_chans_good))
        subset_pos = chn_pos_good[sample, :]
        mat[:, sample] = _make_interpolation_matrix(subset_pos, chn_pos_good)
        interpolation_mats.append(mat)

    return interpolation_mats


def _ransac_by_window(data, interpolation_mats, win_size, win_count, matlab_strict):
    """Calculate correlations of channels with their RANSAC-predicted values.

    This function calculates RANSAC correlations for each RANSAC window
    individually, requiring RAM equivalent to [channels * sample rate * seconds
    per RANSAC window] to run. Generally, this method will use less RAM than
    :func:`_ransac_by_channel`, with the exception of short recordings with high
    electrode counts.

    Parameters
    ----------
    data : np.ndarray
        A 2-D array containing the EEG signals from currently-good channels.
    interpolation_mats : list of np.ndarray
        A list of interpolation matrices, one for each RANSAC sample of channels.
    win_size : int
        Number of frames/samples of EEG data in each RANSAC correlation window.
    win_count: int
        Number of RANSAC correlation windows.

    Returns
    -------
    channel_correlations : np.ndarray
        Correlations of the given channels to their predicted values within each
        RANSAC window.
    """
    ch_count = data.shape[0]
    ch_correlations = np.ones((win_count, ch_count))

    for window in range(win_count):

        # Get the current window of EEG data
        start = window * win_size
        end = (window + 1) * win_size
        actual = data[:, start:end]

        # Get the median RANSAC-predicted signal for each channel
        predicted = _predict_median_signals(actual, interpolation_mats, matlab_strict)

        # Calculate the actual vs predicted signal correlation for each channel
        ch_correlations[window, :] = _correlate_arrays(actual, predicted, matlab_strict)

    return ch_correlations


def _predict_median_signals(window, interpolation_mats, matlab_strict=False):
    """Calculate the median RANSAC-predicted signal for a given window of data.

    Parameters
    ----------
    window : np.ndarray
        A 2-D window of EEG data with the shape `[channels, samples]`.
    interpolation_mats : list of np.ndarray
        A set of channel interpolation matrices, one for each RANSAC sample of
        channels.
    matlab_strict : bool, optional
        Whether MATLAB PREP's internal logic should be strictly followed (see Notes).
        Defaults to False.

    Returns
    -------
    predicted : np.ndarray
        The median RANSAC-predicted EEG signal for the given window of data.

    Notes
    -----
    In MATLAB PREP, the median signal is calculated by sorting the different
    predictions for each EEG sample/channel from low to high and then taking the value
    at the middle index (as calculated by `int(n_ransac_samples / 2.0)`) for each.
    Because this logic only returns the correct result for odd numbers of samples, the
    current function will instead return the true median signal across predictions
    unless strict MATLAB equivalence is requested.

    """
    ransac_samples = len(interpolation_mats)
    merged_mats = np.concatenate(interpolation_mats, axis=0)

    predictions_per_sample = np.reshape(
        np.matmul(merged_mats, window),
        (ransac_samples, window.shape[0], window.shape[1])
    )

    if matlab_strict:
        # Match MATLAB's rounding logic (.5 always rounded up)
        median_idx = int(_mat_round(ransac_samples / 2.0) - 1)
        predictions_per_sample.sort(axis=0)
        return predictions_per_sample[median_idx, :, :]
    else:
        return np.median(predictions_per_sample, axis=0)


def _ransac_by_channel(
    chans_to_predict,
    random_ch_picks,
    interpolation_mats,
    data,
    n,
    w_correlation,
    matlab_strict,
):
    """Calculate correlations of channels with their RANSAC-predicted values.

    This function calculates RANSAC correlations on one (or more) full channels
    at once, requiring RAM equivalent to [channels per chunk * sample rate *
    length of recording in seconds] to run. Generally, this method will use
    more RAM than :func:`_ransac_by_window`, but may be faster for systems with
    large amounts of RAM.

    Parameters
    ----------
    chans_to_predict : list of int
        Indices of the channels to predict (as they appear in `data`).
    random_ch_picks : list
        Each element is a list of indexes of the channels (as they appear
        in `data`) to use for reconstruction in each of the samples.
    interpolation_mats : list of np.ndarray
        A set of channel interpolation matrices, one for each RANSAC sample of
        channels.
    data : np.ndarray
        2-D EEG data
    n : int
        Number of frames/samples of each window.
    w_correlation: int
        Number of windows.
    matlab_strict : bool
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.

    Returns
    -------
    channel_correlations : np.ndarray
        correlations of the given channels to their RANSAC-predicted values.

    """
    # Preallocate
    channel_correlations = np.ones((w_correlation, len(chans_to_predict)))

    # Make the ransac predictions
    ransac_eeg = _predict_median_signals_channelwise(
        chans_to_predict=chans_to_predict,
        random_ch_picks=random_ch_picks,
        interpolation_mats=interpolation_mats,
        data=data,
        matlab_strict=matlab_strict,
    )

    # Correlate ransac prediction and eeg data

    # For the actual data
    data_window = data[chans_to_predict, : n * w_correlation]
    data_window = data_window.reshape(len(chans_to_predict), w_correlation, n)
    data_window = data_window.swapaxes(1, 0)

    # For the ransac predicted eeg
    pred_window = ransac_eeg[: len(chans_to_predict), : n * w_correlation]
    pred_window = pred_window.reshape(len(chans_to_predict), w_correlation, n)
    pred_window = pred_window.swapaxes(1, 0)

    # Perform correlations
    for k in range(w_correlation):
        data_portion = data_window[k, :, :]
        pred_portion = pred_window[k, :, :]
        R = _correlate_arrays(data_portion, pred_portion, matlab_strict)
        channel_correlations[k, :] = R

    return channel_correlations


def _predict_median_signals_channelwise(
    chans_to_predict,
    random_ch_picks,
    interpolation_mats,
    data,
    matlab_strict,
):
    """Calculate the median RANSAC-predicted signal for a given chunk of channels.

    Parameters
    ----------
    chans_to_predict : list of int
        Indices of the channels to predict (as they appear in `data`).
    random_ch_picks : list
        Each element is a list of indexes of the channels (as they appear
        in `data`) to use for reconstruction in each of the samples.
    interpolation_mats : list of np.ndarray
        A set of channel interpolation matrices, one for each RANSAC sample of
        channels.
    data : np.ndarray
        2-D EEG data
    matlab_strict : bool
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.

    Returns
    -------
    ransac_eeg : np.ndarray
        The EEG data predicted by RANSAC

    """
    # n_chns, n_timepts = data.shape
    # 2 next lines should be equivalent but support single channel processing
    ransac_samples = len(interpolation_mats)
    n_timepts = data.shape[1]
    n_chns = len(chans_to_predict)

    # Before running, make sure we have enough memory
    verify_free_ram(data, ransac_samples, n_chns)

    # Memory seems to be fine ...
    # Make the predictions
    eeg_predictions = np.zeros((n_chns, n_timepts, ransac_samples))
    for sample in range(ransac_samples):
        # Get the random channels & interpolation matrix for the current sample
        reconstr_idx = random_ch_picks[sample]
        interp_mat = interpolation_mats[sample][:, reconstr_idx]
        # Predict the EEG signals for the current RANSAC sample / channel chunk
        eeg_predictions[..., sample] = np.matmul(interp_mat, data[reconstr_idx, :])

    # Form median from all predictions
    if matlab_strict:
        # Match MATLAB's rounding logic (.5 always rounded up)
        median_idx = int(_mat_round(ransac_samples / 2.0) - 1)
        eeg_predictions.sort(axis=-1)
        ransac_eeg = eeg_predictions[:, :, median_idx]
    else:
        ransac_eeg = np.median(eeg_predictions, axis=-1, overwrite_input=True)

    return ransac_eeg
