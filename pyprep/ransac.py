"""RANSAC bad channel identification."""

import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix
from mne.utils import ProgressBar, check_random_state, logger

from pyprep.utils import (
    _correlate_arrays,
    _get_random_subset,
    _mat_round,
    _split_list,
    _verify_free_ram,
)


def find_bad_by_ransac(
    data,
    sample_rate,
    complete_chn_labs,
    chn_pos,
    exclude,
    n_samples=50,
    sample_prop=0.25,
    corr_thresh=0.75,
    frac_bad=0.4,
    corr_window_secs=5.0,
    channel_wise=False,
    max_chunk_size=None,
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
    frac_bad : float, optional
        The minimum fraction of bad (i.e., below-threshold) RANSAC windows for a
        channel to be considered bad-by-RANSAC. Defaults to ``0.4``.
    corr_window_secs : float, optional
        The duration (in seconds) of each RANSAC correlation window. Defaults to
        5 seconds.
    channel_wise : bool, optional
        Whether RANSAC should predict signals for chunks of channels over the
        entire signal length ("channel-wise RANSAC", see `max_chunk_size`
        parameter). If ``False``, RANSAC will instead predict signals for all
        channels at once but over a number of smaller time windows instead of
        over the entire signal length ("window-wise RANSAC"). Channel-wise
        RANSAC generally has higher RAM demands than window-wise RANSAC
        (especially if `max_chunk_size` is ``None``), but can be faster on
        systems with lots of RAM to spare. Defaults to ``False``.
    max_chunk_size : {int, None}, optional
        The maximum number of channels to predict at once during channel-wise
        RANSAC. If ``None``, RANSAC will use the largest chunk size that will
        fit into the available RAM, which may slow down other programs on the
        host system. If using window-wise RANSAC (the default), this parameter
        has no effect. Defaults to ``None``.
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
    .. [1] Fischler, M.A., Bolles, R.C. (1981). Random sample consensus: A
        Paradigm for Model Fitting with Applications to Image Analysis and
        Automated Cartography. Communications of the ACM, 24, 381-395
    .. [2] Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
        (2017). Autoreject: Automated Artifact Rejection for MEG and EEG
        Data. NeuroImage, 159, 417-429

    """
    # Check we find channel positions
    if np.isnan(chn_pos).any():
        raise ValueError(
            "Found NaN in channel positions. Did you supply a montage for the raw data?"
        )

    # First, check that the argument types are valid
    if not isinstance(n_samples, int):
        raise TypeError(
            f"Argument 'n_samples' must be an int (got {type(n_samples).__name__})"
        )

    complete_chn_labs = np.asarray(complete_chn_labs)

    # Get all channel positions and the position subset of "clean channels"
    # Exclude should be the bad channels from other methods
    # That is, identify all bad channels by other means
    good_idx = mne.pick_channels(list(complete_chn_labs), include=[], exclude=exclude)
    n_chans_good = good_idx.shape[0]
    chn_pos_good = chn_pos[good_idx, :]

    # Check if we have enough remaining channels
    # after exclusion of bad channels
    n_chans = data.shape[0]
    n_pred_chns = int(np.around(sample_prop * n_chans))

    if n_pred_chns <= 3:
        sample_pct = int(sample_prop * 100)
        raise OSError(
            "Too few channels in the original data to reliably perform RANSAC"
            f"(minimum {int(np.floor(4.0 / sample_prop))} for a sample size "
            f"of {sample_pct}%)."
        )
    elif n_chans_good < (n_pred_chns + 1):
        raise OSError(
            "Too many noisy channels in the data to reliably perform RANSAC "
            f"(only {n_chans_good} good channels remaining, "
            f"need at least {n_pred_chns + 1})."
        )

    # Before running, make sure we have enough memory when using the
    # smallest possible chunk size
    if channel_wise:
        _verify_free_ram(data, n_samples, 1)
    else:
        window_size = int(sample_rate * corr_window_secs)
        _verify_free_ram(data[:, :window_size], n_samples, n_chans_good)

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

    logger.info("Executing RANSAC\nThis may take a while, so be patient...")

    # If enabled, run window-wise RANSAC
    if not channel_wise:
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
            if not max_chunk_size or i <= max_chunk_size:
                chunk_sizes.append(i)

    chunk_size = chunk_sizes.pop()
    mem_error = True
    job = list(range(n_chans_good))

    # If not using window-wise RANSAC, do channel-wise RANSAC
    while mem_error and channel_wise:
        try:
            channel_chunks = _split_list(job, chunk_size)
            total_chunks = len(channel_chunks)
            current = 1
            for chunk in channel_chunks:
                interp_mats_for_chunk = [mat[chunk, :] for mat in interp_mats]
                channel_correlations[:, good_idx[chunk]] = _ransac_by_channel(
                    data[good_idx, :],
                    interp_mats_for_chunk,
                    win_size,
                    win_count,
                    chunk,
                    random_ch_picks,
                    matlab_strict,
                )
                if chunk == channel_chunks[0]:
                    # If it gets here, it means it is the optimal
                    logger.info("Finding optimal chunk size : %s", chunk_size)
                    logger.info("Total # of chunks: %s", total_chunks)
                    logger.info("Current chunk:")

                logger.info(current)
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
    bad_ransac_channels_idx = np.argwhere(frac_bad_corr_windows > frac_bad)
    bad_ransac_channels_name = complete_chn_labs[bad_ransac_channels_idx.astype(int)]
    bad_by_ransac = [i[0] for i in bad_ransac_channels_name]
    logger.info("\nRANSAC done!")

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
        A 2-D array containing the EEG signals from all currently-good channels.
    interpolation_mats : list of np.ndarray
        A list of interpolation matrices, one for each RANSAC sample of channels.
    win_size : int
        Number of frames/samples of EEG data in each RANSAC correlation window.
    win_count: int
        Number of RANSAC correlation windows.
    matlab_strict : bool
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.

    Returns
    -------
    correlations : np.ndarray
        Correlations of the given channels to their predicted values within each
        RANSAC window.

    """
    ch_count = data.shape[0]
    correlations = np.ones((win_count, ch_count))

    pb = ProgressBar(range(win_count))
    for window in pb:
        # Get the current window of EEG data
        start = window * win_size
        end = (window + 1) * win_size
        actual = data[:, start:end]

        # Get the median RANSAC-predicted signal for each channel
        predicted = _predict_median_signals(actual, interpolation_mats, matlab_strict)

        # Calculate the actual vs predicted signal correlation for each channel
        correlations[window, :] = _correlate_arrays(actual, predicted, matlab_strict)

    return correlations


def _predict_median_signals(window, interpolation_mats, matlab_strict=False):
    """Calculate the median RANSAC-predicted signal for a given window of data.

    Parameters
    ----------
    window : np.ndarray
        A 2-D window of EEG data with the shape `[channels, samples]`.
    interpolation_mats : list of np.ndarray
        A set of channel interpolation matrices, one for each RANSAC sample of
        channels.
    matlab_strict : bool
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.

    Returns
    -------
    predicted : np.ndarray
        The median RANSAC-predicted EEG signal for the given window of data.

    Notes
    -----
    In MATLAB PREP, the median signal is calculated by sorting the different
    predictions for each EEG sample/channel from low to high and then taking the value
    at the middle index (as calculated by ``int(n_ransac_samples / 2.0)``) for each.
    Because this logic only returns the correct result for odd numbers of samples, the
    current function will instead return the true median signal across predictions
    unless strict MATLAB equivalence is requested.

    """
    ransac_samples = len(interpolation_mats)
    merged_mats = np.concatenate(interpolation_mats, axis=0)

    predictions_per_sample = np.reshape(
        np.matmul(merged_mats, window),
        (ransac_samples, window.shape[0], window.shape[1]),
    )

    if matlab_strict:
        # Match MATLAB's rounding logic (.5 always rounded up)
        median_idx = int(_mat_round(ransac_samples / 2.0) - 1)
        predictions_per_sample.sort(axis=0)
        return predictions_per_sample[median_idx, :, :]
    else:
        return np.median(predictions_per_sample, axis=0)


def _ransac_by_channel(
    data,
    interpolation_mats,
    win_size,
    win_count,
    chans_to_predict,
    random_ch_picks,
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
    data : np.ndarray
        A 2-D array containing the EEG signals from all currently-good channels.
    interpolation_mats : list of np.ndarray
        A set of channel interpolation matrices, one for each RANSAC sample of
        channels.
    win_size : int
        Number of frames/samples of EEG data in each RANSAC correlation window.
    win_count: int
        Number of RANSAC correlation windows.
    chans_to_predict : list of int
        Indices of the channels to predict (as they appear in `data`) within the
        current chunk.
    random_ch_picks : list of list of int
        A list containing multiple random subsets of currently-good channels.
    matlab_strict : bool
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.

    Returns
    -------
    correlations : np.ndarray
        Correlations of the given channels to their predicted values within each
        RANSAC window.

    """
    # Preallocate RANSAC correlation matrix for current chunk
    chunk_size = len(chans_to_predict)
    correlations = np.ones((win_count, chunk_size))

    # Get median RANSAC predictions for each channel in the current chunk
    predicted_chans = _predict_median_signals_channelwise(
        data=data,
        interpolation_mats=interpolation_mats,
        random_ch_picks=random_ch_picks,
        chunk_size=len(chans_to_predict),
        matlab_strict=matlab_strict,
    )

    # Correlate ransac prediction and eeg data

    # For the actual data
    data_window = data[chans_to_predict, : win_size * win_count]
    data_window = data_window.reshape(chunk_size, win_count, win_size)
    data_window = data_window.swapaxes(1, 0)

    # For the ransac predicted eeg
    pred_window = predicted_chans[:chunk_size, : win_size * win_count]
    pred_window = pred_window.reshape(chunk_size, win_count, win_size)
    pred_window = pred_window.swapaxes(1, 0)

    # Perform correlations
    for k in range(win_count):
        data_portion = data_window[k, :, :]
        pred_portion = pred_window[k, :, :]
        R = _correlate_arrays(data_portion, pred_portion, matlab_strict)
        correlations[k, :] = R

    return correlations


def _predict_median_signals_channelwise(
    data,
    interpolation_mats,
    random_ch_picks,
    chunk_size,
    matlab_strict,
):
    """Calculate the median RANSAC-predicted signal for a given chunk of channels.

    Parameters
    ----------
    data : np.ndarray
        A 2-D array containing the EEG signals from all currently-good channels.
    interpolation_mats : list of np.ndarray
        A set of channel interpolation matrices, one for each RANSAC sample of
        channels.
    random_ch_picks : list of list of int
        A list containing multiple random subsets of currently-good channels.
    chunk_size : int
        The number of channels to predict in the current chunk.
    matlab_strict : bool
        Whether or not RANSAC should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.

    Returns
    -------
    predicted_chans : np.ndarray
        The median RANSAC-predicted EEG signals for the given chunk of channels.

    """
    # n_chns, n_timepts = data.shape
    # 2 next lines should be equivalent but support single channel processing
    ransac_samples = len(interpolation_mats)
    n_timepts = data.shape[1]

    # Before running, make sure we have enough memory
    _verify_free_ram(data, ransac_samples, chunk_size)

    # Memory seems to be fine ...
    # Make the predictions
    eeg_predictions = np.zeros((chunk_size, n_timepts, ransac_samples))
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
        return eeg_predictions[:, :, median_idx]
    else:
        return np.median(eeg_predictions, axis=-1, overwrite_input=True)
