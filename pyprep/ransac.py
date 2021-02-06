"""RANSAC bad channel identification."""
import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix
from mne.utils import check_random_state

from pyprep.utils import split_list, verify_free_ram


def find_bad_by_ransac(
    data,
    sample_rate,
    signal_len,
    complete_chn_labs,
    chn_pos,
    exclude,
    n_samples=50,
    fraction_good=0.25,
    corr_thresh=0.75,
    fraction_bad=0.4,
    corr_window_secs=5.0,
    channel_wise=False,
    random_state=None,
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
    data : np.ndarray
        2-D EEG data, should be detrended.
    sample_rate : float
        sample rate of the EEG data
    signal_len : float
        number of total samples in the signal (the length of the signal).
    complete_chn_labs : array_like
        labels of the channels in data in the same order
    chn_pos : np.ndarray
        3-D coordinates of all the channels in the order of data
    exclude : list
        labels of the channels to ignore in the ransac. In example bad channels
        from other methods.
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

    Returns
    -------
    bad_by_ransac : list
        List of channels labels marked bad by ransac.
    channel_correlations : np.ndarray
        Array of shape (windows,channels) holding the correlations of
        the channels to their predicted ransac value in each of the windows.

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
    good_chn_labs = complete_chn_labs[good_idx]
    n_chans_good = good_idx.shape[0]
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

    # Before running, make sure we have enough memory when using the
    # smallest possible chunk size
    verify_free_ram(data, n_samples, 1)

    # Generate random channel picks for each RANSAC sample
    random_ch_picks = []
    good_chans = np.arange(chn_pos_good.shape[0])
    rng = check_random_state(random_state)
    for i in range(n_samples):
        # Pick a random subset of clean channels to use for interpolation
        picks = rng.choice(good_chans, size=n_pred_chns, replace=False)
        random_ch_picks.append(picks)

    # Correlation windows setup
    correlation_frames = corr_window_secs * sample_rate
    correlation_window = np.arange(correlation_frames)
    n = correlation_window.shape[0]
    correlation_offsets = np.arange(
        0, (signal_len - correlation_frames), correlation_frames
    )
    w_correlation = correlation_offsets.shape[0]

    # Preallocate
    n_chans_complete = len(complete_chn_labs)
    channel_correlations = np.ones((w_correlation, n_chans_complete))
    # Notice self.EEGData.shape[0] = self.n_chans_new
    # Is now data.shape[0] = n_chans_complete
    # They came from the same drop of channels

    print("Executing RANSAC\nThis may take a while, so be patient...")

    # Calculate smallest chunk size for each possible chunk count
    chunk_sizes = []
    chunk_count = 0
    for i in range(1, n_chans_complete + 1):
        n_chunks = int(np.ceil(n_chans_complete / i))
        if n_chunks != chunk_count:
            chunk_count = n_chunks
            chunk_sizes.append(i)

    chunk_size = chunk_sizes.pop()
    mem_error = True
    job = list(range(n_chans_complete))

    if channel_wise:
        chunk_size = 1
    while mem_error:
        try:
            channel_chunks = split_list(job, chunk_size)
            total_chunks = len(channel_chunks)
            current = 1
            for chunk in channel_chunks:
                channel_correlations[:, chunk] = _ransac_correlations(
                    chunk,
                    random_ch_picks,
                    chn_pos,
                    chn_pos_good,
                    good_chn_labs,
                    complete_chn_labs,
                    data,
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
            if len(chunk_sizes):
                chunk_size = chunk_sizes.pop()
            else:  # pragma: no cover
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
    bad_ransac_channels_name = complete_chn_labs[bad_ransac_channels_idx.astype(int)]
    bad_by_ransac = [i[0] for i in bad_ransac_channels_name]
    print("\nRANSAC done!")

    return bad_by_ransac, channel_correlations


def _ransac_correlations(
    chans_to_predict,
    random_ch_picks,
    chn_pos,
    chn_pos_good,
    good_chn_labs,
    complete_chn_labs,
    data,
    n_samples,
    n,
    w_correlation,
):
    """Get correlations of channels to their ransac predicted values.

    Parameters
    ----------
    chans_to_predict : list of int
        Indexes of the channels to predict as they appear in chn_pos.
    random_ch_picks : list
        each element is a list of indexes of the channels (as they appear
        in good_chn_labs) to use for reconstruction in each of the samples.
    chn_pos : np.ndarray
        3-D coordinates of the electrode positions to predict
    chn_pos_good : np.ndarray
        3-D coordinates of all the channels not detected noisy so far
    good_chn_labs : array_like
        channel labels for the ch_pos_good channels
    complete_chn_labs : array_like
        labels of the channels in data in the same order
    data : np.ndarray
        2-D EEG data
    n_samples : int
        Number of samples used for computation of ransac.
    n : int
        Number of frames/samples of each window.
    w_correlation: int
        Number of windows.

    Returns
    -------
    channel_correlations : np.ndarray
        correlations of the given channels to their ransac predicted values.

    """
    # Preallocate
    channel_correlations = np.ones((w_correlation, len(chans_to_predict)))

    # Make the ransac predictions
    ransac_eeg = _run_ransac(
        n_samples=n_samples,
        random_ch_picks=random_ch_picks,
        chn_pos=chn_pos[chans_to_predict, :],
        chn_pos_good=chn_pos_good,
        good_chn_labs=good_chn_labs,
        complete_chn_labs=complete_chn_labs,
        data=data,
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


def _run_ransac(
    n_samples,
    random_ch_picks,
    chn_pos,
    chn_pos_good,
    good_chn_labs,
    complete_chn_labs,
    data,
):
    """Detect noisy channels apart from the ones described previously.

    It creates a random subset of the so-far good channels
    and predicts the values of the channels not in the subset.

    Parameters
    ----------
    n_samples : int
        number of interpolations from which a median will be computed
    random_ch_picks : list
        each element is a list of indexes of the channels (as they appear
        in good_chn_labs) to use for reconstruction in each of the samples.
    chn_pos : np.ndarray
        3-D coordinates of the electrode position
    chn_pos_good : np.ndarray
        3-D coordinates of all the channels not detected noisy so far
    good_chn_labs : array_like
        channel labels for the ch_pos_good channels
    complete_chn_labs : array_like
        labels of the channels in data in the same order
    data : np.ndarray
        2-D EEG data

    Returns
    -------
    ransac_eeg : np.ndarray
        The EEG data predicted by RANSAC

    """
    # n_chns, n_timepts = data.shape
    # 2 next lines should be equivalent but support single channel processing
    n_timepts = data.shape[1]
    n_chns = chn_pos.shape[0]

    # Before running, make sure we have enough memory
    verify_free_ram(data, n_samples, n_chns)

    # Memory seems to be fine ...
    # Make the predictions
    eeg_predictions = np.zeros((n_chns, n_timepts, n_samples))
    for sample in range(n_samples):
        # Get the random channel selection for the current sample
        reconstr_idx = random_ch_picks[sample]
        eeg_predictions[..., sample] = _get_ransac_pred(
            chn_pos, chn_pos_good, good_chn_labs, complete_chn_labs, reconstr_idx, data
        )

    # Form median from all predictions
    ransac_eeg = np.median(eeg_predictions, axis=-1, overwrite_input=True)
    return ransac_eeg


def _get_ransac_pred(
    chn_pos, chn_pos_good, good_chn_labs, complete_chn_labs, reconstr_idx, data
):
    """Perform RANSAC prediction.

    Parameters
    ----------
    chn_pos : np.ndarray
        3-D coordinates of the electrode position
    chn_pos_good : np.ndarray
        3-D coordinates of all the channels not detected noisy so far
    good_chn_labs : array_like
        channel labels for the ch_pos_good channels
    complete_chn_labs : array_like
        labels of the channels in data in the same order
    reconstr_idx : array_like
        indexes of the channels in good_chn_labs to use for reconstruction
    data : np.ndarray
        2-D EEG data

    Returns
    -------
    ransac_pred : np.ndarray
        Single RANSAC prediction

    """
    # Get positions and according labels
    reconstr_labels = good_chn_labs[reconstr_idx]
    reconstr_pos = chn_pos_good[reconstr_idx, :]

    # Map the labels to their indices within the complete data
    # Do not use mne.pick_channels, because it will return a sorted list.
    reconstr_picks = [
        list(complete_chn_labs).index(chn_lab) for chn_lab in reconstr_labels
    ]

    # Interpolate
    interpol_mat = _make_interpolation_matrix(reconstr_pos, chn_pos)
    ransac_pred = np.matmul(interpol_mat, data[reconstr_picks, :])
    return ransac_pred
