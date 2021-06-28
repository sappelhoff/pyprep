"""Module contains frequently used functions dealing with channel lists."""
import math
from cmath import sqrt

import mne
import numpy as np
import scipy.interpolate
from mne.surface import _normalize_vectors
from numpy.polynomial.legendre import legval
from psutil import virtual_memory
from scipy import linalg
from scipy.signal import firwin, lfilter, lfilter_zi


def _union(list1, list2):
    return list(set(list1 + list2))


def _set_diff(list1, list2):
    return list(set(list1) - set(list2))


def _mat_round(x):
    """Round a number to the nearest whole number.

    Parameters
    ----------
    x : float
        The number to round.

    Returns
    -------
    rounded : float
        The input value, rounded to the nearest whole number.

    Notes
    -----
    MATLAB rounds all numbers ending in .5 up to the nearest integer, whereas
    Python (and Numpy) rounds them to the nearest even number. This function
    mimics MATLAB's behaviour.
    """
    return np.ceil(x) if x % 1 >= 0.5 else np.floor(x)


def _mat_quantile(arr, q, axis=None):
    """Calculate the numeric value at quantile (`q`) for a given distribution.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing samples from the distribution to summarize. Must
        be either a 1-D or 2-D array.
    q : float
        The quantile to calculate for the input data. Must be between 0 and 1,
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis along which quantile values should be calculated. Defaults to
        calculating the value at the given quantile for the entire array.

    Returns
    -------
    quantile : scalar or np.ndarray
        If no axis is specified, returns the value at quantile (q) for the full
        input array as a single numeric value. Otherwise, returns an
        ``np.ndarray`` containing the values at quantile (q) for each row along
        the specified axis.

    Notes
    -----
    MATLAB calculates quantiles using different logic than Numpy: Numpy treats
    the provided values as a whole population, whereas MATLAB treats them as a
    sample from a population of unknown size and adjusts quantiles accordingly.
    This function mimics MATLAB's logic to produce identical results.

    """
    # Sort the array in ascending order along the given axis (any NaNs go to the end)
    # Return NaN if array is empty.
    if len(arr) == 0:
        return np.NaN
    arr_sorted = np.sort(arr, axis=axis)

    # Ensure array is either 1D or 2D
    if arr_sorted.ndim > 2:
        e = "Only 1D and 2D arrays are supported (input has {0} dimensions)"
        raise ValueError(e.format(arr_sorted.ndim))

    # Reshape data into a 2D array with the shape (num_axes, data_per_axis)
    if axis is None:
        arr_sorted = arr_sorted.reshape(-1, 1)
    else:
        arr_sorted = np.moveaxis(arr_sorted, axis, 0)

    # Initialize quantile array with values for non-usable (n < 2) axes.
    # Sets quantile to only non-NaN value if n == 1, or NaN if n == 0
    quantiles = arr_sorted[0, :]

    # Get counts of non-NaN values for each axis and determine which have n > 1
    n = np.sum(np.isfinite(arr_sorted), axis=0)
    n_usable = n[n > 1]

    if np.any(n > 1):
        # Calculate MATLAB-style sample-adjusted quantile values
        q = np.asarray(q, dtype=np.float64)
        q_adj = ((q - 0.5) * n_usable / (n_usable - 1)) + 0.5

        # Get the exact (float) index position of the quantile for each usable axis, as
        # well as the indices of the values below and above it (if not a whole number)
        exact_idx = (n_usable - 1) * np.clip(q_adj, 0, 1)
        pre_idx = np.floor(exact_idx).astype(np.int32)
        post_idx = np.ceil(exact_idx).astype(np.int32)

        # Interpolate exact quantile values for each usable axis
        axis_idx = np.arange(len(n))[n > 1]
        pre = arr_sorted[pre_idx, axis_idx]
        post = arr_sorted[post_idx, axis_idx]
        quantiles[n > 1] = pre + (post - pre) * (exact_idx - pre_idx)

    return quantiles[0] if quantiles.size == 1 else quantiles


def _mat_iqr(arr, axis=None):
    """Calculate the inter-quartile range (IQR) for a given distribution.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing samples from the distribution to summarize.
    axis : {int, tuple of int, None}, optional
        Axis along which IQRs should be calculated. Defaults to calculating the
        IQR for the entire array.

    Returns
    -------
    iqr : scalar or np.ndarray
        If no axis is specified, returns the IQR for the full input array as a
        single numeric value. Otherwise, returns an ``np.ndarray`` containing
        the IQRs for each row along the specified axis.

    Notes
    -----
    See notes for :func:`utils._mat_quantile`.

    """
    return _mat_quantile(arr, 0.75, axis) - _mat_quantile(arr, 0.25, axis)


def _eeglab_create_highpass(cutoff, srate):
    """Create a high-pass FIR filter using Hamming windows.

    Parameters
    ----------
    cutoff : float
        The lower pass-band edge of the filter, in Hz.
    srate : float
        The sampling rate of the EEG signal, in Hz.

    Returns
    -------
    filter : np.ndarray
        A 1-dimensional array of FIR filter coefficients.

    Notes
    -----
    In MATLAB PREP, the internal ``removeTrend`` function uses EEGLAB's
    ``pop_eegfiltnew`` to high-pass the EEG data to remove slow drifts.
    Because MNE's ``mne.filter.filter_data`` and EEGLAB's ``pop_eegfiltnew``
    calculate filter parameters slightly differently, this function is
    used to precisely match EEGLAB & MATLAB PREP's filtering method.

    """
    TRANSITION_WIDTH_RATIO = 0.25
    HAMMING_CONSTANT = 3.3  # note: not entirely clear what this represents

    # Calculate parameters for constructing filter
    trans_bandwidth = cutoff if cutoff < 2 else cutoff * TRANSITION_WIDTH_RATIO
    order = HAMMING_CONSTANT / (trans_bandwidth / srate)
    order = int(np.ceil(order / 2) * 2)  # ensure order is even
    stop = cutoff - trans_bandwidth
    transition = (stop + cutoff) / srate

    # Generate highpass filter
    N = order + 1
    filt = np.zeros(N)
    filt[N // 2] = 1
    filt -= firwin(N, transition, window="hamming", nyq=1)
    return filt


def _eeglab_fir_filter(data, filt):
    """Apply an FIR filter to a 2-D array of EEG data.

    Parameters
    ----------
    data : np.ndarray
        A 2-D array of EEG data to filter.
    filt : np.ndarray
        A 1-D array of FIR filter coefficients.

    Returns
    -------
    filtered : np.ndarray
        A 2-D array of FIR-filtered EEG data.

    Notes
    -----
    Produces identical output to EEGLAB's ``firfilt`` function (for non-epoched
    data). For internal use within :mod:`pyprep.removeTrend`.

    """
    # Initialize parameters for FIR filtering
    frames_per_window = 2000
    group_delay = int((len(filt) - 1) / 2)
    n_samples = data.shape[1]
    n_windows = int(np.ceil((n_samples - group_delay) / frames_per_window))
    pad_len = min(group_delay, n_samples)

    # Prepare initial state of filter, using padding at start of data
    start_pad_idx = np.zeros(pad_len, dtype=np.uint8)
    start_padded = np.concatenate((data[:, start_pad_idx], data[:, :pad_len]), axis=1)
    zi_init = lfilter_zi(filt, 1) * np.take(start_padded, [0], axis=0)
    _, zi = lfilter(filt, 1, start_padded, axis=1, zi=zi_init)

    # Iterate over windows of signal, filtering in chunks
    out = np.zeros_like(data)
    for w in range(n_windows):
        start = group_delay + w * frames_per_window
        end = min(start + frames_per_window, n_samples)
        start_out = start - group_delay
        end_out = end - group_delay
        out[:, start_out:end_out], zi = lfilter(
            filt, 1, data[:, start:end], axis=1, zi=zi
        )

    # Finish filtering data, using padding at end to calculate final values
    end_pad_idx = np.zeros(pad_len, dtype=np.uint8) + (n_samples - 1)
    end, _ = lfilter(filt, 1, data[:, end_pad_idx], axis=1, zi=zi)
    out[:, (n_samples - pad_len) :] = end[:, (group_delay - pad_len) :]

    return out


def _eeglab_calc_g(pos_from, pos_to, stiffness=4, num_lterms=7):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The electrode positions to interpolate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The electrode positions to interpolate.
    stiffness : float
        Stiffness of the spline.
    num_lterms : int
        Number of Legendre terms to evaluate.

    Returns
    -------
    G : np.ndarray of float, shape(n_channels, n_channels)
        The G matrix.

    Notes
    -----
    Produces identical output to the private ``computeg`` function in EEGLAB's
    ``eeg_interp.m``.

    """
    # https://github.com/sccn/eeglab/blob/167dfc8/functions/popfunc/eeg_interp.m#L347

    n_to = pos_to.shape[0]
    n_from = pos_from.shape[0]

    # Calculate the Euclidian distances between the 'to' and 'from' electrodes
    dxyz = []
    for i in range(0, 3):
        d1 = np.repeat(pos_to[:, i], n_from).reshape((n_to, n_from))
        d2 = np.repeat(pos_from[:, i], n_to).reshape((n_from, n_to)).T
        dxyz.append((d1 - d2) ** 2)
    elec_dists = np.sqrt(sum(dxyz))

    # Subtract all the Euclidian electrode distances from 1 (why?)
    EI = np.ones([n_to, n_from]) - elec_dists

    # Calculate Legendre coefficients for the given degree and stiffness
    factors = [0]
    for n in range(1, num_lterms + 1):
        f = (2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness * 4 * np.pi)
        factors.append(f)

    return legval(EI, factors)


def _eeglab_interpolate(data, pos_from, pos_to):
    """Interpolate bad channels using EEGLAB's custom method.

    Parameters
    ----------
    data : np.ndarray
        A 2-D array containing signals from currently-good EEG channels with
        which to interpolate signals for bad channels.
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The electrode positions to interpolate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The electrode positions to interpolate.

    Returns
    -------
    interpolated : np.ndarray
        The interpolated signals for all bad channels.

    Notes
    -----
    Produces identical output to the private ``spheric_spline`` function in
    EEGLAB's ``eeg_interp.m`` (with minor rounding errors).

    """
    # https://github.com/sccn/eeglab/blob/167dfc8/functions/popfunc/eeg_interp.m#L314

    # Calculate G for distances between good electrodes + between goods & bads
    G_from = _eeglab_calc_g(pos_from, pos_from)
    G_to_from = _eeglab_calc_g(pos_from, pos_to)

    # Get average reference signal for all good channels and subtract from data
    avg_ref = np.mean(data, axis=0)
    data_tmp = data - avg_ref

    # Calculate interpolation matrix from electrode locations
    pad_ones = np.ones((1, pos_from.shape[0]))
    C_inv = linalg.pinv(np.vstack([G_from, pad_ones]))
    interp_mat = np.matmul(G_to_from, C_inv[:, :-1])

    # Interpolate bad channels and add average good reference to them
    interpolated = np.matmul(interp_mat, data_tmp) + avg_ref

    return interpolated


def _eeglab_interpolate_bads(raw):
    """Interpolate bad channels using EEGLAB's custom method.

    This method modifies the provided Raw object in place.

    Parameters
    ----------
    raw : mne.io.Raw
        An MNE Raw object for which channels marked as "bad" should be
        interpolated.

    Notes
    -----
    Produces identical results as EEGLAB's ``eeg_interp`` function when using
    the default spheric spline method (with minor rounding errors). This method
    appears to be loosely based on the same general Perrin et al. (1989) method
    as MNE's interpolation, but there are several quirks with the implementation
    that cause it to produce fairly different numbers.

    """
    # Get the indices of good and bad EEG channels
    eeg_chans = mne.pick_types(raw.info, eeg=True, exclude=[])
    good_idx = mne.pick_types(raw.info, eeg=True, exclude="bads")
    bad_idx = sorted(_set_diff(eeg_chans, good_idx))

    # Get the spatial coordinates of the good and bad electrodes
    elec_pos = raw._get_channel_positions(picks=eeg_chans)
    pos_good = elec_pos[good_idx, :].copy()
    pos_bad = elec_pos[bad_idx, :].copy()
    _normalize_vectors(pos_good)
    _normalize_vectors(pos_bad)

    # Interpolate bad channels
    interp = _eeglab_interpolate(raw._data[good_idx, :], pos_good, pos_bad)
    raw._data[bad_idx, :] = interp

    # Clear all bad EEG channels
    eeg_bad_names = [raw.info["ch_names"][i] for i in bad_idx]
    bads_non_eeg = _set_diff(raw.info["bads"], eeg_bad_names)
    raw.info["bads"] = bads_non_eeg


def _get_random_subset(x, size, rand_state):
    """Get a random subset of items from a list or array, without replacement.

    Parameters
    ----------
    x : list or np.ndarray
        One-dimensional array of items to sample from.
    size : int
        The number of items to sample. Must be less than the number of input
        items.
    rand_state : np.random.RandomState
        A random state object to use for random number generation.

    Returns
    -------
    sample : list
        A random subset of the input items.

    Notes
    -----
    This function generates random subsets identical to the internal
    ``randsample`` function in MATLAB PREP's ``findNoisyChannels.m``, allowing
    the same random seed to produce identical results across both PyPREP and
    MATLAB PREP.

    """
    sample = []
    remaining = list(x)
    for val in rand_state.rand(size):
        index = round(1 + (len(remaining) - 1) * val) - 1
        pick = remaining.pop(index)
        sample.append(pick)
    return sample


def _correlate_arrays(a, b, matlab_strict=False):
    """Calculate correlations between two equally-sized 2-D arrays of EEG data.

    Both input arrays must be in the shape (channels, samples).

    Parameters
    ----------
    a : np.ndarray
        A 2-D array to correlate with `a`.
    b : np.ndarray
        A 2-D array to correlate with `b`.
    matlab_strict : bool, optional
        Whether or not correlations should be calculated identically to MATLAB
        PREP (i.e., without mean subtraction) instead of by traditional Pearson
        product-moment correlation (see Notes for details). Defaults to
        ``False`` (Pearson correlation).

    Returns
    -------
    correlations : np.ndarray
        A one-dimensional array containing the correlations of the two input arrays
        along the second axis.

    Notes
    -----
    In MATLAB PREP, RANSAC channel predictions are correlated with actual data
    using a non-standard method: essentially, it uses the standard Pearson
    correlation formula but without subtracting the channel means from each channel
    before calculating sums of squares, i.e.,::

       SSa = np.sum(a ** 2)
       SSb = np.sum(b ** 2)
       correlation = np.sum(a * b) / (np.sqrt(SSa) * np.sqrt(SSb))

    Because EEG data is roughly mean-centered to begin with, this produces similar
    values to normal Pearson correlation. However, to avoid making any assumptions
    about the signal for any given channel/window, PyPREP defaults to normal
    Pearson correlation unless strict MATLAB equivalence is requested.

    """
    if matlab_strict:
        SSa = np.sum(a ** 2, axis=1)
        SSb = np.sum(b ** 2, axis=1)
        SSab = np.sum(a * b, axis=1)
        return SSab / (np.sqrt(SSa) * np.sqrt(SSb))
    else:
        n_chan = a.shape[0]
        return np.diag(np.corrcoef(a, b)[:n_chan, n_chan:])


def _mad(x, axis=None):
    """Calculate median absolute deviations from the median (MAD) for an array.

    Parameters
    ----------
    x : np.ndarray
        A 1-D or 2-D numeric array to summarize.
    axis : {int, tuple of int, None}, optional
        Axis along which MADs should be calculated. If ``None``, the MAD will
        be calculated for the full input array. Defaults to ``None``.

    Returns
    -------
    mad : scalar or np.ndarray
        If no axis is specified, returns the MAD for the full input array as a
        single numeric value. Otherwise, returns an ``np.ndarray`` containing
        the MAD for each index along the specified axis.

    """
    # Ensure array is either 1D or 2D
    x = np.asarray(x)
    if x.ndim > 2:
        e = "Only 1D and 2D arrays are supported (input has {0} dimensions)"
        raise ValueError(e.format(x.ndim))

    # Calculate the median absolute deviation from the median
    med = np.median(x, axis=axis)
    if axis == 1:
        med = med.reshape(-1, 1)  # Transposes array to allow subtraction below
    mad = np.median(np.abs(x - med), axis=axis)

    return mad


def _filter_design(N_order, amp, freq):
    """Create FIR low-pass filter for EEG data using frequency sampling method.

    Parameters
    ----------
    N_order : int
        Order of the filter.
    amp : list of int
        Amplitude vector for the frequencies.
    freq : list of int
        Frequency vector for which amplitude can be either 0 or 1.

    Returns
    -------
    kernel : np.ndarray
        Filter kernel.

    """
    nfft = np.maximum(512, 2 ** (np.ceil(math.log(100) / math.log(2))))
    hamming_window = np.subtract(
        0.54,
        np.multiply(
            0.46,
            np.cos(
                np.divide(np.multiply(2 * math.pi, np.arange(N_order + 1)), N_order)
            ),
        ),
    )
    pchip_interpolate = scipy.interpolate.PchipInterpolator(
        np.round(np.multiply(nfft, freq)), amp
    )
    freq = pchip_interpolate(np.arange(nfft + 1))
    freq = np.multiply(
        freq,
        np.exp(
            np.divide(
                np.multiply(-(0.5 * N_order) * sqrt(-1) * math.pi, np.arange(nfft + 1)),
                nfft,
            )
        ),
    )
    kernel = np.real(
        np.fft.ifft(np.concatenate([freq, np.conj(freq[len(freq) - 2 : 0 : -1])]))
    )
    kernel = np.multiply(kernel[0 : N_order + 1], (np.transpose(hamming_window[:])))
    return kernel


def _split_list(mylist, chunk_size):
    """Split list in chunks.

    Parameters
    ----------
    my_list: list
        list to split.
    chunk_size: int
        size of the lists returned.

    Returns
    -------
    list
        list of the chunked lists.

    See: https://stackoverflow.com/a/312466/5201771
    """
    return [
        mylist[offs : offs + chunk_size] for offs in range(0, len(mylist), chunk_size)
    ]


def _print_progress(current, end, start=None, stepsize=1, every=0.1):
    """Print the current progress in a loop.

    Parameters
    ----------
    current: {int, float}
        The index or numeric value of the current position in the loop.
    end: {int, float}
        The final index or numeric value in the loop.
    start: {int, float, None}, optional
        The first index or numeric value in the loop. If ``None``, the start
        index will assumed to be `stepsize` (i.e., 3 if `stepsize` is 3).
        Defaults to ``None``.
    stepsize: {int, float}, optional
        The fixed amount by which `current` increases every iteration of the
        loop. Defaults to ``1``.
    every: float, optional
        The frequency with which to print progress updates during the loop,
        as a proportion between 0 and 1, exclusive. Defaults to ``0.1``, which
        prints a progress update after every 10%.

    """
    start = stepsize if not start else start
    end = end - start + 1
    current = current - start + 1

    if current == 1:
        print("Progress:", end=" ", flush=True)
    elif current == end:
        print("100%")
    elif current > 0:
        progress = float(current) / end
        last = float(current - stepsize) / end
        if int(progress / every) > int(last / every):
            pct = int(progress / every) * every * 100
            print("{0}%...".format(int(pct)), end=" ", flush=True)


def _verify_free_ram(data, n_samples, n_channels, max_prop=0.95):
    """Check if enough memory is free to run RANSAC with the given parameters.

    Parameters
    ----------
    data : np.ndarray
        2-D EEG data
    n_samples : int
        Number of samples to use for computation of RANSAC.
    n_channels : int
        Number of channels to process per chunk.
    max_prop : float
        The maximum proportion of available memory that RANSAC is allowed to
        use.

    Raises
    ------
    MemoryError
        If insufficient free memory to perform RANSAC with the given data and
        parameters.

    """
    available_gb = virtual_memory().available * 1e-9 * max_prop
    needed_gb = (data[0, :].nbytes * 1e-9) * n_samples * n_channels
    if available_gb < needed_gb:
        ram_diff = needed_gb - available_gb
        raise MemoryError(
            "For given data of shape {} and the requested"
            " number of {} samples, {} GB of additional memory"
            " would be needed. You could close other programs,"
            " downsample the data, or reduce the number"
            " of requested samples."
            "".format(data.shape, n_samples, ram_diff)
        )
