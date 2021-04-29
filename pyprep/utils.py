"""Module contains frequently used functions dealing with channel lists."""
import math
from cmath import sqrt

import mne
import numpy as np
import scipy.interpolate
from scipy.stats import iqr
from scipy.signal import firwin, lfilter, lfilter_zi
from psutil import virtual_memory


def _union(list1, list2):
    return list(set(list1 + list2))


def _set_diff(list1, list2):
    return list(set(list1) - set(list2))


def _intersect(list1, list2):
    return list(set(list1).intersection(set(list2)))


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
        Input array containing samples from the distribution to summarize.
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
    q = np.asarray(q, dtype=np.float64)
    n = len(arr)
    q_adj = ((q - 0.5) * n / (n - 1)) + 0.5
    return np.quantile(arr, np.clip(q_adj, 0, 1), axis=axis)


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
    iqr_q = np.asarray([25, 75], dtype=np.float64)
    n = len(arr)
    iqr_adj = ((iqr_q - 50) * n / (n - 1)) + 50
    return iqr(arr, rng=np.clip(iqr_adj, 0, 100), axis=axis)


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
    filt -= firwin(N, transition, window='hamming', nyq=1)
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
    start_padded = np.concatenate(
        (data[:, start_pad_idx], data[:, :pad_len]),
        axis=1
    )
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
    out[:, (n_samples - pad_len):] = end[:, (group_delay - pad_len):]

    return out


def _get_random_subset(x, size, rand_state):
    """Get a random subset of items from a list or array, without replacement.

    Parameters
    ----------
    x : list or np.ndarray
        One-dimensional array of items to sample from.
    size : int
        The number of items to sample. Must be less than the number of input
        items.
    rand_state : np.random.RandState
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
    using a non-standard method: essentialy, it uses the standard Pearson
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


def filter_design(N_order, amp, freq):
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


def split_list(mylist, chunk_size):
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


def print_progress(current, end, start=None, stepsize=1, every=0.1):
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


def make_random_mne_object(
    ch_names,
    ch_types,
    times,
    sfreq,
    n_freq_comps=5,
    freq_range=[10, 60],
    scale=1e-6,
    RNG=np.random.RandomState(1337),
):
    """Make a random MNE object to use for testing.

    Parameters
    ----------
    ch_names : list
        names of channels
    ch_types : list
        types of channels
    times : np.ndarray, shape (1,)
        Time vector to use.
    sfreq : float
        Sampling frequency associated with the time vector.
    n_freq_comps : int
        Number of signal components summed to make a signal.
    freq_range : list, len==2
        Signals will contain freqs from this range.
    scale : float
        Scaling factor applied to the signal in volts. For example 1e-6 to
        get microvolts.
    RNG : np.random.RandomState
        Random state seed.

    Returns
    -------
    raw : mne.io.Raw
        The mne object for performing the tests.
    n_freq_comps : int
        Number of random frequency components to introduce.
    freq_range : tuple
        The low (`freq_range[0]`) and high (`freq_range[1]`) endpoints of
        a frequency range from which random draws will be made to
        introduce frequency components in the test data.
    """
    n_chans = len(ch_names)
    signal_len = times.shape[0]
    # Make a random signal
    signal = np.zeros((n_chans, signal_len))
    low = freq_range[0]
    high = freq_range[1]
    for chan in range(n_chans):
        # Each channel signal is a sum of random freq sine waves
        for freq_i in range(n_freq_comps):
            freq = RNG.randint(low, high, signal_len)
            signal[chan, :] += np.sin(2 * np.pi * times * freq)

    signal *= scale  # scale

    # Make mne object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(signal, info)
    return raw, n_freq_comps, freq_range


def verify_free_ram(data, n_samples, n_channels, max_prop=0.95):
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
