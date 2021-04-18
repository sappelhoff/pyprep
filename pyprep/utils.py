"""Module contains frequently used functions dealing with channel lists."""
import math
from cmath import sqrt

import mne
import numpy as np
import scipy.interpolate
from scipy.stats import iqr
from psutil import virtual_memory


def _union(list1, list2):
    return list(set(list1 + list2))


def _set_diff(list1, list2):
    return list(set(list1) - set(list2))


def _intersect(list1, list2):
    return list(set(list1).intersection(set(list2)))


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
