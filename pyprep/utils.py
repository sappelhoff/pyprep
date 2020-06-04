"""Module contains frequently used functions dealing with channel lists."""
import math
from cmath import sqrt

import numpy as np
import scipy.interpolate


def _union(list1, list2):
    return list(set(list1 + list2))


def _set_diff(list1, list2):
    return list(set(list1) - set(list2))


def _intersect(list1, list2):
    return list(set(list1).intersection(set(list2)))


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
    kernel : ndarray
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
