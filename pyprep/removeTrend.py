"""High-pass filter and locally detrend the EEG signal."""
import logging

import mne
import numpy as np

from pyprep.utils import _eeglab_create_highpass, _eeglab_fir_filter


def removeTrend(
    EEG,
    sample_rate,
    detrendType="high pass",
    detrendCutoff=1.0,
    detrendChannels=None,
    matlab_strict=False,
):
    """Remove trends (i.e., slow drifts in baseline) from an array of EEG data.

    Parameters
    ----------
    EEG : np.ndarray
        A 2-D array of EEG data to detrend.
    sample_rate : float
        The sample rate (in Hz) of the input EEG data.
    detrendType : str, optional
        Type of detrending to be performed: must be one of 'high pass',
        'high pass sinc, or 'local detrend'. Defaults to 'high pass'.
    detrendCutoff : float, optional
        The high-pass cutoff frequency (in Hz) to use for detrending. Defaults
        to 1.0 Hz.
    detrendChannels : {list, None}, optional
        List of the indices of all channels that require detrending/filtering.
        If ``None``, all channels are used (default).
    matlab_strict : bool, optional
        Whether or not detrending should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code
        (see :ref:`matlab-diffs` for more details). Defaults to ``False``.

    Returns
    -------
    EEG : np.ndarray
        A 2-D array containing the filtered/detrended EEG data.

    Notes
    -----
    High-pass filtering is implemented using the MNE filter function
    :func:``mne.filter.filter_data`` unless `matlab_strict` is ``True``, in
    which case it is performed using a minimal re-implementation of EEGLAB's
    ``pop_eegfiltnew``. Local detrending is performed using a Python
    re-implementation of the ``runline`` function from the Chronux package for
    MATLAB [1]_.

    References
    ----------
    .. [1] http://chronux.org/

    """
    if len(EEG.shape) == 1:
        EEG = np.reshape(EEG, (1, EEG.shape[0]))

    if detrendType.lower() == "high pass":
        if matlab_strict:
            picks = detrendChannels if detrendChannels else range(EEG.shape[0])
            filt = _eeglab_create_highpass(detrendCutoff, sample_rate)
            EEG[picks, :] = _eeglab_fir_filter(EEG[picks, :], filt)
        else:
            EEG = mne.filter.filter_data(
                EEG,
                sfreq=sample_rate,
                l_freq=detrendCutoff,
                h_freq=None,
                picks=detrendChannels
            )

    elif detrendType.lower() == "high pass sinc":
        fOrder = np.round(14080 * sample_rate / 512)
        fOrder = np.int(fOrder + fOrder % 2)
        EEG = mne.filter.filter_data(
            data=EEG,
            sfreq=sample_rate,
            l_freq=1,
            h_freq=None,
            picks=detrendChannels,
            filter_length=fOrder,
            fir_window="blackman",
        )

    elif detrendType.lower() == "local detrend":
        if detrendChannels is None:
            detrendChannels = np.arange(0, EEG.shape[0])
        windowSize = 1.5 / detrendCutoff
        windowSize = np.minimum(windowSize, EEG.shape[1])
        stepSize = 0.02
        EEG = np.transpose(EEG)
        n = np.round(sample_rate * windowSize)
        dn = np.round(sample_rate * stepSize)

        if dn > n or dn < 1:
            logging.error(
                "Step size should be less than the window size and "
                "contain at least 1 sample"
            )
        if n == EEG.shape[0]:
            # data = scipy.signal.detrend(EEG, axis=0)
            pass
        else:
            for ch in detrendChannels:
                EEG[:, ch] = runline(EEG[:, ch], np.int(n), np.int(dn))
        EEG = np.transpose(EEG)

    else:
        logging.warning(
            "No filtering/detreding performed since the detrend type did not match"
        )

    return EEG


def runline(y, n, dn):
    """Perform local linear regression on a channel of EEG data.

    A re-implementation of the ``runline`` function from the Chronux package
    for MATLAB [1]_.

    Parameters
    ----------
    y : np.ndarray
        A 1-D array of data from a single EEG channel.
    n : int
        Length of the detrending window.
    dn : int
        Length of the window step size.

    Returns
    -------
    y: np.ndarray
       The detrended signal for the given EEG channel.

    References
    ----------
    .. [1] http://chronux.org/

    """
    nt = y.shape[0]
    y_line = np.zeros((nt, 1))
    norm = np.zeros((nt, 1))
    nwin = np.int(np.ceil((nt - n) / dn))
    yfit = np.zeros((nwin, n))
    xwt = (np.arange(1, n + 1) - n / 2) / (n / 2)
    wt = np.power(1 - np.power(np.absolute(xwt), 3), 3)
    for j in range(0, nwin):
        tseg = y[dn * j : dn * j + n]
        y1 = np.mean(tseg)
        y2 = np.mean(np.multiply(np.arange(1, n + 1), tseg)) * (2 / (n + 1))
        a = np.multiply(np.subtract(y2, y1), 6 / (n - 1))
        b = np.subtract(y1, a * (n + 1) / 2)
        yfit[j, :] = np.multiply(np.arange(1, n + 1), a) + b
        y_line[j * dn : j * dn + n] = y_line[j * dn : j * dn + n] + np.reshape(
            np.multiply(yfit[j, :], wt), (n, 1)
        )
        norm[j * dn : j * dn + n] = norm[j * dn : j * dn + n] + np.reshape(wt, (n, 1))

    for i in range(0, len(norm)):
        if norm[i] > 0:
            y_line[i] = y_line[i] / norm[i]
    indx = (nwin - 1) * dn + n - 1
    npts = len(y) - indx + 1
    y_line[indx - 1 :] = np.reshape(
        (np.multiply(np.arange(n + 1, n + npts + 1), a) + b), (npts, 1)
    )
    for i in range(0, len(y_line)):
        y[i] = y[i] - y_line[i]
    return y
