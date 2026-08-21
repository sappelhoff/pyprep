"""Configure tests."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import mne
import numpy as np
import pytest
from mne.datasets import eegbci


@pytest.fixture(scope="session")
def montage():
    """Fixture for standard EEG montage."""
    montage_kind = "standard_1020"
    montage = mne.channels.make_standard_montage(montage_kind)
    return montage


@pytest.fixture(scope="session")
def raw():
    """Return an `mne.io.Raw` object for use with unit tests.

    This fixture downloads and reads in subject 4, run 1 from the Physionet
    BCI2000 (eegbci) open dataset. This recording is quite noisy and is thus a
    good candidate for testing the PREP pipeline.

    File attributes:
    - Channels: 64 EEG
    - Sample rate: 160 Hz
    - Duration: 61 seconds

    This is only run once per session to save time downloading.

    """
    mne.set_log_level("WARNING")

    # Download and read S004R01.edf from the BCI2000 dataset
    edf_fpath = eegbci.load_data(4, 1, update_path=True)[0]
    raw = mne.io.read_raw_edf(edf_fpath, preload=True)
    eegbci.standardize(raw)  # Fix non-standard channel names

    return raw


@pytest.fixture(scope="session")
def raw_clean(montage):
    """Return an `mne.io.Raw` object with no bad channels for use with tests.

    This fixture downloads and reads in subject 30, run 2 from the Physionet
    BCI2000 (eegbci) open dataset, which contains no bad channels on an initial
    pass of :class:`pyprep.NoisyChannels`. Intended for use with tests where
    channels are made artificially bad.

    File attributes:
    - Channels: 64 EEG
    - Sample rate: 160 Hz
    - Duration: 61 seconds

    This is only run once per session to save time downloading.

    """
    mne.set_log_level("WARNING")

    # Download and read S030R02.edf from the BCI2000 dataset
    edf_fpath = eegbci.load_data(30, 2, update_path=True)[0]
    raw = mne.io.read_raw_edf(edf_fpath, preload=True)
    eegbci.standardize(raw)  # Fix non-standard channel names

    # Set a montage for use with RANSAC
    raw.set_montage(montage)

    return raw


def make_random_mne_object(
    ch_names,
    ch_types,
    times,
    sfreq,
    n_freq_comps=5,
    freq_range=[10, 60],
    scale=1e-6,
    rng=np.random.default_rng(1337),
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
    rng : np.random.Generator
        The random number generator object. Must be created with
        ``np.random.default_rng``.

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
        for _ in range(n_freq_comps):
            freq = rng.integers(low, high, signal_len)
            signal[chan, :] += np.sin(2 * np.pi * times * freq)

    signal *= scale  # scale

    # Make mne object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(signal, info)
    return raw, n_freq_comps, freq_range
