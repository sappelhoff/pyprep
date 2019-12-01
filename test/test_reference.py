"""Test Robust Reference"""
import pytest

import numpy as np
import mne
import random
from pyprep.reference import robust_reference, remove_reference
from pyprep.noisy import Noisydata


def make_reference_mne_object(sfreq=1000., t_secs=200, n_freq_comps=5,
                              freq_range=[10, 40]):
    """Make a random MNE object to use for testing.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.

    t_secs : int
        Recording length in seconds.

    n_freq_comps : int
        Number of signal components summed to make a signal.

    freq_range : list, len==2
        Signals will contain freqs from this range.

    Returns
    -------
    raw : mne raw object
        The mne object for performing the tests.

    n_freq_comps : int

    freq_range : list, len==2

    """
    t = np.arange(0, t_secs, 1./sfreq)
    signal_len = t.shape[0]
    ch_names = ['Fpz', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    ch_types = ['eeg' for chn in ch_names]
    n_chans = len(ch_names)

    # Make a random signal
    signal = np.zeros((n_chans, signal_len))
    low = freq_range[0]
    high = freq_range[1]

    # Each channel signal is a weighted sum of random freq sine waves
    for freq_i in range(n_freq_comps):
        freq = np.random.randint(low, high, signal_len)
        for chan in range(n_chans):
            weight = np.random.random()*5
            signal[chan, :] += np.sin(2*np.pi*t*freq)*weight


    signal *= 1e-6  # scale to Volts

    # Make mne object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=ch_types)
    raw = mne.io.RawArray(signal, info)
    return raw, n_freq_comps, freq_range


# Make new random mne objects until we have one without inherent bad chans after referencing by mean
found_good_test_object = False
while not found_good_test_object:
    raw, n_freq_comps, freq_range = make_reference_mne_object()
    reference_signal = np.median(raw.get_data(), axis=0)
    referenced_signal = remove_reference(raw.get_data(), reference_signal)
    raw_copy = raw.copy()
    raw_copy._data = referenced_signal
    nd = Noisydata(raw)
    nd.find_all_bads(ransac=False)
    nd_copy = Noisydata(raw_copy)
    nd_copy.find_all_bads(ransac=False)
    if not nd.find_all_bads(ransac=False) and not nd_copy.find_all_bads(ransac=False):
        found_good_test_object = True


def test_all_good_reference(raw=raw):
    """Test robust reference when all channels are good"""
    ch_names = raw.info['ch_names']
    params = {'ref_chs': ch_names, 'eval_chs': ch_names}
    noisy_channels, reference = robust_reference(raw, params, ransac=False)
    assert noisy_channels == {'bad_by_nan': [], 'bad_by_flat': [],
                              'bad_by_deviation': [], 'bad_by_hf_noise': [],
                              'bad_by_correlation': [], 'bad_by_ransac': [],
                              'bad_all': []}


def test_all_bad_reference(raw=raw):
    """Test robust reference when all reference channels are bad"""
    ch_names = raw.info['ch_names']
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape

    # Randomly set some channels as bad
    [nan_chn_idx, flat_chn_idx] = random.sample(set(np.arange(0, m)), 2)

    # Insert a nan value for a random channel
    nan_chn_lab = raw_tmp.ch_names[nan_chn_idx]
    raw_tmp._data[nan_chn_idx, n - 1] = np.nan

    # Insert one random flat channel
    flat_chn_lab = raw_tmp.ch_names[flat_chn_idx]
    raw_tmp._data[flat_chn_idx, :] = np.ones_like(raw_tmp._data[1, :])*1e-6

    reference_channels = [ch_names[nan_chn_idx], ch_names[flat_chn_idx]]
    params = {'ref_chs': reference_channels, 'eval_chs': reference_channels}
    with pytest.raises(ValueError):
        noisy_channels, reference = robust_reference(raw_tmp, params, ransac=False)


def test_remove_reference():
    signal = np.array([[1, 2, 3, 4],
                       [0, 1, 2, 3],
                       [3, 4, 5, 6]])
    reference = np.array([1, 1, 2, 2])
    with pytest.raises(ValueError):
        signal_new = remove_reference(reference, reference)
    with pytest.raises(ValueError):
        signal_new = remove_reference(signal, signal)
    with pytest.raises(ValueError):
        signal_new = remove_reference(signal, reference[0:3])
    with pytest.raises(TypeError):
        signal_new = remove_reference(signal, reference, np.array([1, 2]))
    assert np.array_equal(remove_reference(signal, reference, [1, 2]),
                          np.array([[1, 2, 3, 4], [-1, 0, 0, 1], [2, 3, 3, 4]]))
