"""Test the noisy module."""

from nose.tools import assert_raises

import numpy as np
import mne

from pyprep.noisy import Noisydata


# Make a random MNE object
sfreq = 1000.
t = np.arange(0, 600, 1./sfreq)  # 10 minutes recording
signal_len = t.shape[0]
n_chans = 4
ch_names = ['Fz', 'Cz', 'Pz', 'Oz']

# Make a random signal
signal = np.zeros((n_chans, signal_len))
for chan in range(n_chans):
    # Each channel signal is a sum of random sine waves
    for freq_i in range(10):
        freq = np.random.randint(0, 100, signal_len)
        signal[chan, :] += np.sin(2*np.pi*t*freq)

signal *= 1e-6  # scale to Volts

# Make mne object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                       ch_types=['eeg']*n_chans)
raw = mne.io.RawArray(signal, info)


def test_init(raw=raw):
    """Test the class initialization."""
    nd = Noisydata(raw)
    assert nd


def test_find_by_nan(raw=raw):
    """Test find_by_nan."""
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape

    # Insert a nan value for a random channel
    rand_chn_idx = int(np.random.randint(0, 5, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    raw_tmp._data[rand_chn_idx, n-1] = np.nan

    # Now find it and assert it's the correct one.
    nd = Noisydata(raw_tmp)
    nd.find_bad_by_nan()
    assert nd.bad_by_nan == [rand_chn_lab]


def test_ransac_too_few_preds(raw=raw):
    """Test that ransac throws an arror for few predictors."""
    chns = np.random.choice(raw.ch_names, size=3, replace=False)
    raw.pick_channels(chns)
    nd = Noisydata(raw)
    assert_raises(IOError, nd.find_bad_by_ransac)


def test_ransac_too_little_ram(raw=raw):
    """Test that ransac throws a memory error if not enough available."""
    nd = Noisydata(raw)
    chn_pos = nd.chn_pos
    chn_pos_good = chn_pos
    good_chn_labs = raw.ch_names
    n_pred_chns = 4
    data = raw._data
    n_samples = 99e99
    assert_raises(MemoryError, nd._run_ransac, chn_pos, chn_pos_good,
                  good_chn_labs, n_pred_chns, data, n_samples)
