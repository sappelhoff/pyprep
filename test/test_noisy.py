"""Test the noisy module."""

from nose.tools import assert_raises

import numpy as np
import mne

from pyprep.noisy import Noisydata


# Make a random MNE object
sfreq = 1000.
t = np.arange(0, 600, 1./sfreq)  # 10 minutes recording
signal_len = t.shape[0]
n_chans = 3
ch_names = ['Cz', 'Pz', 'Oz']

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


def test_ransac_too_few_preds(raw=raw):
    """Test that ransac throws an arror for few predictors."""
    nd = Noisydata(raw)
    assert_raises(IOError, nd.find_bad_by_ransac)
