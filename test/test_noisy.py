"""Test the noisy module."""

import numpy as np
import mne

from pyprep.noisy import Noisydata


def test_init():
    """Test the class initialization."""
    # Make a random raw mne object
    sfreq = 1000.
    t = np.arange(0, 10, 1./sfreq)
    n_chans = 3
    ch_names = ['Cz', 'Pz', 'Oz']
    X = np.random.random((n_chans, t.shape[0]))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=['eeg']*n_chans)
    raw = mne.io.RawArray(X, info)
    nd = Noisydata(raw)
    assert nd
