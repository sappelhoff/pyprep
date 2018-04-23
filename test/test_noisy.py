"""Test the noisy module."""

import numpy as np
import mne

from pyprep.noisy import Noisydata


def test_init():
    """Test the class initialization."""
    # Make a random raw mne object
    sfreq = 1000.
    t = np.arange(0, 10, 1./sfreq)
    n_chns = 3
    X = np.random.random((n_chns, t.shape[0]))
    info = mne.create_info(n_chns, sfreq)
    raw = mne.io.RawArray(X, info)
    nd = Noisydata(raw)
    assert nd
