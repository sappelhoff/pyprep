"""Test Robust Reference"""
import random

import mne
import numpy as np
import pytest
from mne.datasets import eegbci

from pyprep.reference import Reference

# load in subject 1, run 1 dataset
edf_fpath = eegbci.load_data(1, 1)[0]

# using sample EEG data (https://physionet.org/content/eegmmidb/1.0.0/)
raw = mne.io.read_raw_edf(edf_fpath, preload=True)
raw.rename_channels(lambda s: s.strip("."))
raw.rename_channels(
    lambda s: s.replace("c", "C")
    .replace("o", "O")
    .replace("f", "F")
    .replace("t", "T")
    .replace("Tp", "TP")
    .replace("Cp", "CP")
)
ch_names = raw.info["ch_names"]
montage = mne.channels.make_standard_montage(kind="standard_1020")
raw.set_montage(montage)


def test_basic_input():
    """Test Reference output data type"""
    raw_tmp = raw.copy()
    params = {"ref_chs": ch_names, "reref_chs": ch_names}
    reference = Reference(raw_tmp, params, ransac=False)
    reference.perform_reference()
    assert type(reference.noisy_channels) == dict
    assert type(reference.noisy_channels_original) == dict
    assert type(reference.bad_before_interpolation) == list
    assert type(reference.reference_signal) == np.ndarray
    assert type(reference.interpolated_channels) == list
    assert type(reference.still_noisy_channels) == list
    assert type(reference.raw) == mne.io.edf.edf.RawEDF


def test_all_bad_input():
    """Test robust reference when all reference channels are bad"""
    raw_tmp = raw.copy()
    m, n = raw_tmp.get_data().shape

    # Randomly set some channels as bad
    [nan_chn_idx, flat_chn_idx] = random.sample(set(np.arange(0, m)), 2)

    # Insert a nan value for a random channel
    nan_chn_lab = raw_tmp.ch_names[nan_chn_idx]
    raw_tmp._data[nan_chn_idx, n - 1] = np.nan

    # Insert one random flat channel
    flat_chn_lab = raw_tmp.ch_names[flat_chn_idx]
    raw_tmp._data[flat_chn_idx, :] = np.ones_like(raw_tmp._data[1, :]) * 1e-6

    reference_channels = [ch_names[nan_chn_idx], ch_names[flat_chn_idx]]
    params = {"ref_chs": reference_channels, "reref_chs": reference_channels}
    reference = Reference(raw_tmp, params, ransac=False)
    with pytest.raises(ValueError):
        reference.robust_reference()


def test_remove_reference():
    signal = np.array([[1, 2, 3, 4], [0, 1, 2, 3], [3, 4, 5, 6]])
    reference = np.array([1, 1, 2, 2])
    with pytest.raises(ValueError):
        signal_new = Reference.remove_reference(reference, reference)
    with pytest.raises(ValueError):
        signal_new = Reference.remove_reference(signal, signal)
    with pytest.raises(ValueError):
        signal_new = Reference.remove_reference(signal, reference[0:3])
    with pytest.raises(TypeError):
        signal_new = Reference.remove_reference(signal, reference, np.array([1, 2]))
    assert np.array_equal(
        Reference.remove_reference(signal, reference, [1, 2]),
        np.array([[1, 2, 3, 4], [-1, 0, 0, 1], [2, 3, 3, 4]]),
    )
