"""Test Robust Reference."""
import random
from unittest import mock

import mne
import numpy as np
import pytest

from pyprep.reference import Reference


@pytest.mark.usefixtures("raw", "montage")
def test_basic_input(raw, montage):
    """Test Reference output data type."""
    ch_names = raw.info["ch_names"]

    raw_tmp = raw.copy()
    raw_tmp.set_montage(montage)
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

    # Make sure the set of reference channels weren't modified by re-referencing
    assert params["ref_chs"] == reference.reference_channels


@pytest.mark.usefixtures("raw_clean")
def test_clean_input(raw_clean):
    """Test robust referencing with a clean input signal."""
    ch_names = raw_clean.info["ch_names"]
    params = {"ref_chs": ch_names, "reref_chs": ch_names}

    # Here we monkey-patch Reference to skip bad channel detection, ensuring
    # a run with all clean channels is tested
    with mock.patch("pyprep.NoisyChannels.find_all_bads", return_value=True):
        reference = Reference(raw_clean, params, ransac=False)
        reference.robust_reference()

    assert len(reference.unusable_channels) == 0
    assert len(reference.noisy_channels_original['bad_all']) == 0
    assert len(reference.noisy_channels['bad_all']) == 0


@pytest.mark.usefixtures("raw", "montage")
def test_all_bad_input(raw, montage):
    """Test robust reference when all reference channels are bad."""
    ch_names = raw.info["ch_names"]

    raw_tmp = raw.copy()
    raw_tmp.set_montage(montage)
    m, n = raw_tmp.get_data().shape

    # Randomly set some channels as bad
    [nan_chn_idx, flat_chn_idx] = random.sample(set(np.arange(0, m)), 2)

    # Insert a nan value for a random channel
    # nan_chn_lab = raw_tmp.ch_names[nan_chn_idx]
    raw_tmp._data[nan_chn_idx, n - 1] = np.nan

    # Insert one random flat channel
    # flat_chn_lab = raw_tmp.ch_names[flat_chn_idx]
    raw_tmp._data[flat_chn_idx, :] = np.ones_like(raw_tmp._data[1, :]) * 1e-6

    reference_channels = [ch_names[nan_chn_idx], ch_names[flat_chn_idx]]
    params = {"ref_chs": reference_channels, "reref_chs": reference_channels}
    reference = Reference(raw_tmp, params, ransac=False)
    with pytest.raises(ValueError):
        reference.robust_reference()


def test_remove_reference():
    """Test removing the reference."""
    signal = np.array([[1, 2, 3, 4], [0, 1, 2, 3], [3, 4, 5, 6]])
    reference = np.array([1, 1, 2, 2])
    with pytest.raises(ValueError):
        Reference.remove_reference(reference, reference)
    with pytest.raises(ValueError):
        Reference.remove_reference(signal, signal)
    with pytest.raises(ValueError):
        Reference.remove_reference(signal, reference[0:3])
    with pytest.raises(TypeError):
        Reference.remove_reference(signal, reference, np.array([1, 2]))
    assert np.array_equal(
        Reference.remove_reference(signal, reference, [1, 2]),
        np.array([[1, 2, 3, 4], [-1, 0, 0, 1], [2, 3, 3, 4]]),
    )
