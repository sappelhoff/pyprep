"""Test Robust Reference."""
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
    assert len(reference.noisy_channels_original["bad_all"]) == 0
    assert len(reference.noisy_channels["bad_all"]) == 0


@pytest.mark.usefixtures("raw_clean")
def test_all_bad_input(raw_clean):
    """Test robust reference when all reference channels are bad."""
    ch_names = raw_clean.info["ch_names"]
    params = {"ref_chs": ch_names, "reref_chs": ch_names}

    # Define a mock function to make all channels bad by deviation
    def _bad_by_dev(self):
        self.bad_by_deviation = self.ch_names_original.tolist()

    # Here we monkey-patch Reference to make all channels bad by deviation, allowing
    # us to test the 'too-few-good-channels' exception
    with mock.patch("pyprep.NoisyChannels.find_bad_by_deviation", new=_bad_by_dev):
        reference = Reference(raw_clean, params, ransac=False)
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
