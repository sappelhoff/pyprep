"""Test the full PREP pipeline."""
import mne
import numpy as np
import pytest

from pyprep.prep_pipeline import PrepPipeline

from .conftest import make_random_mne_object


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline(raw, montage):
    """Test prep pipeline."""
    eeg_index = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
    raw_copy = raw.copy()
    ch_names = raw_copy.info["ch_names"]
    ch_names_eeg = list(np.asarray(ch_names)[eeg_index])
    sample_rate = raw_copy.info["sfreq"]
    prep_params = {
        "ref_chs": ch_names_eeg,
        "reref_chs": ch_names_eeg,
        "line_freqs": np.arange(60, sample_rate / 2, 60),
    }
    prep = PrepPipeline(raw_copy, prep_params, montage, random_state=42)
    prep.fit()


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline_non_eeg(raw, montage):
    """Test prep pipeline with non eeg channels."""
    raw_copy = raw.copy()

    # make arbitrary non eeg channels from the register
    sfreq = raw_copy.info["sfreq"]  # Sampling frequency
    times = np.array(list(range(raw_copy.get_data().shape[1])))
    ch_names_non_eeg = ["misc" + str(i) for i in range(4)]
    ch_types_non_eeg = ["misc" for i in range(4)]
    raw_non_eeg, _, _ = make_random_mne_object(
        ch_names_non_eeg,
        ch_types_non_eeg,
        times,
        sfreq,
        RNG=np.random.RandomState(1337),
    )

    raw_copy.add_channels([raw_non_eeg], force_update_info=True)

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, sfreq / 2, 60),
    }
    prep = PrepPipeline(raw_copy, prep_params, montage, random_state=42)

    prep.fit()

    # correct non-eeg channels configured in init
    assert set(prep.ch_names_non_eeg) == set(ch_names_non_eeg)
    # original (all) channel names same as full_raw names
    assert set(prep.raw.ch_names) == set(raw_copy.ch_names)
    # names of raw (only eeg)  same as full names - non eeg names
    assert set(prep.raw_eeg.ch_names) == set(raw_copy.ch_names) - set(ch_names_non_eeg)
    # quantity of raw (only eeg) same as quantity of all - non eeg lists
    assert prep.raw_eeg.get_data().shape[0] == len(raw_copy.ch_names) - len(
        prep.ch_names_non_eeg
    )
    # quantity of channels in is the same as qty of full raw
    assert raw_copy.get_data().shape[0] == prep.raw.get_data().shape[0]


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline_filter_kwargs(raw, montage):
    """Test prep pipeline with filter kwargs."""
    eeg_index = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
    raw_copy = raw.copy()
    ch_names = raw_copy.info["ch_names"]
    ch_names_eeg = list(np.asarray(ch_names)[eeg_index])
    sample_rate = raw_copy.info["sfreq"]
    prep_params = {
        "ref_chs": ch_names_eeg,
        "reref_chs": ch_names_eeg,
        "line_freqs": np.arange(60, sample_rate / 2, 60),
    }
    filter_kwargs = {
        "mt_bandwidth": 3,
        "p_value": 0.05,
    }

    prep = PrepPipeline(
        raw_copy, prep_params, montage, random_state=42, filter_kwargs=filter_kwargs
    )
    prep.remove_line_noise(prep_params["line_freqs"])
