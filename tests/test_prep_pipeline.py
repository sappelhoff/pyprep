"""Test the full PREP pipeline."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import warnings
from unittest import mock

import numpy as np
import pytest

from pyprep.prep_pipeline import PrepPipeline

from .conftest import make_random_mne_object


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
        rng=np.random.default_rng(1337),
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
    eeg_index = np.array(
        [idx for idx, typ in enumerate(raw.get_channel_types()) if typ == "eeg"]
    )
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
        "method": "fir",
        "phase": "zero-double",
    }

    prep = PrepPipeline(
        raw_copy, prep_params, montage, random_state=42, filter_kwargs=filter_kwargs
    )
    prep.fit()

    # The input is purely EEG, so there are no non-EEG channels to split off
    # and the `raw` property returns the EEG data unchanged.
    assert prep.raw_non_eeg is None
    assert prep.raw.ch_names == prep.raw_eeg.ch_names


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline_stages_match_fit(raw, montage):
    """Running the stages manually should be identical to calling fit()."""
    sfreq = raw.info["sfreq"]
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, sfreq / 2, 60),
    }

    # The all-in-one fit() path
    prep_fit = PrepPipeline(
        raw.copy(), prep_params, montage, ransac=False, random_state=42
    )
    prep_fit.fit()

    # The equivalent manual, staged path
    prep_staged = PrepPipeline(
        raw.copy(), prep_params, montage, ransac=False, random_state=42
    )
    prep_staged.remove_line_noise()
    prep_staged.robust_reference()

    # The processed signals and detected bad channels should be identical
    np.testing.assert_allclose(
        prep_fit.raw.get_data(), prep_staged.raw.get_data()
    )
    assert prep_fit.bad_before_interpolation == prep_staged.bad_before_interpolation
    assert prep_fit.interpolated_channels == prep_staged.interpolated_channels
    assert prep_fit.still_noisy_channels == prep_staged.still_noisy_channels


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline_skip_interpolation(raw, montage):
    """Robust referencing with interpolate_bads=False should skip interpolation."""
    sfreq = raw.info["sfreq"]
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, sfreq / 2, 60),
    }

    # Full pipeline (with interpolation) for reference
    prep_full = PrepPipeline(
        raw.copy(), prep_params, montage, ransac=False, random_state=42
    )
    prep_full.fit()

    # Same pipeline, but with the final interpolation disabled
    prep_noint = PrepPipeline(
        raw.copy(), prep_params, montage, ransac=False, random_state=42
    )
    prep_noint.remove_line_noise()
    prep_noint.robust_reference(interpolate_bads=False)

    # Bad channels detected post-reference are the same (interpolation is later)
    assert prep_noint.bad_before_interpolation == prep_full.bad_before_interpolation
    assert prep_noint.reference_before_interpolation is not None
    assert prep_noint.EEG_before_interpolation is not None

    # The post-interpolation outputs were never computed
    assert prep_noint.interpolated_channels is None
    assert prep_noint.still_noisy_channels is None
    assert prep_noint.reference_after_interpolation is None
    assert prep_noint.noisy_channels_after_interpolation is None


@pytest.mark.usefixtures("raw", "montage")
def test_robust_reference_warns_without_line_noise(raw, montage):
    """robust_reference warns if line noise was not removed first, fit() does not."""
    sfreq = raw.info["sfreq"]
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, sfreq / 2, 60),
    }

    # Detection results are irrelevant here, so skip them for speed
    with mock.patch("pyprep.NoisyChannels.find_all_bads", return_value=True):
        # Calling robust_reference directly, out of order, warns
        prep = PrepPipeline(
            raw.copy(), prep_params, montage, ransac=False, random_state=42
        )
        with pytest.warns(UserWarning, match="without prior line-noise"):
            prep.robust_reference(interpolate_bads=False)

        # Running the full pipeline through fit() must not emit that warning
        prep2 = PrepPipeline(
            raw.copy(), prep_params, montage, ransac=False, random_state=42
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            prep2.fit()
        assert not any(
            "without prior line-noise" in str(w.message) for w in record
        )


@pytest.mark.usefixtures("raw", "montage")
def test_fit_without_line_freqs_does_not_warn(raw, montage):
    """An empty line_freqs is a deliberate skip and must not trigger a warning."""
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": [],
    }

    with mock.patch("pyprep.NoisyChannels.find_all_bads", return_value=True):
        prep = PrepPipeline(
            raw.copy(), prep_params, montage, ransac=False, random_state=42
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            prep.fit()
        assert not any(
            "without prior line-noise" in str(w.message) for w in record
        )
