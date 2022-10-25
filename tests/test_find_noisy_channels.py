"""Test the find_noisy_channels module."""
import numpy as np
import pytest
from numpy.random import RandomState

from pyprep.find_noisy_channels import NoisyChannels
from pyprep.ransac import find_bad_by_ransac
from pyprep.removeTrend import removeTrend

# Set a fixed random seed for reproducible test results

RNG = RandomState(30)


# Define some fixtures and utility functions for use across multiple tests


@pytest.fixture(scope="session")
def raw_clean_detrend(raw_clean):
    """Return a pre-detrended `mne.io.Raw` object with no bad channels.

    Based on the data from the `raw_clean` fixture, which uses the data for
    subject 1, run 1 from the Physionet BCI2000 dataset.

    This is only run once per session to save time.

    """
    raw_clean_detrended = raw_clean.copy()
    raw_clean_detrended._data = removeTrend(
        raw_clean.get_data(), raw_clean.info["sfreq"]
    )
    return raw_clean_detrended


@pytest.fixture
def raw_tmp(raw_clean_detrend):
    """Return an unmodified copy of the `raw_clean_detrend` fixture.

    This is run once per NoisyChannels test, to keep any modifications to
    `raw_tmp` during a test from affecting `raw_tmp` in any others.

    """
    raw_tmp = raw_clean_detrend.copy()
    return raw_tmp


def _generate_signal(fmin, fmax, timepoints, fcount=1):
    """Generate an EEG signal from one or more sine waves in a frequency range."""
    signal = np.zeros_like(timepoints)
    for freq in RNG.randint(fmin, fmax + 1, fcount):
        signal += np.sin(2 * np.pi * timepoints * freq)
    return signal * 1e-6


# Run unit tests for each bad channel type detected by NoisyChannels


def test_bad_by_nan(raw_tmp):
    """Test the detection of channels containing any NaN values."""
    # Insert a NaN value into a random channel
    n_chans = raw_tmp._data.shape[0]
    nan_idx = int(RNG.randint(0, n_chans, 1))
    raw_tmp._data[nan_idx, 3] = np.nan

    # Test automatic detection of NaN channels on NoisyChannels init
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    assert nd.bad_by_nan == [raw_tmp.ch_names[nan_idx]]

    # Test manual re-running of NaN channel detection
    nd.find_bad_by_nan_flat()
    assert nd.bad_by_nan == [raw_tmp.ch_names[nan_idx]]


def test_bad_by_flat(raw_tmp):
    """Test the detection of channels with flat or very weak signals."""
    # Make the signal for a random channel extremely weak
    n_chans = raw_tmp._data.shape[0]
    flat_idx = int(RNG.randint(0, n_chans, 1))
    raw_tmp._data[flat_idx, :] = raw_tmp._data[flat_idx, :] * 1e-12

    # Test automatic detection of flat channels on NoisyChannels init
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    assert nd.bad_by_flat == [raw_tmp.ch_names[flat_idx]]

    # Test manual re-running of flat channel detection
    nd.find_bad_by_nan_flat()
    assert nd.bad_by_flat == [raw_tmp.ch_names[flat_idx]]

    # Test detection when channel is completely flat
    raw_tmp._data[flat_idx, :] = 0
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    assert nd.bad_by_flat == [raw_tmp.ch_names[flat_idx]]


def test_bad_by_deviation(raw_tmp):
    """Test detection of channels with relatively high or low amplitudes."""
    # Set scaling factors for high and low deviation test channels
    low_dev_factor = 0.1
    high_dev_factor = 4.0

    # Make the signal for a random channel have a very high amplitude
    n_chans = raw_tmp._data.shape[0]
    high_dev_idx = int(RNG.randint(0, n_chans, 1))
    raw_tmp._data[high_dev_idx, :] *= high_dev_factor

    # Test detection of abnormally high-amplitude channels
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_deviation()
    assert nd.bad_by_deviation == [raw_tmp.ch_names[high_dev_idx]]

    # Make the signal for a different channel have a very low amplitude
    low_dev_idx = (high_dev_idx - 1) if high_dev_idx > 0 else 1
    raw_tmp._data[low_dev_idx, :] *= low_dev_factor

    # Test detection of abnormally low-amplitude channels
    # NOTE: The default z-score threshold (5.0) is too strict to allow detection
    # of abnormally low-amplitude channels in some datasets. Using a relaxed Z
    # threshold of 3.29 (p < 0.001, two-tailed) until a better solution is found.
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_deviation(deviation_threshold=3.29)
    bad_by_dev_idx = [low_dev_idx, high_dev_idx]
    assert nd.bad_by_deviation == [raw_tmp.ch_names[i] for i in bad_by_dev_idx]


def test_bad_by_hf_noise(raw_tmp):
    """Test detection of channels with high-frequency noise."""
    # Add some noise between 70 & 80 Hz to the signal of a random channel
    n_chans = raw_tmp._data.shape[0]
    hf_noise_idx = int(RNG.randint(0, n_chans, 1))
    hf_noise = _generate_signal(70, 80, raw_tmp.times, 5) * 10
    raw_tmp._data[hf_noise_idx, :] += hf_noise

    # Test detection of channels with high-frequency noise
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_hfnoise()
    assert nd.bad_by_hf_noise == [raw_tmp.ch_names[hf_noise_idx]]

    # Test lack of high-frequency noise detection when sample rate < 100 Hz
    raw_tmp.resample(80)  # downsample from 160 Hz to 80 Hz
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_hfnoise()
    assert len(nd.bad_by_hf_noise) == 0
    assert nd._extra_info["bad_by_hf_noise"]["median_channel_noisiness"] == 0
    assert nd._extra_info["bad_by_hf_noise"]["channel_noisiness_sd"] == 1


def test_bad_by_dropout(raw_tmp):
    """Test detection of channels with excessive portions of flat signal."""
    # Add large dropout portions to the signal of a random channel
    n_chans, n_samples = raw_tmp._data.shape
    dropout_idx = int(RNG.randint(0, n_chans, 1))
    x1, x2 = (int(n_samples / 10), int(2 * n_samples / 10))
    raw_tmp._data[dropout_idx, x1:x2] = 0  # flatten 10% of signal

    # Test detection of channels that have excessive dropout regions
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_correlation()
    assert nd.bad_by_dropout == [raw_tmp.ch_names[dropout_idx]]


def test_bad_by_correlation(raw_tmp):
    """Test detection of channels that correlate poorly with others."""
    # Replace a random channel's signal with uncorrelated values
    n_chans, n_samples = raw_tmp._data.shape
    low_corr_idx = int(RNG.randint(0, n_chans, 1))
    raw_tmp._data[low_corr_idx, :] = _generate_signal(10, 30, raw_tmp.times, 5)

    # Test detection of channels that correlate poorly with others
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_correlation()
    assert nd.bad_by_correlation == [raw_tmp.ch_names[low_corr_idx]]

    # Add a channel with dropouts to see if correlation detection still works
    dropout_idx = (low_corr_idx - 1) if low_corr_idx > 0 else 1
    x1, x2 = (int(n_samples / 10), int(2 * n_samples / 10))
    raw_tmp._data[dropout_idx, x1:x2] = 0  # flatten 10% of signal

    # Re-test detection of channels that correlate poorly with others
    # (only new bad-by-correlation channel should be dropout)
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_correlation()
    assert raw_tmp.ch_names[low_corr_idx] in nd.bad_by_correlation
    assert len(nd.bad_by_correlation) <= 2


def test_bad_by_SNR(raw_tmp):
    """Test detection of channels that have low signal-to-noise ratios."""
    # Replace a random channel's signal with uncorrelated values
    n_chans = raw_tmp._data.shape[0]
    low_snr_idx = int(RNG.randint(0, n_chans, 1))
    raw_tmp._data[low_snr_idx, :] = _generate_signal(10, 30, raw_tmp.times, 5)

    # Add some high-frequency noise to the uncorrelated channel
    hf_noise = _generate_signal(70, 80, raw_tmp.times, 5) * 10
    raw_tmp._data[low_snr_idx, :] += hf_noise

    # Test detection of channels with a low signal-to-noise ratio
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.find_bad_by_SNR()
    assert nd.bad_by_SNR == [raw_tmp.ch_names[low_snr_idx]]


def test_find_bad_by_ransac(raw_tmp):
    """Test the RANSAC component of NoisyChannels."""
    # Set a consistent random seed for all RANSAC runs
    RANSAC_RNG = 435656

    # RANSAC identifies channels that go bad together and are highly correlated.
    # Inserting highly correlated signal in channels 0 through 6 at 30 Hz
    raw_tmp._data[0:6, :] = _generate_signal(30, 30, raw_tmp.times)

    # Run different variations of RANSAC on the same data
    test_matrix = {
        # List items represent [matlab_strict, channel_wise, max_chunk_size]
        "by_window": [False, False, None],
        "by_channel": [False, True, None],
        "by_channel_maxchunk": [False, True, 2],
        "by_window_strict": [True, False, None],
        "by_channel_strict": [True, True, None],
    }
    bads = {}
    corr = {}
    for name, args in test_matrix.items():
        nd = NoisyChannels(
            raw_tmp, do_detrend=False, random_state=RANSAC_RNG, matlab_strict=args[0]
        )
        nd.find_bad_by_ransac(channel_wise=args[1], max_chunk_size=args[2])
        # Save bad channels and RANSAC correlation matrix for later comparison
        bads[name] = nd.bad_by_ransac
        corr[name] = nd._extra_info["bad_by_ransac"]["ransac_correlations"]

    # Test whether all methods detected bad channels properly
    assert bads["by_window"] == raw_tmp.ch_names[0:6]
    assert bads["by_channel"] == raw_tmp.ch_names[0:6]
    assert bads["by_channel_maxchunk"] == raw_tmp.ch_names[0:6]
    assert bads["by_window_strict"] == raw_tmp.ch_names[0:6]
    assert bads["by_channel_strict"] == raw_tmp.ch_names[0:6]

    # Make sure non-strict correlation matrices all match
    assert np.allclose(corr["by_window"], corr["by_channel"])
    assert np.allclose(corr["by_window"], corr["by_channel_maxchunk"])

    # Make sure MATLAB-strict correlation matrices match
    assert np.allclose(corr["by_window_strict"], corr["by_channel_strict"])

    # Make sure strict and non-strict matrices differ
    assert not np.allclose(corr["by_window"], corr["by_window_strict"])

    # Ensure that RANSAC doesn't change random state if in MATLAB-strict mode
    rng = RandomState(RANSAC_RNG)
    init_state = rng.get_state()[2]
    nd = NoisyChannels(raw_tmp, do_detrend=False, random_state=rng, matlab_strict=True)
    nd.find_bad_by_ransac()
    assert rng.get_state()[2] == init_state

    # Test calling the find_bad_by_ransac function directly
    chn_pos = np.stack([ch["loc"][0:3] for ch in raw_tmp.info["chs"]])
    bads, corr = find_bad_by_ransac(
        raw_tmp._data, raw_tmp.info["sfreq"], raw_tmp.info["ch_names"], chn_pos, []
    )


def test_find_bad_by_ransac_err(raw_tmp):
    """Test error handling in the `find_bad_by_ransac` method."""
    # Set n_samples very very high to trigger a memory error
    n_samples = int(1e100)
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    with pytest.raises(MemoryError):
        nd.find_bad_by_ransac(n_samples=n_samples)

    # Set n_samples to a float to trigger a type error
    n_samples = 35.5
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    with pytest.raises(TypeError):
        nd.find_bad_by_ransac(n_samples=n_samples)

    # Test IOError when too few good channels for RANSAC sample size
    n_chans = raw_tmp._data.shape[0]
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    nd.bad_by_deviation = raw_tmp.info["ch_names"][0 : int(n_chans * 0.8)]
    with pytest.raises(IOError):
        nd.find_bad_by_ransac()

    # Test IOError when not enough channels for RANSAC predictions
    raw_tmp._data[0 : (n_chans - 2), :] = 0  # make all channels flat except 2
    nd = NoisyChannels(raw_tmp, do_detrend=False)
    with pytest.raises(IOError):
        nd.find_bad_by_ransac()
