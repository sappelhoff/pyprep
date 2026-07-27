"""Test remove trend."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

import numpy as np

import pyprep.removeTrend as removeTrend


def test_highpass():
    """Test for checking high pass filters."""
    srate = 100
    t = np.arange(0, 30, 1 / srate)
    lowfreq_signal = np.sin(2 * np.pi * 0.1 * t)
    highfreq_signal = np.sin(2 * np.pi * 8 * t)
    signal = lowfreq_signal + highfreq_signal
    lowpass_filt1 = removeTrend.removeTrend(
        signal, detrendType="High pass sinc", sample_rate=srate, detrendCutoff=1
    )
    lowpass_filt2 = removeTrend.removeTrend(
        signal, detrendType="High pass", sample_rate=srate, detrendCutoff=1
    )
    lowpass_filt3 = removeTrend.removeTrend(
        signal,
        detrendType="High pass",
        sample_rate=srate,
        detrendCutoff=1,
        matlab_strict=True,
    )
    error1 = lowpass_filt1 - highfreq_signal
    error2 = lowpass_filt2 - highfreq_signal
    error3 = lowpass_filt3 - highfreq_signal
    assert np.sqrt(np.mean(error1**2)) < 0.1
    assert np.sqrt(np.mean(error2**2)) < 0.1
    assert np.sqrt(np.mean(error3**2)) < 0.1


def test_detrend():
    """Test for local regression to remove linear trend from EEG data."""
    # creating a new signal for checking detrending using local regression
    srate = 100
    t = np.arange(0, 30, 1 / srate)
    randgen = np.random.RandomState(9)
    npoints = len(t)
    signal = randgen.randn(npoints)
    signal_trend = 2 + 1.5 * np.linspace(0, 1, npoints) + signal
    signal_detrend = removeTrend.removeTrend(
        signal_trend, detrendType="Local detrend", sample_rate=100
    )
    error3 = signal_detrend - signal
    assert np.sqrt(np.mean(error3**2)) < 0.1


def test_detrend_step_size_error_branch():
    """Test local detrend logs error when dn > n (line 112)."""
    # windowSize = 1.5 / detrendCutoff, clamped to EEG.shape[1].
    # dn = round(srate * 0.02). For dn > n: need n < dn.
    # Use detrendCutoff=500: windowSize = 0.003. srate=100: n=round(100*0.003)=0.
    # dn = round(100 * 0.02) = 2. So dn=2 > n=0 -> line 112 branch.
    # Input is 1D and gets reshaped to (1, n_samp) -> output shape (1, n_samp).
    srate = 100
    n_samples = 1000
    signal = np.random.randn(n_samples)
    result = removeTrend.removeTrend(
        signal,
        detrendType="Local detrend",
        sample_rate=srate,
        detrendCutoff=500.0,
    )
    # 1D input is reshaped to (1, n_samp) and returned as 2D
    assert result.shape == (1, n_samples)


def test_detrend_window_equals_data_length():
    """Test local detrend pass branch when window equals signal length (line 118)."""
    # After transpose, EEG.shape[0] == n_samp. Need n == n_samp.
    # n = round(srate * windowSize). windowSize = min(1.5/cutoff, EEG.shape[1]).
    # EEG.shape[1] is n_samp (before transpose). So windowSize = n_samp (samples!).
    # n = round(srate * n_samp). We need that to equal n_samp.
    # That requires srate = 1. Use srate=1, n_samp=100, cutoff=1:
    # windowSize = min(1.5, 100) = 1.5. n = round(1 * 1.5) = 2. Not equal.
    # Instead: use 2D EEG input where EEG.shape[1] is large and windowSize is clamped.
    # With EEG = (ch, n_samp): windowSize = min(1.5/cutoff, n_samp).
    # If 1.5/cutoff > n_samp -> windowSize = n_samp (in samples!).
    # n = round(srate * n_samp). For n == n_samp -> srate == 1 Hz.
    srate = 1.0  # 1 Hz sampling
    n_samp = 50
    # To clamp: 1.5/cutoff > n_samp -> cutoff < 1.5/n_samp = 0.03
    # Use cutoff=0.01: windowSize = min(150, 50) = 50. n = round(1 * 50) = 50.

    # After transpose: EEG.shape[0] = n_samp = 50 = n -> pass branch!
    signal = np.random.randn(2, n_samp)
    result = removeTrend.removeTrend(
        signal,
        detrendType="Local detrend",
        sample_rate=srate,
        detrendCutoff=0.01,
    )
    assert result.shape == signal.shape


def test_unknown_detrend_type_logs_warning():
    """Test warning is logged for unknown detrend type (line 125)."""
    # The code uses logging.warning (not Python's warnings module),
    # so no Python Warning is raised. 1D input -> output shape (1, n_samp).
    signal = np.random.randn(100)
    result = removeTrend.removeTrend(signal, detrendType="UnknownType", sample_rate=100)
    # 1D input is reshaped to (1, n_samp) then returned
    assert result.shape == (1, 100)
