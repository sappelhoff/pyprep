import numpy as np
import pyprep.removeTrend as removeTrend
import pytest

def test_highpass():
    """Test for checking high pass filters"""

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
    error1 = lowpass_filt1 - highfreq_signal
    error2 = lowpass_filt2 - highfreq_signal
    assert np.sqrt(np.mean(error1 ** 2)) < 0.1
    assert np.sqrt(np.mean(error2 ** 2)) < 0.1


def test_detrend():
    """Test for local regression to remove linear trend from EEG data"""

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
    assert np.sqrt(np.mean(error3 ** 2)) < 0.1

