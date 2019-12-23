import mne
import numpy as np
from mne.datasets import eegbci

from pyprep.find_noisy_channels import NoisyChannels


def test_findnoisychannels():
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
    a = mne.channels.make_standard_montage(kind="standard_1020")
    mne.set_log_level("WARNING")
    raw.set_montage(a)
    nd = NoisyChannels(raw)
    nd.find_all_bads(ransac=True)
    bads = nd.get_bads()
    iterations = (
        10  # remove any noisy channels by interpolating the bads for 10 iterations
    )
    for iter in range(0, iterations):
        raw.info["bads"] = bads
        raw.interpolate_bads()
        nd = NoisyChannels(raw)
        nd.find_all_bads(ransac=True)
        bads = nd.get_bads()

    # make sure no bad channels exist in the data
    if bads != []:
        raw.drop_channels(ch_names=bads)

    # Test for NaN and flat channels
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    # Insert a nan value for a random channel
    rand_chn_idx1 = int(np.random.randint(0, m, 1))
    rand_chn_idx2 = int(np.random.randint(0, m, 1))
    rand_chn_lab1 = raw_tmp.ch_names[rand_chn_idx1]
    rand_chn_lab2 = raw_tmp.ch_names[rand_chn_idx2]
    raw_tmp._data[rand_chn_idx1, n - 1] = np.nan
    raw_tmp._data[rand_chn_idx2, :] = np.ones(n)
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_nan_flat()
    assert nd.bad_by_nan == [rand_chn_lab1]
    assert nd.bad_by_flat == [rand_chn_lab2]

    # Test for high and low deviations in EEG data
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    # Now insert one random channel with very low deviations
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    raw_tmp._data[rand_chn_idx, :] = raw_tmp._data[rand_chn_idx, :] / 10
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_deviation()
    assert rand_chn_lab in nd.bad_by_deviation
    # Inserting one random channel with a high deviation
    raw_tmp = raw.copy()
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    arbitrary_scaling = 5
    raw_tmp._data[rand_chn_idx, :] *= arbitrary_scaling
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_deviation()
    assert rand_chn_lab in nd.bad_by_deviation

    # Test for correlation between EEG channels
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    # Use cosine instead of sine to create a signal
    low = 10
    high = 30
    n_freq = 5
    signal = np.zeros((1, n))
    for freq_i in range(n_freq):
        freq = np.random.randint(low, high, n)
        signal[0, :] += np.cos(2 * np.pi * raw.times * freq)
    raw_tmp._data[rand_chn_idx, :] = signal * 1e-6
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_correlation()
    assert rand_chn_lab in nd.bad_by_correlation

    # Test for high freq noise detection
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    # Use freqs between 90 and 100 Hz to insert hf noise
    signal = np.zeros((1, n))
    for freq_i in range(n_freq):
        freq = np.random.randint(90, 100, n)
        signal[0, :] += np.sin(2 * np.pi * raw.times * freq)
    raw_tmp._data[rand_chn_idx, :] = signal * 1e-6
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_hfnoise()
    assert rand_chn_lab in nd.bad_by_hf_noise

    # Test for signal to noise ratio in EEG data
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    # inserting an uncorrelated high frequency (90 Hz) signal in one channel
    raw_tmp[rand_chn_idx, :] = np.sin(2 * np.pi * raw.times * 90) * 1e-6
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_SNR()
    assert rand_chn_lab in nd.bad_by_SNR

    # Test for finding bad channels by RANSAC
    raw_tmp = raw.copy()
    # Ransac identifies channels that go bad together and are highly correlated.
    # Inserting highly correlated signal in channels 0 through 3 at 30 Hz
    raw_tmp._data[0:6, :] = np.cos(2 * np.pi * raw.times * 30) * 1e-6
    nd = NoisyChannels(raw_tmp)
    np.random.seed(30)
    nd.find_bad_by_ransac()
    bads = nd.bad_by_ransac
    assert bads == raw_tmp.ch_names[0:6]
