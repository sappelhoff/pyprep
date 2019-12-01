import numpy as np
from find_noisy_channels import NoisyChannels
import mne

# using sample EEG data (https://physionet.org/content/eegmmidb/1.0.0/)
raw = mne.io.read_raw_edf("C:\\Users\\Aamna\\Desktop\\NDD\\S001R01.edf", preload=True)
raw.rename_channels(lambda s: s.strip("."))
a = mne.channels.read_montage(kind="standard_1020", ch_names=raw.info["ch_names"])
mne.set_log_level("WARNING")
raw.set_montage(a)
nd = NoisyChannels(raw)
nd.find_all_noisy_channels(ransac=True)
bads = nd.get_bads()
iterations = 3  # remove any noisy channels by interpolating the bads for 10 iterations
for iter in range(0, iterations):
    raw.info["bads"] = bads
    raw.interpolate_bads()
    nd = NoisyChannels(raw)
    nd.find_all_noisy_channels(ransac=True)
    bads = nd.get_bads()

# make sure no bad channels exist in the data
if bads != []:
    raw.drop_channels(ch_names=bads)


def test_find_bad_by_nan_flat(raw=raw):
    """Test find_bad_by_nan_flat.

    Parameters
    __________
    raw: raw mne object
         raw mne object having the EEG data and other fields
    """

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


def test_find_bad_by_deviation(raw=raw):
    """Test find_bad_by_deviation.

    Parameters
    __________
    raw: raw mne object
         raw mne object having the EEG data and other fields
    """
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    # Now insert one random channel with very low deviations
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    raw_tmp._data[rand_chn_idx, :] = raw_tmp._data[rand_chn_idx, :] / 10
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_deviation()
    assert nd.bad_by_deviation == [rand_chn_lab]
    # Inserting one random channel with a high deviation
    raw_tmp = raw.copy()
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    arbitrary_scaling = 5
    raw_tmp._data[rand_chn_idx, :] *= arbitrary_scaling
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_deviation()
    assert nd.bad_by_deviation == [rand_chn_lab]


def test_find_bad_by_correlation(raw=raw, low_freq=10.0, high_freq=40.0, n_freq=5):
    """Test find_bad_by_correlation.

    Parameters
    __________
    raw: raw mne object
         raw mne object having the EEG data and other fields

    low_freq: float
              lowest frequency in the signal used for testing channels
    high_freq: float
               highest frequency in the signal used for testing channels
    n_freq: int
            number of frequency components in the test signal
    """
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    # Use cosine instead of sine to create a signal
    low = low_freq
    high = high_freq
    signal = np.zeros((1, n))
    for freq_i in range(n_freq):
        freq = np.random.randint(low, high, n)
        signal[0, :] += np.cos(2 * np.pi * raw.times * freq)
    raw_tmp._data[rand_chn_idx, :] = signal * 1e-6
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_correlation()
    assert nd.bad_by_correlation == [rand_chn_lab]


def test_find_bad_by_hf_noise(raw=raw, n_freq=5):

    """Test find_bad_by_hf_noise.

    Parameters
    __________
    raw: raw mne object
         raw mne object having the EEG data and other fields
    n_freq: int
            number of frequency components used to construct the noise signal
    """
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
    assert nd.bad_by_hf_noise == [rand_chn_lab]


def test_find_bad_by_SNR(raw=raw):
    """Test find_bad_by_SNR

    Parameters
    _________
    raw: raw mne object
         raw mne object having the EEG data and other fields
    """
    raw_tmp = raw.copy()
    m, n = raw_tmp._data.shape
    rand_chn_idx = int(np.random.randint(0, m, 1))
    rand_chn_lab = raw_tmp.ch_names[rand_chn_idx]
    # inserting an uncorrelated high frequency (90 Hz) signal in one channel
    raw_tmp[rand_chn_idx, :] = np.sin(2 * np.pi * raw.times * 90) * 1e-6
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_SNR()
    assert nd.bad_by_SNR == [rand_chn_lab]


def test_find_bad_by_ransac(raw=raw):
    """Test find_bad_by_ransac.

    Parameters
    __________
    raw: raw mne object
         raw mne object having the EEG data and other fields
    """
    raw_tmp = raw.copy()
    # Ransac identifies channels that go bad together and are highly correlated.
    # Inserting highly correlated signal in channels 0 through 3 at 30 Hz
    raw_tmp._data[0:6, :] = np.cos(2 * np.pi * raw.times * 30) * 1e-6
    nd = NoisyChannels(raw_tmp)
    nd.find_bad_by_ransac()
    bads = nd.bad_by_ransac
    assert bads == raw_tmp.ch_names[0:6]


test_find_bad_by_nan_flat(raw)
test_find_bad_by_ransac(raw)
test_find_bad_by_hf_noise(raw)
test_find_bad_by_correlation(raw)
test_find_bad_by_SNR(raw)
