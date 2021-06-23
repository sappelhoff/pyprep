"""
===============================================
RANSAC comparison between pyprep and autoreject
===============================================

Next to the RANSAC implementation in ``pyprep``,
there is another implementation that makes use of MNE-Python.
That alternative RANSAC implementation can be found in the
`"autoreject" package <https://github.com/autoreject/autoreject/>`_.

In this example, we make a basic comparison between the two implementations.

#. by running them on the same simulated data
#. by running them on the same "real" data


.. currentmodule:: pyprep
"""

# Authors: Yorguin Mantilla <yjmantilla@gmail.com>
#
# License: MIT

# %%
# First we import what we need for this example.
import numpy as np
import mne
from scipy import signal as signal
from time import perf_counter
from autoreject import Ransac
import pyprep.ransac as ransac_pyprep


# %%
# Now let's make some arbitrary MNE raw object for demonstration purposes.
# We will think of good channels as sine waves and bad channels correlated with
# each other as sawtooths. The RANSAC will be biased towards sines in its
# prediction (they are the majority) so it will identify the sawtooths as bad.

# Set a random seed to make this example reproducible
random_state = 435656
rng = np.random.RandomState(random_state)

# start defining some key aspects for our simulated data
sfreq = 1000.0
montage = mne.channels.make_standard_montage("standard_1020")
ch_names = montage.ch_names
n_chans = len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_chans)
time = np.arange(0, 30, 1.0 / sfreq)  # 30 seconds of recording

# randomly pick some "bad" channels (sawtooths)
n_bad_chans = 3
bad_channels = rng.choice(np.arange(n_chans), n_bad_chans, replace=False)
bad_channels = [int(i) for i in bad_channels]
bad_ch_names = [ch_names[i] for i in bad_channels]

# The frequency components to use in the signal for good and bad channels
freq_good = 20
freq_bad = 20

# Generate the data: sinewaves for "good", sawtooths for "bad" channels
X = [
    signal.sawtooth(2 * np.pi * freq_bad * time)
    if i in bad_channels
    else np.sin(2 * np.pi * freq_good * time)
    for i in range(n_chans)
]

# Scale the signal amplitude and add noise.
X = 2e-5 * np.array(X) + 1e-5 * np.random.random((n_chans, time.shape[0]))

# Finally, put it all together as an mne "Raw" object.
raw = mne.io.RawArray(X, info)
raw.set_montage(montage, verbose=False)

# Print, which channels are simulated as "bad"
print(bad_ch_names)

# %%
# Configure RANSAC parameters
n_samples = 50
fraction_good = 0.25
corr_thresh = 0.75
fraction_bad = 0.4
corr_window_secs = 5.0

# %%
# autoreject's RANSAC
ransac_ar = Ransac(
    picks=None,
    n_resample=n_samples,
    min_channels=fraction_good,
    min_corr=corr_thresh,
    unbroken_time=fraction_bad,
    n_jobs=1,
    random_state=random_state,
)
epochs = mne.make_fixed_length_epochs(
    raw,
    duration=corr_window_secs,
    preload=True,
    reject_by_annotation=False,
    verbose=None,
)

start_time = perf_counter()
ransac_ar = ransac_ar.fit(epochs)
print("--- %s seconds ---" % (perf_counter() - start_time))

corr_ar = ransac_ar.corr_
bad_by_ransac_ar = ransac_ar.bad_chs_

# Check channels that go bad together by RANSAC
print("autoreject bad chs:", bad_by_ransac_ar)
assert set(bad_ch_names) == set(bad_by_ransac_ar)

# %%
# pyprep's RANSAC

start_time = perf_counter()
bad_by_ransac_pyprep, corr_pyprep = ransac_pyprep.find_bad_by_ransac(
    data=raw._data.copy(),
    sample_rate=raw.info["sfreq"],
    complete_chn_labs=np.asarray(raw.info["ch_names"]),
    chn_pos=raw._get_channel_positions(),
    exclude=[],
    n_samples=n_samples,
    sample_prop=fraction_good,
    corr_thresh=corr_thresh,
    frac_bad=fraction_bad,
    corr_window_secs=corr_window_secs,
    channel_wise=False,
    random_state=rng,
)
print("--- %s seconds ---" % (perf_counter() - start_time))

# Check channels that go bad together by RANSAC
print("pyprep bad chs:", bad_by_ransac_pyprep)
assert set(bad_ch_names) == set(bad_by_ransac_pyprep)

# %%
# Now we test the algorithms on real EEG data.
# Let's download some data for testing.
data_paths = mne.datasets.eegbci.load_data(subject=4, runs=1, update_path=True)
fname_test_file = data_paths[0]

# %%
# Load data and prepare it

raw = mne.io.read_raw_edf(fname_test_file, preload=True)

# The eegbci data has non-standard channel names. We need to rename them:
mne.datasets.eegbci.standardize(raw)

# Add a montage to the data
montage_kind = "standard_1005"
montage = mne.channels.make_standard_montage(montage_kind)
raw.set_montage(montage)


# %%
# autoreject's RANSAC
ransac_ar = Ransac(
    picks=None,
    n_resample=n_samples,
    min_channels=fraction_good,
    min_corr=corr_thresh,
    unbroken_time=fraction_bad,
    n_jobs=1,
    random_state=random_state,
)
epochs = mne.make_fixed_length_epochs(
    raw,
    duration=corr_window_secs,
    preload=True,
    reject_by_annotation=False,
    verbose=None,
)

start_time = perf_counter()
ransac_ar = ransac_ar.fit(epochs)
print("--- %s seconds ---" % (perf_counter() - start_time))

corr_ar = ransac_ar.corr_
bad_by_ransac_ar = ransac_ar.bad_chs_

# Check channels that go bad together by RANSAC
print("autoreject bad chs:", bad_by_ransac_ar)


# %%
# pyprep's RANSAC

start_time = perf_counter()
bad_by_ransac_pyprep, corr_pyprep = ransac_pyprep.find_bad_by_ransac(
    data=raw._data.copy(),
    sample_rate=raw.info["sfreq"],
    complete_chn_labs=np.asarray(raw.info["ch_names"]),
    chn_pos=raw._get_channel_positions(),
    exclude=[],
    n_samples=n_samples,
    sample_prop=fraction_good,
    corr_thresh=corr_thresh,
    frac_bad=fraction_bad,
    corr_window_secs=corr_window_secs,
    channel_wise=False,
    random_state=rng,
)
print("--- %s seconds ---" % (perf_counter() - start_time))

# Check channels that go bad together by RANSAC
print("pyprep bad chs:", bad_by_ransac_pyprep)
