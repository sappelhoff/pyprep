"""
=================
Run the full PREP
=================


In this example we show how to run PREP with ``pyprep``. We also compare
:class:`prep_pipeline.PrepPipeline` with PREP's results in Matlab.

We use sample EEG data from Physionet EEG Motor Movement/Imagery Dataset:
`https://physionet.org/content/eegmmidb/1.0.0/ <https://physionet.org/content/eegmmidb/1.0.0/>`_

"""

# Authors: Aamna Lawrence <aamna.lawrence@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Victor Xiang <xiangliang3151@gmail.com>
#
# License: MIT
#

###############################################################################
#
# .. warning:: This functionality is work in progress.
#              Contributions are welcome!
#

###############################################################################
# First we import what we need for this example.
import os
import pathlib

import mne
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pyprep.prep_pipeline import PrepPipeline

###############################################################################
# Let's download some data for testing. Picking the 1st run of subject 4 here.

data_paths = mne.datasets.eegbci.load_data(subject=4, runs=1, update_path=True)

###############################################################################
# General settings and file paths
mne.set_log_level("WARNING")

# Raw data
fname_test_file = data_paths[0]

# mat files for validation
here = pathlib.Path("__file__").parent.absolute()

fname_mat1 = os.path.join(here, "matlab_results", "EEG_raw.mat")
fname_mat2 = os.path.join(here, "matlab_results", "EEGNew.mat")
fname_mat3 = os.path.join(here, "matlab_results", "EEG.mat")
fname_mat4 = os.path.join(here, "matlab_results", "EEGref.mat")
fname_mat5 = os.path.join(here, "matlab_results", "EEGinterp.mat")

###############################################################################
# Load data and prepare it
# ------------------------

raw = mne.io.read_raw_edf(fname_test_file, preload=True)

# The eegbci data has non-standard channel names. We need to rename them:
mne.datasets.eegbci.standardize(raw)

# Add a montage to the data
montage_kind = "standard_1005"
montage = mne.channels.make_standard_montage(montage_kind)

# Extract some info
sample_rate = raw.info["sfreq"]

# Make a copy of the data
raw_copy = raw.copy()

###############################################################################
# Set PREP parameters and run PREP
# --------------------------------
#
# Notes: We keep all the default parameter settings as described in the PREP,
# though for this case we think that the default fraction of bad time windows
# being 0.01 is too sensitive since it gots only 60 time windows (the EEG data
# is 60s long). As a result this example returns a lot of interpolated channels.

# Fit prep
prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(60, sample_rate / 2, 60),
}

prep = PrepPipeline(raw_copy, prep_params, montage)
prep.fit()

###############################################################################
# Results
# -------
#
# You can check the detected bad channels in each step of PREP.

print("Bad channels: {}".format(prep.interpolated_channels))
print("Bad channels original: {}".format(prep.noisy_channels_original["bad_all"]))
print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))

# Matlab's results
# ----------------
# Bad channels: Fc5, Fc3, Fc1, C3, Cp3, Cp4, Af3, Afz, Af8, F7, F5, F6, F8,
#               Ft8, P7, P2
# Bad channels original: Af3, Af4, Af7, Af8, Fp1, Fp2, Fpz, Ft8
# Bad channels after interpolation: Cp5, Fp2, Af7, F1

###############################################################################
# Validation
# ----------
#
# To validate each step of pyprep's results, we compare results after each step
# with the results from EEGLAB's PREP. To make it easy to compare, we rescale
# the EEG data to [-1, 1] (devided the data by maximum absolute value) when
# making the plot.

EEG_raw = raw_copy.get_data(picks="eeg") * 1e6
EEG_raw_max = np.max(abs(EEG_raw), axis=None)
EEG_raw_matlab = sio.loadmat(fname_mat1)
EEG_raw_matlab = EEG_raw_matlab["save_data"]
EEG_raw_diff = EEG_raw - EEG_raw_matlab
EEG_raw_mse = (EEG_raw_diff / EEG_raw_max ** 2).mean(axis=None)

fig, axs = plt.subplots(5, 3, "all", figsize=(16, 12))
plt.setp(fig, facecolor=[1, 1, 1])
fig.suptitle("Python versus Matlab PREP results", fontsize=16)

im = axs[0, 0].imshow(
    EEG_raw / EEG_raw_max,
    aspect="auto",
    extent=[0, (EEG_raw.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[0, 0].set_title("Python", fontsize=14)
axs[0, 1].imshow(
    EEG_raw_matlab / EEG_raw_max,
    aspect="auto",
    extent=[0, (EEG_raw_matlab.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[0, 1].set_title("Matlab", fontsize=14)
axs[0, 2].imshow(
    EEG_raw_diff / EEG_raw_max,
    aspect="auto",
    extent=[0, (EEG_raw_diff.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[0, 2].set_title("Difference", fontsize=14)
axs[0, 1].set_title("Original EEG", fontsize=14)
# axs[0, 0].set_ylabel('Channel Number', fontsize=14)
cb = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.04)
cb.set_label("\u03BCVolt", fontsize=14)

EEG_new_matlab = sio.loadmat(fname_mat2)
EEG_new_matlab = EEG_new_matlab["save_data"]
EEG_new = prep.EEG_new
EEG_new_max = np.max(abs(EEG_new), axis=None)
EEG_new_diff = EEG_new - EEG_new_matlab
EEG_new_mse = ((EEG_new_diff / EEG_new_max) ** 2).mean(axis=None)
axs[1, 0].imshow(
    EEG_new / EEG_new_max,
    aspect="auto",
    extent=[0, (EEG_new.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[1, 1].imshow(
    EEG_new_matlab / EEG_new_max,
    aspect="auto",
    extent=[0, (EEG_new_matlab.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[1, 2].imshow(
    EEG_new_diff / EEG_new_max,
    aspect="auto",
    extent=[0, (EEG_new_diff.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[1, 1].set_title("High pass filtered EEG", fontsize=14)
# axs[1, 0].set_ylabel('Channel Number', fontsize=14)

EEG_clean_matlab = sio.loadmat(fname_mat3)
EEG_clean_matlab = EEG_clean_matlab["save_data"]
EEG_clean = prep.EEG
EEG_max = np.max(abs(EEG_clean), axis=None)
EEG_diff = EEG_clean - EEG_clean_matlab
EEG_mse = ((EEG_diff / EEG_max) ** 2).mean(axis=None)
axs[2, 0].imshow(
    EEG_clean / EEG_max,
    aspect="auto",
    extent=[0, (EEG_clean.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[2, 1].imshow(
    EEG_clean_matlab / EEG_max,
    aspect="auto",
    extent=[0, (EEG_clean_matlab.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[2, 2].imshow(
    EEG_diff / EEG_max,
    aspect="auto",
    extent=[0, (EEG_diff.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[2, 1].set_title("Line-noise removed EEG", fontsize=14)
axs[2, 0].set_ylabel("Channel Number", fontsize=14)

EEG = prep.EEG_before_interpolation
EEG_max = np.max(abs(EEG), axis=None)
EEG_ref_mat = sio.loadmat(fname_mat4)
EEG_ref_matlab = EEG_ref_mat["save_EEG"]
reference_matlab = EEG_ref_mat["save_reference"]
EEG_ref_diff = EEG - EEG_ref_matlab
EEG_ref_mse = ((EEG_ref_diff / EEG_max) ** 2).mean(axis=None)
reference_signal = prep.reference_before_interpolation
reference_max = np.max(abs(reference_signal), axis=None)
reference_diff = reference_signal - reference_matlab
reference_mse = ((reference_diff / reference_max) ** 2).mean(axis=None)
axs[3, 0].imshow(
    EEG / EEG_max,
    aspect="auto",
    extent=[0, (EEG.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[3, 1].imshow(
    EEG_ref_matlab / EEG_max,
    aspect="auto",
    extent=[0, (EEG_ref_matlab.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[3, 2].imshow(
    EEG_ref_diff / EEG_max,
    aspect="auto",
    extent=[0, (EEG_ref_diff.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[3, 1].set_title("Referenced EEG", fontsize=14)
# axs[3, 0].set_ylabel('Channel Number', fontsize=14)

EEG_final = prep.raw.get_data() * 1e6
EEG_final_max = np.max(abs(EEG_final), axis=None)
EEG_final_matlab = sio.loadmat(fname_mat5)
EEG_final_matlab = EEG_final_matlab["save_data"]
EEG_final_diff = EEG_final - EEG_final_matlab
EEG_final_mse = ((EEG_final_diff / EEG_final_max) ** 2).mean(axis=None)
axs[4, 0].imshow(
    EEG_final / EEG_final_max,
    aspect="auto",
    extent=[0, (EEG_final.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[4, 1].imshow(
    EEG_final_matlab / EEG_final_max,
    aspect="auto",
    extent=[0, (EEG_final_matlab.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[4, 2].imshow(
    EEG_final_diff / EEG_final_max,
    aspect="auto",
    extent=[0, (EEG_final_diff.shape[1] / sample_rate), 63, 0],
    vmin=-1,
    vmax=1,
    cmap=plt.get_cmap("RdBu"),
)
axs[4, 1].set_title("Interpolated EEG", fontsize=14)
# axs[4, 0].set_ylabel('Channel Number', fontsize=14)
axs[4, 1].set_xlabel("Time(s)", fontsize=14)

###############################################################################
# Mean square error of each step:

print("Raw data MSE: {}".format(EEG_raw_mse))
print("Filtered data MSE: {}".format(EEG_new_mse))
print("Line-noise removed data MSE: {}".format(EEG_mse))
print("Referenced data MSE: {}".format(EEG_ref_mse))
print("Interpolated data MSE: {}".format(EEG_final_mse))

###############################################################################
# Discussion
# ----------
#
# It can be seen the results match well on each step except the final step.
# This is due to the difference of find_noisy_channel functions, since the
# channels with relatively large error corrsponds to the channels that are only
# interpolated in Python or Matlab.
#
# We think the differences mainly arise from
#
# 1. Difference in bad channels from Ransac criteria, including the random
#    number generator
# 2. Difference in some internal functions of Python and Matlab (e.g., filter
#    and interpolation function)
