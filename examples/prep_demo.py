"""
===================================
01. Run Pyprep On EEG Data Demo
===================================

In this example, we use pyprep to run on a sample of EEG data.

"""

# Authors: Liang Xiang, Aamna Lawrence, Adam Li <adam2392@gmail.com>
import os

import mne
import numpy as np
import scipy.io as sio
from mne.datasets import eegbci

from pyprep.prep_pipeline import PrepPipeline

mne.set_log_level("WARNING")

# download subject 1, runs 1,2,3,4
# eegbci.load_data(1, [1, 2, 3, 4])
# eegbci.load_data(2, [1], DATASET_DIR)
# eegbci.load_data(3, 1, DATASET_DIR)
edf_fpaths = eegbci.load_data(4, 1)

raw = mne.io.read_raw_edf(edf_fpaths[0], preload=True)
raw.rename_channels(lambda s: s.strip("."))
montage_kind = "standard_1020"

eeg_index = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
raw_copy = raw.copy()
ch_names = raw_copy.info["ch_names"]
ch_names_eeg = list(np.asarray(ch_names)[eeg_index])
sample_rate = raw_copy.info["sfreq"]

###############################################################################
# Step 1: Set PREP parameters and run PREP
#
# Notes: we keep all the default parameter settings as described in [PREP paper](https://www.ncbi.nlm.nih.gov/pubmed/26150785)
# except one, the fraction of bad time windows (we change it from 0.01 to 0.1), because the EEG data is 60s long,
# which means it gots only 60 time windows. We think the algorithm will be too sensitive if using the default setting.

prep_params = {
    "ref_chs": ch_names_eeg,
    "reref_chs": ch_names_eeg,
    "line_freqs": np.arange(60, sample_rate / 2, 60),
}
prep = PrepPipeline(raw_copy, prep_params)
prep.fit()

# check results
print("Bad channels: {}".format(prep.interpolated_channels))
print("Bad channels original: {}".format(prep.noisy_channels_original["bad_all"]))
print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))

###############################################################################
# Step 2: Validation
#
# To validate each step of pyprep's results, we compare results after each step with the results from EEGLAB's PREP
#
# To make it easy to compare, we rescale the EEG data to [-1, 1] (devided the data by maximum absolute value) when making the plot

EEG_raw = raw_copy.get_data(picks="eeg") * 1e6
EEG_raw_max = np.max(abs(EEG_raw), axis=None)
EEG_raw_matlab = sio.loadmat("./pyprep/examples/Matlab results/EEG_raw.mat")
EEG_raw_matlab = EEG_raw_matlab["save_data"]
EEG_raw_diff = EEG_raw - EEG_raw_matlab
EEG_raw_mse = (EEG_raw_diff / EEG_raw_max ** 2).mean(axis=None)

EEG_new_matlab = sio.loadmat("./pyprep/examples/Matlab results/EEGNew.mat")
EEG_new_matlab = EEG_new_matlab["save_data"]
EEG_new = prep.EEG_new
EEG_new_max = np.max(abs(EEG_new), axis=None)
EEG_new_diff = EEG_new - EEG_new_matlab
EEG_new_mse = ((EEG_new_diff / EEG_new_max) ** 2).mean(axis=None)

EEG_clean_matlab = sio.loadmat("./pyprep/examples/Matlab results/EEG.mat")
EEG_clean_matlab = EEG_clean_matlab["save_data"]
EEG_clean = prep.EEG
EEG_max = np.max(abs(EEG_clean), axis=None)
EEG_diff = EEG_clean - EEG_clean_matlab
EEG_mse = ((EEG_diff / EEG_max) ** 2).mean(axis=None)

EEG = prep.EEG_before_interpolation
EEG_max = np.max(abs(EEG), axis=None)
EEG_ref_mat = sio.loadmat("./pyprep/examples/Matlab results/EEGref.mat")
EEG_ref_matlab = EEG_ref_mat["save_EEG"]
reference_matlab = EEG_ref_mat["save_reference"]
EEG_ref_diff = EEG - EEG_ref_matlab
EEG_ref_mse = ((EEG_ref_diff / EEG_max) ** 2).mean(axis=None)
reference_signal = prep.reference_before_interpolation
reference_max = np.max(abs(reference_signal), axis=None)
reference_diff = reference_signal - reference_matlab
reference_mse = ((reference_diff / reference_max) ** 2).mean(axis=None)

EEG_final = prep.raw.get_data() * 1e6
EEG_final_max = np.max(abs(EEG_final), axis=None)
EEG_final_matlab = sio.loadmat("./pyprep/examples/Matlab results/EEGinterp.mat")
EEG_final_matlab = EEG_final_matlab["save_data"]
EEG_final_diff = EEG_final - EEG_final_matlab
EEG_final_mse = ((EEG_final_diff / EEG_final_max) ** 2).mean(axis=None)

print("Raw data MSE: {}".format(EEG_raw_mse))
print("Filtered data MSE: {}".format(EEG_new_mse))
print("Line-noise removed data MSE: {}".format(EEG_mse))
print("Referenced data MSE: {}".format(EEG_ref_mse))
print("Interpolated data MSE: {}".format(EEG_final_mse))
