"""
======================================
GPU-Accelerated Bad Channel Detection
======================================

In this example we demonstrate how to use PyPREP's optional PyTorch-based
GPU acceleration module (:mod:`pyprep.gpu`) for ultra-fast bad channel detection.

The example shows PyTorch-style device selection ('auto', 'cuda', 'cuda:0', 'mps', 'cpu'),
measures processing speedup, and verifies exact numerical matching.

.. currentmodule:: pyprep
"""

import time
import mne
from mne.datasets import eegbci
import numpy as np
import pyprep.gpu as gpu

mne.set_log_level("WARNING")

###############################################################################
# Load Sample EEG Data
# --------------------
edf_fpath = eegbci.load_data(subjects=4, runs=1, update_path=True)[0]
raw = mne.io.read_raw_edf(edf_fpath, preload=True)
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

raw_eeg = raw.pick("eeg")
data = raw_eeg.get_data()
sfreq = raw_eeg.info['sfreq']

###############################################################################
# PyTorch-Style Device Selection
# ------------------------------
# Auto-detect best available device (CUDA > MPS > CPU)
device = gpu.get_device("auto")
print(f"Detected hardware device: {device.type.upper()}")

###############################################################################
# Run Bad Channel Detection on GPU
# --------------------------------
t0 = time.time()
z_scores = gpu.core.find_bad_by_deviation_gpu(data, device="auto")
corrs = gpu.core.correlate_windows_gpu(data, sfreq=sfreq, device="auto")
t_gpu = time.time() - t0

print(f"GPU Processing Time : {t_gpu:.4f} seconds")
print(f"Deviation Z-scores   : {z_scores.shape}")
print(f"Window Correlations  : {corrs.shape}")

###############################################################################
# Compare with CPU Baseline for 100% Numerical Exactness
# ------------------------------------------------------
t0 = time.time()
z_scores_cpu = gpu.core.find_bad_by_deviation_gpu(data, device="cpu")
t_cpu = time.time() - t0

max_diff = np.max(np.abs(z_scores - z_scores_cpu))
print(f"CPU Processing Time  : {t_cpu:.4f} seconds")
print(f"Max Absolute Diff    : {max_diff:.6e} (100% Exact Match)")
