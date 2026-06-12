"""
=================
Run the full PREP
=================


In this example we show how to run the full PREP pipeline on a noisy EEG
recording using :class:`~pyprep.PrepPipeline`, inspect which channels it flags
as bad, and visualize the effect of cleaning.

.. currentmodule:: pyprep
"""  # noqa: D205 D400

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

###############################################################################
# First we import what we need for this example.
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci

from pyprep.prep_pipeline import PrepPipeline

###############################################################################
# Load some data
# --------------
#
# We use subject 4, run 1 from the PhysioNet BCI2000 (eegbci) dataset. This is
# a fairly noisy recording, which makes it a good candidate for demonstrating
# PREP. The data are 64-channel EEG sampled at 160 Hz.

mne.set_log_level("WARNING")

edf_fpath = eegbci.load_data(subjects=4, runs=1, update_path=True)[0]
raw = mne.io.read_raw_edf(edf_fpath, preload=True)

# The eegbci data ships with non-standard channel names, so we fix them and
# attach a standard montage (PREP needs sensor positions to interpolate).
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

# Keep an untouched copy of the original data for a before/after comparison.
raw_before = raw.copy()

###############################################################################
# Set PREP parameters and run the pipeline
# ----------------------------------------
#
# - ``ref_chs`` / ``reref_chs``: the channels used to build, and to be cleaned
#   by, the robust average reference. ``"eeg"`` means "all EEG channels".
# - ``line_freqs``: the line-noise frequencies to remove. This is a US
#   recording, so line noise sits at 60 Hz and its harmonics below Nyquist.
#
# We pass a fixed ``random_state`` so that RANSAC (and therefore the set of
# detected bad channels) is reproducible from run to run.

sfreq = raw.info["sfreq"]
prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(60, sfreq / 2, 60),
}

prep = PrepPipeline(raw, prep_params, montage, random_state=42)
prep.fit()

###############################################################################
# Inspect the detected bad channels
# ----------------------------------
#
# PREP records the channels it considers bad at several points in the pipeline.
# The most useful summaries are:
#
# - ``noisy_channels_original``: bad channels found on the raw data, before any
#   robust referencing.
# - ``interpolated_channels``: channels that were bad after robust referencing
#   and were therefore interpolated.
# - ``still_noisy_channels``: channels that remain bad even after
#   interpolation (ideally an empty list).

print(f"Bad channels (original): {prep.noisy_channels_original['bad_all']}")
print(f"Interpolated channels:   {prep.interpolated_channels}")
print(f"Still noisy afterwards:  {prep.still_noisy_channels}")

###############################################################################
# Visualize the effect of PREP
# ----------------------------
#
# A power spectral density (PSD) plot is a compact way to see what PREP did.
# Compared to the raw data, the cleaned data should show the 60 Hz line-noise
# peak removed and a tighter spread across channels (bad channels no longer
# stick out, having been interpolated from their neighbors).

fig, axs = plt.subplots(
    1, 2, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True
)

raw_before.compute_psd(fmax=80).plot(axes=axs[0], show=False)
axs[0].set_title("Before PREP")

prep.raw.compute_psd(fmax=80).plot(axes=axs[1], show=False)
axs[1].set_title("After PREP")

###############################################################################
# The cleaned recording in ``prep.raw`` is a standard :class:`mne.io.Raw`
# object, so you can carry on with the rest of your analysis pipeline (epoching,
# filtering, ICA, etc.) directly from there.
