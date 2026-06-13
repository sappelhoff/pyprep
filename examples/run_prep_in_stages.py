"""
======================
Run PREP in stages
======================


:meth:`~pyprep.PrepPipeline.fit` runs the whole PREP pipeline in one call, but
sometimes you want to drive the individual stages yourself, for example to log
progress, plot intermediate results, or decide what to do with bad channels
*before* committing to interpolation.

This example shows how to run PREP stage by stage with the
:meth:`~pyprep.PrepPipeline.remove_line_noise` and
:meth:`~pyprep.PrepPipeline.robust_reference` methods, and how to skip the final
bad-channel interpolation via ``interpolate_bads=False``.

.. currentmodule:: pyprep
"""  # noqa: D205 D400

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

###############################################################################
# First we import what we need for this example.
import mne
import numpy as np
from mne.datasets import eegbci

from pyprep.prep_pipeline import PrepPipeline

###############################################################################
# Load some data
# --------------
#
# As in the "Run the full PREP" example, we use subject 4, run 1 from the
# PhysioNet BCI2000 (eegbci) dataset: a fairly noisy 64-channel recording
# sampled at 160 Hz.

mne.set_log_level("WARNING")

edf_fpath = eegbci.load_data(subjects=4, runs=1, update_path=True)[0]
raw = mne.io.read_raw_edf(edf_fpath, preload=True)

# Fix the non-standard channel names and attach a montage (PREP needs sensor
# positions to interpolate).
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

###############################################################################
# Set up the pipeline
# -------------------
#
# We fix the RANSAC random seed to make the example reproducible. As in the
# other examples, this is a single high-entropy value drawn once via
# ``secrets.randbits(31)`` rather than a hand-picked "magic" number, to avoid
# (even unconsciously) cherry-picking a flattering seed. See NumPy's RNG
# guidance: https://numpy.org/doc/stable/reference/random/index.html
random_state = 1507900087

sfreq = raw.info["sfreq"]
prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(60, sfreq / 2, 60),  # 60 Hz line noise + harmonics
}

prep = PrepPipeline(raw, prep_params, montage, random_state=random_state)

###############################################################################
# Stage 1: line-noise removal
# ---------------------------
#
# :meth:`~pyprep.PrepPipeline.remove_line_noise` runs the adaptive line-noise
# removal step on its own. With no argument it uses the ``line_freqs`` from
# ``prep_params``. This is exactly what :meth:`~pyprep.PrepPipeline.fit` does as
# its first step.

prep.remove_line_noise()

###############################################################################
# Stage 2: robust referencing *without* interpolation
# ----------------------------------------------------
#
# :meth:`~pyprep.PrepPipeline.robust_reference` estimates and applies the robust
# average reference and detects the channels that are still bad afterwards. By
# passing ``interpolate_bads=False`` we stop *before* interpolating them, so we
# can inspect (and decide what to do with) the bad channels ourselves.

prep.robust_reference(interpolate_bads=False)

# Channels detected as bad after robust referencing, but not yet interpolated.
# We coerce names to plain ``str`` for tidy printing (some detection methods
# return ``numpy.str_``).
bad_after_reference = sorted(map(str, prep.bad_before_interpolation))
print(f"Bad after robust referencing: {bad_after_reference}")

# Because we skipped interpolation, the post-interpolation outputs are not
# populated yet:
print(f"interpolated_channels: {prep.interpolated_channels}")
print(f"still_noisy_channels:  {prep.still_noisy_channels}")

###############################################################################
# Decide what to do with the bad channels
# ----------------------------------------
#
# At this point you have full control. You could, for example, simply mark the
# bad channels on the (robustly referenced) data and drop or handle them
# yourself in MNE, instead of interpolating:

raw_reref = prep.raw.copy()
raw_reref.info["bads"] = bad_after_reference
# e.g. ``raw_reref.drop_channels(bad_after_reference)`` to discard them.

###############################################################################
# Stage 3 (optional): interpolate the remaining bad channels
# ----------------------------------------------------------
#
# If instead you *do* want PREP's default behaviour, you would simply leave
# ``interpolate_bads`` at its default of ``True`` (or just call
# :meth:`~pyprep.PrepPipeline.fit`). Running the three stages manually like this
# is equivalent to a single :meth:`~pyprep.PrepPipeline.fit` call::
#
#     prep = PrepPipeline(raw, prep_params, montage, random_state=random_state)
#     prep.remove_line_noise()
#     prep.robust_reference()  # interpolate_bads=True by default
#
# is the same as::
#
#     prep = PrepPipeline(raw, prep_params, montage, random_state=random_state)
#     prep.fit()
#
# The staged API just lets you step in between.
