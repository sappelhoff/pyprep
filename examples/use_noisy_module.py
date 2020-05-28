"""
====================
Use the Noisy module
====================

In this example we demonstrate how to use the :mod:`pyprep.noisy` module.

"""

# Authors: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
# License: MIT

###############################################################################
# First we import what we need for this example.
import numpy as np
import mne

from pyprep.noisy import Noisydata

###############################################################################
# Now let's make some arbitrary MNE raw object for demonstration purposes.

sfreq = 1000.0
n_chans = 6
ch_names = ["Fpz", "Fz", "FCz", "Cz", "Pz", "Oz"]

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_chans)

time = np.arange(0, 60, 1.0 / sfreq)  # 60 seconds of recording
X = np.random.random((n_chans, time.shape[0]))
raw = mne.io.RawArray(X, info)
print(raw)


###############################################################################
# Assign the mne object to the :class:`Noisydata` class. The resulting object
# will be the place where all following methods are performed.

nd = Noisydata(raw)


###############################################################################
# Find all bad channels and print a summary
nd.find_all_bads(ransac=False)
bads = nd.get_bads(verbose=True)

###############################################################################
# Now the bad channels are saved in `bads` and we can continue processing our
# `raw` object. For more information, we can access attributes of the ``nd``
# instance:

# Check the high frequency noise per channel
print(nd._channel_hf_noise)

# and so on ...

###############################################################################
# For finding bad epochs, it is recommended to highpass the data or apply
# baseline correction. Furthermore, bad channels should be identified and
# removed or interpolated beforehand.

from pyprep.noisy import find_bad_epochs  # noqa: E402

# Turn our arbitrary data into epochs
events = np.array(
    [[10000, 0, 1], [20000, 0, 2], [30000, 0, 1], [40000, 0, 2], [50000, 0, 1]]
)

epochs = mne.Epochs(raw, events)

###############################################################################
# Now find the bad epochs
# You can also define picks, and the threshold to be used
bads = find_bad_epochs(epochs, picks=None, thresh=3.29053)

print(bads)
