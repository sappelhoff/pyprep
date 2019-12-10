"""Module for PREP pipeline."""
import mne
import numpy as np

from pyprep.utilities import _union, _set_diff
from pyprep.removeTrend import removeTrend
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.reference import Reference


class PrepPipeline:
    """Early stage preprocessing (PREP) of EEG data.

    This class implements the functionality  of the PREP (preprocessing pipeline) for EEG data described in [1].

    Parameters
    ----------
        raw : raw mne object
        prep_params : dict
            Parameters of PREP which include at least the following keys:
            ref_chs : list
                A list of channel names to be used for rereferencing [default: all channels]
            reref_chs: list
                A list of channel names to be used for line-noise removed, and referenced [default: all channels]
            line_freqs : 1d array
                A list of line frequencies to be removed
        montage_kind : str
            Which kind of montage should be used to infer the electrode
            positions? E.g., 'standard_1020'
        ransac : boolean
            Whether or not to use ransac

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       EEG analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(self, raw, prep_params, montage_kind="standard_1020", ransac=True):
        """Initialize PREP class."""
        self.raw = raw.copy()
        self.ch_names = self.raw.ch_names
        montage = mne.channels.read_montage(kind=montage_kind, ch_names=self.ch_names)
        self.raw.set_montage(montage)
        self.raw.pick_types(eeg=True, eog=False, meg=False)
        self.ch_names_eeg = self.raw.ch_names
        self.EEG_raw = self.raw.get_data() * 1e6
        self.prep_params = prep_params
        self.sfreq = self.raw.info["sfreq"]
        self.ransac = ransac

    def fit(self):
        """Run the whole PREP pipeline."""
        noisy_detector = NoisyChannels(self.raw)
        noisy_detector.find_bad_by_nan_flat()
        unusable_channels = _union(
            noisy_detector.bad_by_nan, noisy_detector.bad_by_flat
        )
        reference_channels = _set_diff(self.prep_params["ref_chs"], unusable_channels)
        # Step 1: 1Hz high pass filtering
        self.EEG_new = removeTrend(self.EEG_raw, sample_rate=self.sfreq)

        # Step 2: Removing line noise
        linenoise = self.prep_params["line_freqs"]
        self.EEG_clean = mne.filter.notch_filter(
            self.EEG_new,
            Fs=self.sfreq,
            freqs=linenoise,
            method="spectrum_fit",
            mt_bandwidth=2,
            p_value=0.01,
        )

        # Add Trend back
        self.EEG = self.EEG_raw - self.EEG_new + self.EEG_clean
        self.raw._data = self.EEG * 1e-6

        # Step 3: Referencing
        reference = Reference(self.raw, self.prep_params, ransac=self.ransac)
        self.raw = reference.perform_reference()
        self.noisy_channels_original = reference.noisy_channels_original
        self.bad_before_interpolation = reference.bad_before_interpolation
        self.EEG_before_interpolation = reference.EEG_before_interpolation
        self.reference_before_interpolation = reference.reference_signal
        self.reference_after_interpolation = reference.reference_signal_new
        self.interpolated_channels = reference.interpolated_channels
        self.still_noisy_channels = reference.still_noisy_channels

        return self
