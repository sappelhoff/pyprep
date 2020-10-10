"""Module for PREP pipeline."""
import mne
from mne.utils import check_random_state

from pyprep.find_noisy_channels import NoisyChannels
from pyprep.reference import Reference
from pyprep.removeTrend import removeTrend
from pyprep.utils import _set_diff, _union  # noqa: F401


class PrepPipeline:
    """Early stage preprocessing (PREP) of EEG data.

    This class implements the functionality  of the PREP (preprocessing
    pipeline) for EEG data described in [1]_.

    Parameters
    ----------
    raw : mne.raw
        The data. Channel types must be correctly assigned (e.g.,
        ocular channels are assigned the type 'eog').
    prep_params : dict
        Parameters of PREP which include at least the following keys:
        - ref_chs : list | 'eeg'
            - A list of channel names to be used for rereferencing.
              Can be a str 'eeg' to use all EEG channels.
        - reref_chs : list | 'eeg'
            - A list of channel names to be used for line-noise removed, and
              referenced. Can be a str 'eeg' to use all EEG channels.
        - line_freqs : array_like
            - list of floats indicating frequencies to be removed.
              For example for 60Hz you may specify ``np.arange(60, sfreq / 2, 60)``.
              Specify an empty list to skip the line noise removal step.
    montage : DigMontage
        Digital montage of EEG data.
    ransac : bool
        Whether or not to use ransac.
    random_state : int | None | np.random.mtrand.RandomState
        The random seed at which to initialize the class. If random_state is
        an int, it will be used as a seed for RandomState.
        If None, the seed will be obtained from the operating system
        (see RandomState for details). Default is None.
    filter_kwargs : dict | None
        Optional keywords arguments to be passed on to mne.filter.notch_filter.
        Do not set the "x", Fs", and "freqs" arguments via the filter_kwargs
        parameter, but use the "raw" and "prep_params" parameters instead.
        If None is passed, the pyprep default settings for filtering are used
        instead.

    Attributes
    ----------
    raw : mne.raw
        The data including eeg and non eeg channels. It is unprocessed if
        accessed before the fit method, processed if accessed after a
        succesful fit method.
    raw_eeg : mne.raw
        The only-eeg part of the data. It is unprocessed if accessed before
        the fit method, processed if accessed after a succesful fit method.
    raw_non_eeg : mne.raw | None
        The non-eeg part of the data. It is not processed when calling
        the fit method. If the input was only EEG it will be None.
    noisy_channels_original : list
        bad channels before robust reference
    bad_before_interpolation : list
        bad channels after robust reference but before interpolation
    EEG_before_interpolation : ndarray
        EEG data in uV before the interpolation
    reference_before_interpolation : ndarray
        Reference signal in uV before interpolation.
    reference_after_interpolation : ndarray
        Reference signal in uV after interpolation.
    interpolated_channels : list
        Names of the interpolated channels.
    still_noisy_channels : list
        Names of the noisy channels after interpolation.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       EEG analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(
        self,
        raw,
        prep_params,
        montage,
        ransac=True,
        random_state=None,
        filter_kwargs=None,
    ):
        """Initialize PREP class."""
        self.raw_eeg = raw.copy()

        # split eeg and non eeg channels
        self.ch_names_all = raw.ch_names.copy()
        self.ch_types_all = raw.get_channel_types()
        self.ch_names_eeg = [
            self.ch_names_all[i]
            for i in range(len(self.ch_names_all))
            if self.ch_types_all[i] == "eeg"
        ]
        self.ch_names_non_eeg = list(set(self.ch_names_all) - set(self.ch_names_eeg))
        self.raw_eeg.pick_channels(self.ch_names_eeg)
        if self.ch_names_non_eeg == []:
            self.raw_non_eeg = None
        else:
            self.raw_non_eeg = raw.copy()
            self.raw_non_eeg.pick_channels(self.ch_names_non_eeg)

        self.raw_eeg.set_montage(montage)
        # raw_non_eeg may not be compatible with the montage
        # so it is not set for that object

        self.EEG_raw = self.raw_eeg.get_data() * 1e6
        self.prep_params = prep_params
        if self.prep_params["ref_chs"] == "eeg":
            self.prep_params["ref_chs"] = self.ch_names_eeg
        if self.prep_params["reref_chs"] == "eeg":
            self.prep_params["reref_chs"] = self.ch_names_eeg
        self.sfreq = self.raw_eeg.info["sfreq"]
        self.ransac = ransac
        self.random_state = check_random_state(random_state)
        self.filter_kwargs = filter_kwargs

    @property
    def raw(self):
        """Return a version of self.raw_eeg that includes the non-eeg channels."""
        full_raw = self.raw_eeg.copy()
        if self.raw_non_eeg is None:
            return full_raw
        else:
            return full_raw.add_channels([self.raw_non_eeg])

    def fit(self):
        """Run the whole PREP pipeline."""
        noisy_detector = NoisyChannels(self.raw_eeg, random_state=self.random_state)
        noisy_detector.find_bad_by_nan_flat()
        # unusable_channels = _union(
        #     noisy_detector.bad_by_nan, noisy_detector.bad_by_flat
        # )
        # reference_channels = _set_diff(self.prep_params["ref_chs"], unusable_channels)
        # Step 1: 1Hz high pass filtering
        if len(self.prep_params["line_freqs"]) != 0:
            self.EEG_new = removeTrend(self.EEG_raw, sample_rate=self.sfreq)

            # Step 2: Removing line noise
            linenoise = self.prep_params["line_freqs"]
            if self.filter_kwargs is None:
                self.EEG_clean = mne.filter.notch_filter(
                    self.EEG_new,
                    Fs=self.sfreq,
                    freqs=linenoise,
                    method="spectrum_fit",
                    mt_bandwidth=2,
                    p_value=0.01,
                    filter_length="10s",
                )
            else:
                self.EEG_clean = mne.filter.notch_filter(
                    self.EEG_new,
                    Fs=self.sfreq,
                    freqs=linenoise,
                    **self.filter_kwargs,
                )

            # Add Trend back
            self.EEG = self.EEG_raw - self.EEG_new + self.EEG_clean
            self.raw_eeg._data = self.EEG * 1e-6

        # Step 3: Referencing
        reference = Reference(
            self.raw_eeg,
            self.prep_params,
            ransac=self.ransac,
            random_state=self.random_state,
        )
        reference.perform_reference()
        self.raw_eeg = reference.raw
        self.noisy_channels_original = reference.noisy_channels_original
        self.bad_before_interpolation = reference.bad_before_interpolation
        self.EEG_before_interpolation = reference.EEG_before_interpolation
        self.reference_before_interpolation = reference.reference_signal
        self.reference_after_interpolation = reference.reference_signal_new
        self.interpolated_channels = reference.interpolated_channels
        self.still_noisy_channels = reference.still_noisy_channels

        return self
