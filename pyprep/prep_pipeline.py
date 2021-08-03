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
    raw : mne.io.Raw
        The data. Channel types must be correctly assigned (e.g.,
        ocular channels are assigned the type 'eog').
    prep_params : dict
        Parameters of PREP which include at least the following keys:

        - ref_chs : {list, 'eeg'}
            - A list of channel names to be used for rereferencing.
              Can be a str 'eeg' to use all EEG channels.
        - reref_chs : {list, 'eeg'}
            - A list of channel names to be used for line-noise removed,
              and referenced. Can be a str 'eeg' to use all EEG channels.
        - line_freqs : {np.ndarray, list}
            - list of floats indicating frequencies to be removed.
              For example, for 60Hz you may specify
              ``np.arange(60, sfreq / 2, 60)``. Specify an empty list to
              skip the line noise removal step.
        - max_iterations : int, optional
            - The maximum number of iterations of noisy channel removal to
              perform during robust referencing. Defaults to ``4``.
    montage : mne.channels.DigMontage
        Digital montage of EEG data.
    ransac : bool, optional
        Whether or not to use RANSAC for noisy channel detection in addition to
        the other methods in :class:`~pyprep.NoisyChannels`. Defaults to True.
    channel_wise : bool, optional
        Whether RANSAC should predict signals for chunks of channels over the
        entire signal length ("channel-wise RANSAC", see `max_chunk_size`
        parameter). If ``False``, RANSAC will instead predict signals for all
        channels at once but over a number of smaller time windows instead of
        over the entire signal length ("window-wise RANSAC"). Channel-wise
        RANSAC generally has higher RAM demands than window-wise RANSAC
        (especially if `max_chunk_size` is ``None``), but can be faster on
        systems with lots of RAM to spare. Has no effect if not using RANSAC.
        Defaults to ``False``.
    max_chunk_size : {int, None}, optional
        The maximum number of channels to predict at once during channel-wise
        RANSAC. If ``None``, RANSAC will use the largest chunk size that will
        fit into the available RAM, which may slow down other programs on the
        host system. If using window-wise RANSAC (the default) or not using
        RANSAC at all, this parameter has no effect. Defaults to ``None``.
    random_state : {int, None, np.random.RandomState}, optional
        The random seed at which to initialize the class. If random_state is
        an int, it will be used as a seed for RandomState.
        If None, the seed will be obtained from the operating system
        (see RandomState for details). Default is None.
    filter_kwargs : {dict, None}, optional
        Optional keywords arguments to be passed on to mne.filter.notch_filter.
        Do not set the "x", Fs", and "freqs" arguments via the filter_kwargs
        parameter, but use the "raw" and "prep_params" parameters instead.
        If None is passed, the pyprep default settings for filtering are used
        instead.
    matlab_strict : bool, optional
        Whether or not PyPREP should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code
        (see :ref:`matlab-diffs` for more details). Defaults to False.

    Attributes
    ----------
    raw : mne.io.Raw
        The data including eeg and non eeg channels. It is unprocessed if
        accessed before the fit method, processed if accessed after a
        successful fit method.
    raw_eeg : mne.io.Raw
        The only-eeg part of the data. It is unprocessed if accessed before
        the fit method, processed if accessed after a successful fit method.
    raw_non_eeg : {mne.io.Raw, None}
        The non-eeg part of the data. It is not processed when calling
        the fit method. If the input was only EEG it will be None.
    noisy_channels_original : dict
       Detailed bad channels in each criteria before robust reference.
    noisy_channels_before_interpolation : dict
        Detailed bad channels in each criteria just before interpolation.
    noisy_channels_after_interpolation : dict
        Detailed bad channels in each criteria just after interpolation.
    bad_before_interpolation : list
        bad channels after robust reference but before interpolation
    EEG_before_interpolation : np.ndarray
        EEG data in uV before the interpolation
    reference_before_interpolation : np.ndarray
        Reference signal in uV before interpolation.
    reference_after_interpolation : np.ndarray
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
        channel_wise=False,
        max_chunk_size=None,
        random_state=None,
        filter_kwargs=None,
        matlab_strict=False,
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

        self.EEG_raw = self.raw_eeg.get_data()
        self.prep_params = prep_params
        if self.prep_params["ref_chs"] == "eeg":
            self.prep_params["ref_chs"] = self.ch_names_eeg
        if self.prep_params["reref_chs"] == "eeg":
            self.prep_params["reref_chs"] = self.ch_names_eeg
        if "max_iterations" not in prep_params.keys():
            self.prep_params["max_iterations"] = 4
        self.sfreq = self.raw_eeg.info["sfreq"]
        self.ransac_settings = {
            "ransac": ransac,
            "channel_wise": channel_wise,
            "max_chunk_size": max_chunk_size,
        }
        self.random_state = check_random_state(random_state)
        self.filter_kwargs = filter_kwargs
        self.matlab_strict = matlab_strict

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
            self.EEG_new = removeTrend(
                self.EEG_raw, self.sfreq, matlab_strict=self.matlab_strict
            )

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
            self.raw_eeg._data = self.EEG

        # Step 3: Referencing
        reference = Reference(
            self.raw_eeg,
            self.prep_params,
            random_state=self.random_state,
            matlab_strict=self.matlab_strict,
            **self.ransac_settings,
        )
        reference.perform_reference(self.prep_params["max_iterations"])
        self.raw_eeg = reference.raw
        self.noisy_channels_original = reference.noisy_channels_original
        self.noisy_channels_before_interpolation = (
            reference.noisy_channels_before_interpolation
        )
        self.noisy_channels_after_interpolation = (
            reference.noisy_channels_after_interpolation
        )
        self.bad_before_interpolation = reference.bad_before_interpolation
        self.EEG_before_interpolation = reference.EEG_before_interpolation
        self.reference_before_interpolation = reference.reference_signal
        self.reference_after_interpolation = reference.reference_signal_new
        self.interpolated_channels = reference.interpolated_channels
        self.still_noisy_channels = reference.still_noisy_channels

        return self
