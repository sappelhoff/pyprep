"""Module for PREP pipeline."""
import mne
from mne.utils import check_random_state

from pyprep.reference import Reference
from pyprep.removeTrend import removeTrend


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
              These channels will be used to construct the reference
              signal.
              Can be a str 'eeg' to use all EEG channels.
        - reref_chs : {list, 'eeg'}
            - A list of channel names to define from which channels the
              reference signal will be subtracted.
              Can be a str 'eeg' to use all EEG channels.
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
        raw.load_data()
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

        # Initialize attributes to be filled in later
        self.EEG_raw = self.raw_eeg.get_data()
        self.EEG_filtered = None
        self.EEG_post_reference = None

        # NOTE: 'original' refers to before initial average reference, not first
        # pass afterwards. Not necessarily comparable to later values?
        self.noisy_info = {
            "original": None, "post-reference": None, "post-interpolation": None
        }
        self.bad_channels = {
            "original": None, "post-reference": None, "post-interpolation": None
        }
        self.interpolated_channels = None
        self.robust_reference_signal = None
        self._interpolated_reference_signal = None

    @property
    def raw(self):
        """Return a version of self.raw_eeg that includes the non-eeg channels."""
        full_raw = self.raw_eeg.copy()
        if self.raw_non_eeg is not None:
            full_raw.add_channels([self.raw_non_eeg], force_update_info=True)
        return full_raw

    @property
    def current_noisy_info(self):
        post_ref = self.noisy_info["post-reference"]
        post_interp = self.noisy_info["post-interpolation"]
        return post_interp if post_interp else post_ref

    @property
    def remaining_bad_channels(self):
        post_ref = self.bad_channels["post-reference"]
        post_interp = self.bad_channels["post-interpolation"]
        return post_interp if post_interp else post_ref

    @property
    def current_reference_signal(self):
        post_ref = self.robust_reference_signal
        post_interp = self._interpolated_reference_signal
        return post_interp if post_interp else post_ref

    def get_raw(self, stage=None):
        """Retrieve the full recording data at a given stage of the pipeline.

        Valid pipeline stages include 'unprocessed' (the raw data prior to running
        the pipeline), 'filtered' (the data following adaptive line noise
        removal), 'post-reference' (the data after robust referencing, prior to any
        bad channel interpolation), and 'post-interpolation' (the data after robust
        referencing and bad channel interpolation).

        Parameters
        ----------
        stage : str, optional
            The stage of the pipeline for which the full data will be retrieved. If
            not specified, the current state of the data will be retrieved.

        Returns
        -------
        full_raw: mne.io.Raw
            An MNE Raw object containing the EEG data for the given stage of the
            pipeline, along with any non-EEG channels that were present in the
            original input data.

        """
        interpolated = self.interpolated_channels is not None
        stages = {
            "unprocessed": self.EEG_raw,
            "filtered": self.EEG_filtered,
            "post-reference": self.EEG_post_reference,
            "post-interpolation": self.raw_eeg._data if interpolated else None,
        }
        if stage is not None and stage.lower() not in stages.keys():
            raise ValueError(
                "'{stage}' is not a valid pipeline stage. Valid stages are "
                "'unprocessed', 'filtered', 'post-reference', and 'post-interpolation'."
            )

        eeg_data = self.raw_eeg._data  # Default to most recent stage of pipeline
        if stage:
            eeg_data = stages[stage.lower()]
            if not eeg_data:
                raise ValueError(
                    "Could not retrieve {stage} data, as that stage of the pipeline "
                    "has not yet been performed."
                )
        full_raw = self.raw_eeg.copy()
        full_raw._data = eeg_data
        if self.raw_non_eeg is not None:
            full_raw.add_channels([self.raw_non_eeg])

        return full_raw

    def remove_line_noise(self, line_freqs):
        """Remove line noise from all EEG channels using multi-taper decomposition.

        This filtering method attempts to isolate and remove line noise from the
        signal while preserving unrelated background signal in the same frequency
        ranges. This is done to minimize distortions in the power-spectral density
        curves due to line noise removal.

        Parameters
        ----------
        line_freqs: {np.ndarray, list}
            A list of the frequencies (in Hz) at which line noise should be removed
            (e.g., ``np.arange(60, sfreq / 2, 60)`` for a recording with a powerline
            noise of 60 Hz).

        """
        # Define default settings for filter and apply any kwargs overrides
        settings = {"mt_bandwidth": 2, "p_value": 0.01, "filter_length": "10s"}
        if isinstance(self.filter_kwargs, dict):
            settings.update(self.filter_kwargs)

        # Remove slow drifts from the recording prior to filtering
        eeg_detrended = removeTrend(
            self.EEG_raw, self.sfreq, matlab_strict=self.matlab_strict
        )

        # Remove line noise and add the removed slow drifts back
        eeg_cleaned = mne.filter.notch_filter(
            eeg_detrended,
            Fs=self.sfreq,
            freqs=line_freqs,
            method="spectrum_fit",
            **settings,
            # Add support for parallel jobs if joblib installed?
        )
        self.EEG_filtered = (self.EEG_raw - eeg_detrended) + eeg_cleaned
        self.raw_eeg._data = self.EEG_filtered

    def robust_reference(self, max_iterations=4, interpolate_bads=True):
        """Perform robust referencing on the EEG signal and detect bad channels.

        This method uses an iterative approach to estimate a robust average
        reference signal free of contamination from bad channels, as detected
        automatically using the methods of :class:`~pyprep.NoisyChannels`. Once
        estimated, the robust average reference is applied to the data and bad
        channel detection is re-run to flag any noisy or unusable channels
        post-reference.

        By default, this method will also interpolate the signals of any channels
        detected as bad following robust referencing, re-reference the data
        accordingly, and re-detect any remaining bad channels.

        Parameters
        ----------
        max_iterations : int, optional
            The maximum number of iterations of noisy channel removal to perform
            during robust referencing. Defaults to ``4``.
        interpolate_bads : bool, optional
            Whether or not any remaining bad channels following robust referencing
            should be interpolated. Defaults to ``True``.

        """
        # Perform robust referencing on the signal
        ref = Reference(
            self.raw_eeg,
            self.prep_params,
            random_state=self.random_state,
            matlab_strict=self.matlab_strict,
            **self.ransac_settings,
        )
        ref.perform_reference(max_iterations, interpolate_bads)

        self.raw_eeg = ref.raw
        self.EEG_post_reference = ref.EEG_before_interpolation
        self.robust_reference_signal = ref.reference_signal
        self._interpolated_reference_signal = ref.reference_signal_new

        self.noisy_info["original"] = ref.noisy_channels_original
        self.noisy_info["post-reference"] = ref.noisy_channels_before_interpolation
        self.noisy_info["post-interpolation"] = ref.noisy_channels_after_interpolation

        self.bad_channels["original"] = ref.noisy_channels_original["bad_all"]
        self.bad_channels["post-reference"] = ref.bad_before_interpolation
        self.bad_channels["post-interpolation"] = ref.still_noisy_channels
        self.interpolated_channels = ref.interpolated_channels

    def fit(self):
        """Run the whole PREP pipeline."""
        # Step 1: Adaptive line noise removal
        if len(self.prep_params["line_freqs"]) != 0:
            self.remove_line_noise(self.prep_params["line_freqs"])

        # Step 2: Robust Referencing
        self.robust_reference(self.prep_params["max_iterations"])

        return self
