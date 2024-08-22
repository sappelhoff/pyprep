"""functions of referencing part of PREP."""
import logging

import numpy as np
from mne.utils import check_random_state

from pyprep.find_noisy_channels import NoisyChannels
from pyprep.removeTrend import removeTrend
from pyprep.utils import _eeglab_interpolate_bads, _set_diff, _union

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Reference:
    """Estimate the 'true' reference with all the bad channels interpolated.

    This class implements the functionality of the `performReference` function
    as part of the PREP (preprocessing pipeline) for EEG data described in [1]_.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    params : dict
        Parameters of PREP which include at least the following keys:
        - ``ref_chs``
        - ``reref_chs``
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
    matlab_strict : bool, optional
        Whether or not PyPREP should strictly follow MATLAB PREP's internal
        math, ignoring any improvements made in PyPREP over the original code.
        Defaults to False.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       raw analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(
        self,
        raw,
        params,
        ransac=True,
        channel_wise=False,
        max_chunk_size=None,
        random_state=None,
        matlab_strict=False,
    ):
        """Initialize the class."""
        raw.load_data()
        self.raw = raw.copy()
        self.ch_names = self.raw.ch_names
        self.raw.pick(picks="eeg")
        self.ch_names_eeg = self.raw.ch_names
        self.EEG = self.raw.get_data()
        self.reference_channels = params["ref_chs"]
        self.rereferenced_channels = params["reref_chs"]
        self.sfreq = self.raw.info["sfreq"]
        self.ransac_settings = {
            "ransac": ransac,
            "channel_wise": channel_wise,
            "max_chunk_size": max_chunk_size,
        }
        self.random_state = check_random_state(random_state)
        self._extra_info = {}
        self.matlab_strict = matlab_strict

    def perform_reference(self, max_iterations=4):
        """Estimate the true signal mean and interpolate bad channels."""
        # Phase 1: Estimate the true signal mean with robust referencing
        self.robust_reference(max_iterations)

        # Create a copy of raw data to estimate reference signal
        dummy = self.raw.copy()
        dummy.info["bads"] = self.noisy_channels["bad_all"]

        if self.matlab_strict:
            _eeglab_interpolate_bads(dummy)
        else:
            dummy.interpolate_bads()

        # Calculate the reference signal
        self.reference_signal = np.nanmean(
            dummy.get_data(picks=self.reference_channels), axis=0
        )
        del dummy

        rereferenced_index = [
            self.ch_names_eeg.index(ch) for ch in self.rereferenced_channels
        ]
        self.EEG = self.remove_reference(
            self.EEG, self.reference_signal, rereferenced_index
        )

        # Phase 2: Find the bad channels and interpolate
        self.raw._data = self.EEG
        noisy_detector = NoisyChannels(
            self.raw, random_state=self.random_state, matlab_strict=self.matlab_strict
        )
        noisy_detector.find_all_bads(**self.ransac_settings)

        # Record noisy channels and EEG before interpolation
        self.bad_before_interpolation = noisy_detector.get_bads(verbose=True)
        self.EEG_before_interpolation = self.EEG.copy()
        self.noisy_channels_before_interpolation = noisy_detector.get_bads(as_dict=True)
        self._extra_info["interpolated"] = noisy_detector._extra_info

        # Handle both cases: list or dict
        if isinstance(self.bad_before_interpolation, dict):
            bad_channels_from_dict = self.bad_before_interpolation.get("bad_all", [])
        else:
            bad_channels_from_dict = self.bad_before_interpolation

        # Ensure 'bads' is a list of channel names
        bad_channels = _union(bad_channels_from_dict, self.unusable_channels)
        valid_bad_channels = [
            ch for ch in bad_channels if ch in self.raw.info["ch_names"]
        ]
        self.raw.info["bads"] = valid_bad_channels

        if self.matlab_strict:
            _eeglab_interpolate_bads(self.raw)
        else:
            self.raw.interpolate_bads()

        # Correct the reference signal after interpolation
        reference_correct = np.nanmean(
            self.raw.get_data(picks=self.reference_channels), axis=0
        )
        self.EEG = self.raw.get_data()
        self.EEG = self.remove_reference(
            self.EEG, reference_correct, rereferenced_index
        )

        # Update the reference signal after interpolation
        self.reference_signal_new = self.reference_signal + reference_correct

        # MNE Raw object after interpolation
        self.raw._data = self.EEG

        # Still noisy channels after interpolation
        self.interpolated_channels = valid_bad_channels
        noisy_detector = NoisyChannels(
            self.raw, random_state=self.random_state, matlab_strict=self.matlab_strict
        )
        noisy_detector.find_all_bads(**self.ransac_settings)
        self.still_noisy_channels = noisy_detector.get_bads()
        valid_still_noisy_channels = [
            ch for ch in self.still_noisy_channels if ch in self.raw.info["ch_names"]
        ]
        self.raw.info["bads"] = valid_still_noisy_channels
        self.noisy_channels_after_interpolation = noisy_detector.get_bads(as_dict=True)
        self._extra_info["remaining_bad"] = noisy_detector._extra_info

        return self

    def robust_reference(self, max_iterations=4):
        """Detect bad channels and estimate the robust reference signal.

        This function implements the functionality of the `robustReference` function
        as part of the PREP pipeline on mne raw object.

        Parameters
        ----------
        max_iterations : int, optional
            The maximum number of iterations of noisy channel removal to perform
            during robust referencing. Defaults to ``4``.

        Returns
        -------
        noisy_channels: dict
            A dictionary of names of noisy channels detected from all methods
            after referencing.
        reference_signal: np.ndarray, shape(n, )
            Estimation of the 'true' signal mean
        """
        # Copy and detrend the data
        raw = self.raw.copy()
        raw._data = removeTrend(
            raw.get_data(), self.sfreq, matlab_strict=self.matlab_strict
        )

        # Detect initial noisy channels
        noisy_detector = NoisyChannels(
            raw,
            do_detrend=False,
            random_state=self.random_state,
            matlab_strict=self.matlab_strict,
        )
        noisy_detector.find_all_bads(**self.ransac_settings)
        self.noisy_channels_original = noisy_detector.get_bads(as_dict=True)
        self._extra_info["initial_bad"] = noisy_detector._extra_info
        logger.info(f"Initial bad channels: {self.noisy_channels_original}")

        # Determine channels to exclude from reference estimation
        self.unusable_channels = _union(
            noisy_detector.bad_by_nan + noisy_detector.bad_by_flat,
            noisy_detector.bad_by_SNR,
        )
        reference_channels = _set_diff(self.reference_channels, self.unusable_channels)

        # Initialize structure to store noisy channels
        noisy = {
            "bad_by_nan": noisy_detector.bad_by_nan,
            "bad_by_flat": noisy_detector.bad_by_flat,
            "bad_by_deviation": [],
            "bad_by_hf_noise": [],
            "bad_by_correlation": [],
            "bad_by_SNR": [],
            "bad_by_dropout": [],
            "bad_by_ransac": [],
            "bad_all": [],
        }

        # Get initial reference signal
        self.reference_signal = np.nanmedian(
            raw.get_data(picks=reference_channels), axis=0
        )
        reference_index = [self.ch_names_eeg.index(ch) for ch in reference_channels]
        signal_tmp = self.remove_reference(
            raw.get_data(), self.reference_signal, reference_index
        )

        # Iteratively update the reference signal and noisy channels
        iterations = 0
        previous_bads = set()
        raw_tmp = raw.copy()

        while iterations < max_iterations:
            raw_tmp._data = signal_tmp

            # Detect noisy channels
            noisy_detector = NoisyChannels(
                raw_tmp,
                do_detrend=False,
                random_state=self.random_state,
                matlab_strict=self.matlab_strict,
            )
            noisy_detector.find_all_bads(**self.ransac_settings)
            noisy_new = noisy_detector.get_bads(as_dict=True)

            # Update noisy channels, excluding certain types if needed
            bad_chans = set()
            for bad_type, channels in noisy_new.items():
                noisy[bad_type] = _union(noisy[bad_type], channels)
                if bad_type not in {"bad_by_SNR", "bad_all"} or not self.matlab_strict:
                    bad_chans.update(noisy[bad_type])

            noisy["bad_all"] = list(bad_chans)
            logger.info(f"Updated bad channels: {noisy}")

            # Stop if no new bad channels or maximum iterations reached
            if bad_chans == previous_bads or len(bad_chans) == 0:
                logger.info("Robust reference completed.")
                break

            if len(bad_chans) >= raw_tmp.info["nchan"] - 2:
                raise ValueError(
                    "RobustReference:TooManyBad "
                    "Not enough good channels left to perform robust referencing."
                )

            # Interpolate bad channels
            raw_tmp._data = raw.get_data().copy()
            raw_tmp.info["bads"] = list(bad_chans)
            if self.matlab_strict:
                _eeglab_interpolate_bads(raw_tmp)
            else:
                raw_tmp.interpolate_bads()

            # Update the reference signal
            self.reference_signal = np.nanmean(
                raw_tmp.get_data(picks=reference_channels), axis=0
            )
            signal_tmp = self.remove_reference(
                raw.get_data(), self.reference_signal, reference_index
            )

            iterations += 1
            logger.info(f"Iteration {iterations} completed.")

        # Store the final set of noisy channels
        self.noisy_channels = noisy

        return self.noisy_channels, self.reference_signal

    @staticmethod
    def remove_reference(signal, reference, index=None):
        """Remove the reference signal from the original EEG signal.

        This function implements the functionality of the `removeReference` function
        as part of the PREP pipeline on mne raw object.

        Parameters
        ----------
        signal : np.ndarray, shape(channels, times)
            The original EEG signal.
        reference : np.ndarray, shape(times,)
            The reference signal.
        index : {list, None}, optional
            A list of channel indices from which the reference signal should be
            subtracted. Defaults to all channels in `signal`.

        Returns
        -------
        np.ndarray, shape(channels, times)
            The referenced EEG signal.

        """
        if np.ndim(signal) != 2:
            raise ValueError(
                "RemoveReference: EEG signal must be 2D array (channels * times)"
            )
        if np.ndim(reference) != 1:
            raise ValueError("RemoveReference: Reference signal must be 1D array")
        if np.shape(signal)[1] != np.shape(reference)[0]:
            raise ValueError(
                "RemoveReference: The second dimension of EEG signal must be "
                "the same with the length of reference signal"
            )
        if index is None:
            signal_referenced = signal - reference
        else:
            if not isinstance(index, list):
                raise TypeError(
                    "RemoveReference: Expected type list, got {} instead".format(
                        type(index)
                    )
                )
            signal_referenced = signal.copy()
            signal_referenced[np.asarray(index), :] = (
                signal[np.asarray(index), :] - reference
            )
        return signal_referenced
