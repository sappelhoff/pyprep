"""functions of referencing part of PREP."""

# Authors: The PyPREP developers
# SPDX-License-Identifier: MIT

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
    ransac : bool | None
        Whether or not to use RANSAC for noisy channel detection in addition to
        the other methods in :class:`~pyprep.NoisyChannels`. Defaults to True.
    channel_wise : bool | None
        Whether RANSAC should predict signals for chunks of channels over the
        entire signal length ("channel-wise RANSAC", see `max_chunk_size`
        parameter). If ``False``, RANSAC will instead predict signals for all
        channels at once but over a number of smaller time windows instead of
        over the entire signal length ("window-wise RANSAC"). Channel-wise
        RANSAC generally has higher RAM demands than window-wise RANSAC
        (especially if `max_chunk_size` is ``None``), but can be faster on
        systems with lots of RAM to spare. Has no effect if not using RANSAC.
        Defaults to ``False``.
    max_chunk_size : {int, None} | None
        The maximum number of channels to predict at once during channel-wise
        RANSAC. If ``None``, RANSAC will use the largest chunk size that will
        fit into the available RAM, which may slow down other programs on the
        host system. If using window-wise RANSAC (the default) or not using
        RANSAC at all, this parameter has no effect. Defaults to ``None``.
    random_state : {int, None, np.random.RandomState} | None
        The random seed at which to initialize the class. If random_state is
        an int, it will be used as a seed for RandomState.
        If None, the seed will be obtained from the operating system
        (see RandomState for details). Default is None.
    reject_by_annotation : {None, 'omit'} | None
        How to handle BAD-annotated time segments (annotations starting with
        "BAD" or "bad") during channel quality assessment. If ``'omit'``,
        annotated segments are excluded. Defaults to ``None`` (ignore).
    matlab_strict : bool | None
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
        reject_by_annotation=None,
        matlab_strict=False,
    ):
        """Initialize the class."""
        raw.load_data()
        self.raw = raw.copy()
        self.ch_names = self.raw.ch_names
        self.raw.pick("eeg", exclude=[])  # include previously marked bads
        self.bads_manual = raw.info["bads"]
        self.ch_names_eeg = self.raw.ch_names
        self.EEG = self.raw.get_data()
        self.reference_channels = params["ref_chs"]
        self.rereferenced_channels = params["reref_chs"]
        self.sfreq = self.raw.info["sfreq"]
        self.ransac_settings = {
            "ransac": ransac,
            "channel_wise": channel_wise,
            "max_chunk_size": max_chunk_size,
            "reject_by_annotation": reject_by_annotation,
        }
        self.random_state = check_random_state(random_state)
        self._extra_info = {}
        self.matlab_strict = matlab_strict

    def perform_reference(self, max_iterations=4):
        """Estimate the true signal mean and interpolate bad channels.

        Parameters
        ----------
        max_iterations : int | None
            The maximum number of iterations of noisy channel removal to perform
            during robust referencing. Defaults to ``4``.

        This function implements the functionality of the `performReference` function
        as part of the PREP pipeline on mne raw object.

        Notes
        -----
        This function calls ``robust_reference`` first.
        Currently this function only implements the functionality of default
        settings, i.e., ``doRobustPost``.

        """
        # Phase 1: Estimate the true signal mean with robust referencing
        self.robust_reference(max_iterations)
        # If we interpolate the raw here we would be interpolating
        # more than what we later actually account for (in interpolated channels).
        dummy = self.raw.copy()
        dummy.info["bads"] = self.noisy_channels["bad_all"]
        if self.matlab_strict:
            _eeglab_interpolate_bads(dummy)
        else:
            dummy.interpolate_bads()
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
            self.raw,
            random_state=self.random_state,
            matlab_strict=self.matlab_strict,
            reject_by_annotation=self.ransac_settings.get("reject_by_annotation"),
        )
        noisy_detector.find_all_bads(**self.ransac_settings)

        # Record Noisy channels and EEG before interpolation
        self.bad_before_interpolation = noisy_detector.get_bads(verbose=True)
        self.EEG_before_interpolation = self.EEG.copy()
        self.noisy_channels_before_interpolation = noisy_detector.get_bads(as_dict=True)
        self.noisy_channels_before_interpolation["bad_by_manual"] = self.bads_manual
        self._extra_info["interpolated"] = noisy_detector._extra_info

        bad_channels = _union(self.bad_before_interpolation, self.unusable_channels)
        self.raw.info["bads"] = bad_channels
        if self.matlab_strict:
            _eeglab_interpolate_bads(self.raw)
        else:
            self.raw.interpolate_bads()
        reference_correct = np.nanmean(
            self.raw.get_data(picks=self.reference_channels), axis=0
        )
        self.EEG = self.raw.get_data()
        self.EEG = self.remove_reference(
            self.EEG, reference_correct, rereferenced_index
        )
        # reference signal after interpolation
        self.reference_signal_new = self.reference_signal + reference_correct
        # MNE Raw object after interpolation
        self.raw._data = self.EEG

        # Still noisy channels after interpolation
        self.interpolated_channels = bad_channels
        noisy_detector = NoisyChannels(
            self.raw,
            random_state=self.random_state,
            matlab_strict=self.matlab_strict,
            reject_by_annotation=self.ransac_settings.get("reject_by_annotation"),
        )
        noisy_detector.find_all_bads(**self.ransac_settings)
        self.still_noisy_channels = noisy_detector.get_bads()
        self.raw.info["bads"] = self.still_noisy_channels
        self.noisy_channels_after_interpolation = noisy_detector.get_bads(as_dict=True)
        self._extra_info["remaining_bad"] = noisy_detector._extra_info

        return self

    def robust_reference(self, max_iterations=4):
        """Detect bad channels and estimate the robust reference signal.

        This function implements the functionality of the `robustReference` function
        as part of the PREP pipeline on mne raw object.

        Parameters
        ----------
        max_iterations : int | None
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
        raw = self.raw.copy()
        raw._data = removeTrend(
            raw.get_data(), self.sfreq, matlab_strict=self.matlab_strict
        )

        # Determine unusable channels and remove them from the reference channels
        noisy_detector = NoisyChannels(
            raw,
            do_detrend=False,
            random_state=self.random_state,
            matlab_strict=self.matlab_strict,
            reject_by_annotation=self.ransac_settings.get("reject_by_annotation"),
        )
        noisy_detector.find_all_bads(**self.ransac_settings)
        self.noisy_channels_original = noisy_detector.get_bads(as_dict=True)
        self._extra_info["initial_bad"] = noisy_detector._extra_info
        logger.info(f"Bad channels: {self.noisy_channels_original}")

        # Determine channels to use/exclude from initial reference estimation
        self.unusable_channels = _union(
            noisy_detector.bad_by_nan + noisy_detector.bad_by_flat,
            noisy_detector.bad_by_SNR + self.bads_manual,
        )
        reference_channels = _set_diff(self.reference_channels, self.unusable_channels)

        # Initialize channels to permanently flag as bad during referencing
        noisy = {
            "bad_by_nan": noisy_detector.bad_by_nan,
            "bad_by_flat": noisy_detector.bad_by_flat,
            "bad_by_deviation": [],
            "bad_by_hf_noise": [],
            "bad_by_correlation": [],
            "bad_by_SNR": [],
            "bad_by_dropout": [],
            "bad_by_psd": [],
            "bad_by_ransac": [],
            "bad_by_manual": self.bads_manual,
            "bad_all": [],
        }

        # Get initial estimate of the reference by the specified method
        signal = raw.get_data()
        self.reference_signal = np.nanmedian(
            raw.get_data(picks=reference_channels), axis=0
        )
        reference_index = [self.ch_names_eeg.index(ch) for ch in reference_channels]
        signal_tmp = self.remove_reference(
            signal, self.reference_signal, reference_index
        )

        # Remove reference from signal, iteratively interpolating bad channels
        raw_tmp = raw.copy()
        iterations = 0
        previous_bads = set()

        while True:
            raw_tmp._data = signal_tmp
            noisy_detector = NoisyChannels(
                raw_tmp,
                do_detrend=False,
                random_state=self.random_state,
                matlab_strict=self.matlab_strict,
                reject_by_annotation=self.ransac_settings.get("reject_by_annotation"),
            )
            # Detrend applied at the beginning of the function.

            # Detect all currently bad channels
            noisy_detector.find_all_bads(**self.ransac_settings)
            noisy_new = noisy_detector.get_bads(as_dict=True)

            # Specify bad channel types to ignore when updating noisy channels
            # NOTE: MATLAB PREP ignores dropout channels, possibly by mistake?
            # see: https://github.com/VisLab/EEG-Clean-Tools/issues/28
            ignore = ["bad_by_SNR", "bad_all"]
            if self.matlab_strict:
                ignore += ["bad_by_dropout"]

            # Update set of all noisy channels detected so far with any new ones
            bad_chans = set()
            for bad_type in noisy_new.keys():
                noisy[bad_type] = _union(noisy[bad_type], noisy_new[bad_type])
                if bad_type not in ignore:
                    bad_chans.update(noisy[bad_type])
            noisy["bad_by_manual"] = self.bads_manual
            noisy["bad_all"] = list(bad_chans) + self.bads_manual
            logger.info(f"Bad channels: {noisy}")

            if (
                iterations > 1
                and (len(bad_chans) == 0 or bad_chans == previous_bads)
                or iterations > max_iterations
            ):
                logger.info("Robust reference done")
                self.noisy_channels = noisy
                break
            previous_bads = bad_chans.copy()

            if raw_tmp.info["nchan"] - len(bad_chans) < 2:
                raise ValueError(
                    "RobustReference:TooManyBad "
                    "Could not perform a robust reference -- not enough good channels"
                )

            if len(bad_chans) > 0:
                raw_tmp._data = signal.copy()
                raw_tmp.info["bads"] = list(bad_chans)
                if self.matlab_strict:
                    _eeglab_interpolate_bads(raw_tmp)
                else:
                    raw_tmp.interpolate_bads()

            self.reference_signal = np.nanmean(
                raw_tmp.get_data(picks=reference_channels), axis=0
            )

            signal_tmp = self.remove_reference(
                signal, self.reference_signal, reference_index
            )
            iterations = iterations + 1
            logger.info(f"Iterations: {iterations}")

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
        index : {list, None} | None
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
                    f"RemoveReference: Expected type list, got {type(index)} instead"
                )
            signal_referenced = signal.copy()
            signal_referenced[np.asarray(index), :] = (
                signal[np.asarray(index), :] - reference
            )
        return signal_referenced
