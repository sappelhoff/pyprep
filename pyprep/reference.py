"""functions of referencing part of PREP."""
import logging

import numpy as np
from mne.utils import check_random_state

# from pyprep.noisy import Noisydata
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.removeTrend import removeTrend
from pyprep.utils import _set_diff, _union

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
    raw : raw mne object
        The raw data.
    params : dict
        Parameters of PREP which include at least the following keys:
        - ``ref_chs``
        - ``reref_chs``
    ransac : bool
        Whether or not to use ransac.
    random_state : int | None | np.random.mtrand.RandomState
        The random seed at which to initialize the class. If random_state is
        an int, it will be used as a seed for RandomState.
        If None, the seed will be obtained from the operating system
        (see RandomState for details). Default is None.

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       raw analysis. Frontiers in Neuroinformatics, 9, 16.

    """

    def __init__(self, raw, params, ransac=True, random_state=None):
        """Initialize the class."""
        self.raw = raw.copy()
        self.ch_names = self.raw.ch_names
        self.raw.pick_types(eeg=True, eog=False, meg=False)
        self.ch_names_eeg = self.raw.ch_names
        self.EEG = self.raw.get_data() * 1e6
        self.reference_channels = params["ref_chs"]
        self.rereferenced_channels = params["reref_chs"]
        self.sfreq = self.raw.info["sfreq"]
        self.ransac = ransac
        self.random_state = check_random_state(random_state)

    def perform_reference(self):
        """Estimate the true signal mean and interpolate bad channels.

        This function implements the functionality of the `performReference` function
        as part of the PREP pipeline on mne raw object.

        Notes
        -----
        This function calls ``robust_reference`` first.
        Currently this function only implements the functionality of default
        settings, i.e., ``doRobustPost``.

        """
        # Phase 1: Estimate the true signal mean with robust referencing
        self.robust_reference()
        # If we interpolate the raw here we would be interpolating
        # more than what we later actually account for (in interpolated channels).
        dummy = self.raw.copy()
        dummy.info["bads"] = self.noisy_channels["bad_all"]
        dummy.interpolate_bads()
        self.reference_signal = (
            np.nanmean(dummy.get_data(picks=self.reference_channels), axis=0) * 1e6
        )
        del dummy
        rereferenced_index = [
            self.ch_names_eeg.index(ch) for ch in self.rereferenced_channels
        ]
        self.EEG = self.remove_reference(
            self.EEG, self.reference_signal, rereferenced_index
        )

        # Phase 2: Find the bad channels and interpolate
        self.raw._data = self.EEG * 1e-6
        noisy_detector = NoisyChannels(self.raw, random_state=self.random_state)
        noisy_detector.find_all_bads(ransac=self.ransac)

        # Record Noisy channels and EEG before interpolation
        self.bad_before_interpolation = noisy_detector.get_bads(verbose=True)
        self.EEG_before_interpolation = self.EEG.copy()

        bad_channels = _union(self.bad_before_interpolation, self.unusable_channels)
        self.raw.info["bads"] = bad_channels
        self.raw.interpolate_bads()
        reference_correct = (
            np.nanmean(self.raw.get_data(picks=self.reference_channels), axis=0) * 1e6
        )
        self.EEG = self.raw.get_data() * 1e6
        self.EEG = self.remove_reference(
            self.EEG, reference_correct, rereferenced_index
        )
        # reference signal after interpolation
        self.reference_signal_new = self.reference_signal + reference_correct
        # MNE Raw object after interpolation
        self.raw._data = self.EEG * 1e-6

        # Still noisy channels after interpolation
        self.interpolated_channels = bad_channels
        noisy_detector = NoisyChannels(self.raw, random_state=self.random_state)
        noisy_detector.find_all_bads(ransac=self.ransac)
        self.still_noisy_channels = noisy_detector.get_bads()
        self.raw.info["bads"] = self.still_noisy_channels
        return self

    def robust_reference(self):
        """Detect bad channels and estimate the robust reference signal.

        This function implements the functionality of the `robustReference` function
        as part of the PREP pipeline on mne raw object.

        Parameters
        ----------
        ransac : bool
            Whether or not to use ransac

        Returns
        -------
        noisy_channels: dict
            A dictionary of names of noisy channels detected from all methods
            after referencing.
        reference_signal: ndarray, shape(n, )
            Estimation of the 'true' signal mean

        """
        raw = self.raw.copy()
        raw._data = removeTrend(raw.get_data(), sample_rate=self.sfreq)

        # Determine unusable channels and remove them from the reference channels
        noisy_detector = NoisyChannels(
            raw, do_detrend=False, random_state=self.random_state
        )
        noisy_detector.find_all_bads(ransac=self.ransac)
        self.noisy_channels_original = {
            "bad_by_nan": noisy_detector.bad_by_nan,
            "bad_by_flat": noisy_detector.bad_by_flat,
            "bad_by_deviation": noisy_detector.bad_by_deviation,
            "bad_by_hf_noise": noisy_detector.bad_by_hf_noise,
            "bad_by_correlation": noisy_detector.bad_by_correlation,
            "bad_by_ransac": noisy_detector.bad_by_ransac,
            "bad_all": noisy_detector.get_bads(),
        }
        self.noisy_channels = self.noisy_channels_original.copy()
        logger.info("Bad channels: {}".format(self.noisy_channels))

        self.unusable_channels = _union(
            noisy_detector.bad_by_nan, noisy_detector.bad_by_flat
        )

        # According to the Matlab Implementation (see robustReference.m)
        # self.unusable_channels = _union(self.unusable_channels,
        # noisy_detector.bad_by_SNR)
        # but maybe this makes no difference...

        self.reference_channels = _set_diff(
            self.reference_channels, self.unusable_channels
        )

        # Get initial estimate of the reference by the specified method
        signal = raw.get_data() * 1e6
        self.reference_signal = (
            np.nanmedian(raw.get_data(picks=self.reference_channels), axis=0) * 1e6
        )
        reference_index = [
            self.ch_names_eeg.index(ch) for ch in self.reference_channels
        ]
        signal_tmp = self.remove_reference(
            signal, self.reference_signal, reference_index
        )

        # Remove reference from signal, iteratively interpolating bad channels
        raw_tmp = raw.copy()
        iterations = 0
        noisy_channels_old = []
        max_iteration_num = 4

        while True:
            raw_tmp._data = signal_tmp * 1e-6
            noisy_detector = NoisyChannels(
                raw_tmp, do_detrend=False, random_state=self.random_state
            )
            # Detrend applied at the beginning of the function.
            noisy_detector.find_all_bads(ransac=self.ransac)
            self.noisy_channels["bad_by_nan"] = _union(
                self.noisy_channels["bad_by_nan"], noisy_detector.bad_by_nan
            )
            self.noisy_channels["bad_by_flat"] = _union(
                self.noisy_channels["bad_by_flat"], noisy_detector.bad_by_flat
            )
            self.noisy_channels["bad_by_deviation"] = _union(
                self.noisy_channels["bad_by_deviation"], noisy_detector.bad_by_deviation
            )
            self.noisy_channels["bad_by_hf_noise"] = _union(
                self.noisy_channels["bad_by_hf_noise"], noisy_detector.bad_by_hf_noise
            )
            self.noisy_channels["bad_by_correlation"] = _union(
                self.noisy_channels["bad_by_correlation"],
                noisy_detector.bad_by_correlation,
            )
            self.noisy_channels["bad_by_ransac"] = _union(
                self.noisy_channels["bad_by_ransac"], noisy_detector.bad_by_ransac
            )
            self.noisy_channels["bad_all"] = _union(
                self.noisy_channels["bad_all"], noisy_detector.get_bads()
            )
            logger.info("Bad channels: {}".format(self.noisy_channels))

            if (
                iterations > 1
                and (
                    not self.noisy_channels["bad_all"]
                    or set(self.noisy_channels["bad_all"]) == set(noisy_channels_old)
                )
                or iterations > max_iteration_num
            ):
                break
            noisy_channels_old = self.noisy_channels["bad_all"].copy()

            if raw_tmp.info["nchan"] - len(self.noisy_channels["bad_all"]) < 2:
                raise ValueError(
                    "RobustReference:TooManyBad "
                    "Could not perform a robust reference -- not enough good channels"
                )

            if self.noisy_channels["bad_all"]:
                raw_tmp._data = signal * 1e-6
                raw_tmp.info["bads"] = self.noisy_channels["bad_all"]
                raw_tmp.interpolate_bads()
                signal_tmp = raw_tmp.get_data() * 1e6
            else:
                signal_tmp = signal
            self.reference_signal = (
                np.nanmean(raw_tmp.get_data(picks=self.reference_channels), axis=0)
                * 1e6
            )

            signal_tmp = self.remove_reference(
                signal, self.reference_signal, reference_index
            )
            iterations = iterations + 1
            logger.info("Iterations: {}".format(iterations))

        logger.info("Robust reference done")
        return self.noisy_channels, self.reference_signal

    @staticmethod
    def remove_reference(signal, reference, index=None):
        """Remove the reference signal from the original EEG signal.

        This function implements the functionality of the `removeReference` function
        as part of the PREP pipeline on mne raw object.

        Parameters
        ----------
        signal : ndarray, shape(channels, times)
            The original EEG signal.
        reference : ndarray, shape(times,)
            The reference signal.
        index : list | None
            A list channel index from which the signal was removed.

        Returns
        -------
        ndarray, shape(channels, times)
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
