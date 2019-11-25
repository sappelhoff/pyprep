import mne
import numpy as np
import logging
from pyprep.utilities import union, set_diff
from pyprep.noisy import Noisydata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def robust_reference(raw, params, montage_kind='standard_1020', ransac=True):
    """
    Detect bad channels by robust referencing
    This function implements the functionality of the `robustReference` function
    as part of the PREP pipeline for raw data described in [1].

    Parameters
    ----------
    raw : raw mne object
    params : dict
        Parameters of PREP which include at least the following keys:
        ref_chs
        eval_chs
    montage_kind : str
        Which kind of montage should be used to infer the electrode
        positions? E.g., 'standard_1020'
    ransac : boolean
        Whether or not to use ransac

    Returns
    -------
    noisy_channels: dictionary
        A dictionary of names of noisy channels detected from all methods
    reference_signal: 1D Array
        Estimation of the 'true' signal mean

    References
    ----------
    .. [1] Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., Robbins, K. A.
       (2015). The PREP pipeline: standardized preprocessing for large-scale
       raw analysis. Frontiers in Neuroinformatics, 9, 16.
    """
    raw_copy = raw.copy()
    montage = mne.channels.read_montage(kind=montage_kind, ch_names=raw_copy.ch_names)
    raw_copy.set_montage(montage)
    raw_copy.pick_types(eeg=True, eog=False, meg=False)
    # raw.rename_channels(lambda s: s.strip("."))
    ch_names = raw_copy.info['ch_names']

    # Warn if evaluation and reference channels are not the same for robust
    if not set(params['ref_chs']) == set(params['eval_chs']):
        logger.warning('robustReference: Reference channels and'
                       'evaluation channels should be same for robust reference')

    # Determine unusable channels and remove them from the reference channels
    noisy_detector = Noisydata(raw_copy, montage_kind=montage_kind)
    noisy_detector.find_all_bads(ransac=ransac)
    noisy_channels = {'bad_by_nan': noisy_detector.bad_by_nan,
                      'bad_by_flat': noisy_detector.bad_by_flat,
                      'bad_by_deviation': noisy_detector.bad_by_deviation,
                      'bad_by_hf_noise': noisy_detector.bad_by_hf_noise,
                      'bad_by_correlation': noisy_detector.bad_by_correlation,
                      'bad_by_ransac': noisy_detector.bad_by_ransac,
                      'bad_all': noisy_detector.get_bads()}
    logger.info('Bad channels: {}'.format(noisy_channels))

    unusable_channels = union(noisy_detector.bad_by_nan, noisy_detector.bad_by_flat)
    reference_channels = set_diff(params['ref_chs'], unusable_channels)

    # Get initial estimate of the reference by the specified method
    signal = raw_copy.get_data()
    reference_signal = np.nanmean(raw_copy.get_data(picks=reference_channels), axis=0)
    reference_index = [ch_names.index(ch) for ch in reference_channels]
    signal_tmp = remove_reference(signal, reference_signal, reference_index)

    # Remove reference from signal, iteratively interpolating bad channels
    raw_tmp = raw_copy.copy()

    iterations = 0
    noisy_channels_old = []
    max_iteration_num = 4

    while True:
        raw_tmp._data = signal_tmp
        noisy_detector = Noisydata(raw_tmp, montage_kind=montage_kind)
        noisy_detector.find_all_bads(ransac=ransac)
        noisy_channels['bad_by_nan'] = union(noisy_channels['bad_by_nan'], noisy_detector.bad_by_nan)
        noisy_channels['bad_by_flat'] = union(noisy_channels['bad_by_flat'], noisy_detector.bad_by_flat)
        noisy_channels['bad_by_deviation'] = union(noisy_channels['bad_by_deviation'], noisy_detector.bad_by_deviation)
        noisy_channels['bad_by_hf_noise'] = union(noisy_channels['bad_by_hf_noise'], noisy_detector.bad_by_hf_noise)
        noisy_channels['bad_by_correlation'] = union(noisy_channels['bad_by_correlation'],
                                                     noisy_detector.bad_by_correlation)
        noisy_channels['bad_by_ransac'] = union(noisy_channels['bad_by_ransac'], noisy_detector.bad_by_ransac)
        noisy_channels['bad_all'] = union(noisy_channels['bad_all'], noisy_detector.get_bads())
        logger.info('Bad channels: {}'.format(noisy_channels))

        if iterations > 1 and (not noisy_channels['bad_all'] or
                               set(noisy_channels['bad_all']) == set(noisy_channels_old)) or \
                iterations > max_iteration_num:
            break
        noisy_channels_old = noisy_channels['bad_all'].copy()

        if raw_tmp.info['nchan']-len(noisy_channels['bad_all']) < 2:
            raise ValueError('RobustReference:TooManyBad '
                             'Could not perform a robust reference -- not enough good channels')

        if noisy_channels['bad_all']:
            raw_tmp._data = signal
            raw_tmp.info['bads'] = noisy_channels['bad_all']
            raw_tmp.interpolate_bads()
            signal_tmp = raw_tmp.get_data()
        else:
            signal_tmp = signal
        reference_signal = np.nanmean(raw_tmp.get_data(picks=reference_channels), axis=0)
        signal_tmp = remove_reference(signal, reference_signal, reference_index)
        iterations = iterations + 1
        logger.info('Iterations: {}'.format(iterations))

    logger.info('Robust reference done')
    return noisy_channels, reference_signal


def remove_reference(signal, reference, index=None):
    """
    Remove the reference signal from the original EEG signal,
    with some unusable channels excluded.

    Parameters
    ----------
    signal : 2D array (channels * times)
        Original EEG signal
    reference : 1D array (length times)
        Reference signal
    index : list | None
        A list channel index from which the signal was removed

    Returns
    -------
    2D array (channels * times)
        The referenced EEG signal

    """
    if np.ndim(signal) != 2:
        raise ValueError('RemoveReference: EEG signal must be 2D array (channels * times)')
    if np.ndim(reference) != 1:
        raise ValueError('RemoveReference: Reference signal must be 1D array')
    if np.shape(signal)[1] != np.shape(reference)[0]:
        raise ValueError('RemoveReference: The second dimension of EEG signal must be '
                         'the same with the length of reference signal')
    if index is None:
        signal_referenced = signal - reference
    else:
        if not isinstance(index, list):
            raise TypeError('RemoveReference: Expected type list, got {} instead'.format(type(index)))
        signal_referenced = signal
        signal_referenced[np.asarray(index), :] = signal[np.asarray(index), :] - reference
    return signal_referenced
