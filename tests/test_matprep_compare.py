"""Compare PyPREP results to MATLAB PREP."""
from urllib.request import urlopen
import numpy as np
import scipy
import mne

import pytest

from pyprep.find_noisy_channels import NoisyChannels
from pyprep.removeTrend import removeTrend


# Define some fixtures for things that will be used across multiple tests

@pytest.fixture(scope='session')
def matprep_artifacts(tmpdir_factory):
    """Fixture for downloading & using MATLAB PREP artifacts from CI.

    Downloads the latest set of CI-generated MATLAB PREP artifacts and saves
    them to a temporary folder for use with various tests, returning the paths
    to each artifact in a {name: path} dict. The temporary folder will be
    automatically cleaned up by pytest once all tests have completed.

    """
    base_url = "https://github.com/a-hurst/matprep_artifacts/releases/latest/download/"
    artifacts = [
        "1_matprep_raw.set",
        "2_matprep_removetrend.set",
        "3_matprep_cleanline.set",
        "4_matprep_pre_reference.set",
        "5_matprep_post_reference.set",
        "matprep_info.mat"
    ]
    dirpath = tmpdir_factory.mktemp("artifacts")

    artifact_paths = {}
    for f in artifacts:
        artifact_name = f.split(".")[0]
        outfile = str(dirpath.join(f))
        dl = urlopen(base_url + f)
        with open(outfile, 'wb') as out:
            out.write(dl.read())
        artifact_paths[artifact_name] = outfile

    return artifact_paths


@pytest.fixture(scope='session')
def matprep_noisy(matprep_artifacts):
    """Import and preprocess artifact containing MATLAB PREP runtime info.

    This fixture only parses and retains data from the first pass of noisy
    channel detection during re-referencing, since it's easiest to compare with
    PyPREP. It also adds a new key to the imported struct, 'bads', which
    contains the names of the channels flagged as bad by each detection
    method (as opposed to just channel indices).

    """
    # Read in and parse noisy channel info artifact from MATLAB PREP
    info_path = matprep_artifacts['matprep_info']
    matprep_info = scipy.io.loadmat(info_path, simplify_cells=True)['prep_info']
    matprep_noisy_all = matprep_info['reference']
    matprep_noisy = matprep_noisy_all['noisyStatisticsOriginal']

    # Gather bad channel names from MatPREP, converting numbers to labels
    ch_names = matprep_info['originalChannelLabels']
    bad_types = {
        'badChannelsFromNaNs': 'by_nan',
        'badChannelsFromNoData': 'by_flat',
        'badChannelsFromDeviation': 'by_deviation',
        'badChannelsFromHFNoise': 'by_hf_noise',
        'badChannelsFromCorrelation': 'by_correlation',
        'badChannelsFromLowSNR': 'by_SNR',
        'badChannelsFromDropOuts': 'by_dropout',
        'badChannelsFromRansac': 'by_ransac',
        'all': 'all',
    }
    matprep_bads = {}
    for bad_type, name in bad_types.items():
        bads_idx = matprep_noisy['noisyChannels'][bad_type]
        bads_idx = [bads_idx] if isinstance(bads_idx, int) else bads_idx
        matprep_bads[name] = [ch_names[i-1] for i in bads_idx]
    matprep_noisy['bads'] = matprep_bads

    return matprep_noisy


@pytest.fixture(scope='session')
def pyprep_noisy(matprep_artifacts):
    """Get original NoisyChannels results for comparison with MATLAB PREP.

    This fixture uses an artifact from MATLAB PREP of the CleanLined and
    detrended EEG signal right before MATLAB PREP runs its first iteration of
    NoisyChannels during re-referncing. As such, any differences in test results
    will be due to actual differences in the noisy channel detection code rather
    than differences at an earlier stage of the pipeline.

    """
    # Import pre-reference MATLAB PREP data
    preref_path = matprep_artifacts['4_matprep_pre_reference']
    matprep_preref = mne.io.read_raw_eeglab(preref_path, preload=True)

    # Run NoisyChannels on MATLAB data and extract internal noisy info
    matprep_seed = 435656
    pyprep_noisy = NoisyChannels(
        matprep_preref,
        do_detrend=False,
        random_state=matprep_seed,
        matlab_strict=True
    )
    pyprep_noisy.find_all_bads()

    return pyprep_noisy


# Define MATLAB comparison tests for each main component of PyPREP

@pytest.mark.usefixtures('matprep_artifacts')
def test_compare_removeTrend(matprep_artifacts):
    """Test the numeric equivalence of removeTrend to MATLAB PREP."""
    # Get paths of MATLAB .set files
    raw_path = matprep_artifacts['1_matprep_raw']
    detrend_path = matprep_artifacts['2_matprep_removetrend']

    # Load relevant MATLAB data
    matprep_raw = mne.io.read_raw_eeglab(raw_path, preload=True)
    matprep_detrended = mne.io.read_raw_eeglab(detrend_path, preload=True)
    sample_rate = matprep_raw.info["sfreq"]

    # Apply removeTrend to raw artifact to get expected and actual signals
    expected = matprep_detrended._data
    actual = removeTrend(
        matprep_raw._data, sample_rate, detrendType="high pass", matlab_strict=True
    )

    # Check MATLAB equivalence at start of recording
    win_size = 500  # window of samples to check
    assert np.allclose(
        actual[:, :win_size], expected[:, :win_size], equal_nan=True
    )

    # Check MATLAB equivalence in middle of recording
    win_start = int(actual.shape[1] / 2)
    win_end = win_start + win_size
    assert np.allclose(
        actual[:, win_start:win_end], expected[:, win_start:win_end], equal_nan=True
    )

    # Check MATLAB equivalence at end of recording
    assert np.allclose(
        actual[:, -win_size:], expected[:, -win_size:], equal_nan=True
    )


class TestCompareNoisyChannels(object):
    """Compare the results of NoisyChannels to the equivalent MatPREP code.

    These comparisons use input data that's already had adaptive line noise
    removal and high-pass trend removal done to the signal, so any differences
    found will be due to differences in the noisy channel detection code rather
    than any differences at an earlier stage of the pipeline.

    """

    def test_bad_by_nan(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-NaN results between PyPREP and MatPREP."""
        # Compare names of bad-by-NaN channels
        assert pyprep_noisy.bad_by_nan == matprep_noisy['bads']['by_nan']

    def test_bad_by_flat(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-flat results between PyPREP and MatPREP."""
        # Compare names of bad-by-flat channels
        assert pyprep_noisy.bad_by_flat == matprep_noisy['bads']['by_flat']

    def test_bad_by_deviation(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-deviation results between PyPREP and MatPREP."""
        # Gather PyPREP deviation info and MATLAB equivalents
        mat = matprep_noisy
        matprep_info = {
            'median_channel_deviation': mat['channelDeviationMedian'],
            'channel_deviation_sd': mat['channelDeviationSD'],
            'robust_channel_deviations': mat['robustChannelDeviation'],
            'channel_deviations': mat['channelDeviations']
        }
        pyprep_info = pyprep_noisy._extra_info['bad_by_deviation']

        # Compare overall medians and SDs for channel deviations
        median_dev = 'median_channel_deviation'
        dev_sd = 'channel_deviation_sd'
        assert np.isclose(pyprep_info[median_dev] * 1e6, matprep_info[median_dev])
        assert np.isclose(pyprep_info[dev_sd] * 1e6, matprep_info[dev_sd])

        # Compare robust deviations across channels
        dev_by_chan = 'robust_channel_deviations'
        assert np.allclose(pyprep_info[dev_by_chan], matprep_info[dev_by_chan])

        # Compare windows of channel deviations across recording
        chan_devs = 'channel_deviations'
        assert np.allclose(pyprep_info[chan_devs] * 1e6, matprep_info[chan_devs].T)

        # Compare names of bad-by-deviation channels
        assert pyprep_noisy.bad_by_deviation == matprep_noisy['bads']['by_deviation']

    def test_bad_by_hf_noise(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-HF-noise results between PyPREP and MatPREP."""
        # Gather PyPREP high-frequency noise info and MATLAB equivalents
        mat = matprep_noisy
        matprep_info = {
            'median_channel_noisiness': mat['noisinessMedian'],
            'channel_noisiness_sd': mat['noisinessSD'],
            'hf_noise_zscores': mat['zscoreHFNoise'],
            'noise_levels': mat['noiseLevels']
        }
        pyprep_info = pyprep_noisy._extra_info['bad_by_hf_noise']

        # Compare overall medians and SDs for channel noisiness
        median_noise = 'median_channel_noisiness'
        noise_sd = 'channel_noisiness_sd'
        assert np.isclose(pyprep_info[median_noise], matprep_info[median_noise])
        assert np.isclose(pyprep_info[noise_sd], matprep_info[noise_sd])

        # Compare noisiness z-scores across channels
        TOL = 1e-5  # NOTE: Some diffs > 1e-6 (default), maybe slight diff in math?
        noise_z = 'hf_noise_zscores'
        assert np.allclose(pyprep_info[noise_z], matprep_info[noise_z], atol=TOL)

        # Compare windows of noisiness per channel across recording
        noise_win = 'noise_levels'
        assert np.allclose(pyprep_info[noise_win], matprep_info[noise_win].T)

        # Compare names of bad-by-HF-noise channels
        assert pyprep_noisy.bad_by_hf_noise == matprep_noisy['bads']['by_hf_noise']

    def test_bad_by_correlation(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-correlation results between PyPREP and MatPREP."""
        # Gather PyPREP correlation info and MATLAB equivalents
        mat = matprep_noisy
        matprep_info = {
            'max_correlations': mat['maximumCorrelations'],
            'median_max_correlations': mat['medianMaxCorrelation']
        }
        pyprep_info = pyprep_noisy._extra_info['bad_by_correlation']

        # Compare median maximum correlations across channels
        med_max_corr = 'median_max_correlations'
        assert np.allclose(pyprep_info[med_max_corr], matprep_info[med_max_corr])

        # Compare max correlations per channel across recording
        max_corr = 'max_correlations'
        assert np.allclose(pyprep_info[max_corr], matprep_info[max_corr])

        # Compare names of bad-by-correlation channels
        matprep_bad_by_corr = matprep_noisy['bads']['by_correlation']
        assert pyprep_noisy.bad_by_correlation == matprep_bad_by_corr

    def test_bad_by_SNR(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-SNR results between PyPREP and MatPREP."""
        pyprep_bads_snr = sorted(pyprep_noisy.bad_by_SNR)
        matprep_bads_snr = sorted(matprep_noisy['bads']['by_SNR'])
        assert pyprep_bads_snr == matprep_bads_snr

    def test_bad_by_dropout(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-dropout results between PyPREP and MatPREP."""
        # Gather PyPREP and MATLAB PREP dropout info
        matprep_dropouts = matprep_noisy['dropOuts']
        pyprep_dropouts = pyprep_noisy._extra_info['bad_by_dropout']['dropouts']

        # Compare dropout windows per channel across recording
        assert np.allclose(pyprep_dropouts, matprep_dropouts)

        # Compare names of bad-by-dropout channels
        assert pyprep_noisy.bad_by_dropout == matprep_noisy['bads']['by_dropout']

    def test_bad_by_ransac(self, pyprep_noisy, matprep_noisy):
        """Compare bad-by-RANSAC results between PyPREP and MatPREP."""
        # Gather PyPREP RANSAC correlation info and MATLAB equivalents
        mat = matprep_noisy
        matprep_info = {
            'ransac_correlations': mat['ransacCorrelations'],
            'bad_window_fractions': mat['ransacBadWindowFraction']
        }
        pyprep_info = pyprep_noisy._extra_info['bad_by_ransac']

        # Compare fractions of bad RANSAC windows across channels
        bad_fracs = 'bad_window_fractions'
        assert np.allclose(pyprep_info[bad_fracs], matprep_info[bad_fracs])

        # Compare RANSAC window correlations per channel across recording
        ransac_corr = 'ransac_correlations'
        assert np.allclose(pyprep_info[ransac_corr], matprep_info[ransac_corr].T)

        # Compare names of bad-by-RANSAC channels
        assert pyprep_noisy.bad_by_ransac == matprep_noisy['bads']['by_ransac']

    def test_all_bads(self, pyprep_noisy, matprep_noisy):
        """Compare names of all bad channels between PyPREP and MatPREP."""
        pyprep_bads_all = sorted(pyprep_noisy.get_bads())
        matprep_bads_all = sorted(matprep_noisy['bads']['all'])
        assert pyprep_bads_all == matprep_bads_all
