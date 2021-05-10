"""Configure tests."""
import mne
import pytest
from mne.datasets import eegbci


@pytest.fixture(scope="session")
def montage():
    """Fixture for standard EEG montage."""
    montage_kind = "standard_1020"
    montage = mne.channels.make_standard_montage(montage_kind)
    return montage


@pytest.fixture(scope="session")
def raw():
    """Return an `mne.io.Raw` object for use with unit tests.

    This fixture downloads and reads in subject 4, run 1 from the Physionet
    BCI2000 (eegbci) open dataset. This recording is quite noisy and is thus a
    good candidate for testing the PREP pipeline.

    File attributes:
    - Channels: 64 EEG
    - Sample rate: 160 Hz
    - Duration: 61 seconds

    This is only run once per session to save time downloading.

    """
    mne.set_log_level("WARNING")

    # Download and read S004R01.edf from the BCI2000 dataset
    edf_fpath = eegbci.load_data(4, 1, update_path=True)[0]
    raw = mne.io.read_raw_edf(edf_fpath, preload=True)
    eegbci.standardize(raw)  # Fix non-standard channel names

    return raw


@pytest.fixture(scope="session")
def raw_clean(montage):
    """Return an `mne.io.Raw` object with no bad channels for use with tests.

    This fixture downloads and reads in subject 1, run 1 from the Physionet
    BCI2000 (eegbci) open dataset, interpolates its two bad channels (T10 & F8),
    and performs average referencing on the data. Intended for use with tests
    where channels are made artifically bad.

    File attributes:
    - Channels: 64 EEG
    - Sample rate: 160 Hz
    - Duration: 61 seconds

    This is only run once per session to save time downloading.

    """
    mne.set_log_level("WARNING")

    # Download and read S001R01.edf from the BCI2000 dataset
    edf_fpath = eegbci.load_data(1, 1, update_path=True)[0]
    raw = mne.io.read_raw_edf(edf_fpath, preload=True)
    eegbci.standardize(raw)  # Fix non-standard channel names

    # Interpolate the file's few bad channels to produce a clean dataset
    raw.set_montage(montage)
    raw.info["bads"] = ["T10", "F8"]
    raw.interpolate_bads()

    # Re-reference the data after interpolating bad channels
    mne.set_eeg_reference(raw, 'average', copy=False, ch_type='eeg')

    return raw
