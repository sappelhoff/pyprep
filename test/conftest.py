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
    """Fixture for physionet EEG subject 4, dataset 1."""
    mne.set_log_level("WARNING")
    # load in subject 1, run 1 dataset
    edf_fpath = eegbci.load_data(4, 1)[0]

    # using sample EEG data (https://physionet.org/content/eegmmidb/1.0.0/)
    raw = mne.io.read_raw_edf(edf_fpath, preload=True)
    raw.rename_channels(lambda s: s.strip("."))
    raw.rename_channels(
        lambda s: s.replace("c", "C")
            .replace("o", "O")
            .replace("f", "F")
            .replace("t", "T")
            .replace("Tp", "TP")
            .replace("Cp", "CP")
    )
    return raw
