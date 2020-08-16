"""Test the full PREP pipeline."""
import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
import scipy.io as sio

from pyprep.prep_pipeline import PrepPipeline

RNG = np.random.RandomState(1337)


def make_random_mne_object(
    ch_names, ch_types, times, sfreq, n_freq_comps=5, freq_range=[10, 60], scale=1e-6
):
    """Make a random MNE object to use for testing.

    Parameters
    ----------
    ch_names : list
        names of channels
    ch_types : list
        types of channels
    times : 1d numpy array
        Time vector to use.
    sfreq : float
        Sampling frequency associated with the time vector.
    n_freq_comps : int
        Number of signal components summed to make a signal.
    freq_range : list, len==2
        Signals will contain freqs from this range.
    scale: float
        scale of the signal in volts. (ie 1e-6 for microvolts).

    Returns
    -------
    raw : mne raw object
        The mne object for performing the tests.

    n_freq_comps : int

    freq_range : list, len==2

    """
    n_chans = len(ch_names)
    signal_len = times.shape[0]
    # Make a random signal
    signal = np.zeros((n_chans, signal_len))
    low = freq_range[0]
    high = freq_range[1]
    for chan in range(n_chans):
        # Each channel signal is a sum of random freq sine waves
        for freq_i in range(n_freq_comps):
            freq = RNG.randint(low, high, signal_len)
            signal[chan, :] += np.sin(2 * np.pi * times * freq)

    signal *= scale  # scale

    # Make mne object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(signal, info)
    return raw, n_freq_comps, freq_range


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline(raw, montage):
    """Test prep pipeline."""
    eeg_index = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
    raw_copy = raw.copy()
    ch_names = raw_copy.info["ch_names"]
    ch_names_eeg = list(np.asarray(ch_names)[eeg_index])
    sample_rate = raw_copy.info["sfreq"]
    prep_params = {
        "ref_chs": ch_names_eeg,
        "reref_chs": ch_names_eeg,
        "line_freqs": np.arange(60, sample_rate / 2, 60),
    }
    prep = PrepPipeline(raw_copy, prep_params, montage, random_state=42)
    prep.fit()

    EEG_raw = raw_copy.get_data(picks="eeg") * 1e6
    EEG_raw_max = np.max(abs(EEG_raw), axis=None)
    EEG_raw_matlab = sio.loadmat("./examples/matlab_results/EEG_raw.mat")
    EEG_raw_matlab = EEG_raw_matlab["save_data"]
    EEG_raw_diff = EEG_raw - EEG_raw_matlab
    # EEG_raw_mse = (EEG_raw_diff / EEG_raw_max ** 2).mean(axis=None)

    fig, axs = plt.subplots(5, 3, "all")
    plt.setp(fig, facecolor=[1, 1, 1])
    fig.suptitle("Python versus Matlab PREP results", fontsize=16)

    im = axs[0, 0].imshow(
        EEG_raw / EEG_raw_max,
        aspect="auto",
        extent=[0, (EEG_raw.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[0, 0].set_title("Python", fontsize=14)
    axs[0, 1].imshow(
        EEG_raw_matlab / EEG_raw_max,
        aspect="auto",
        extent=[0, (EEG_raw_matlab.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[0, 1].set_title("Matlab", fontsize=14)
    axs[0, 2].imshow(
        EEG_raw_diff / EEG_raw_max,
        aspect="auto",
        extent=[0, (EEG_raw_diff.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[0, 2].set_title("Difference", fontsize=14)
    # axs[0, 0].set_title('Original EEG', loc='left', fontsize=14)
    # axs[0, 0].set_ylabel('Channel Number', fontsize=14)
    cb = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.04)
    cb.set_label("\u03BCVolt", fontsize=14)

    EEG_new_matlab = sio.loadmat("./examples/matlab_results/EEGNew.mat")
    EEG_new_matlab = EEG_new_matlab["save_data"]
    EEG_new = prep.EEG_new
    EEG_new_max = np.max(abs(EEG_new), axis=None)
    EEG_new_diff = EEG_new - EEG_new_matlab
    # EEG_new_mse = ((EEG_new_diff / EEG_new_max) ** 2).mean(axis=None)
    axs[1, 0].imshow(
        EEG_new / EEG_new_max,
        aspect="auto",
        extent=[0, (EEG_new.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[1, 1].imshow(
        EEG_new_matlab / EEG_new_max,
        aspect="auto",
        extent=[0, (EEG_new_matlab.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[1, 2].imshow(
        EEG_new_diff / EEG_new_max,
        aspect="auto",
        extent=[0, (EEG_new_diff.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    # axs[1, 0].set_title('High pass filter', loc='left', fontsize=14)
    # axs[1, 0].set_ylabel('Channel Number', fontsize=14)

    EEG_clean_matlab = sio.loadmat("./examples/matlab_results/EEG.mat")
    EEG_clean_matlab = EEG_clean_matlab["save_data"]
    EEG_clean = prep.EEG
    EEG_max = np.max(abs(EEG_clean), axis=None)
    EEG_diff = EEG_clean - EEG_clean_matlab
    # EEG_mse = ((EEG_diff / EEG_max) ** 2).mean(axis=None)
    axs[2, 0].imshow(
        EEG_clean / EEG_max,
        aspect="auto",
        extent=[0, (EEG_clean.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[2, 1].imshow(
        EEG_clean_matlab / EEG_max,
        aspect="auto",
        extent=[0, (EEG_clean_matlab.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[2, 2].imshow(
        EEG_diff / EEG_max,
        aspect="auto",
        extent=[0, (EEG_diff.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    # axs[2, 0].set_title('Line-noise removal', loc='left', fontsize=14)
    axs[2, 0].set_ylabel("Channel Number", fontsize=14)

    EEG = prep.EEG_before_interpolation
    EEG_max = np.max(abs(EEG), axis=None)
    EEG_ref_mat = sio.loadmat("./examples/matlab_results/EEGref.mat")
    EEG_ref_matlab = EEG_ref_mat["save_EEG"]
    # reference_matlab = EEG_ref_mat["save_reference"]
    EEG_ref_diff = EEG - EEG_ref_matlab
    # EEG_ref_mse = ((EEG_ref_diff / EEG_max) ** 2).mean(axis=None)
    # reference_signal = prep.reference_before_interpolation
    # reference_max = np.max(abs(reference_signal), axis=None)
    # reference_diff = reference_signal - reference_matlab
    # reference_mse = ((reference_diff / reference_max) ** 2).mean(axis=None)
    axs[3, 0].imshow(
        EEG / EEG_max,
        aspect="auto",
        extent=[0, (EEG.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[3, 1].imshow(
        EEG_ref_matlab / EEG_max,
        aspect="auto",
        extent=[0, (EEG_ref_matlab.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[3, 2].imshow(
        EEG_ref_diff / EEG_max,
        aspect="auto",
        extent=[0, (EEG_ref_diff.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    # axs[3, 0].set_title('Referencing', loc='left', fontsize=14)
    # axs[3, 0].set_ylabel('Channel Number', fontsize=14)

    EEG_final = prep.raw_eeg.get_data() * 1e6
    EEG_final_max = np.max(abs(EEG_final), axis=None)
    EEG_final_matlab = sio.loadmat("./examples/matlab_results/EEGinterp.mat")
    EEG_final_matlab = EEG_final_matlab["save_data"]
    EEG_final_diff = EEG_final - EEG_final_matlab
    # EEG_final_mse = ((EEG_final_diff / EEG_final_max) ** 2).mean(axis=None)
    axs[4, 0].imshow(
        EEG_final / EEG_final_max,
        aspect="auto",
        extent=[0, (EEG_final.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[4, 1].imshow(
        EEG_final_matlab / EEG_final_max,
        aspect="auto",
        extent=[0, (EEG_final_matlab.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    axs[4, 2].imshow(
        EEG_final_diff / EEG_final_max,
        aspect="auto",
        extent=[0, (EEG_final_diff.shape[1] / sample_rate), 63, 0],
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("RdBu"),
    )
    # axs[4, 0].set_title('Interpolation', loc='left', fontsize=14)
    # axs[4, 0].set_ylabel('Channel Number', fontsize=14)
    axs[4, 1].set_xlabel("Time(s)", fontsize=14)


def test_prep_pipeline_non_eeg(raw, montage):
    """Test prep pipeline with non eeg channels."""
    raw_copy = raw.copy()

    # make arbitrary non eeg channels from the register
    sfreq = raw_copy.info["sfreq"]  # Sampling frequency
    times = list(range(raw_copy._data.shape[1]))
    ch_names_non_eeg = ["misc" + str(i) for i in range(4)]
    ch_types_non_eeg = ["misc" for i in range(4)]
    raw_non_eeg, _, _ = make_random_mne_object(
        ch_names_non_eeg, ch_types_non_eeg, times, sfreq
    )

    raw_copy.add_channels([raw_non_eeg])

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, sfreq / 2, 60),
    }
    prep = PrepPipeline(raw_copy, prep_params, montage, random_state=42)

    prep.fit()

    # correct non-eeg channels configured in init
    assert set(prep.ch_names_non_eeg) == set(ch_names_non_eeg)
    # original (all) channel names same as full_raw names
    assert set(prep.raw.ch_names_all) == set(raw_copy.ch_names)
    # names of raw (only eeg)  same as full names - non eeg names
    assert set(prep.raw.ch_names_all) == set(raw_copy.ch_names) - set(ch_names_non_eeg)
    # quantity of raw (only eeg) same as quantity of all - non eeg lists
    assert prep.raw_eeg._data.shape[0] == len(raw_copy.ch_names) - len(
        prep.ch_names_non_eeg
    )
    # quantity of channels in is the same as qty of full raw
    assert raw_copy._data.shape[0] == prep.raw._data.shape[0]
