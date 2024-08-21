"""Test the full PREP pipeline."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyprep import PrepPipeline

from .conftest import make_random_mne_object


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline(raw, montage):
    """Test prep pipeline."""
    # Pick only EEG channels
    raw.pick(picks="eeg")

    # Create a copy of raw data
    raw_copy = raw.copy()

    # Get channel names (after picking EEG channels)
    ch_names_eeg = raw_copy.info["ch_names"]

    # Setup preprocessing parameters
    sample_rate = raw_copy.info["sfreq"]
    prep_params = {
        "ref_chs": ch_names_eeg,
        "reref_chs": ch_names_eeg,
        "line_freqs": np.arange(60, sample_rate / 2, 60),
    }

    # Initialize and fit PrepPipeline
    prep = PrepPipeline(raw_copy, prep_params, montage, random_state=42)
    prep.fit()

    # Load MATLAB results

    # Extract data from the pipeline
    data = {
        "EEG_raw": raw_copy.get_data(picks="eeg") * 1e6,
        "EEG_new": prep.EEG_new,
        "EEG_clean": prep.EEG,
        "EEG_before_interpolation": prep.EEG_before_interpolation,
        "EEG_final": prep.raw.get_data() * 1e6,
    }

    # Calculate maximum values for normalization
    data_max = {key: np.max(np.abs(value)) for key, value in data.items()}

    # Create plots
    fig, axs = plt.subplots(5, 3, sharex="all", figsize=(15, 15))
    plt.setp(fig, facecolor=[1, 1, 1])
    fig.suptitle("Python versus Matlab PREP results", fontsize=16)

    def plot_data(ax, data_key, matlab_key, title):
        im = ax.imshow(
            data[data_key] / data_max[data_key],
            aspect="auto",
            extent=[0, (data[data_key].shape[1] / sample_rate), 63, 0],
            vmin=-1,
            vmax=1,
            cmap=plt.get_cmap("RdBu"),
        )
        ax.set_title(title, fontsize=14)
        return im

    # Plot each stage of data
    im = plot_data(axs[0, 0], "EEG_raw", "EEG_raw", "Python")
    plot_data(axs[0, 1], "EEG_raw", "EEG_raw", "Matlab")
    plot_data(axs[0, 2], "EEG_raw", "EEG_raw", "Difference")

    plot_data(axs[1, 0], "EEG_new", "EEGNew", "Python")
    plot_data(axs[1, 1], "EEG_new", "EEGNew", "Matlab")
    plot_data(axs[1, 2], "EEG_new", "EEGNew", "Difference")

    plot_data(axs[2, 0], "EEG_clean", "EEG", "Python")
    plot_data(axs[2, 1], "EEG_clean", "EEG", "Matlab")
    plot_data(axs[2, 2], "EEG_clean", "EEG", "Difference")

    plot_data(axs[3, 0], "EEG_before_interpolation", "EEGref", "Python")
    plot_data(axs[3, 1], "EEG_before_interpolation", "EEGref", "Matlab")
    plot_data(axs[3, 2], "EEG_before_interpolation", "EEGref", "Difference")

    plot_data(axs[4, 0], "EEG_final", "EEGinterp", "Python")
    plot_data(axs[4, 1], "EEG_final", "EEGinterp", "Matlab")
    plot_data(axs[4, 2], "EEG_final", "EEGinterp", "Difference")

    # Colorbar and labels
    cb = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.04)
    cb.set_label("\u03BCVolt", fontsize=14)
    axs[4, 1].set_xlabel("Time (s)", fontsize=14)
    axs[2, 0].set_ylabel("Channel Number", fontsize=14)

    plt.show()


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline_non_eeg(raw, montage):
    """Test prep pipeline with non eeg channels."""
    raw_copy = raw.copy()

    # make arbitrary non eeg channels from the register
    sfreq = raw_copy.info["sfreq"]  # Sampling frequency
    times = np.array(list(range(raw_copy.get_data().shape[1])))
    ch_names_non_eeg = ["misc" + str(i) for i in range(4)]
    ch_types_non_eeg = ["misc" for i in range(4)]
    raw_non_eeg, _, _ = make_random_mne_object(
        ch_names_non_eeg,
        ch_types_non_eeg,
        times,
        sfreq,
        rng=np.random.default_rng(1337),
    )

    raw_copy.add_channels([raw_non_eeg], force_update_info=True)

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
    assert set(prep.raw.ch_names) == set(raw_copy.ch_names)
    # names of raw (only eeg)  same as full names - non eeg names
    assert set(prep.raw_eeg.ch_names) == set(raw_copy.ch_names) - set(ch_names_non_eeg)
    # quantity of raw (only eeg) same as quantity of all - non eeg lists
    assert prep.raw_eeg.get_data().shape[0] == len(raw_copy.ch_names) - len(
        prep.ch_names_non_eeg
    )
    # quantity of channels in is the same as qty of full raw
    assert raw_copy.get_data().shape[0] == prep.raw.get_data().shape[0]


@pytest.mark.usefixtures("raw", "montage")
def test_prep_pipeline_filter_kwargs(raw, montage):
    """Test prep pipeline with filter kwargs."""
    # Get EEG channel indices
    raw.pick(picks="eeg")  # Picks only EEG channels

    # Create a copy of raw data to avoid modifying the original
    raw_copy = raw.copy()

    # Extract EEG channel names
    ch_names_eeg = raw_copy.ch_names  # Since we picked only EEG channels, use all names

    # Get the sample rate from the copied raw data
    sample_rate = raw_copy.info["sfreq"]

    # Prepare parameters for the pipeline
    prep_params = {
        "ref_chs": ch_names_eeg,
        "reref_chs": ch_names_eeg,
        "line_freqs": np.arange(60, sample_rate / 2, 60),
    }

    # Define filter kwargs
    filter_kwargs = {
        "method": "fir",
        "phase": "zero-double",
    }

    # Initialize and fit the PrepPipeline
    prep = PrepPipeline(
        raw_copy, prep_params, montage, random_state=42, filter_kwargs=filter_kwargs
    )
    prep.fit()

    # Initialize and fit the PrepPipeline
    prep = PrepPipeline(
        raw_copy, prep_params, montage, random_state=42, filter_kwargs=filter_kwargs
    )
    prep.fit()
