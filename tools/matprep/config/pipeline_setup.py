"""Set up MATLAB PREP CI environment."""

# Authors: Austin Hurst
#          Stefan Appelhoff

import shutil
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from mne.datasets import eegbci


def download(url, dest, retries=5, timeout=60):
    """Download ``url`` to ``dest``, retrying on transient network errors.

    Used for the EEGLAB/PREP software archives (the EEG recording itself is
    fetched via MNE's ``eegbci`` loader). A single ``urlopen`` with no timeout
    is fragile in CI: a momentary connectivity blip (e.g. an IPv6 route that
    blackholes the connection) hangs for the full kernel timeout and then kills
    the build. So bound each attempt with a timeout and retry with backoff. A
    browser-like ``User-Agent`` is sent because some servers reject the default
    ``Python-urllib`` agent.

    Parameters
    ----------
    url : str
        The URL to download.
    dest : str | pathlib.Path
        Local path to write the downloaded bytes to.
    retries : int
        Maximum number of attempts before giving up.
    timeout : int
        Per-attempt socket timeout, in seconds.

    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (pyprep CI)"})
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            return
        except (URLError, TimeoutError, ConnectionError) as err:
            last_err = err
            print(f"  attempt {attempt}/{retries} failed ({err})")
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"Failed to download '{url}' after {retries} attempts: {last_err}"
    )


# Initialize required directories

download_dir = Path("temp")
package_dir = Path("deps")
artifact_dir = Path("artifacts")

download_dir.mkdir(exist_ok=True)
package_dir.mkdir(exist_ok=True)
artifact_dir.mkdir(exist_ok=True)

# Download test EEG data (S004R01 from the BCI2000 / EEGBCI dataset).
#
# Fetched via MNE's eegbci loader rather than a raw download: it is pooch-backed
# (retries on transient failures and verifies the bytes against MNE's hash
# registry) and pulls the recording from the same PhysioNet source the rest of
# pyprep's test suite uses, so both sides see the identical file.

subject = 4
run = 1
eeg_filename = f"S{subject:03d}R{run:02d}.edf"

print("* Downloading EEG test data...")
eeg_paths = eegbci.load_data(subject, run, path=str(package_dir), update_path=False)
shutil.copy(eeg_paths[0], eeg_filename)

# Download and extract EEGLAB and MATLAB PREP.
#
# These versions are pinned on purpose so the generated MATLAB PREP artifacts
# (our numeric "ground truth" for ``tests/test_matprep_compare.py``) stay
# reproducible. Bumping a version is a deliberate change: update the URL here in
# a dedicated PR and regenerate the artifacts, so any shift in the reference
# output is reviewed rather than silently absorbed.
#
# - EEGLAB daily builds: https://sccn.ucsd.edu/eeglab/download/daily/
# - PREP (EEG-Clean-Tools): pinned to a specific commit rather than a branch,
#   since ``master`` is a moving target.
pkgs = {
    "EEGLAB": "https://sccn.ucsd.edu/eeglab/download/daily/eeglab2024.2.zip",
    "PREP": (
        "https://github.com/VisLab/EEG-Clean-Tools/archive/"
        "0914132081ea7d534ee19f96973791fd81825755.zip"
    ),
}

for name, url in pkgs.items():
    print(f"* Downloading {name}...")

    # Download .zip to temporary folder
    download_path = download_dir / f"{name}.zip"
    download(url, download_path)

    # Unzip downloaded file to MATLAB package directory
    with ZipFile(download_path, "r") as z:
        z.extractall(package_dir)
        root_dir = z.namelist()[0].split("/")[0]
        outpath = package_dir / root_dir

    # Rename unzipped package folder to package name for consistency
    pkg_path = package_dir / name
    shutil.move(outpath, pkg_path)
