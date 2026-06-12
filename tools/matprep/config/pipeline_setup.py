"""Set up MATLAB PREP CI environment."""

# Authors: Austin Hurst
#          Stefan Appelhoff

import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import ZipFile

# Initialize required directories

download_dir = Path("temp")
package_dir = Path("deps")
artifact_dir = Path("artifacts")

download_dir.mkdir(exist_ok=True)
package_dir.mkdir(exist_ok=True)
artifact_dir.mkdir(exist_ok=True)

# Download test EEG data (currently using S004R01 from the BCI2000 dataset)

subject = "S004"
run = "R01"
eeg_filename = f"{subject}{run}.edf"
eeg_url = f"https://www.physionet.org/files/eegmmidb/1.0.0/{subject}/{eeg_filename}?download"

print("* Downloading EEG test data...")
try:
    mod_http = urlopen(eeg_url)
    with open(eeg_filename, "wb") as out:
        out.write(mod_http.read())
except (URLError, HTTPError):
    raise RuntimeError(f"Failed to download '{eeg_filename}'")

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
    try:
        mod_http = urlopen(url)
        with open(download_path, "wb") as out:
            out.write(mod_http.read())
    except (URLError, HTTPError):
        raise RuntimeError(f"Failed to download '{url}'")

    # Unzip downloaded file to MATLAB package directory
    with ZipFile(download_path, "r") as z:
        z.extractall(package_dir)
        root_dir = z.namelist()[0].split("/")[0]
        outpath = package_dir / root_dir

    # Rename unzipped package folder to package name for consistency
    pkg_path = package_dir / name
    shutil.move(outpath, pkg_path)
