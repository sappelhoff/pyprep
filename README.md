[![Build Status](https://travis-ci.org/sappelhoff/pyprep.svg?branch=master)](https://travis-ci.org/sappelhoff/pyprep)


[![codecov](https://codecov.io/gh/sappelhoff/pyprep/branch/master/graph/badge.svg)](https://codecov.io/gh/sappelhoff/pyprep)



# pyprep
A python implementation of the Preprocessing Pipeline (PREP) for EEG data.

Working with [MNE Python](https://www.martinos.org/mne/stable/index.html) for EEG data processing and analysis.

# Installation

Probably easiest through:

- `pip install git+https://github.com/sappelhoff/pyprep.git`

Alternatively:

0. Activate your python environment
1. `git clone https://github.com/sappelhoff/pyprep` ... clone pyprep locally
2. `cd pyprep` ... go to pyprep directory
3. `pip install -r requirements.txt` ... installs all dependencies
4. `pip install -e .` ... installs pyprep

# Usage
TBD, for now, see the docstrings ... but a minimal example is:

```python
from pyprep.noisy import find_noisy_channels

raw = *load some data*

bads = find_noisy_channels(raw)

print(bads)
>> ['Cz', 'Pz', 'FT10']

```

# Reference
Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16. [10.3389/fninf.2015.00016](https://doi.org/10.3389/fninf.2015.00016)
