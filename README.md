[![Build Status](https://travis-ci.org/sappelhoff/pyprep.svg?branch=master)](https://travis-ci.org/sappelhoff/pyprep) [![codecov](https://codecov.io/gh/sappelhoff/pyprep/branch/master/graph/badge.svg)](https://codecov.io/gh/sappelhoff/pyprep) [![Documentation Status](https://readthedocs.org/projects/pyprep/badge/?version=latest)](http://pyprep.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/pyprep.svg)](https://badge.fury.io/py/pyprep)


# pyprep

A python implementation of the Preprocessing Pipeline (PREP) for EEG data.

Working with [MNE-Python](https://www.martinos.org/mne/stable/index.html) for EEG data processing and analysis.

For a basic use example, see [the documentation.](http://pyprep.readthedocs.io/en/latest/examples.html)

Also contains a function to detect outlier epochs inspired by the FASTER algorithm.

# Installation

Probably easiest through:

`pip install pyprep`

For development version:


```bash
git clone https://github.com/sappelhoff/pyprep #clone pyprep locally
cd pyprep #go to pyprep directory
pip install -r requirements.txt #install all dependencies
pip install -e . #install pyprep
```
# Reference
Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16. doi: [10.3389/fninf.2015.00016](https://doi.org/10.3389/fninf.2015.00016)

Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER: fully automated statistical thresholding for EEG artifact rejection. Journal of neuroscience methods, 192(1), 152-162. doi: [10.1016/j.jneumeth.2010.07.015](https://doi.org/10.1016/j.jneumeth.2010.07.015)
