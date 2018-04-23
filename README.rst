.. image:: https://travis-ci.org/sappelhoff/pyprep.svg?branch=master
    :target: https://travis-ci.org/sappelhoff/pyprep

.. image:: https://codecov.io/gh/sappelhoff/pyprep/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/sappelhoff/pyprep


pyprep
======
A python implementation of the Preprocessing Pipeline (PREP) for EEG data.

Working with `MNE Python<https://www.martinos.org/mne/stable/index.html>`_ for EEG data processing and analysis.

Installation
============
Probably easiest through:

::
    
    pip install git+https://github.com/sappelhoff/pyprep.git

Alternatively:
==============
0. Activate your python environment

.. code-block:: bash

   git clone https://github.com/sappelhoff/pyprep #clone pyprep locally
    cd pyprep #go to pyprep directory
    pip install -r requirements.txt #install all dependencies
    pip install -e . #install pyprep

# Reference
===========
Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics, 9, 16. `doi: 10.3389/fninf.2015.00016<https://doi.org/10.3389/fninf.2015.00016>`_
