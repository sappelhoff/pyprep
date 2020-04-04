

.. image:: https://github.com/sappelhoff/pyprep/workflows/Python%20tests/badge.svg
   :target: https://github.com/sappelhoff/pyprep/actions?query=workflow%3A%22Python+tests%22
   :alt: Python tests


.. image:: https://codecov.io/gh/sappelhoff/pyprep/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/sappelhoff/pyprep
   :alt: codecov


.. image:: https://readthedocs.org/projects/pyprep/badge/?version=latest
   :target: http://pyprep.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://badge.fury.io/py/pyprep.svg
   :target: https://badge.fury.io/py/pyprep
   :alt: PyPI version


pyprep
======

See `the documentation <http://pyprep.readthedocs.io/en/latest/>`_!

.. docs_readme_include_label

``pyprep`` is a python implementation of the
`Preprocessing Pipeline (PREP) <https://doi.org/10.3389/fninf.2015.00016>`_ for
EEG data, working with `MNE-Python <https://www.martinos.org/mne/stable/index.html>`_
for EEG data processing and analysis. Also contains a function to detect
outlier epochs inspired by the FASTER algorithm.

Installation
============

``pyprep`` requires Python version ``3.6`` or higher to run properly. It is
furthermore recommended to run ``pyprep`` in a dedicated virtual environment
(using e.g., `conda <https://docs.conda.io/en/latest/miniconda.html>`_).

For installing the **stable** version of ``pyprep``, simply call
``pip install pyprep``. This should install dependencies automatically. You
can also install the dependencies yourself by running
``pip install -r requirements.txt`` from the project root.

For installation of the **development** version use:

.. code-block:: Text

   git clone https://github.com/sappelhoff/pyprep
   cd pyprep
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .

Contributions
=============

Contributions are welcome! You should have read the references below. After
that, feel free to submit pull requests. Be sure to always include tests for
all new code that you introduce (whenever possible).

References
==========

1. Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A.
   (2015). The PREP pipeline: standardized preprocessing for large-scale EEG
   analysis. Frontiers in Neuroinformatics, 9, 16. doi:
   `10.3389/fninf.2015.00016 <https://doi.org/10.3389/fninf.2015.00016>`_

2. Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER: fully automated
   statistical thresholding for EEG artifact rejection. Journal of neuroscience
   methods, 192(1), 152-162. doi:
   `10.1016/j.jneumeth.2010.07.015 <https://doi.org/10.1016/j.jneumeth.2010.07.015>`_
