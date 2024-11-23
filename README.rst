

.. image:: https://github.com/sappelhoff/pyprep/workflows/Python%20build/badge.svg
   :target: https://github.com/sappelhoff/pyprep/actions?query=workflow%3A%22Python+build%22
   :alt: Python build


.. image:: https://github.com/sappelhoff/pyprep/workflows/Python%20tests/badge.svg
   :target: https://github.com/sappelhoff/pyprep/actions?query=workflow%3A%22Python+tests%22
   :alt: Python tests


.. image:: https://codecov.io/gh/sappelhoff/pyprep/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/sappelhoff/pyprep
   :alt: Test coverage

.. image:: https://readthedocs.org/projects/pyprep/badge/?version=latest
   :target: https://pyprep.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://badge.fury.io/py/pyprep.svg
   :target: https://badge.fury.io/py/pyprep
   :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/pyprep.svg
   :target: https://anaconda.org/conda-forge/pyprep
   :alt: Conda version

.. image:: https://zenodo.org/badge/129240824.svg
   :target: https://zenodo.org/badge/latestdoi/129240824
   :alt: Zenodo archive

PyPREP
======

For documentation, see the:

- `stable documentation <https://pyprep.readthedocs.io/en/stable/>`_
- `latest (development) documentation <https://pyprep.readthedocs.io/en/latest/>`_

.. docs_readme_include_label

``pyprep`` is a Python implementation of the
`Preprocessing Pipeline (PREP) <https://doi.org/10.3389/fninf.2015.00016>`_
for EEG data, working with `MNE-Python <https://mne.tools>`_.

Installation
============

``pyprep`` runs on Python version 3.9 or higher.

We recommend to run ``pyprep`` in a dedicated virtual environment
(for example using `conda <https://docs.conda.io/en/latest/miniconda.html>`_).

For installing the **stable** version of ``pyprep``, call:

.. code-block:: Text

   python -m pip install --upgrade pyprep

or if you use `conda <https://docs.conda.io/en/latest/miniconda.html>`_:

.. code-block:: Text

   conda install --channel conda-forge pyprep

For installing the **latest (development)** version of ``pyprep``, call:

.. code-block:: Text

   python -m pip install --upgrade https://github.com/sappelhoff/pyprep/archive/refs/heads/main.zip

Both the *stable* and the *latest* installation will additionally install
all required dependencies automatically.
The dependencies are defined in the ``pyproject.toml`` file under the
``dependencies`` and ``project.optional-dependencies`` sections.

Contributing
============

The development of ``pyprep`` is taking place on
`GitHub <https://github.com/sappelhoff/pyprep>`_.

For more information, please see
`CONTRIBUTING.md <https://github.com/sappelhoff/pyprep/blob/main/.github/CONTRIBUTING.md>`_.

Citing
======

If you use this software in academic work, please cite it using the `Zenodo entry <https://zenodo.org/badge/latestdoi/129240824>`_.
Please also consider citing the original publication on PREP (see "References" below).
Metadata is encoded in the `CITATION.cff` file.

References
==========

1. Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A.
   (2015). The PREP pipeline: standardized preprocessing for large-scale EEG
   analysis. Frontiers in Neuroinformatics, 9, 16. doi:
   `10.3389/fninf.2015.00016 <https://doi.org/10.3389/fninf.2015.00016>`_
