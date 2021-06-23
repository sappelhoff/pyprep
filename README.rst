

.. image:: https://github.com/sappelhoff/pyprep/workflows/Python%20build/badge.svg
   :target: https://github.com/sappelhoff/pyprep/actions?query=workflow%3A%22Python+build%22
   :alt: Python build


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

For documentation, see the:

- `stable documentation <http://pyprep.readthedocs.io/en/stable/>`_
- `latest (development) documentation <http://pyprep.readthedocs.io/en/latest/>`_

.. docs_readme_include_label

``pyprep`` is a Python implementation of the
`Preprocessing Pipeline (PREP) <https://doi.org/10.3389/fninf.2015.00016>`_
for EEG data, working with `MNE-Python <https://mne.tools>`_.

**ALPHA SOFTWARE.**
**This package is currently in its early stages of iteration.**
**It may change both its internals or its user-facing API in the near future.**
**Any feedback and ideas on how to improve either of these is more than welcome!**
**Use this software at your own risk.**

Installation
============

``pyprep`` requires Python version ``3.7`` or higher to run properly.
We recommend to run ``pyprep`` in a dedicated virtual environment
(for example using `conda <https://docs.conda.io/en/latest/miniconda.html>`_).

For installing the **stable** version of ``pyprep``, call:

.. code-block:: Text

   pip install pyprep

**We currently recommend that you install the latest development version**

For installing the **latest (development)** version of ``pyprep``, call:

.. code-block:: Text

   pip install git+https://github.com/sappelhoff/pyprep.git@master


Both the *stable* and the *latest* installation will additionally install
all required dependencies automatically.
The dependencies are defined in the ``setup.cfg`` file under the
``options.install_requires`` section.

Contributions
=============

**We are actively looking for contributors!**

Please chime in with your ideas on how to improve this software by opening
a GitHub issue, or submitting a pull request.

See also our `CONTRIBUTING.md <https://github.com/sappelhoff/pyprep/blob/master/.github/CONTRIBUTING.md>`_
file for help with submitting a pull request.

Potential contributors should install ``pyprep`` in the following way:

#. First they should fork ``pyprep`` to their own GitHub account.
#. Then they should run the following commands,
   adequately replacing ``<gh-username>`` with their GitHub username.

.. code-block:: Text

   git clone https://github.com/<gh-username>/pyprep
   cd pyprep
   pip install -r requirements-dev.txt
   pre-commit install
   pip install -e .

References
==========

1. Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A.
   (2015). The PREP pipeline: standardized preprocessing for large-scale EEG
   analysis. Frontiers in Neuroinformatics, 9, 16. doi:
   `10.3389/fninf.2015.00016 <https://doi.org/10.3389/fninf.2015.00016>`_
