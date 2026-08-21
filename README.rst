

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

``pyprep`` runs on Python version 3.10 or higher.

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

Logging
=======

``pyprep`` logs through the standard ``logging`` module, on a logger named
``pyprep``, and configures nothing on import.
It is quiet, but not silent: warnings and errors still reach ``stderr`` without
any setup, because Python falls back to ``logging.lastResort`` when it finds no
handler.
``"INFO"`` records, which is where the pipeline reports what it decided, stay
hidden until you ask for them.

To see them without writing handler boilerplate, for example in a script or a
notebook:

.. code-block:: python

   import pyprep

   pyprep.setup_logging("info")

This attaches a handler on ``sys.stdout`` to the ``pyprep`` logger only, and
stops those records from propagating to the root logger so they are not printed
twice.
Pass ``"warning"`` to keep only warnings and errors, or ``"debug"`` for more
detail.

If your application already routes logging somewhere of its own, a file or a
Rich console, do not call ``setup_logging`` at all.
The records reach your handlers by propagation, and ``set_log_level`` will raise
``pyprep``'s verbosity if you want the ``"INFO"`` ones too, without touching the
handler, the stream, the format, or the propagation your application chose:

.. code-block:: python

   pyprep.set_log_level("info")

To take the output back after calling ``setup_logging``, drop the handler again:

.. code-block:: python

   logging.getLogger("pyprep").handlers.clear()

MNE has its own ``mne`` logger, controlled separately through
``mne.set_log_level``.

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
