"""Setup pyprep."""
from setuptools import setup, find_packages
import os
from os import path
import io

# get the version
version = None
with open(os.path.join("pyprep", "__init__.py"), "r") as fid:
    for line in (line.strip() for line in fid):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'")
            break
if version is None:
    raise RuntimeError("Could not determine version")

here = path.abspath(path.dirname(__file__))

# Get long description from README file
with io.open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyprep",
    version=version,
    description=(
        "A Python implementation of the preprocessing pipeline (PREP) for EEG data."
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="http://github.com/sappelhoff/pyprep",
    author="Stefan Appelhoff",
    author_email="stefan.appelhoff@mailbox.org",
    license="MIT",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
    ],
    keywords="EEG artifact preprocessing data",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.14.1",
        "scipy>=1.0.0",
        "statsmodels>=0.8.0",
        "mne>=0.19.0",
        "scikit-learn>=0.19.1",
        "matplotlib>=1.5.1",
        "psutil>=5.4.3",
    ],
    python_requires=">=3.6",
    extras_require={
        "test": ["pytest>=5.3", "codecov", "pytest-cov", "black", "pydocstyle"]
    },
    project_urls={
        "Documentation": "https://pyprep.readthedocs.io/en/latest",
        "Bug Reports": "https://github.com/sappelhoff/pyprep/issues",
        "Source": "https://github.com/sappelhoff/pyprep",
    },
)
