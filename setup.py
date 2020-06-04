"""Setup pyprep."""
from setuptools import setup, find_packages
from os import path
import io

import versioneer

# Get long description from README file
here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyprep",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
        "psutil>=5.4.3",
    ],
    python_requires=">=3.6",
    project_urls={
        "Documentation": "https://pyprep.readthedocs.io/en/latest",
        "Bug Reports": "https://github.com/sappelhoff/pyprep/issues",
        "Source": "https://github.com/sappelhoff/pyprep",
    },
)
