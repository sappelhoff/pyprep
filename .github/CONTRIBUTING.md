# Contributions

Contributions are welcome in the form of feedback and discussion in issues,
or pull requests for changes to the code.

Once the implementation of a piece of functionality is considered to be free of
bugs and properly documented, it can be incorporated into the `main` branch.

To help developing `pyprep`,
you will need a few adjustments to your installation as shown below.

## Install the development version

First make a fork of the repository under your `USERNAME` GitHub account.
Then, in your Python environment follow these steps:

```Shell
git clone https://github.com/USERNAME/pyprep
cd pyprep
git fetch --tags --prune --prune-tags
python -m pip install -e ".[dev]"
pre-commit install
```

You may also clone the repository via ssh, depending on your preferred workflow
(`git clone git@github.com:USERNAME/pyprep.git`).

Note that we are working with "pre-commit hooks".
See https://pre-commit.com/ for more information.

## Running tests and coverage

If you have followed the steps to get the development version,
you can run tests as follows.
From the project root, call:

- `pytest` to run tests and coverage
- `pre-commit run -a` to run style checks (Ruff and some additional hooks)

## Building the documentation

The documentation can be built using [Sphinx](https://www.sphinx-doc.org).

The publicly accessible documentation is built and hosted by
[Read the Docs](https://readthedocs.org/).
Credentials for Read the Docs are currently held by:

- [@sappelhoff](https://github.com/sappelhoff/)

## Info about versioning

We follow a [semantic versioning scheme](https://semver.org/).
This is implemented via [hatch-vcs](https://github.com/ofek/hatch-vcs).

## Making a release on GitHub, PyPi, and Conda-Forge

- needs admin rights
- we are using [semver](https://semver.org/) (see section on versioning)
- we are using [GitHub Actions to deploy](./workflows/release.yml)

Follow this workflow:

1. go to your python environment for `pyprep`
1. make sure all tests pass and the docs are built cleanly
1. update `docs/changelog.rst`, renaming the "current" headline to the new
   version and updating the "Authors" section. "Authors" are all people
   who committed code or in other ways contributed to `pyprep` (e.g., by
   reviewing PRs, moderating discussions).
1. commit the change and `git push` to main (or make a pull request).
   Start your commit message with `[REL]`.
1. make an annotated tag `git tag -a -m "1.2.3" 1.2.3 upstream/main` (This
   assumes that you have a git remote configured with the name "upstream" and
   pointing to https://github.com/sappelhoff/pyprep).
1. `git push --follow-tags upstream`
1. make a release on GitHub, using the git tag from the previous step (e.g.,
   `1.2.3`). Fill the tag name into all fields of the release.

Then the release is done and main has to be prepared for development of the
next release:

1. add a "current" headline to `docs/changelog.rst`
1. commit the changes and `git push` to main (or make a pull request)
