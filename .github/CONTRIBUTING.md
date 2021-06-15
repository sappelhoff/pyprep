# Table of Contents

- [Pull request workflow](#pull-request-workflow)
  * [Syncing your fork's `master` with `upstream master`](#syncing-your-fork-s--master--with--upstream-master-)
  * [Working on a feature (and rebasing)](#working-on-a-feature--and-rebasing-)
    + [Working on a feature](#working-on-a-feature)
    + [Rebasing without conflicts](#rebasing-without-conflicts)
    + [Rebasing WITH conflicts](#rebasing-with-conflicts)
    + [Rebasing ... panic mode (or "the easy way")](#rebasing--panic-mode--or--the-easy-way--)
- [Info about docs](#info-about-docs)
- [Info about versioning](#info-about-versioning)
- [How to make a release](#how-to-make-a-release)

# Pull request workflow

- assuming you are working with `git` from a command line
- assuming your GitHub username is `username`
- `github.com/sappelhoff/pyprep` is `upstream`
- `github.com/username/pyprep` is `origin` (your *fork*)

## Syncing your fork's `master` with `upstream master`

- first, you start with *forking* `upstream`
- then, you continue by *cloning* your fork: `git clone https://github.com/username/pyprep`
    - you'll have your own `master` branch there
- **you always want to make sure that your fork's `master` and `upstream master` are aligned**
    - to do this, you work with *git remotes*
    - Note: this also means that you NEVER work on `master` (unless you know
      what you are doing) ... because you want to always be able to SYNC your
      fork with `upstream`, which would mean losing your own work on `master`
- use `git remote -v` to list your configured remotes
    - initially this will only list `origin` ... a bit like this maybe:

```Text
origin	https://github.com/username/pyprep (fetch)
origin	https://github.com/username/pyprep (push)
```

- Now you want to add `upstream` as a remote. Use
  `git remote add upstream https://github.com/sappelhoff/pyprep`
- again, do `git remote -v`, it should look like this:

```Text
origin	https://github.com/username/pyprep (fetch)
origin	https://github.com/username/pyprep (push)
upstream	https://github.com/sappelhoff/pyprep (fetch)
upstream	https://github.com/sappelhoff/pyprep (push)

```

- Now you can use your `upstream` *remote* to make sure your fork's `master` is
  up to date.
    1.  `git checkout master` to make sure you are on your `master` branch
    1. Make sure you do not have any changes on your `master`, because we will
       discard them!
    1. `git pull upstream master` SYNC your fork and `upstream`
    1. sometimes there are issues, so to be safe, do:
       `git reset --hard upstream/master` ... this makes sure that both
       branches are really synced.
    1. ensure with another `git pull upstream master` ... this should say
       "already up to date"

## Working on a feature (and rebasing)

### Working on a feature

- before working on any feature: always do `git checkout master` and
  `git pull upstream master`
- then make your new branch to work on and check it out, for example
  `git checkout -b my_feature`
    - do your work
    - submit a pull request
    - hope you are lucky and nobody did work in between
- however IF somebody did work in between, we need to *rebase*. Just follow the
  steps below

### Rebasing without conflicts

1. sync `master` through: `git checkout master` and `git pull upstream master`
1. go back to your branch and rebase it: `git checkout my_feature` and then
   `git rebase master`

Now it could be that you are lucky and there no conflicts ... in that case, the
rebase just works and you can then finish up by *force pushing* your rebased
branch: `git push -f my_feature` ... you need to force it, because rebasing
changed the history of your branch. But don't worry, if rebasing "just worked"
without any conflicts, this should be very safe.

### Rebasing WITH conflicts

In case you are unlucky, there are conflicts and you'll have to resolve them
step by step ... `git` will be in *rebase* mode and try to rebase one commit
after another ... for each commit where conflicts are detected, it'll stop.

Then you have to do: `git status` to see conflicting files ... then edit these
files to resolve conflicts ... then `git add <filename>` ... and then
`git rebase --continue` to go on to the next commit, rinse and repeat.

**NOTE: the conflict resolution part is the dangerous part that can get very
messy and where you can actually lose stuff ... so make backups of your branch
before.**

After everything is resolved, you can again do `git push -f my_feature`.

If you screw up **during** rebasing and you panic, you can do
`git rebase --abort` and start again.

### Rebasing ... panic mode (or "the easy way")

If nothing helps and you just don't know how to resolve the issues and
conflicts that arise during rebasing, just make a new branch:
    1. `git checkout master`
    1. `git pull upstream master`
    1. `git checkout -b my_feature_2nd_attempt`

... and apply your changes manually.

This method is not really a `git` workflow, ... but in cases where there are
only few changes, this is often a practical solution.

# Info about versioning

The versioning of `pyprep` is done with [versioneer](https://github.com/python-versioneer/python-versioneer).

The following files are controlled by `versioneer` and should not be modified manually:

- `./versioneer.py`
- `./pyprep/_version.py`

The same is true for the following lines in `./pyprep/__init__.py`:

```Python
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
```

To update the `versioneer` software, follow the instructions on their documentation page.
For the day-to-day development of `pyprep`, no interaction with `versioneer` is neeeded,
because the software picks up all information from the `git` commands.

# Info about docs

The documentation is build and hosted by [https://readthedocs.org/](https://readthedocs.org/).

Admin credentials are needed to access the setup.

# How to make a release

- needs admin rights
- we are using [semver](https://semver.org/) (see section on versioning)
- we are using [GitHub Actions to deploy](./workflows/python_publish.yml)
- PyPi credentials are stored as GitHub secrets

Follow this workflow:

1. go to your python environment for `pyprep`
1. make sure all tests pass and the docs are built cleanly
1. update `docs/whats_new.rst`, renaming the "current" headline to the new
   version and updating the "Authors" section. "Authors" are all people
   who committed code or in other ways contributed to `pyprep` (e.g., by
   reviewing PRs, moderating discussions).
1. commit the change and `git push` to master (or make a pull request).
   Start your commit message with `[REL]`.
1. make an annotated tag `git tag -a -m "1.2.3" 1.2.3 upstream/master` (This
   assumes that you have a git remote configured with the name "upstream" and
   pointing to https://github.com/sappelhoff/pyprep).
1. `git push --follow-tags upstream`
1. make a release on GitHub, using the git tag from the previous step (e.g.,
   `1.2.3`). Fill the tag name into all fields of the release.

Then the release is done and master has to be prepared for development of the
next release:

1. add a "current" headline to `docs/whats_new.rst`
1. commit the changes and `git push` to master (or make a pull request)
