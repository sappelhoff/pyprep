"""Provide a convenient way to link to GitHub issues and pull requests.

Link to any issue or PR using :gh:`issue-or-pr-number`.

Adapted from:
https://doughellmann.com/blog/2010/05/09/defining-custom-roles-in-sphinx/

"""
from docutils.nodes import reference
from docutils.parsers.rst.roles import set_classes


def gh_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link to a GitHub issue."""
    try:
        # issue/PR mode (issues/PR-num will redirect to pull/PR-num)
        int(text)
    except ValueError:
        # direct link mode
        slug = text
    else:
        slug = "issues/" + text
    text = "#" + text
    ref = "https://github.com/sappelhoff/pyprep/" + slug
    set_classes(options)
    node = reference(rawtext, text, refuri=ref, **options)
    return [node], []


def setup(app):
    """Do setup."""
    app.add_role("gh", gh_role)
    return
