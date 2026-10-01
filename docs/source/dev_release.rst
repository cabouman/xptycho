===================
Releasing a version
===================

xptycho is published to PyPI by the GitHub Actions workflow in
``.github/workflows/release.yml``.  You drive it from your machine with
``dev_scripts/release.sh``: one command opens a pull request for you to review,
and after you merge it a second command tags the release and publishes it.
Uploads use Trusted Publishing, so no API token is ever stored or typed.

.. note::

   xptycho has not been released yet, and the one-time setup below has not
   been done.  Do the one-time setup before the first release.

The examples below release version ``0.1.0``.  Replace ``0.1.0`` with the
version you are releasing.

Branches
========

Work happens on ``prerelease``, and ``main`` is the released code.  After a
release, ``main``, ``prerelease``, and the version tag are all on the same
commit.  New work then goes on ``prerelease``, which stays ahead of ``main``
until the next release.

One-time setup
==============

These steps are done once for the package and never repeated.

1. On GitHub, open the repository settings and create two environments named
   ``pypi`` and ``testpypi``.  Do not add a required reviewer to either one.
   The pull request review is the only approval in the procedure.

2. On `PyPI <https://pypi.org/manage/account/publishing/>`__, add a pending
   publisher with these values:

   - PyPI project name: ``xptycho``
   - Owner: ``cabouman``
   - Repository name: ``xptycho``
   - Workflow name: ``release.yml``
   - Environment name: ``pypi``

3. On `TestPyPI <https://test.pypi.org/manage/account/publishing/>`__, add a
   pending publisher with the same values and the environment name
   ``testpypi``.  TestPyPI and PyPI are separate sites with separate accounts.

Dry run on TestPyPI (optional, recommended the first time)
==========================================================

This proves the whole pipeline on a throwaway upload before the real one.  A
PyPI version number can never be reused, so it is worth doing once.

1. Publish a release candidate::

       dev_scripts/release.sh 0.1.0rc1

   This stamps the version, pushes ``prerelease``, and creates a GitHub
   pre-release tagged ``v0.1.0rc1``.  GitHub Actions builds the package and
   uploads it to TestPyPI.  No approval is needed.

2. Check the upload::

       pip install -i https://test.pypi.org/simple/ xptycho

   If something is wrong, fix it and repeat with ``0.1.0rc2``.

Release to PyPI
===============

1. Open the release pull request::

       dev_scripts/release.sh 0.1.0

   This sets the version to ``0.1.0`` on ``prerelease``, pushes it, and opens a
   pull request from ``prerelease`` to ``main``.  Nothing is published yet.

2. Review the code and accept.  On GitHub, open the pull request and look at the
   diff.  The tests run on the pull request automatically, so their result is
   shown next to the **Merge** button.  If you are happy, click **Merge pull
   request**.  (If you are not, close it and keep working on ``prerelease``.)

3. Publish the release::

       dev_scripts/release.sh 0.1.0 --publish

   This fast-forwards ``prerelease`` up to ``main`` so the two branches are
   identical, updates your local ``main`` to match, and tags the shared commit
   ``v0.1.0``.  GitHub Actions then builds the package and publishes it to PyPI.

4. Confirm it is live::

       pip install xptycho

Notes
=====

- The tag is always ``v`` followed by the version; the workflow fails the build
  if the tag does not match ``__version__``.
- After step 3, ``main``, ``prerelease``, and the ``v0.1.0`` tag are all on the
  same commit.  Tagging adds a label to that commit; it moves no branch, so the
  branches stay in sync and you can go on adding to ``prerelease``.
- Merge the pull request with the default **Create a merge commit** option, so
  that ``prerelease`` can fast-forward up to ``main`` in step 3.
- The version is single-sourced from ``xptycho/__init__.py``; ``pyproject.toml``
  and the docs read it from there.  ``release.sh`` also stamps the version and
  date into ``CITATION.cff`` and the software BibTeX entries.
- Read the Docs caches pages.  After a docs change, reload the page with
  Cmd+Shift+R to see the new build.
