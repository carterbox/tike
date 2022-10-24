#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Tike documentation build configuration file, created by
# sphinx-quickstart on Tue Sep 12 16:06:17 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
import os
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../../src'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.napoleon',
              'sphinx.ext.coverage',
              'sphinx.ext.autosummary',
              'sphinx.ext.imgmath',
              'sphinx.ext.viewcode',
              'sphinxcontrib.bibtex',
              'sphinx.ext.extlinks',
              'nbsphinx',
              ]

# bibtex setting
bibtex_bibfiles = [
    'bibtex/zrefs.bib',
]

# extlinks settings
extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi:'),
}

# Napoleon settings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
Argonne = u'Argonne National Laboratory'
project = u'Tike'
copyright = u'2017-2020, ' + Argonne

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
try:
    release = get_distribution('tike').version
    # The short X.Y version.
    version = '.'.join(release.split('.')[:2])
except DistributionNotFound:
    # package is not installed
    pass

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en_US'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 4,
}

# -- Options for HTMLHelp output ------------------------------------------

htmlhelp_basename = project+'doc'

# -- Options for autodoc output ------------------------------------------

autodoc_mock_imports = [
    'cupy',
    'cupyx',
    'h5py',
    'importlib_resources',
    'matplotlib',
    'matplotlib.pyplot',
    'mpi4py',
    'numpy',
    'scipy',
    'cv2',
]
