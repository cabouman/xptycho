# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'xptycho'
copyright = '2026, Charles A. Bouman'
author = 'Charles A. Bouman'

import xptycho
release = xptycho.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinxext.opengraph',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstrings only.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# The figures are drawn at build time by the scripts in figs/, so no
# image files are stored.  Only the picture is shown: no source, no links.
plot_include_source = False
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [('png', 150)]
plot_rcparams = {'savefig.bbox': 'tight'}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/cabouman/xptycho',
    'use_repository_button': True,
    'logo': {
        'image_light': '_static/logo.png',
        'image_dark': '_static/logo_dark.png',
    },
}
html_title = 'xptycho'
html_static_path = ['_static']

# Open Graph / social link preview.  Pasting a documentation URL into a
# chat, a post, or a message shows the card in _static/og_card.png, made
# by dev_scripts/make_social_card.py.  ogp_site_url makes the card and
# page URLs absolute, which link-preview crawlers require.
ogp_site_url = 'https://xptycho.readthedocs.io/en/latest/'
ogp_image = 'https://xptycho.readthedocs.io/en/latest/_static/og_card.png'
ogp_image_alt = 'xptycho: ptychographic reconstruction with PMACE in PyTorch'
ogp_type = 'website'
ogp_enable_meta_description = False
ogp_description_length = 0
ogp_social_cards = {'enable': False}    # use og_card.png, not per-page cards
