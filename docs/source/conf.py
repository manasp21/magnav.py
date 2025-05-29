# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../')) # Adjusted path to reach project root


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MagNavPy'
copyright = '2020-2023, Original MagNav.jl Authors; 2024-2025, Manas Pandey'
author = 'Original MagNav.jl Authors; Ported to Python by Manas Pandey'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'github_user': 'manasp21',
    'github_repo': 'magnav.py',
    'github_button': True,
    'github_type': 'star',
    'fixed_sidebar': True,
    'page_width': '1000px', # Optional: Adjust page width
    'sidebar_width': '220px', # Optional: Adjust sidebar width
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}