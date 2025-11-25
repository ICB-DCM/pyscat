# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyScat"
copyright = "2025, Daniel Weindl"
author = "Daniel Weindl"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
]

# templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


autodoc_default_options = {
    "members": None,
    "imported-members": ["pyscat"],
    "inherited-members": False,
    "show-inheritance": None,
    "undoc-members": True,
    "special-members": "__init__",
}

# sphinx-autodoc-typehints options
# https://github.com/tox-dev/sphinx-autodoc-typehints
always_use_bars_union = True
typehints_fully_qualified = False

# autodoc_class_signature = "separated"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pypesto": ("https://pypesto.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

typehints_document_rtype = True
autodoc_typehints = "description"

bibtex_bibfiles = ["references.bib"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
