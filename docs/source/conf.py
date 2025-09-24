# -- Project information -----------------------------------------------------
project = "pyROMA"
copyright = "2025, Altynbek Zhubanchaliyev"
author = "Altynbek Zhubanchaliyev et al."
release = "0.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.mathjax",
    # "myst_parser",  # uncomment if you add Markdown docs
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Don’t execute notebooks on build (you already had this):
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# -- HTML --------------------------------------------------------------------
html_theme = "furo"

# If you keep a single logo file in docs/
# (works fine; optional: provide separate light/dark images)
html_logo = "pyroma_logo.png"

# Optional but nice with Furo:
html_theme_options = {
    "sidebar_hide_name": False,
    
    "source_repository": "https://github.com/altyn-bulmers/pyroma/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# Pleasant code highlighting in light/dark:
pygments_style = "tango"
pygments_dark_style = "native"

# Static assets (leave as-is if you don’t have custom CSS/JS yet)
html_static_path = ["_static"]

# Intersphinx (optional – point at common docs)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

