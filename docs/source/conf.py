# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'computer_vision'
copyright = '2025, author_name'
author = 'author_name'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
def setup(app):
    app.add_css_file('custom.css')
    
    html_theme_options = {
    'nosidebar': True,
    'show_search_results_only': False,
    'show_related': False,
    'show_relbars': False,
    'show_powered_by': False,
    'sidebar_collapse': False,
    'fixed_sidebar': False,
    'searchbar': False,  # <-- Ajoute cette ligne pour désactiver la barre de recherche
}


