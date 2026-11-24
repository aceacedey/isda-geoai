"""Sphinx configuration of the website.

Build on this computer:  pip install -r docs/requirements.txt
                         sphinx-build -b html docs docs/_build/html
The pages are notebooks with stored outputs (written by build_site.py); nothing is executed during the build.
"""
project = "GeoAI on Helsinki hexagons"
author = "Subhrasankha Dey"
copyright = "2026, Subhrasankha Dey. Text and figures CC BY-SA 4.0. Map data © OpenStreetMap contributors (ODbL)"
release = "2026.11"

extensions = ["myst_nb"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
nb_execution_mode = "off"            # show the stored outputs; never run a notebook during the build
nb_merge_streams = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]
html_theme = "furo"
html_title = "GeoAI on Helsinki hexagons"
html_static_path = ["_static"]
html_css_files = ["site.css"]
html_js_files = ["site_data.js", "widgets.js"]
html_theme_options = {"navigation_with_keys": True}
