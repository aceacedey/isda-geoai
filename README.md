# GeoAI on Helsinki hexagons

A website for the two lectures *Machine Learning & GeoAI approaches to spatial data analysis* of the course GIS-E1130 Introduction to Spatial Data Analysis (Aalto University). Each page shows one concept of the lectures on the same dataset: the Helsinki region divided into 1,459 H3 hexagons, each with 16 counts of OpenStreetMap points of interest and one land-use label.

| Page | Concept | Slider or widget |
|---|---|---|
| The data | a table and a map; the label is a rule; neighbours are alike | interactive map |
| What is learning? | task, experience, performance; the five ways to learn | |
| Unsupervised learning | k-means, the elbow method, groups without names | k-means step by step; K from 2 to 8 |
| Supervised learning | classification and regression, parameters, training, unseen rows, flexibility | a line with a and b; training; flexibility |
| Semi-supervised learning | self-training from 30 labels | the rounds of self-training |
| Self-supervised learning | pretext task, embedding, similarity search | |
| Reinforcement learning | a courier robot that learns from points | the tries of the robot |
| Spatially explicit models | spatial lag, the invariance test, the size of the unit | |
| Challenges | random and spatial splits, three cities, bias in the data, a research case | |
| Your own data | sources, a recipe, working with an AI assistant | |

## How the site is made

All pages are one notebook, cut into pages.

```
src/NN_slug.txt      the text and the code of one page (edit these)
build_site.py        joins the sources, runs the notebook once, writes docs/<slug>.ipynb and docs/_static/
helpers.py           data loading, maps and plots used by the notebook
data/                the hexagon tables (GeoPackage), resolution 8 and 9
build_dataset.py     builds the tables again from OpenStreetMap (not needed to build the site)
docs/                the Sphinx project that Read the Docs builds: conf.py, index.md, the page notebooks, _static/
notebooks/           the whole notebook in one file, with outputs
```

The page notebooks in `docs/` carry their outputs, so Read the Docs installs only Sphinx, MyST-NB and the Furo theme (`docs/requirements.txt`) and runs no analysis code.

### After an edit

```
pip install -r requirements.txt            # once: the packages of the analysis
python build_site.py                       # after a change of code: runs the notebook again (about four minutes)
python build_site.py text                  # after a change of text only: keeps the stored outputs
pip install -r docs/requirements.txt       # once: Sphinx, MyST-NB, Furo
sphinx-build -b html docs docs/_build/html # the site on this computer: open docs/_build/html/index.html through a local web server
```

The sliders and the interactive map need a web server, because a browser does not load them from a `file://` address: `python -m http.server --directory docs/_build/html` and then <http://localhost:8000>.

Source format: `@@page slug` starts a page, `@@md` a text cell, `@@code` a code cell that is folded on the site, `@@code show` one that is open, `@@code hidden` one that is left out of the site, and `@@code slider` one whose picture is left out because a slider shows the same pictures.

## Publishing on Read the Docs

1. Create an empty repository on GitHub and push this folder to it (`.gitignore` already leaves out the virtual environment, the local build and the working pictures).
2. On <https://app.readthedocs.org>, choose *Add project*, pick the repository and keep the defaults. The file `.readthedocs.yaml` tells Read the Docs what to do.
3. The site is rebuilt after every push.

## Data and licences

- Points of interest and land-use polygons: © OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0. The tables in `data/` are derived from them and are under the same licence.
- Hexagons: cells of the H3 grid, resolutions 8 and 9.
- The interactive map loads its background from the tile server of OpenStreetMap.
- Text and figures: CC BY-SA 4.0. Code: see `LICENSE` once a licence has been chosen.
- The research case on the page *Challenges* quotes published and aggregated results only; no travel-demand data are in this repository.
