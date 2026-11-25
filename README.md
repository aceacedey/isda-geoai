# GeoAI on Helsinki hexagons

A worked example, as a website of short notebooks, for the two lectures *Machine Learning & GeoAI approaches to spatial data analysis* of the course GIS-E1130 Introduction to Spatial Data Analysis (Aalto University). Each page shows one concept of the lectures on the same dataset: the Helsinki region divided into 1,459 H3 hexagons, each with 16 counts of OpenStreetMap points of interest and one land-use label.

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

Every page is a short notebook that runs by itself: a line of text, a code cell that is shown, its output, the result in one line. A page loads the data in its first cell (`hexes, X, y, W = h.load()`) and uses nothing from another page.

```
src/NN_slug.txt      the text and the code of one page (edit these)
build_site.py        makes one notebook per page, runs each in its own kernel, writes docs/<slug>.ipynb and docs/_static/
helpers.py           loading, maps and plots that the notebooks call
data/                the hexagon tables (GeoPackage), resolution 8 and 9
build_dataset.py     builds the tables again from OpenStreetMap (not needed to build the site)
docs/                the Sphinx project that Read the Docs builds: conf.py, index.md, the page notebooks, _static/
```

`requirements.txt` lists what a student needs to run the notebooks; `requirements-build.txt` adds what the teacher needs to build the site and the dataset. The page *Getting started* (`docs/setup.md`) is written by hand. The pictures in `docs/_static/slides/` are slides of the lecture deck, exported at 1280 × 720 pixels; the deck itself is not in this repository.

The page notebooks in `docs/` carry their outputs, so Read the Docs installs only Sphinx, MyST-NB and the Furo theme (`docs/requirements.txt`) and runs no analysis code. For download, `docs/_static/notebooks/` holds every notebook without outputs, and `docs/_static/geoai_hexagons.zip` holds all of them with `helpers.py`, `requirements.txt` and the data.

### After an edit

```
pip install -r requirements-build.txt      # once: the packages of the notebooks, and those to build the site and the dataset
python build_site.py                       # runs every page again, four at a time
python build_site.py data supervised       # runs only these pages
python build_site.py text                  # after a change of text only: keeps the stored outputs
pip install -r docs/requirements.txt       # once: Sphinx, MyST-NB, Furo
sphinx-build -b html docs docs/_build/html # the site on this computer: open docs/_build/html/index.html through a local web server
```

The sliders and the interactive map need a web server, because a browser does not load them from a `file://` address: `python -m http.server --directory docs/_build/html` and then <http://localhost:8000>.

Source format: `@@page slug` starts a page, `@@md` a text cell, `@@md site` a text cell that only the website shows (the lecture slides), `@@code` a code cell that is shown, `@@code fold` one that is folded, and `@@code hidden` one that is left out of the site and of the notebook for download. Hidden cells hold only the teacher's plumbing: the frames of a slider, the interactive map, numbers for the slides.

Rules of a page: at most six steps; one to three sentences before a code cell; a shown code cell of at most 12 lines (a learning step that is written out may be longer); drawing code lives in `helpers.py`; a page stays under about 450 words of text.

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
