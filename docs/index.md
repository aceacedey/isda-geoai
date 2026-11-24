# GeoAI on Helsinki hexagons

This site belongs to the two lectures *Machine Learning & GeoAI approaches to spatial data analysis* of the course GIS-E1130 Introduction to Spatial Data Analysis at Aalto University. The lectures introduce the ways in which a machine can learn. Each page of this site takes one of those ways and shows it on the same small dataset: the Helsinki region divided into 1,459 hexagons, each with 16 counts of points of interest and one land-use label from OpenStreetMap.

<div class="roadmap"><div class="rm-question"><b>One dataset, every concept of the two lectures</b><span>A hexagon is one row of a table and one place on a map. Every page asks one question of that table and that map, and answers it with a figure.</span></div><div class="rm-steps"><div class="rm-step"><b>The data</b><span>a table and a map</span></div><div class="rm-step"><b>Part 1</b><span>what learning is; learning without and with answers</span></div><div class="rm-step"><b>Part 2</b><span>few answers, no answers, rewards; models that use location; what can go wrong</span></div><div class="rm-step"><b>Your own data</b><span>the same notebook on another table</span></div></div></div>

## How to read a page

- Every page opens with a road map: what the pages before it showed, the question of the page, and the steps that answer it.
- Every figure is followed by a short paragraph headed *What did we just see?*
- The code that made a figure is folded above it. Click *Show code cell source* to read it.
- Some figures have a slider. The pictures behind a slider were drawn beforehand, so nothing is computed while you move it.

## The pages

```{toctree}
:caption: The data
:maxdepth: 1

data
```

```{toctree}
:caption: Part 1 of the lecture
:maxdepth: 1

learning
unsupervised
supervised
```

```{toctree}
:caption: Part 2 of the lecture
:maxdepth: 1

semi_supervised
self_supervised
reinforcement
spatially_explicit
challenges
```

```{toctree}
:caption: More
:maxdepth: 1

own_data
```

## Run it yourself

All pages together are one Jupyter notebook: <a href="_static/GeoAI_Helsinki_hexagons.ipynb" download>download GeoAI_Helsinki_hexagons.ipynb</a>. The notebook needs the files `helpers.py` and `data/helsinki_hex_r8.gpkg` of the repository next to it, and the Python packages listed in `requirements.txt`. It runs in about four minutes on a laptop. The page [Your own data](own_data) says what to change for another city or another table.

## Data and licence

The points of interest and the land-use polygons are © OpenStreetMap contributors and available under the Open Database License (ODbL). The hexagons are cells of the H3 grid at resolution 8. The script `build_dataset.py` of the repository builds the table from these sources. The text and the figures of this site are licensed CC BY-SA 4.0.
