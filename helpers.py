"""Helpers for the week-10 GeoAI tutorial: loading the hexagons, the neighbour matrix, maps, figure export.

The learning algorithms themselves are written out in the notebook. This file only holds the plumbing.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

FEATURES = ["food", "shop_daily", "shop_other", "education", "health", "car", "bike", "civic", "culture",
            "finance", "street_furniture", "sport", "play", "office", "lodging", "craft_industry"]
CLASSES = ["green", "residential", "work"]

# Colours. The three class colours were checked for colour-blind separation as a set.
COLORS = {"green": "#1baf7a", "residential": "#eb6834", "work": "#2a78d6"}
UNKNOWN = "#e4e3dd"      # hexagons whose label is hidden
INK = "#0b0b0b"
MUTED = "#898781"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
HIGHLIGHT = "#0b0b0b"

# A few places for orientation on the maps (lon, lat).
PLACES = {"Helsinki centre": (24.941, 60.170), "Otaniemi": (24.827, 60.186), "Tikkurila": (25.044, 60.292),
          "Espoon keskus": (24.656, 60.205), "Itäkeskus": (25.082, 60.210), "Nuuksio": (24.550, 60.300)}

plt.rcParams.update({
    "figure.dpi": 90, "savefig.dpi": 150, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
})


# ------------------------------------------------------------------------------------------------ data
def load_hexagons(res=8):
    """One row per H3 hexagon, in metres (ETRS-TM35FIN). Built by build_dataset.py from OpenStreetMap."""
    path = DATA / f"helsinki_hex_r{res}.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run:  python build_dataset.py {res}")
    return gpd.read_file(path).to_crs("EPSG:3067").reset_index(drop=True)


def neighbour_matrix(gdf):
    """Row-standardised spatial weights matrix W (as in week 3): W[i, j] = 1/k if j is one of the k
    hexagons touching i. Then  W @ X  is the average of the neighbours' values: the spatial lag."""
    xy = gdf[["x", "y"]].to_numpy()
    spacing = np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1])
    A = NearestNeighbors(radius=1.2 * spacing).fit(xy).radius_neighbors_graph(xy).tolil()
    A.setdiag(0)
    A = sparse.csr_matrix(A)
    k = np.asarray(A.sum(axis=1)).ravel()
    return sparse.diags(1.0 / np.maximum(k, 1)) @ A


def hexagon_at(gdf, lon, lat):
    """Row number of the hexagon that contains a lon/lat point."""
    pt = gpd.GeoSeries.from_xy([lon], [lat], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
    hit = np.where(gdf.contains(pt))[0]
    if len(hit) == 0:
        raise ValueError("That point is outside the study area (or in a hexagon that was dropped).")
    return int(hit[0])


# ------------------------------------------------------------------------------------------------ maps
def _finish_map(ax, gdf, title, places=True):
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if places:
        pts = gpd.GeoSeries.from_xy([p[0] for p in PLACES.values()], [p[1] for p in PLACES.values()],
                                    crs="EPSG:4326").to_crs(gdf.crs)
        for name, p in zip(PLACES, pts):
            ax.plot(p.x, p.y, "o", ms=3, color=INK, zorder=5)
            ax.annotate(name, (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK,
                        zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def map_classes(gdf, labels, ax=None, title="", outline=None, places=True, legend=True):
    """Colour hexagons by class. Hidden labels (None / NaN) are drawn in light grey.
    outline: row numbers to draw with a dark edge (e.g. the hexagons whose label we know)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.2))
    labels = pd.Series(labels, index=gdf.index, dtype="object")
    colors = labels.map(COLORS).fillna(UNKNOWN)
    gdf.plot(ax=ax, color=colors.tolist(), edgecolor="white", linewidth=0.25)
    if outline is not None and len(outline):
        gdf.iloc[np.asarray(outline)].plot(ax=ax, facecolor="none", edgecolor=HIGHLIGHT, linewidth=1.1, zorder=4)
    if legend:
        handles = [Patch(fc=COLORS[c], label=f"{c} ({int((labels == c).sum())})") for c in CLASSES]
        n_hidden = int(labels.isna().sum())
        if n_hidden:
            handles.append(Patch(fc=UNKNOWN, label=f"label hidden ({n_hidden})"))
        ax.legend(handles=handles, loc="upper left", fontsize=9, handlelength=1.2)
    _finish_map(ax, gdf, title, places)
    return ax


def map_values(gdf, values, ax=None, title="", label="", cmap="Blues", vmin=None, vmax=None, outline=None, places=True):
    """Colour hexagons by a number (one hue, light to dark)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.2))
    gdf.assign(_v=np.asarray(values)).plot(ax=ax, column="_v", cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="white",
                                           linewidth=0.25, legend=True,
                                           legend_kwds={"shrink": 0.55, "label": label, "pad": 0.01})
    if outline is not None and len(outline):
        gdf.iloc[np.asarray(outline)].plot(ax=ax, facecolor="none", edgecolor=HIGHLIGHT, linewidth=1.4, zorder=4)
    _finish_map(ax, gdf, title, places)
    return ax


def map_groups(gdf, groups, ax=None, title="", highlight=None, places=False):
    """Draw hexagons in light grey and the hexagons of one group (e.g. the test fold) in dark."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.2))
    groups = np.asarray(groups)
    colors = np.where(groups == highlight, "#2a78d6", UNKNOWN)
    gdf.plot(ax=ax, color=colors.tolist(), edgecolor="white", linewidth=0.25)
    ax.legend(handles=[Patch(fc="#2a78d6", label=f"test ({int((groups == highlight).sum())})"),
                       Patch(fc=UNKNOWN, label=f"train ({int((groups != highlight).sum())})")],
              loc="upper left", fontsize=9, handlelength=1.2)
    _finish_map(ax, gdf, title, places)
    return ax


def bars(ax, names, values, colors=None, title="", xlabel="accuracy", xlim=(0, 1), fmt="{:.0%}"):
    """Horizontal bars with the value written at the end of each bar."""
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, height=0.55, color=colors or "#2a78d6")
    for yi, v in zip(y, values):
        ax.text(v + 0.012, yi, fmt.format(v), va="center", fontsize=10, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    return ax


# ------------------------------------------------------------------------------------------------ part 0
PILES = {"training": "#cfcdc6", "validation": "#bb16a3", "test": "#2a78d6"}


ACCENT = "#bb16a3"          # the accent colour of the slides


def accent_cmap():
    """One hue, light to dark, for maps of a number (it avoids the three colours of the land-use classes)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("accent", ["#f1efe9", "#e7b3dc", ACCENT, "#5c0a50"])


def number_hexagons(ax, gdf, rows, size=10):
    """Outline the given rows on a map and write 1, 2, 3 ... next to them."""
    gdf.iloc[list(rows)].plot(ax=ax, facecolor="none", edgecolor=HIGHLIGHT, linewidth=2.0 * size / 10, zorder=7)
    for n, i in enumerate(rows, 1):
        ax.annotate(str(n), (gdf["x"].iloc[i], gdf["y"].iloc[i]), xytext=(0.9 * size, -1.3 * size), textcoords="offset points",
                    fontsize=size, fontweight="bold", color="white", ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=HIGHLIGHT, ec="white", lw=0.8))


def inset_map(gdf, labels=None, values=None, rows=None, title="", vmin=None, vmax=None, figsize=(4.2, 3.3)):
    """A small map without a legend, for the corner of a slide: hexagons coloured by class (labels) or by a number
    (values, light to dark). rows: hexagons to outline and number, so that a table, a plot and the map show the same rows."""
    fig, ax = plt.subplots(figsize=figsize)
    if labels is not None:
        colors = pd.Series(labels, index=gdf.index, dtype="object").map(COLORS).fillna(UNKNOWN)
        gdf.plot(ax=ax, color=colors.tolist(), edgecolor="white", linewidth=0.1)
    else:
        gdf.assign(_v=np.asarray(values, float)).plot(ax=ax, column="_v", cmap=accent_cmap(), vmin=vmin, vmax=vmax,
                                                      edgecolor="white", linewidth=0.1)
    if rows is not None:
        number_hexagons(ax, gdf, rows, size=9)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9 if title else 1)
    return fig


def count_ticks(ax, axis, counts, to_axis=np.log1p):
    """Write the ticks of a logarithmic axis as plain counts. to_axis turns a count into its position on the axis."""
    pos = [float(to_axis(c)) for c in counts]
    labels = [f"{c:,}" for c in counts]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(pos, labels)


def class_regions(ax, P, labels, share=0.5):
    """For every class, shade the part of the plane that holds the densest `share` of its points, and mark its average
    with a star. A cloud of 1,459 dots hides where each class sits; these regions show it."""
    from scipy.stats import gaussian_kde

    labels = np.asarray(labels)
    lo, hi = P.min(axis=0) - 0.5, P.max(axis=0) + 0.5
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 170), np.linspace(lo[1], hi[1], 170))
    for c in CLASSES:
        pts = P[labels == c].T
        kde = gaussian_kde(pts, bw_method=0.35)
        z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        cut = np.sort(kde(pts))[int((1 - share) * pts.shape[1])]
        ax.contourf(gx, gy, z, levels=[cut, z.max()], colors=[COLORS[c]], alpha=0.2)
        ax.contour(gx, gy, z, levels=[cut], colors=[COLORS[c]], linewidths=2.2)


def feature_plane(values, labels, names, rows=None, ax=None, title="", counts=(0, 1, 3, 10, 30, 100)):
    """Two features as two axes: one dot per hexagon, and for every class the region that holds the densest half of it.
    values are log(1 + count); the ticks are written as counts."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 4.8))
    values, labels = np.asarray(values, float), np.asarray(labels)
    P = spread(values, 0.16)
    for c in CLASSES:
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=7, color=COLORS[c], alpha=0.4, linewidths=0)
    class_regions(ax, P, labels)
    for c in CLASSES:
        m = labels == c
        ax.scatter([], [], s=60, color=COLORS[c], label=f"{c} ({m.sum()})")
    if rows is not None:
        for n, i in enumerate(rows, 1):
            ax.scatter(*P[i], s=170, facecolor=COLORS[labels[i]], edgecolor=HIGHLIGHT, linewidths=1.6, zorder=9)
            ax.annotate(str(n), P[i], ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=10)
    count_ticks(ax, "x", counts), count_ticks(ax, "y", counts)
    ax.set_xlim(-0.35, values[:, 0].max() + 0.35), ax.set_ylim(-0.35, values[:, 1].max() + 0.35)
    ax.set_xlabel(names[0]), ax.set_ylabel(names[1])
    ax.legend(loc="upper left", fontsize=9, handletextpad=0.2)
    ax.set_title(title)
    return ax


def spread(values, amount=0.13, seed=0):
    """Shift every point by a small random amount, for drawing only. Counts are whole numbers, so many
    hexagons have exactly the same values and would hide behind each other in a scatter plot."""
    values = np.asarray(values, float)
    return values + np.random.default_rng(seed).uniform(-amount, amount, values.shape)


def feature_space(values, labels, names, rows=None, azim=-58, elev=20, ax=None, title=""):
    """Three features as three axes: one dot per hexagon, coloured by its label."""
    if ax is None:
        fig = plt.figure(figsize=(7.2, 6))
        ax = fig.add_subplot(projection="3d")
        fig.subplots_adjust(left=0.0, right=0.94, bottom=0.03, top=0.97 if title else 1.0)
    P = spread(values)
    labels = np.asarray(labels)
    for c in CLASSES:
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=9, color=COLORS[c], alpha=0.55, linewidths=0, label=f"{c} ({m.sum()})",
                   depthshade=False)
    if rows is not None:
        for n, i in enumerate(rows, 1):
            ax.scatter(*P[i], s=150, facecolor=COLORS[labels[i]], edgecolor=HIGHLIGHT, linewidths=1.8, depthshade=False, zorder=9)
            ax.text(P[i, 0], P[i, 1], P[i, 2] + 0.32, str(n), fontsize=11, fontweight="bold", ha="center", zorder=10)
    ax.set_xlabel(names[0]), ax.set_ylabel(names[1]), ax.set_zlabel(names[2])
    ax.view_init(elev=elev, azim=azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#f7f6f2")
        axis.pane.set_edgecolor(GRID)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=9, markerscale=2.2, handletextpad=0.2)
    ax.set_title(title)
    return ax


def regions(model, values, labels, names, ax=None, title="", rows=None, counts=None):
    """Colour the plane of two features by what a fitted model predicts there, and draw the hexagons on top.
    counts: if given, write the ticks of both axes as these counts (the values are log(1 + count))."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.8))
    from matplotlib.colors import ListedColormap

    values, labels = np.asarray(values, float), np.asarray(labels)
    hi = values.max(axis=0) + 0.35
    gx, gy = np.meshgrid(np.linspace(-0.3, hi[0], 260), np.linspace(-0.3, hi[1], 260))
    z = pd.Series(model.predict(np.c_[gx.ravel(), gy.ravel()])).map({c: k for k, c in enumerate(CLASSES)}).to_numpy()
    ax.pcolormesh(gx, gy, z.reshape(gx.shape), cmap=ListedColormap([COLORS[c] for c in CLASSES]), vmin=0, vmax=2,
                  alpha=0.28, shading="auto", rasterized=True)
    P = spread(values)
    for c in CLASSES:
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=7, color=COLORS[c], alpha=0.7, linewidths=0)
    if rows is not None:
        for n, i in enumerate(rows, 1):
            ax.scatter(*P[i], s=170, facecolor=COLORS[labels[i]], edgecolor=HIGHLIGHT, linewidths=1.6, zorder=9)
            ax.annotate(str(n), P[i], ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=10)
    if counts is not None:
        count_ticks(ax, "x", counts), count_ticks(ax, "y", counts)
    ax.set_xlim(-0.3, hi[0]), ax.set_ylim(-0.3, hi[1])
    ax.set_xlabel(names[0]), ax.set_ylabel(names[1])
    ax.set_title(title)
    return ax


def map_piles(gdf, piles, ax=None, title=""):
    """Colour hexagons by the pile they were dealt to: training, validation or test."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.2))
    piles = pd.Series(piles, index=gdf.index)
    gdf.plot(ax=ax, color=piles.map(PILES).tolist(), edgecolor="white", linewidth=0.25)
    ax.legend(handles=[Patch(fc=col, label=f"{name} ({int((piles == name).sum())})") for name, col in PILES.items()],
              loc="upper left", fontsize=9, handlelength=1.2)
    _finish_map(ax, gdf, title, places=False)
    return ax


def spin_gif(draw, out_name, turns=36, ms=140):
    """Turn a 3-D plot once around and save the frames as a GIF. draw(azim) must return a figure."""
    import io

    from PIL import Image

    frames = []
    for k in range(turns):
        fig = draw(-58 + 360 * k / turns)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, facecolor=SURFACE)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / out_name
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    return out


def frames_gif(draw, steps, out_name, ms=900, hold_last=3):
    """Draw one figure per step and join the figures into a GIF. draw(step) must return a figure.
    The last frame is repeated hold_last times, so that the end state stays on screen."""
    import io

    from PIL import Image

    frames = []
    for step in steps:
        fig = draw(step)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, facecolor=SURFACE)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    frames += [frames[-1]] * (hold_last - 1)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / out_name
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    return out


# ------------------------------------------------------------------------------------------------ export
NOTED = {}


def note(name, value, enabled=True):
    """Keep a number that a slide quotes. For the teacher (enabled), all kept numbers are written to
    figures/slide_numbers.json, from which the slide deck is built: the slides then cannot drift from the notebook."""
    NOTED[name] = value
    if enabled:
        import json

        FIGURES.mkdir(exist_ok=True)
        plain = lambda o: o.item() if hasattr(o, "item") else str(o)      # numpy numbers as plain numbers
        (FIGURES / "slide_numbers.json").write_text(json.dumps(NOTED, indent=1, ensure_ascii=False, default=plain), encoding="utf-8")


def save(fig, name, enabled=True, untitled=False):
    """Save a figure into figures/ at slide resolution (used to make the slides).
    untitled: leave the title out of the saved file, because the slide has its own; the figure on screen keeps it."""
    if not enabled:
        return None
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    if untitled:
        for ax in fig.axes:
            ax.set_title("", loc="left")
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    for ax, title in zip(fig.axes, titles):
        ax.set_title(title, loc="left")
    return path


def make_gif(png_paths, out_name, seconds=1.0, hold_last=3.0):
    """Join PNG frames into a GIF. Frames are repeated instead of relying on GIF timing,
    because some viewers (and PowerPoint versions) ignore per-frame durations."""
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in png_paths]
    w, h = frames[0].size
    frames = [f.resize((w, h)) for f in frames]
    step = 0.5
    seq = []
    for i, f in enumerate(frames):
        seq += [f] * int(round((hold_last if i == len(frames) - 1 else seconds) / step))
    out = FIGURES / out_name
    seq[0].save(out, save_all=True, append_images=seq[1:], duration=int(step * 1000), loop=0, optimize=True)
    return out


# ------------------------------------------------------------------------------------------------ for the website
CLUSTER_COLORS = ["#bb16a3", "#52514e", "#d98e00", "#3f8fb0", "#7a5195", "#9aa33a", "#c0504d", "#8c8c8c"]


def map_clusters(gdf, groups, ax=None, title=""):
    """Colour hexagons by the group a clustering put them in. The colours carry no meaning: a group has no name."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.2))
    groups = np.asarray(groups)
    gdf.plot(ax=ax, color=[CLUSTER_COLORS[g % len(CLUSTER_COLORS)] for g in groups], edgecolor="white", linewidth=0.25)
    ax.legend(handles=[Patch(fc=CLUSTER_COLORS[g % len(CLUSTER_COLORS)], label=f"group {g + 1} ({int((groups == g).sum())})") for g in np.unique(groups)],
              loc="upper left", fontsize=9, handlelength=1.2)
    _finish_map(ax, gdf, title, places=False)
    return ax


def frames(draw, steps, name, dpi=110):
    """Save one picture per step into figures/frames/<name>/, for a slider on the website. draw(step) must return a figure."""
    folder = FIGURES / "frames" / name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for k, step in enumerate(steps):
        fig = draw(step)
        fig.savefig(folder / f"{k:02d}.png", dpi=dpi, facecolor=SURFACE)
        plt.close(fig)
    return folder


def web_map(gdf, columns, name="hexagons"):
    """An interactive map of the hexagons, coloured by land use; pointing at a hexagon shows its row of the table.
    `columns` maps a column of the table to the words shown for it. Written to figures/maps/<name>.html for the website.
    Needs the package folium."""
    import folium

    shown = gdf[["landuse", "geometry"] + list(columns)].to_crs("EPSG:4326")
    centre = shown.geometry.union_all().centroid if hasattr(shown.geometry, "union_all") else shown.geometry.unary_union.centroid
    m = folium.Map(location=[centre.y, centre.x], zoom_start=10, tiles="OpenStreetMap")
    folium.GeoJson(
        shown,
        style_function=lambda f: {"fillColor": COLORS[f["properties"]["landuse"]], "color": "white", "weight": 0.5, "fillOpacity": 0.5},
        highlight_function=lambda f: {"weight": 2, "color": "#0b0b0b"},
        tooltip=folium.GeoJsonTooltip(fields=["landuse"] + list(columns), aliases=["land-use label"] + list(columns.values())),
    ).add_to(m)
    folder = FIGURES / "maps"
    folder.mkdir(parents=True, exist_ok=True)
    m.save(folder / f"{name}.html")
    return folder / f"{name}.html"
