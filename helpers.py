"""Helpers for the notebooks of "GeoAI on Helsinki hexagons": loading the hexagons, the neighbour matrix, maps and plots.

The learning steps themselves are written out in the notebooks. This file holds what would distract from them:
loading, drawing, and (for the teacher) saving pictures and numbers for the website and the slides.
"""
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")

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


SAVE = os.environ.get("GEOAI_SAVE_FIGS") == "1"     # only for the teacher: write slider frames, the web map and numbers for the slides
PAGE = os.environ.get("GEOAI_PAGE", "notebook")     # the page that is being built

# Five hexagons that the notebooks follow (lon, lat of a point inside each).
FIVE_PLACES = {"Otaniemi": (24.827, 60.186), "Tikkurila": (25.044, 60.292), "Kontula": (25.080, 60.235),
               "Seurasaari": (24.885, 60.183), "Nuuksio": (24.550, 60.300)}


# ------------------------------------------------------------------------------------------------ data
def load(res=8):
    """Everything a notebook starts from:
    hexes  the table, one row per hexagon (a GeoDataFrame in metres)
    X      the features: log(1 + count) of the 16 kinds of points of interest
    y      the land-use label of each hexagon
    W      the spatial weights matrix of week 3: W @ X is the average over the hexagons that touch each hexagon"""
    hexes = load_hexagons(res)
    return hexes, np.log1p(hexes[FEATURES].to_numpy(float)), hexes["landuse"].to_numpy(), neighbour_matrix(hexes)


def five(gdf):
    """The five hexagons that the notebooks follow: place name -> row number."""
    return pd.Series({name: hexagon_at(gdf, lon, lat) for name, (lon, lat) in FIVE_PLACES.items()})


def label_with(gdf, rule=0.20):
    """The land-use label under another rule: at least `rule` built-up land makes a hexagon residential or work,
    whichever covers more; everything else is green. The dataset uses 0.20."""
    built = gdf["share_residential"] + gdf["share_work"]
    return np.where(built < rule, "green", np.where(gdf["share_work"] >= gdf["share_residential"], "work", "residential"))


def three_piles(y, seed):
    """Deal the hexagons at random into training (60%), validation (20%) and test (20%). Returns three arrays of row numbers."""
    from sklearn.model_selection import train_test_split

    rest, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    train, valid = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=seed)
    return train, valid, test


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


def note(name, value, enabled=None):
    """Keep a number that a slide quotes. For the teacher, the numbers of a page are written to
    figures/numbers/<page>.json; build_site.py joins them into figures/slide_numbers.json."""
    NOTED[name] = value
    if SAVE if enabled is None else enabled:
        import json

        (FIGURES / "numbers").mkdir(parents=True, exist_ok=True)
        plain = lambda o: o.item() if hasattr(o, "item") else str(o)      # numpy numbers as plain numbers
        (FIGURES / "numbers" / f"{PAGE}.json").write_text(json.dumps(NOTED, indent=1, ensure_ascii=False, default=plain), encoding="utf-8")


def save(fig, name, enabled=None, untitled=False):
    """Save a figure into figures/ at slide resolution (for the teacher).
    untitled: leave the title out of the saved file, because the slide has its own; the figure on screen keeps it."""
    if not (SAVE if enabled is None else enabled):
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


def profile(gdf, row, ax=None):
    """The 16 counts of one hexagon as bars."""
    values = gdf.loc[row, FEATURES].astype(float).sort_values()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(values.index, values.values, height=0.6, color="#2a78d6")
    for name, v in values.items():
        ax.text(v + 1, name, f"{int(v)}", va="center", fontsize=9)
    ax.set_title(f"The 16 counts of one hexagon ({gdf.loc[row, 'landuse']})")
    ax.set_xlabel("number of points of interest")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    return ax


# ------------------------------------------------------------------------------------------------ page: What is learning?
# Drawing for the page "learning". helpers.py loads this file into its own namespace.


def plot_learning_curve(learning, baseline, n_held):
    """Accuracy against the number of labelled hexagons the program was given.
    learning: a table with the columns mean, min and max of the draws, one row per number of labelled hexagons.
    baseline: the accuracy of always answering 'green'.   n_held: the number of hexagons the accuracy was measured on."""
    sizes = list(learning.index)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.fill_between(learning.index, learning["min"], learning["max"], color=ACCENT, alpha=0.15, label="range of 20 draws")
    ax.plot(learning.index, learning["mean"], "o-", color=ACCENT, lw=2.5, label="mean of 20 draws")
    ax.axhline(baseline, color=MUTED, ls="--", lw=1.5)
    ax.text(sizes[-1], baseline + 0.012, "always answering 'green'", ha="right", fontsize=9, color=SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("experience E: hexagons whose land use the program was given")
    ax.set_ylabel(f"performance P: accuracy on {n_held} other hexagons")
    ax.set_title("More experience, better performance")
    ax.legend(loc="center right")
    return ax


# ------------------------------------------------------------------------------------------------ page: Unsupervised learning
# Drawing functions of the page "unsupervised". helpers.py loads this file into its own namespace.


def plot_kmeans_step(points, history, i, names=("parking and fuel in the hexagon", "playgrounds and picnic sites in the hexagon")):
    """One step of k-means on two counts: every hexagon in the colour of its group, the centres as stars.
    history is the list that kmeans_steps returns: for every step the centres and the group of every point. Returns the figure."""
    centres, group = history[i]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    P = spread(points, 0.16)
    ax.scatter(P[:, 0], P[:, 1], s=8, color=[CLUSTER_COLORS[g] for g in group], alpha=0.5, linewidths=0)
    ax.scatter(centres[:, 0], centres[:, 1], s=320, marker="*", color=[CLUSTER_COLORS[g] for g in range(len(centres))],
               edgecolor=INK, linewidths=1.5, zorder=5)
    count_ticks(ax, "x", (0, 1, 3, 10, 30, 100)), count_ticks(ax, "y", (0, 1, 3, 10, 30, 100))
    ax.set_xlabel(names[0]), ax.set_ylabel(names[1])
    changed = int((group != history[i - 1][1]).sum()) if i else None
    ax.set_title(f"step {i + 1}: every hexagon joins its nearest centre (star)" if i == 0 else
                 f"step {i + 1}: {changed} hexagons changed their group" if changed else f"step {i + 1}: no hexagon changed its group")
    fig.tight_layout()
    return fig


def map_kmeans(gdf, groups, k):
    """The groups of k-means with K = k on one map. Returns the figure."""
    fig, ax = plt.subplots(figsize=(8.6, 6))
    map_clusters(gdf, groups, ax=ax, title=f"k-means with K = {k}")
    return fig


def map_kmeans_and_labels(gdf, groups, labels):
    """Two maps side by side: the groups that k-means found, and the land-use label that it never saw. Returns the figure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    map_clusters(gdf, groups, ax=axes[0], title=f"{len(np.unique(groups))} groups found by k-means (no label was used)")
    map_classes(gdf, labels, ax=axes[1], title="The land-use label, for comparison", places=False)
    fig.tight_layout()
    return fig


def plot_elbow(wcss):
    """The elbow method: the within-cluster sum of squares (a Series, K -> value) against K. Returns the figure."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(wcss.index, wcss.values, "o-", color=ACCENT, lw=2.5)
    ax.set_xlabel("K, the number of groups")
    ax.set_ylabel("within-cluster sum of squares")
    ax.set_title("The elbow method")
    return fig


# ------------------------------------------------------------------------------------------------ page: Supervised learning
# Drawing for the page "supervised". helpers.py loads this file into its own namespace.
POI_TICKS = [0, 1, 10, 100, 1000]       # counts written on an axis of log10(1 + count)


def plot_two_tasks(gdf, labels, built_up_pct, rows):
    """Two maps side by side: the land-use label (classification) and the share of built-up land (regression).
    rows: the hexagons to outline and number on both maps."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    map_classes(gdf, labels, ax=axes[0], title="Classification: which land use?", places=False)
    map_values(gdf, built_up_pct, ax=axes[1], title="Regression: how much built-up land?", label="built-up land (%)",
               cmap=accent_cmap(), places=False)
    for ax in axes:
        number_hexagons(ax, gdf, rows)
    fig.tight_layout()
    return fig


def plot_toy_lines(x, t, choices):
    """Three choices of a and b for the line y = a + b x on a few made-up points. The vertical strokes are the errors,
    point by point; the title gives the typical error (root mean square). choices: a list of (a, b)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (a, b), colour in zip(axes, choices, [SECONDARY, "#d98e00", ACCENT]):
        error = float(np.sqrt(((a + b * x - t) ** 2).mean()))
        ax.vlines(x, t, a + b * x, color=colour, lw=1.6, alpha=0.7)
        ax.plot([0, 10], [a, a + 10 * b], color=colour, lw=3.4)
        ax.scatter(x, t, s=70, color=INK, zorder=3)
        ax.set_title(f"a = {a},  b = {b}\ntypical error {error:.0f}", color=colour, fontsize=14)
        ax.set_xlim(0, 10), ax.set_ylim(0, 105)
        ax.set_xlabel("x", fontsize=13)
    axes[0].set_ylabel("y", fontsize=13)
    fig.tight_layout()
    return fig


def _poi_axis(ax):
    count_ticks(ax, "x", POI_TICKS, lambda c: np.log10(1 + c))
    ax.set_xlabel("points of interest in the hexagon")


def plot_poi_line(poi_log, built_up_pct, rows, a=None, b=None, title="More points of interest, more built-up land"):
    """Every hexagon as a dot: points of interest across (a logarithmic axis), built-up land upwards. The given rows are
    numbered. With a and b, the line y = a + b x is drawn and its typical error is written under the title."""
    x = np.asarray(poi_log, float).ravel()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(spread(x, 0.02), built_up_pct, s=7, color=MUTED, alpha=0.5, linewidths=0)
    for n, i in enumerate(rows, 1):
        ax.scatter(x[i], built_up_pct[i], s=190, facecolor="white", edgecolor=INK, linewidths=1.6, zorder=6)
        ax.annotate(str(n), (x[i], built_up_pct[i]), ha="center", va="center", fontsize=9, fontweight="bold", zorder=7)
    _poi_axis(ax)
    ax.set_ylim(-5, 105)
    ax.set_ylabel("built-up land (%)")
    if a is not None:
        xs = np.linspace(0, x.max(), 400)
        error = float(np.sqrt(((a + b * x - built_up_pct) ** 2).mean()))
        ax.plot(xs, a + b * xs, color=ACCENT, lw=3.5)
        title = f"{title}:  a = {a:.1f},  b = {b:.1f}\ntypical error {error:.1f} points"
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_rule_maps(gdf, labels, answers):
    """Two maps side by side: the land-use label, and the answers of a fitted rule."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    map_classes(gdf, labels, ax=axes[0], title="The land-use label", places=False)
    map_classes(gdf, answers, ax=axes[1], title="The answers of the rule on two counts", places=False)
    fig.tight_layout()
    return fig


def plot_three_piles(gdf, valid, test):
    """A map of the three piles: the rows `valid` and `test`, and training for all other hexagons."""
    piles = np.full(len(gdf), "training", dtype=object)
    piles[valid], piles[test] = "validation", "test"
    _, ax = plt.subplots(figsize=(8.6, 6))
    return map_piles(gdf, piles, ax=ax, title="Three piles, dealt at random")


def _flex_panel(ax, model, poi_log, built_up_pct, few, valid, lw=3):
    """The fitted curve, the hexagons it was fitted to (black) and the validation hexagons (grey)."""
    xs = np.linspace(0, poi_log.max(), 400).reshape(-1, 1)
    ax.scatter(poi_log[valid], built_up_pct[valid], s=8, color=PILES["training"], linewidths=0, label="validation hexagons")
    ax.scatter(poi_log[few], built_up_pct[few], s=40, color=INK, zorder=3, label=f"the {len(few)} training hexagons")
    ax.plot(xs, np.clip(model.predict(xs), -40, 140), color=ACCENT, lw=lw)
    _poi_axis(ax)
    ax.set_ylim(-8, 105)


def plot_flexibility(curve, poi_log, built_up_pct, few, valid):
    """Three rules fitted to the same few hexagons: a flat line, a straight line and a curve of degree 6.
    curve(degree) must return a model that is not fitted yet."""
    words = {0: "too rigid: a flat line", 1: "in between: a straight line", 6: "too flexible: a curve"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (degree, word) in zip(axes, words.items()):
        _flex_panel(ax, curve(degree).fit(poi_log[few], built_up_pct[few]), poi_log, built_up_pct, few, valid)
        ax.set_title(word)
    axes[0].set_ylabel("built-up land (%)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_flexibility_frame(degree, curve, poi_log, built_up_pct, few, valid, medians):
    """One frame of the slider: the rule of this degree on the few hexagons, and its median R2 over many deals
    (medians: a table with the columns training and validation, one row per degree)."""
    words = {0: "too rigid", 1: "in between", 2: "in between", 3: "in between", 6: "too flexible", 9: "much too flexible"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
    _flex_panel(ax, curve(degree).fit(poi_log[few], built_up_pct[few]), poi_log, built_up_pct, few, valid, lw=3.5)
    ax.set_ylabel("built-up land (%)")
    ax.set_title(f"{words[degree]}  (may bend {max(degree - 1, 0)} times)" if degree > 1
                 else f"{words[degree]}  ({'a flat line' if degree == 0 else 'a straight line'})")
    tr, va = medians.loc[degree, "training"], medians.loc[degree, "validation"]
    bx.bar([0, 1], [max(tr, 0), max(va, 0)], color=[MUTED, ACCENT], width=0.6)
    bx.text(0, max(tr, 0) + 0.02, f"{tr:.2f}", ha="center", fontsize=12, fontweight="bold")
    bx.text(1, max(va, 0) + 0.02, f"{va:.2f}" if va > -0.1 else "below 0", ha="center", fontsize=12, fontweight="bold")
    bx.set_xticks([0, 1], [f"its own\n{len(few)} hexagons", "validation\npile"])
    bx.set_ylim(0, 0.75)
    bx.set_title("R²")
    bx.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ page: Semi-supervised learning
# Functions of the page "semi_supervised" (the first three are also used by "self_supervised").
# helpers.py loads this file into its own namespace.


def new_model(C=1.0):
    """The classifier of the pages with few labels: logistic regression, the decision boundary of the page on
    supervised learning. C = 1 is the default of scikit-learn; a smaller C gives a more cautious model."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, C=C)


def draw_labelled(y, n, seed, per_class=False):
    """Row numbers of the n hexagons whose label we are given: n drawn at random from all hexagons,
    or (per_class=True) n // 3 drawn from every class."""
    r = np.random.default_rng(seed)
    if per_class:
        return np.concatenate([r.choice(np.where(y == c)[0], n // 3, replace=False) for c in CLASSES])
    return r.choice(len(y), n, replace=False)


def hide_all_but(y, labelled):
    """The labels as a model with few labels sees them: the class name in the rows `labelled`, None in all other rows."""
    y_start = np.full(len(y), None, dtype=object)
    y_start[labelled] = y[labelled]
    return y_start


def map_two_samples(gdf, y, samples):
    """One map per sample of labelled hexagons (name -> row numbers): the known labels in colour, the hidden ones in grey.
    Returns the figure."""
    fig, axes = plt.subplots(1, len(samples), figsize=(7.5 * len(samples), 5.6))
    for ax, (name, labelled) in zip(np.atleast_1d(axes), samples.items()):
        map_classes(gdf, hide_all_but(y, labelled), ax=ax, outline=labelled, title=f"What we know at the start: {name}")
    fig.tight_layout()
    return fig


def map_self_training(gdf, y, y_start, pred_supervised, pred_self):
    """Four maps: the labels at the start, the prediction of the supervised model, the prediction after self-training,
    and the labels that were hidden. The accuracies in the titles are counted on the hidden hexagons. Returns the figure."""
    hidden = pd.isna(y_start)
    labelled = np.where(~hidden)[0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    map_classes(gdf, y_start, ax=axes[0, 0], title=f"1. Start: {len(labelled)} labels", outline=labelled)
    map_classes(gdf, pred_supervised, ax=axes[0, 1],
                title=f"2. Supervised on {len(labelled)} labels: accuracy {(pred_supervised[hidden] == y[hidden]).mean():.0%}")
    map_classes(gdf, pred_self, ax=axes[1, 0], title=f"3. After self-training: accuracy {(pred_self[hidden] == y[hidden]).mean():.0%}")
    map_classes(gdf, y, ax=axes[1, 1], title="4. The labels we had hidden")
    fig.tight_layout()
    return fig


def map_round(gdf, y, history, r):
    """One round of self-training on the map: the hexagons that carry a label after round r, the others in grey.
    history is the list that self_train returns; the hexagons labelled at the start are outlined. Returns the figure."""
    hidden = pd.isna(history[0])
    y_r = history[r]
    adopted = pd.notna(y_r) & hidden
    right = f"{(y_r[adopted] == y[adopted]).mean():.0%} of the pseudo-labels right" if adopted.any() else "only the labels we were given"
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    map_classes(gdf, y_r, ax=ax, outline=np.where(~hidden)[0],
                title=f"Round {r}: {pd.notna(y_r).sum():,} of {len(y):,} hexagons labelled · {right}")
    ax.title.set_fontsize(12)
    fig.tight_layout()
    return fig


def plot_draws(draws, true_work):
    """The repeated experiment. Left: the accuracy of the supervised and of the self-trained model in every draw, by sample
    (a line joins the two models of one draw, a bar marks the mean). Right: the share of hexagons that each model predicts
    as work, against the share on the map (true_work). draws has the columns sample, supervised, self-training,
    'work, supervised' and 'work, self-training'. Returns the figure."""
    means = draws.groupby("sample", sort=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for g, (name, part) in enumerate(draws.groupby("sample", sort=False)):
        x0, x1 = g * 3, g * 3 + 1
        for _, row in part.iterrows():
            ax.plot([x0, x1], [row["supervised"], row["self-training"]], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(np.full(len(part), x0), part["supervised"], s=42, color="#2a78d6", edgecolor="white", linewidth=1, zorder=3,
                   label="supervised" if g == 0 else None)
        ax.scatter(np.full(len(part), x1), part["self-training"], s=42, color="#eb6834", edgecolor="white", linewidth=1, zorder=3,
                   label="self-training" if g == 0 else None)
        for x, col, side in ((x0, "supervised", -1), (x1, "self-training", 1)):
            ax.hlines(part[col].mean(), x - 0.3, x + 0.3, color=INK, linewidth=2.2, zorder=4)
            ax.text(x + side * 0.38, part[col].mean(), f"{part[col].mean():.0%}", ha="right" if side < 0 else "left", va="center",
                    fontsize=11, fontweight="bold", zorder=5)
        ax.text((x0 + x1) / 2, -0.13, name, ha="center", va="top", fontsize=11, transform=ax.get_xaxis_transform())
    ticks = [x for g in range(len(means)) for x in (g * 3, g * 3 + 1)]
    ax.set_xticks(ticks, ["supervised", "self-training"] * len(means), fontsize=9)
    ax.set_xlim(-0.8, ticks[-1] + 0.8)
    ax.set_ylabel("accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.grid(True, color=GRID)
    ax.set_title(f"Accuracy in {len(draws) // len(means)} draws (lines join the same draw)")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    labels = [f"{name}\n{model}" for name in means.index for model in ("supervised", "self-training")]
    values = [means.loc[name, f"work, {model}"] for name in means.index for model in ("supervised", "self-training")]
    ax.bar(range(len(values)), values, width=0.6, color=["#2a78d6", "#eb6834"] * len(means))
    for i, v in enumerate(values):
        ax.text(i, v + 0.004, f"{v:.0%}", ha="center", fontsize=10)
    ax.axhline(true_work, color=INK, linestyle="--", linewidth=1.2)
    ax.text(len(values) - 0.55, true_work + 0.004, f"on the map: {true_work:.0%}", ha="right", fontsize=9)
    ax.set_xticks(range(len(values)), labels, fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Share of hexagons predicted as work")
    fig.tight_layout()
    return fig


def map_propagation(gdf, y, y_start, pred):
    """Two maps: the labels that label propagation gives from the hexagons labelled in y_start, and the land-use label.
    The accuracy in the title is counted on the hidden hexagons. Returns the figure."""
    hidden = pd.isna(y_start)
    labelled = np.where(~hidden)[0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    map_classes(gdf, pred, ax=axes[0], outline=labelled,
                title=f"Label propagation from {len(labelled)} labels, geography only: accuracy {(pred[hidden] == y[hidden]).mean():.0%}")
    map_classes(gdf, y, ax=axes[1], title="The land-use label")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ page: Self-supervised learning
# Drawing functions of the page "self_supervised". helpers.py loads this file into its own namespace.
# (new_model and draw_labelled, which the page also uses, are in helpers_add_semi_supervised.py.)


def map_similar(gdf, distance, top, query, name="Otaniemi"):
    """The map of a similarity search: every hexagon coloured by how similar it is to the query hexagon (as the share of
    all hexagons that are less similar), the rows `top` outlined and the query itself in orange.
    distance: the distance of every hexagon to the query in the embedding. Returns the figure."""
    closer_than = 100 * (1 - pd.Series(distance).rank(pct=True).to_numpy())     # 99 = among the 1 % most similar
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    map_values(gdf, closer_than, ax=ax, cmap="Blues", vmin=50, vmax=100,
               label="more similar to the query than ... % of all hexagons", outline=top,
               title=f"The {len(top)} hexagons most like {name} (outlined)")
    gdf.iloc[[query]].plot(ax=ax, facecolor="#eb6834", edgecolor=INK, linewidth=1.6, zorder=5)
    return fig


def plot_few_labels(results, title="Accuracy with 30 random labels (mean of 12 draws)"):
    """Bars of the mean accuracy of each description of a hexagon. results: name -> (mean, spread between draws).
    Returns the figure."""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars(ax, list(results), [m for m, s in results.values()], title=title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ page: Reinforcement learning
# ------------------------------------------------------------------------------------------------ page: reinforcement
# Drawing for the courier robot. `streets` is the street grid of the page:
# {"cols": 7, "rows": 5, "depot": (0, 0), "customer": (6, 4), "roadworks": {(2, 0), ...}}


def courier_streets(streets, route=None, title="", ax=None):
    """The street grid of the courier robot: grey squares are streets, orange squares are roadworks,
    the two dark squares are the depot and the customer. route: the places of one try, drawn as a line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    for col in range(streets["cols"]):
        for row in range(streets["rows"]):
            works = (col, row) in streets["roadworks"]
            ax.add_patch(plt.Rectangle((col - 0.46, row - 0.46), 0.92, 0.92, fc="#eb6834" if works else "#eceae4", ec="none"))
    for (col, row), word in ((streets["depot"], "depot"), (streets["customer"], "customer")):
        ax.add_patch(plt.Rectangle((col - 0.46, row - 0.46), 0.92, 0.92, fc=INK, ec="none"))
        ax.text(col, row, word, color="white", ha="center", va="center", fontsize=6.5, fontweight="bold", zorder=6)
    if route is not None:
        shift = 0.13 if len(route) > 60 else (0.05 if len(route) > 16 else 0.0)      # for drawing only: repeated moves stay visible
        pts = np.asarray(route, float) + np.random.default_rng(0).uniform(-shift, shift, (len(route), 2))
        ax.plot(pts[:, 0], pts[:, 1], color=ACCENT, lw=2.4, alpha=0.75, solid_joinstyle="round", zorder=4)
    ax.set_xlim(-0.6, streets["cols"] - 0.4), ax.set_ylim(-0.6, streets["rows"] - 0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    return ax


def courier_tries(streets, routes, totals, tries=(1, 60, 300)):
    """The routes of a few tries next to each other, each with its number of moves and its points."""
    fig, axes = plt.subplots(1, len(tries), figsize=(5 * len(tries), 3.6))
    for ax, t in zip(np.atleast_1d(axes), tries):
        courier_streets(streets, routes[t - 1], f"try {t}:  {len(routes[t - 1]) - 1} moves,  {totals[t - 1]:+d} points", ax=ax)
    plt.tight_layout()
    return fig


def courier_try(streets, routes, totals, t):
    """One frame of the slider: the route of try t on the street grid, and the points of all tries so far."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.55, 1]})
    courier_streets(streets, routes[t - 1], f"try {t}:  {len(routes[t - 1]) - 1} moves,  {totals[t - 1]:+d} points", ax=ax)
    bx.plot(np.arange(1, t + 1), totals[:t], color=MUTED, lw=1.2)
    bx.scatter([t], [totals[t - 1]], s=70, color=ACCENT, zorder=3)
    bx.axhline(7, color=INK, lw=1, ls=":")
    bx.text(len(totals), 12, "best possible: +7", ha="right", va="bottom", fontsize=9)
    bx.set_xlim(0, len(totals) + 5), bx.set_ylim(min(totals.min(), -100) - 20, 45)
    bx.set_xlabel("try")
    bx.set_title("points per try", fontsize=12)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ page: Spatially explicit models
# ------------------------------------------------------------------------------------------------ page: spatially_explicit
# (new_forest is also used by the page challenges)


def new_forest(trees=200, seed=0):
    """A random forest that is not fitted yet: `trees` decision trees, each grown on a random part of the training rows,
    which vote on the land use of a hexagon. A leaf holds at least two training rows."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, random_state=seed)


def plot_relocation(gdf, labels, occupant):
    """Two maps of the land-use label: the real map, and the map after every hexagon was moved to a random position.
    occupant[p] is the row of the hexagon that now sits at position p."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    map_classes(gdf, labels, ax=axes[0], title="The real map")
    map_classes(gdf, labels[occupant], ax=axes[1], places=False, title="After relocation: same hexagons, random positions")
    plt.tight_layout()
    return fig


def plot_invariance(result):
    """The invariance test as bars, three per model. `result` has one row per model and four columns in this order:
    share of predictions that changed, accuracy on the real map, accuracy of the same model on the relocated map,
    accuracy of a model retrained on the relocated map."""
    names = list(result.index)
    changed, acc_real, acc_same, acc_new = (result.iloc[:, k].tolist() for k in range(4))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    pos = np.arange(len(names))[::-1] * 1.25
    ax.barh(pos + 0.3, acc_real, height=0.27, color="#2a78d6", label="trained and tested on the real map")
    ax.barh(pos, acc_same, height=0.27, color="#eb6834", label="same model, relocated map")
    ax.barh(pos - 0.3, acc_new, height=0.27, color="#1baf7a", label="retrained on the relocated map")
    for p, a, b, c, ch in zip(pos, acc_real, acc_same, acc_new, changed):
        ax.text(a + 0.01, p + 0.3, f"{a:.0%}", va="center", fontsize=10)
        ax.text(b + 0.01, p, f"{b:.0%}   ({ch:.0%} of predictions changed)", va="center", fontsize=10)
        ax.text(c + 0.01, p - 0.3, f"{c:.0%}", va="center", fontsize=10)
    ax.set_yticks(pos, names)
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("accuracy on the test hexagons")
    ax.set_title("The invariance test")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_unit_size(by_size):
    """What the spatial arrangement is worth at each hexagon size, in percentage points of accuracy.
    `by_size` has one row per H3 resolution, with the columns "hexagons" and "worth of the arrangement (points)"."""
    worth = by_size["worth of the arrangement (points)"]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    bars(ax, [f"{name} ({int(n):,} hexagons)" for name, n in by_size["hexagons"].items()], worth.tolist(),
         title="What the spatial arrangement is worth, by hexagon size", xlabel="percentage points of accuracy",
         xlim=(min(0, worth.min()) - 0.5, max(1, worth.max()) * 1.3), fmt="{:+.1f}")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------ page: Challenges
# ------------------------------------------------------------------------------------------------ page: challenges
# (the forest of this page, new_forest, is defined in helpers_add_spatially_explicit.py)


def plot_folds(gdf, random_test, spatial_test):
    """Two maps: the hexagons of the first test fold of a random split and of a spatial split.
    random_test, spatial_test: for every hexagon, the number of the fold in which it is tested."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    map_groups(gdf, random_test, ax=axes[0], highlight=0, title="Random split: one of five test folds")
    map_groups(gdf, spatial_test, ax=axes[1], highlight=0, title="Spatial split: one of five test folds")
    plt.tight_layout()
    return fig


def folds_gif(gdf, random_test, spatial_test, out_name="4_folds_cycle.gif"):
    """The animation of the page (for the teacher): the five test folds of both splits, one after the other."""
    pictures = []
    for k in range(5):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
        map_groups(gdf, random_test, ax=axes[0], highlight=k, title=f"Random split: test fold {k + 1} of 5")
        map_groups(gdf, spatial_test, ax=axes[1], highlight=k, title=f"Spatial split: test fold {k + 1} of 5")
        plt.tight_layout()
        pictures.append(save(fig, f"steps/4_folds_{k + 1}.png", True))
        plt.close(fig)
    return make_gif(pictures, out_name, seconds=1.5, hold_last=1.5)


def plot_two_splits(cv, baseline):
    """Accuracy of every model under the two ways of testing, as pairs of bars.
    cv: one row per model, with the columns "random split" and "spatial split". baseline: the share of the largest class."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    pos = np.arange(len(cv))[::-1]
    ax.barh(pos + 0.2, cv["random split"], height=0.36, color="#2a78d6", label="random split")
    ax.barh(pos - 0.2, cv["spatial split"], height=0.36, color="#eb6834", label="spatial split")
    for p, (r, s) in zip(pos, zip(cv["random split"], cv["spatial split"])):
        ax.text(r + 0.01, p + 0.2, f"{r:.0%}", va="center", fontsize=10)
        ax.text(s + 0.01, p - 0.2, f"{s:.0%}", va="center", fontsize=10)
    ax.axvline(baseline, color=MUTED, linestyle="--", linewidth=1)
    ax.text(baseline, len(cv) - 0.42, f" always guess 'green': {baseline:.0%}", color=MUTED, fontsize=9, va="bottom")
    ax.set_yticks(pos, cv.index)
    ax.set_ylim(-0.6, len(cv) - 0.1)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("accuracy on the held-out hexagons")
    ax.set_title("Same models, same data, two ways of testing")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_layouts(runs, summary):
    """One dot per repeat of the two splits (another seed, another layout of the blocks), and by how much the random split
    overstates each model on average. runs: the columns "features", "split", "accuracy". summary: mean and std per model and split."""
    gap = (summary[("mean", "random split")] - summary[("mean", "spatial split")]) * 100
    order = list(dict.fromkeys(runs["features"]))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, name in enumerate(order[::-1]):
        for split, color, dy in (("random split", "#2a78d6", 0.14), ("spatial split", "#eb6834", -0.14)):
            v = runs[(runs["features"] == name) & (runs["split"] == split)]["accuracy"]
            ax.scatter(v, np.full(len(v), i + dy), s=46, color=color, edgecolor="white", linewidth=1, zorder=3, label=split if i == 0 else None)
        points = round(gap[name])
        ax.text(1.0, i, f"overstated by {points} point{'s' if points != 1 else ''}", va="center", ha="right", fontsize=10)
    ax.set_yticks(range(len(order)), order[::-1])
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("cross-validated accuracy (each dot is one seed or one block layout)")
    ax.set_title("The more a model can use location, the more a random split overstates it")
    ax.xaxis.grid(True, color=GRID)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=10)
    plt.tight_layout()
    plt.show()
