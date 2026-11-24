"""Build the pages of the website from the text sources in src/.

The pages of the site are one notebook, cut into pages. This script
  1. joins src/NN_slug.txt into one notebook and runs it once (about four minutes),
  2. writes the whole notebook to notebooks/GeoAI_Helsinki_hexagons.ipynb,
  3. cuts it into docs/<slug>.ipynb, one page each, with the outputs stored,
  4. copies the pictures, slider frames and the interactive map into docs/_static/.
Read the Docs then shows the stored outputs; it runs no code.

Run:   python build_site.py           everything
       python build_site.py text      after an edit of text only: keep the stored outputs (stops if code changed)
Then:  .venv/Scripts/python -m sphinx -b html docs docs/_build/html        (to look at the site on this computer)

Source format: a line '@@page slug' starts a page; '@@md' starts a text cell; '@@code' starts a code cell that is
folded on the website, '@@code show' one that is open, '@@code hidden' one that is left out of the website,
'@@code slider' one whose picture is left out of the website because a slider next to it shows the same pictures.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).parent
SRC, DOCS, FIG = ROOT / "src", ROOT / "docs", ROOT / "figures"
FULL = ROOT / "notebooks" / "GeoAI_Helsinki_hexagons.ipynb"
TAGS = {"code": ["hide-input"], "code show": [], "code hidden": ["remove-cell"], "code slider": ["hide-input", "remove-output"]}

cells, pages = [], []
for path in sorted(SRC.glob("*.txt")):
    page, kind, buf = None, None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        cell = new_markdown_cell(text) if kind == "md" else new_code_cell(text)
        cell.metadata["page"] = page
        if kind != "md":
            cell.metadata["tags"] = list(TAGS[kind])
        cells.append(cell)

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("@@page "):
            flush()
            page, kind, buf = s.split()[1], None, []
            pages.append(page)
        elif s == "@@md" or s in ("@@" + k for k in TAGS):
            flush()
            kind, buf = s[2:], []
        else:
            buf.append(line)
    flush()

meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nb = new_notebook(cells=cells, metadata=meta)
for k, cell in enumerate(nb.cells):
    cell["id"] = f"cell-{k:03d}"

mode = sys.argv[1] if len(sys.argv) > 1 else "run"
if mode == "text":
    old = nbformat.read(FULL, as_version=4)
    old_code = [c for c in old.cells if c.cell_type == "code"]
    new_code = [c for c in nb.cells if c.cell_type == "code"]
    if [c.source for c in old_code] != [c.source for c in new_code]:
        sys.exit("A code cell changed: run 'python build_site.py' to execute the notebook again.")
    for a, b in zip(new_code, old_code):
        a["outputs"], a["execution_count"] = b["outputs"], b["execution_count"]
    print(f"kept the stored outputs of {len(new_code)} code cells")
else:
    from nbclient import NotebookClient

    os.environ["GEOAI_SAVE_FIGS"] = "1"                # the notebook then writes the pictures, the slider frames and the interactive map
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    finally:
        FULL.parent.mkdir(exist_ok=True)
        nbformat.write(nb, FULL)
    print(f"executed in {time.time() - t0:.0f} s")
FULL.parent.mkdir(exist_ok=True)
nbformat.write(nb, FULL)

# ------------------------------------------------------------------------------------------------ one notebook per page
for page in pages:
    part = new_notebook(cells=[c for c in nb.cells if c.metadata.get("page") == page], metadata=meta)
    nbformat.write(part, DOCS / f"{page}.ipynb")
print("pages:", ", ".join(pages))

# ------------------------------------------------------------------------------------------------ static files
static = DOCS / "_static"
frames = {}
src_frames = FIG / "frames"
if (FIG / "steps").exists():                         # the rounds of self-training are saved as single pictures; they become one more slider
    target = src_frames / "selftraining"
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.png"):
        old.unlink()
    for k, f in enumerate(sorted((FIG / "steps").glob("1_selftraining_round_*.png"))):
        shutil.copy2(f, target / f"{k:02d}.png")
if src_frames.exists():
    shutil.rmtree(static / "frames", ignore_errors=True)
    shutil.copytree(src_frames, static / "frames")
    frames = {d.name: len(list(d.glob("*.png"))) for d in sorted(src_frames.iterdir()) if d.is_dir()}
if (FIG / "maps").exists():
    shutil.rmtree(static / "maps", ignore_errors=True)
    shutil.copytree(FIG / "maps", static / "maps")
shutil.rmtree(static / "figures", ignore_errors=True)
(static / "figures").mkdir(parents=True, exist_ok=True)
page_text = " ".join(c.source for c in nb.cells if c.cell_type == "markdown")
for f in FIG.glob("*.gif"):                          # only the animations that a page shows
    if f"_static/figures/{f.name}" in page_text:
        shutil.copy2(f, static / "figures" / f.name)
numbers = json.loads((FIG / "slide_numbers.json").read_text(encoding="utf-8")) if (FIG / "slide_numbers.json").exists() else {}
(static / "site_data.js").write_text(
    "// written by build_site.py\nwindow.GEOAI = " + json.dumps({"frames": frames, "toy": numbers.get("toy_points", {})}) + ";\n", encoding="utf-8")
shutil.copy2(FULL, static / FULL.name)                # the whole notebook, for the download link on the start page
print("sliders:", frames)
