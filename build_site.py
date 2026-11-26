"""Build the pages of the website from the text sources in src/.

Every page is one notebook that runs by itself: it loads the data in its first cell and uses nothing from another page.
This script
  1. reads src/NN_slug.txt and makes one notebook per page,
  2. runs each notebook in its own kernel (several at a time),
  3. writes docs/<slug>.ipynb with the outputs stored, for the website,
  4. writes docs/_static/notebooks/<slug>.ipynb without outputs and without the hidden cells, for download,
     and docs/_static/geoai_hexagons.zip with the notebooks, helpers.py, the data and requirements.txt,
  5. copies the slider frames, the animations and the interactive map into docs/_static/.
Read the Docs then shows the stored outputs; it runs no code.

Run:   python build_site.py                    every page
       python build_site.py data supervised    only these pages (the others keep what they have)
       python build_site.py text               after an edit of text only: keep the stored outputs (stops if code changed)
Then:  python -m sphinx -b html docs docs/_build/html        (to look at the site on this computer)

Source format: a line '@@page slug' starts a page; '@@md' starts a text cell; '@@md site' a text cell that only the website
shows, not the notebook for download (the lecture slides); '@@code' starts a code cell that is shown
on the website; '@@code fold' one that is folded; '@@code hidden' one that is left out of the website and of the
notebook for download (only for the teacher's plumbing: slider frames, the interactive map, numbers for the slides).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).parent
SRC, DOCS, FIG = ROOT / "src", ROOT / "docs", ROOT / "figures"
STATIC = DOCS / "_static"
TAGS = {"code": [], "code show": [], "code fold": ["hide-input"], "code hidden": ["remove-cell"], "code slider": ["remove-cell"]}
META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
DOWNLOAD = ('\n\n<p class="nb-download">Run it yourself. First time: <a href="_static/geoai_hexagons.zip" download>the zip with all notebooks, '
            'the helpers and the data</a>. Already have that folder: <a href="_static/notebooks/{slug}.ipynb" download>this notebook</a>.</p>')


def read_sources():
    """{slug: notebook} in the order of the files."""
    pages = {}
    for path in sorted(SRC.glob("*.txt")):
        slug, kind, buf, cells = None, None, [], []

        def flush():
            text = "\n".join(buf).strip("\n")
            if kind is None or not text:
                return
            if kind.startswith("md"):
                cell = new_markdown_cell(text)
                if kind == "md site":                     # shown on the website only: the lecture slides
                    cell.metadata["tags"] = ["site-only"]
            else:
                cell = new_code_cell(text)
                cell.metadata["tags"] = list(TAGS[kind])
            cells.append(cell)

        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("@@page "):
                slug = s.split()[1]
            elif s in ("@@md", "@@md site") or s in ("@@" + k for k in TAGS):
                flush()
                kind, buf = s[2:], []
            else:
                buf.append(line)
        flush()
        for k, cell in enumerate(cells):
            cell["id"] = f"{slug}-{k:03d}"
        pages[slug] = new_notebook(cells=cells, metadata=META)
    return pages


def code_of(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def run_one(slug):
    """Run one page in its own kernel and store it with its outputs. Called as:  python build_site.py --one slug"""
    from nbclient import NotebookClient

    nb = read_sources()[slug]
    os.environ["GEOAI_SAVE_FIGS"] = "1"          # the hidden cells then write the slider frames, the map and the numbers
    os.environ["GEOAI_PAGE"] = slug
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    finally:
        nbformat.write(nb, DOCS / f"{slug}.ipynb")


def main(args):
    pages = read_sources()
    if args and args[0] == "--one":
        return run_one(args[1])
    if args == ["text"]:
        changed = []
        for slug, nb in pages.items():
            old = nbformat.read(DOCS / f"{slug}.ipynb", as_version=4) if (DOCS / f"{slug}.ipynb").exists() else None
            if old is None or code_of(old) != code_of(nb):
                changed.append(slug)
                continue
            for a, b in zip([c for c in nb.cells if c.cell_type == "code"], [c for c in old.cells if c.cell_type == "code"]):
                a["outputs"], a["execution_count"] = b["outputs"], b["execution_count"]
            nbformat.write(nb, DOCS / f"{slug}.ipynb")
        if changed:
            sys.exit("Code changed on: " + ", ".join(changed) + ". Run:  python build_site.py " + " ".join(changed))
        print("kept the stored outputs of every page")
    else:
        todo = args or list(pages)
        unknown = [s for s in todo if s not in pages]
        if unknown:
            sys.exit("No such page: " + ", ".join(unknown))
        t0, running, failed = time.time(), {}, []
        queue = list(todo)
        while queue or running:
            while queue and len(running) < 4:
                slug = queue.pop(0)
                log = open(ROOT / f"build_{slug}.log", "w", encoding="utf-8")
                running[slug] = (subprocess.Popen([sys.executable, str(Path(__file__)), "--one", slug], stdout=log, stderr=subprocess.STDOUT), log)
            for slug, (proc, log) in list(running.items()):
                if proc.poll() is not None:
                    log.close()
                    del running[slug]
                    if proc.returncode:
                        failed.append(slug)
                        print(f"  FAILED {slug}: see build_{slug}.log")
                    else:
                        (ROOT / f"build_{slug}.log").unlink()
                        print(f"  ran {slug}  ({time.time() - t0:.0f} s)")
            time.sleep(0.5)
        if failed:
            sys.exit("Pages that did not run: " + ", ".join(failed))

    # ---------------------------------------------------------------------------------- what the website shows
    pages = read_sources()
    for slug in pages:                                   # the stored page, with the link to its notebook under the title
        nb = nbformat.read(DOCS / f"{slug}.ipynb", as_version=4)
        first = next(c for c in nb.cells if c.cell_type == "markdown")
        if "nb-download" not in first.source:
            head, _, rest = first.source.partition("\n")
            first.source = head + DOWNLOAD.format(slug=slug) + ("\n" + rest if rest else "")
        nbformat.write(nb, DOCS / f"{slug}.ipynb")
    for old in DOCS.glob("*.ipynb"):                     # a page that no source describes any more
        if old.stem not in pages:
            old.unlink()

    # ---------------------------------------------------------------------------------- what a student downloads
    out = STATIC / "notebooks"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for slug, nb in pages.items():
        clean = new_notebook(cells=[c for c in nb.cells if not {"remove-cell", "site-only"} & set(c.metadata.get("tags", []))], metadata=META)
        for c in clean.cells:
            c.metadata.pop("tags", None)
        nbformat.write(clean, out / f"{slug}.ipynb")
    with zipfile.ZipFile(STATIC / "geoai_hexagons.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob("*.ipynb")):
            z.write(f, f"geoai_hexagons/{f.name}")
        for f in (ROOT / "helpers.py", ROOT / "requirements.txt", ROOT / "data" / "helsinki_hex_r8.gpkg", ROOT / "data" / "helsinki_hex_r9.gpkg"):
            z.write(f, "geoai_hexagons/" + str(f.relative_to(ROOT)).replace("\\", "/"))
    old_whole = STATIC / "GeoAI_Helsinki_hexagons.ipynb"
    if old_whole.exists():
        old_whole.unlink()

    # ---------------------------------------------------------------------------------- sliders, animations, the map
    frames = {}
    if (FIG / "frames").exists():
        shutil.rmtree(STATIC / "frames", ignore_errors=True)
        shutil.copytree(FIG / "frames", STATIC / "frames")
        frames = {d.name: len(list(d.glob("*.png"))) for d in sorted((FIG / "frames").iterdir()) if d.is_dir()}
    if (FIG / "maps").exists():
        shutil.rmtree(STATIC / "maps", ignore_errors=True)
        shutil.copytree(FIG / "maps", STATIC / "maps")
    shutil.rmtree(STATIC / "figures", ignore_errors=True)
    (STATIC / "figures").mkdir(parents=True, exist_ok=True)
    page_text = " ".join(c.source for nb in pages.values() for c in nb.cells if c.cell_type == "markdown")
    for f in FIG.glob("*.gif"):                          # only the animations that a page shows
        if f"_static/figures/{f.name}" in page_text:
            shutil.copy2(f, STATIC / "figures" / f.name)
    numbers = {}
    for slug in pages:                                   # the numbers that the hidden cells kept, page by page
        part = FIG / "numbers" / f"{slug}.json"
        if part.exists():
            numbers.update(json.loads(part.read_text(encoding="utf-8")))
    (FIG / "slide_numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    (STATIC / "site_data.js").write_text(
        "// written by build_site.py\nwindow.GEOAI = " + json.dumps({"frames": frames, "toy": numbers.get("toy_points", {})}) + ";\n", encoding="utf-8")
    print("pages:", ", ".join(pages))
    print("sliders:", frames)


if __name__ == "__main__":
    main(sys.argv[1:])
