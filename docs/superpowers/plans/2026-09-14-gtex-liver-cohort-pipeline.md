# GTEx Liver Cohort and Centroid Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild this repository as a self-contained pipeline that fetches public GTEx histology metadata, labels a 610-slide liver cohort to CSV, extracts cell centroids from HistoPlus GeoJSON files, and consolidates them into a single AnnData object.

**Architecture:** A `src/gtex_meta` package with one module per stage — `portal` (public API to slide table), `cohort` (slide table to labeled CSV), `centroids` (HistoPlus GeoJSON to per-slide npz), `consolidate` (npz directory to one `.h5ad`). Each stage is a standalone `python -m` entry point reading and writing explicit paths, so stages can run on a laptop or as SLURM jobs. Two sbatch wrappers live in `sbatch/`.

**Tech Stack:** Python 3.14, uv 0.9.10, pandas, numpy, scipy, anndata, geopandas + pyogrio (no fiona), shapely, tqdm, standard-library unittest.

**Spec:** `docs/superpowers/specs/2026-09-14-gtex-liver-cohort-design.md`

## Global Constraints

- Python `>=3.14` per `.python-version`; do not change it.
- Dependencies limited to: `anndata`, `geopandas`, `numpy`, `pandas`, `scipy`, `shapely`, `tqdm`. Do not add others.
- Tests use the standard library `unittest` only; run with `uv run --no-sync python -m unittest discover -s tests -v`.
- Modules are invoked as `uv run --no-sync python -m gtex_meta.<stage>` from the repository root.
- Never import from the LISC `tissuegeometry` repository. This repository is self-contained.
- HistoPlus GeoJSON files are a given input. Missing files produce warnings, never hard failures.
- HistoPlus files are gzipped GeoJSON and must be read with a GDAL `/vsigzip/` path prefix. Verified: `gpd.read_file(path)` raises `DataSourceError`; `gpd.read_file(f"/vsigzip/{path}")` works in 0.9 s for 35,076 features.
- Centroid coordinates are stored as float32. Verified against the LISC output for `GTEX-1JJ6O-0826`: identical class sequence, maximum coordinate difference 0.000973 px.
- `prob` is one scalar per cell (confidence in the assigned class). It is stored, never filtered by default.
- The consolidated AnnData uses: `X` = CSR one-hot float32, `var` = sorted union of class names, `obs` carries `cell_type`, `obsm["spatial"]` = float32 coords, `RangeIndex`, gzip compression.
- Every command in this plan runs from `/Users/simon/Desktop/gtex_meta`.

---

### Task 1: Package scaffolding and dependency trim

**Files:**
- Modify: `pyproject.toml` (full replacement)
- Modify: `.gitignore` (full replacement)
- Create: `src/gtex_meta/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `gtex_meta` with `gtex_meta.__version__ == "0.1.0"`; an editable install in `.venv` so `python -m gtex_meta.*` resolves

- [ ] **Step 1: Write the failing test**

Create `tests/test_package.py`:

```python
import unittest


class PackageTests(unittest.TestCase):
    def test_package_imports_and_exposes_version(self):
        import gtex_meta

        self.assertEqual(gtex_meta.__version__, "0.1.0")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-sync python -m unittest tests.test_package -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gtex_meta'`.

- [ ] **Step 3: Replace `pyproject.toml`**

```toml
[project]
name = "gtex-meta"
version = "0.1.0"
description = "Self-contained GTEx liver cohort and cell-centroid pipeline"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "anndata>=0.13.2",
    "geopandas>=1.1.4",
    "numpy>=2.4.6",
    "pandas>=3.0.5",
    "scipy>=1.18.0",
    "shapely>=2.1.2",
    "tqdm>=4.70.0",
]

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[dependency-groups]
dev = ["ruff>=0.15.19"]

[tool.ruff]
target-version = "py314"
line-length = 100
```

- [ ] **Step 4: Create the package and replace `.gitignore`**

Create `src/gtex_meta/__init__.py`:

```python
"""Self-contained GTEx liver cohort and cell-centroid pipeline."""

__version__ = "0.1.0"
```

Replace `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[oc]
build/
dist/
*.egg-info

# Environments
.venv/

# Editor / agent scratch
.superpowers/
__marimo__/

# Pipeline outputs (regenerable, large, or machine-specific)
data/gtex_portal_slides.csv
data/centroids/
*.npz
*.h5ad
```

- [ ] **Step 5: Install the package**

Run: `uv sync`

Expected: resolves and installs successfully. The trimmed dependency list drops `marimo`, `matplotlib`, `seaborn`, `pydicom`, `idc-index`, and `lazyslide`.

- [ ] **Step 6: Remove the legacy modules that the trim breaks**

`liver.py` imports `matplotlib` and `seaborn`, and both legacy test modules import it, so they cannot survive the dependency trim. They are superseded by this pipeline and are removed now rather than in Task 8.

```bash
git rm liver.py tests/test_liver.py tests/test_import_safety.py
```

- [ ] **Step 7: Run the tests to verify the baseline is clean**

Run: `uv run --no-sync python -m unittest discover -s tests -v`

Expected: `OK` (1 test).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/gtex_meta/__init__.py tests/test_package.py
git commit -m "chore: scaffold gtex_meta package, trim dependencies, drop legacy modules"
```

---

### Task 2: `portal.py` — fetch the public slide table

**Files:**
- Create: `src/gtex_meta/portal.py`
- Test: `tests/test_portal.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `COLUMNS: list[str]` — the nine output column names
  - `fetch_page(page: int, per_page: int = 1000, retries: int = 3) -> dict`
  - `iter_slides(per_page: int = 1000)` — yields API records across pages
  - `to_row(record: dict) -> list[str]` — one API record to nine CSV fields
  - `fetch_slides(tissue: str | None) -> list[list[str]]`
  - `write_csv(rows: list[list[str]], path: str) -> None`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portal.py`:

```python
import csv
import os
import tempfile
import unittest
from unittest import mock

from gtex_meta import portal


def record(slide_id, tissue="Liver", hidden=False, notes=None, categories=None):
    return {
        "histologyImageId": slide_id,
        "tissueSiteDetail": tissue,
        "subjectId": "-".join(slide_id.split("-")[:2]),
        "sex": "male",
        "ageBracket": "60-69",
        "hardyScale": "Slow death",
        "pathologyNotes": notes,
        "pathologyNotesCategories": categories or {},
        "hide": hidden,
    }


class ToRowTests(unittest.TestCase):
    def test_maps_fields_and_joins_true_categories(self):
        row = portal.to_row(
            record(
                "GTEX-AAAA-0126",
                notes="2 pieces; congestion",
                categories={"congestion": True, "steatosis": False},
            )
        )
        self.assertEqual(
            row,
            [
                "GTEX-AAAA-0126",
                "Liver",
                "GTEX-AAAA",
                "male",
                "60-69",
                "Slow death",
                "congestion",
                "2 pieces; congestion",
                "false",
            ],
        )

    def test_marks_hidden_slides(self):
        row = portal.to_row(record("GTEX-AAAA-0126", hidden=True))
        self.assertEqual(row[-1], "true")

    def test_missing_note_and_categories_become_empty_strings(self):
        row = portal.to_row(record("GTEX-AAAA-0126"))
        self.assertEqual(row[6], "")
        self.assertEqual(row[7], "")


class PaginationTests(unittest.TestCase):
    def test_iter_slides_follows_paging_info(self):
        pages = [
            {
                "data": [record("GTEX-AAAA-0126"), record("GTEX-AAA1-0126")],
                "paging_info": {"numberOfPages": 2, "page": 0},
            },
            {
                "data": [record("GTEX-AAA2-0126")],
                "paging_info": {"numberOfPages": 2, "page": 1},
            },
        ]
        with mock.patch.object(portal, "fetch_page", side_effect=pages) as fake:
            slides = list(portal.iter_slides(per_page=2))
        self.assertEqual(len(slides), 3)
        self.assertEqual([call.args[0] for call in fake.call_args_list], [0, 1])

    def test_fetch_slides_filters_by_tissue(self):
        pages = [
            {
                "data": [record("GTEX-AAAA-0126"), record("GTEX-AAAB-0126", tissue="Skin")],
                "paging_info": {"numberOfPages": 1, "page": 0},
            }
        ]
        with mock.patch.object(portal, "fetch_page", side_effect=pages):
            rows = portal.fetch_slides("Liver")
        self.assertEqual([row[0] for row in rows], ["GTEX-AAAA-0126"])


class WriteCsvTests(unittest.TestCase):
    def test_writes_header_and_quotes_every_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "slides.csv")
            portal.write_csv([portal.to_row(record("GTEX-AAAA-0126"))], path)
            with open(path, newline="") as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines[0], ",".join(f'"{name}"' for name in portal.COLUMNS))
        self.assertTrue(lines[1].startswith('"GTEX-AAAA-0126"'))
        self.assertEqual(len(lines), 2)

    def test_main_writes_requested_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "liver.csv")
            rows = [portal.to_row(record("GTEX-AAAA-0126"))]
            with mock.patch.object(portal, "fetch_slides", return_value=rows):
                exit_code = portal.main(["--tissue", "Liver", "--out", path])
            with open(path, newline="") as handle:
                written = list(csv.reader(handle))
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(written), 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync python -m unittest tests.test_portal -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gtex_meta.portal'`.

- [ ] **Step 3: Write the implementation**

Create `src/gtex_meta/portal.py`:

```python
"""Fetch the public GTEx Portal histology slide table.

The Portal histology API is the public source behind the lab's
``GTEx Portal.csv``: 25,713 slides across 40+ tissues, 610 of them liver.
"""

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://gtexportal.org/api/v2/histology/image"
PAGE_SIZE = 1000

COLUMNS = [
    "Tissue Sample ID",
    "Tissue",
    "Subject ID",
    "Sex",
    "Age Bracket",
    "Hardy Scale",
    "Pathology Categories",
    "Pathology Notes",
    "Hidden",
]


def fetch_page(page: int, per_page: int = PAGE_SIZE, retries: int = 3) -> dict:
    """Fetch one page of histology slides, retrying transient failures."""
    query = urllib.parse.urlencode({"page": page, "itemsPerPage": per_page})
    url = f"{API}?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def iter_slides(per_page: int = PAGE_SIZE):
    """Yield every slide record, following the API pagination."""
    page = 0
    while True:
        payload = fetch_page(page, per_page)
        yield from payload["data"]
        page += 1
        if page >= payload["paging_info"]["numberOfPages"]:
            return


def to_row(record: dict) -> list[str]:
    """Convert one API record into the nine output CSV fields."""
    categories = record.get("pathologyNotesCategories") or {}
    return [
        record.get("histologyImageId") or "",
        record.get("tissueSiteDetail") or "",
        record.get("subjectId") or "",
        record.get("sex") or "",
        record.get("ageBracket") or "",
        record.get("hardyScale") or "",
        ", ".join(name for name, flag in categories.items() if flag),
        record.get("pathologyNotes") or "",
        "true" if record.get("hide") else "false",
    ]


def fetch_slides(tissue: str | None = None) -> list[list[str]]:
    """Fetch all slides, optionally restricted to one tissue."""
    rows = []
    for record in iter_slides():
        if tissue and record.get("tissueSiteDetail") != tissue:
            continue
        rows.append(to_row(record))
    return rows


def write_csv(rows: list[list[str]], path: str) -> None:
    """Write rows using the Portal column order, quoting every field."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--tissue",
        default=None,
        help="Only fetch this tissue, e.g. 'Liver' (default: all slides)",
    )
    args = parser.parse_args(argv)

    rows = fetch_slides(args.tissue)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} slides to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync python -m unittest tests.test_portal -v`

Expected: `OK` (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gtex_meta/portal.py tests/test_portal.py
git commit -m "feat: fetch GTEx Portal histology slide table"
```

---

### Task 3: `cohort.py` — label the liver cohort and write the tracked CSVs

**Files:**
- Create: `src/gtex_meta/cohort.py`
- Test: `tests/test_cohort.py`
- Create (generated, tracked): `data/liver_slides.csv`, `data/liver_cohort.csv`

**Interfaces:**
- Consumes: the slide table CSV written by `gtex_meta.portal`
- Produces:
  - `CLEAN_CATEGORIES: set[str]`
  - `split_categories(value) -> set[str]`
  - `has_finding(note: str) -> bool`
  - `label_row(row: pandas.Series) -> tuple[str, str]`
  - `build_cohort(slides: pandas.DataFrame) -> pandas.DataFrame` — adds `label`, `label_reason`, `quality_flag`, `has_finding_text`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cohort.py`:

```python
import unittest

import pandas as pd

from gtex_meta import cohort


def slide(slide_id="GTEX-AAAA-0126", categories="", notes=""):
    return pd.Series(
        {
            "Tissue Sample ID": slide_id,
            "Tissue": "Liver",
            "Pathology Categories": categories,
            "Pathology Notes": notes,
        }
    )


class SplitCategoriesTests(unittest.TestCase):
    def test_splits_on_commas_and_ignores_blanks(self):
        self.assertEqual(
            cohort.split_categories("fibrosis, inflammation"), {"fibrosis", "inflammation"}
        )
        self.assertEqual(cohort.split_categories(""), set())


class HasFindingTests(unittest.TestCase):
    def test_detects_positive_finding(self):
        self.assertTrue(cohort.has_finding("2 pieces; <10% macrovesicular fat"))

    def test_ignores_negated_findings(self):
        self.assertFalse(cohort.has_finding("2 pieces; not congested; no fat"))
        self.assertFalse(cohort.has_finding("2 pieces, no significant steatosis/congestion"))

    def test_detects_findings_that_are_easy_to_miss(self):
        self.assertTrue(cohort.has_finding("several bile duct hamartomas occupy 5%"))
        self.assertTrue(cohort.has_finding("Mild central hypoxia"))
        self.assertTrue(cohort.has_finding("central sinusoidal dilatation; focal fragmentation"))


class LabelRowTests(unittest.TestCase):
    def test_curated_cirrhosis(self):
        label, reason = cohort.label_row(
            slide(categories="cirrhosis, steatosis", notes="cirrhosis")
        )
        self.assertEqual(label, "cirrhosis")
        self.assertEqual(reason, "curated cirrhosis")

    def test_curated_clean(self):
        label, reason = cohort.label_row(
            slide(categories="no_abnormalities", notes="no abnormalities")
        )
        self.assertEqual(label, "healthy")
        self.assertEqual(reason, "curated no_abnormalities")

    def test_uncategorized_note_without_finding_is_healthy(self):
        label, reason = cohort.label_row(slide(notes="2 pieces; not congested; no fat"))
        self.assertEqual(label, "healthy")
        self.assertIn("no finding", reason)

    def test_uncategorized_note_with_finding_is_other(self):
        label, reason = cohort.label_row(slide(notes="2 pieces; ~ 5% macrovesicular fat."))
        self.assertEqual(label, "other")
        self.assertIn("finding", reason)

    def test_empty_note_is_unknown_not_healthy(self):
        label, reason = cohort.label_row(slide(notes=""))
        self.assertEqual(label, "other")
        self.assertIn("unknown", reason)

    def test_autolyzed_slide_is_other(self):
        for note in (
            "2 alquots, ~8x7mm, badly autolyzed",
            "2 pieces ~7x5mm. No steatosis, moderately autolyzed",
            "2 pieces; focally moderate autolysis",
        ):
            label, reason = cohort.label_row(slide(notes=note))
            self.assertEqual(label, "other", note)
            self.assertIn("preservation", reason)


class BuildCohortTests(unittest.TestCase):
    def test_adds_label_columns(self):
        slides = pd.DataFrame(
            [
                {**slide(categories="cirrhosis", notes="cirrhosis").to_dict(),
                 "Tissue Sample ID": "GTEX-A-0126"},
                {**slide(notes="2 pieces").to_dict(),
                 "Tissue Sample ID": "GTEX-B-0126"},
            ]
        )
        result = cohort.build_cohort(slides)
        self.assertEqual(list(result["label"]), ["cirrhosis", "healthy"])
        for column in ("label_reason", "quality_flag", "has_finding_text"):
            self.assertIn(column, result.columns)

    def test_rejects_non_liver_rows(self):
        slides = pd.DataFrame([{**slide().to_dict(), "Tissue": "Skin"}])
        with self.assertRaises(ValueError):
            cohort.build_cohort(slides)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync python -m unittest tests.test_cohort -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gtex_meta.cohort'`.

- [ ] **Step 3: Write the implementation**

Create `src/gtex_meta/cohort.py`:

```python
"""Label the GTEx liver cohort from the Portal histology table.

Rules
-----
cirrhosis
    The curated ``Pathology Categories`` contain ``cirrhosis``.
healthy
    Either the pathologist curated ``no_abnormalities`` / ``clean_specimens``,
    or the slide has no curated category *and* its note is non-empty, contains
    no positive finding, and carries no preservation warning.
other
    Everything else. An empty note is unknown, never healthy.
"""

import argparse
import re
import sys

import pandas as pd

# Findings that disqualify a slide from the control group even when the curator
# assigned no category. Kept wide on purpose: the uncategorized slides do
# contain real findings in their free text.
FINDING_RE = re.compile(
    r"(?i)("
    r"steato|fatty|\bfat\b|congest|sinusoidal|fibro|cirrh|necro|inflamm|hepatit"
    r"|lymphocy|lymphoid|infiltrat|sclero|nodul|atroph|hemorrhag|hyalin|ischem"
    r"|hypox|scar|hyperplas|pigment|balloon|degener|apoptos|granulom|cholang"
    r"|vacuol|amyloid|hamartom|pallor|dilat|cord cell|abscess|metaplas|calcif"
    r"|eosinophil|neutrophil"
    r")"
)

# Tissue-quality problems: not disease, but they confound histology features.
# Both spellings of autolysis count; an earlier prototype matched only
# "autolys" and mislabeled three autolyzed slides as healthy.
QUALITY_RE = re.compile(
    r"(?i)(autoly[sz]|poorly preserved|poor preservation|poor fixation|bad fixation)"
)

# Phrases that negate a finding, e.g. "no fat", "not congested".
NEGATION_RE = re.compile(
    r"(?i)\b(no|not|without|free of|negative for)\b[^.;,]{0,30}"
    r"(fat|steato|congest|fibro|cirrh|necro|inflamm|hepatit|infiltrat|sclero"
    r"|nodul|atroph|hemorrhag|hyalin|ischem|scar|hyperplas|pigment|balloon"
    r"|degener|lesion|abnormal)"
)

CLEAN_CATEGORIES = {"no_abnormalities", "clean_specimens"}


def split_categories(value) -> set[str]:
    """Split a comma-separated category string into a set of names."""
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def has_finding(note: str) -> bool:
    """True if the note describes a finding, after removing negated phrases."""
    stripped = NEGATION_RE.sub(" ", str(note or ""))
    return bool(FINDING_RE.search(stripped))


def label_row(row: pd.Series) -> tuple[str, str]:
    """Return the (label, reason) for one slide."""
    categories = split_categories(row["Pathology Categories"])
    note = str(row["Pathology Notes"] or "").strip()

    if "cirrhosis" in categories:
        return "cirrhosis", "curated cirrhosis"
    if categories & CLEAN_CATEGORIES:
        curated = "/".join(sorted(categories & CLEAN_CATEGORIES))
        return "healthy", f"curated {curated}"
    if not categories:
        if not note:
            return "other", "uncategorized, no note (unknown)"
        if QUALITY_RE.search(note):
            return "other", "uncategorized, preservation warning"
        if has_finding(note):
            return "other", "uncategorized, finding in note"
        return "healthy", "uncategorized, note reports no finding"
    return "other", "curated non-cirrhotic pathology"


def build_cohort(slides: pd.DataFrame) -> pd.DataFrame:
    """Return the slide table with label columns appended."""
    tissues = set(slides["Tissue"].astype(str).unique())
    if tissues != {"Liver"}:
        raise ValueError(f"expected liver-only input, found tissues: {sorted(tissues)}")

    result = slides.copy()
    labels, reasons = zip(*(label_row(row) for _, row in result.iterrows()))
    result["label"] = labels
    result["label_reason"] = reasons
    result["quality_flag"] = result["Pathology Notes"].apply(
        lambda note: "preservation" if QUALITY_RE.search(str(note or "")) else ""
    )
    result["has_finding_text"] = result["Pathology Notes"].apply(has_finding)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", required=True, help="Input slide table CSV")
    parser.add_argument("--out", required=True, help="Output cohort CSV")
    args = parser.parse_args(argv)

    slides = pd.read_csv(args.slides, dtype=str).fillna("")
    labelled = build_cohort(slides)
    labelled.to_csv(args.out, index=False)

    counts = labelled["label"].value_counts()
    print(f"wrote {len(labelled)} slides to {args.out}")
    for name in ("healthy", "cirrhosis", "other"):
        print(f"  {name:10} {counts.get(name, 0)}")

    cirrhosis = labelled[labelled["label"] == "cirrhosis"]
    co_findings: dict[str, int] = {}
    for value in cirrhosis["Pathology Categories"]:
        for category in split_categories(value) - {"cirrhosis"}:
            co_findings[category] = co_findings.get(category, 0) + 1
    for category, count in sorted(co_findings.items(), key=lambda item: -item[1]):
        print(f"  {category:15} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync python -m unittest tests.test_cohort -v`

Expected: `OK` (12 tests).

- [ ] **Step 5: Generate the tracked CSVs from the public API**

Run:

```bash
uv run --no-sync python -m gtex_meta.portal --tissue Liver --out data/liver_slides.csv
uv run --no-sync python -m gtex_meta.cohort --slides data/liver_slides.csv --out data/liver_cohort.csv
```

Expected:

```
wrote 610 slides to data/liver_slides.csv
wrote 610 slides to data/liver_cohort.csv
  healthy    66
  cirrhosis  58
  other      486
  steatosis       21
  fibrosis        18
  hepatitis       10
  nodularity      8
  congestion      8
  inflammation    6
  hyalinization   2
  necrosis        2
  scarring        1
  pigment         1
  sclerotic       1
  atrophy         1
```

- [ ] **Step 6: Verify the cohort CSV against the known distribution**

Run:

```bash
uv run --no-sync python -c "
import pandas as pd
d = pd.read_csv('data/liver_cohort.csv', dtype=str).fillna('')
print('rows', len(d))
print(d.label.value_counts().to_dict())
print('empty notes', (d['Pathology Notes'].str.strip() == '').sum())
print(d[d.label == 'healthy'].label_reason.value_counts().to_dict())
print(d[d.label == 'other'].label_reason.value_counts().to_dict())
"
```

Expected:

```
rows 610
{'other': 486, 'healthy': 66, 'cirrhosis': 58}
empty notes 7
{'uncategorized, note reports no finding': 60, 'curated no_abnormalities': 4, 'curated clean_specimens': 2}
{'curated non-cirrhotic pathology': 445, 'uncategorized, finding in note': 25, 'uncategorized, preservation warning': 9, 'uncategorized, no note (unknown)': 7}
```

- [ ] **Step 7: Commit**

```bash
git add src/gtex_meta/cohort.py tests/test_cohort.py data/liver_slides.csv data/liver_cohort.csv
git commit -m "feat: label the GTEx liver cohort from the public slide table"
```

---

### Task 4: `centroids.py` — HistoPlus GeoJSON to per-slide npz

**Files:**
- Create: `src/gtex_meta/centroids.py`
- Test: `tests/test_centroids.py`

**Interfaces:**
- Consumes: a HistoPlus directory and, optionally, the cohort CSV
- Produces:
  - `METHOD_VERSION: str`
  - `SUFFIX = ".histoplus.geojson.gz"`
  - `gdal_path(path: str) -> str` — prefixes `.gz` paths with `/vsigzip/`
  - `read_slide_ids(csv_path: str) -> list[str]`
  - `extract_one(source_path: str, out_path: str) -> dict` — writes the npz, returns `{"slide_id", "n_cells", "path"}`
  - `select_slides(histoplus_dir, slide_ids) -> tuple[list[tuple[str, str]], list[str]]` — `(sources, missing_ids)`
  - `extract_many(sources, out_dir: str, workers: int, overwrite: bool) -> dict[str, int]` — counts keyed `extracted`, `skipped`, `failed`, `missing`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_centroids.py`:

```python
import gzip
import json
import os
import tempfile
import unittest

import numpy as np

from gtex_meta import centroids

SQUARE = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
TRIANGLE = [[0, 0], [2, 0], [0, 2], [0, 0]]


def feature(cell_id, classification, prob, ring):
    return {
        "type": "Feature",
        "properties": {"cell_id": cell_id, "prob": prob, "classification": classification},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def write_geojson(path, features, compress=False):
    payload = json.dumps({"type": "FeatureCollection", "features": features})
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)


class GdalPathTests(unittest.TestCase):
    def test_prefixes_gzipped_paths_only(self):
        self.assertEqual(
            centroids.gdal_path("/data/a.geojson.gz"), "/vsigzip//data/a.geojson.gz"
        )
        self.assertEqual(centroids.gdal_path("/data/a.geojson"), "/data/a.geojson")


class ExtractOneTests(unittest.TestCase):
    def test_writes_coords_labels_prob_and_class_names(self):
        features = [
            feature("s.00000", "Fibroblasts", 0.9, SQUARE),
            feature("s.00001", "Apoptotic Body", 0.2, TRIANGLE),
            feature("s.00002", "Fibroblasts", 0.7, SQUARE),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "GTEX-AAAA-0126.histoplus.geojson.gz")
            out = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            write_geojson(source, features, compress=True)

            result = centroids.extract_one(source, out)
            with np.load(out, allow_pickle=True) as data:
                coords = data["coords"]
                labels = data["labels"]
                prob = data["prob"]
                class_names = [str(name) for name in data["class_names"]]
                provenance = json.loads(str(data["provenance"]))

        self.assertEqual(result["slide_id"], "GTEX-AAAA-0126")
        self.assertEqual(result["n_cells"], 3)
        self.assertEqual(coords.shape, (3, 2))
        self.assertEqual(coords.dtype, np.float32)
        self.assertEqual(class_names, ["Apoptotic Body", "Fibroblasts"])
        self.assertEqual(list(labels), [1, 0, 1])
        self.assertEqual(labels.dtype, np.int16)
        np.testing.assert_allclose(prob, [0.9, 0.2, 0.7], rtol=1e-6)
        np.testing.assert_allclose(coords[0], [1.0, 1.0], atol=1e-4)
        np.testing.assert_allclose(coords[1], [2 / 3, 2 / 3], atol=1e-4)
        self.assertEqual(provenance["method_version"], centroids.METHOD_VERSION)
        self.assertEqual(provenance["n_cells"], 3)

    def test_rejects_empty_feature_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "GTEX-AAAA-0126.histoplus.geojson")
            out = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            write_geojson(source, [])
            with self.assertRaises(ValueError):
                centroids.extract_one(source, out)

    def test_reads_uncompressed_geojson_without_prefix(self):
        features = [feature("s.00000", "Fibroblasts", 0.5, SQUARE)]
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "GTEX-AAAA-0126.histoplus.geojson")
            out = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            write_geojson(source, features)
            result = centroids.extract_one(source, out)
        self.assertEqual(result["n_cells"], 1)


class SelectSlidesTests(unittest.TestCase):
    def test_reports_missing_slides_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = os.path.join(tmp, "GTEX-AAAA-0126.histoplus.geojson.gz")
            write_geojson(
                present, [feature("s.00000", "Fibroblasts", 0.5, SQUARE)], compress=True
            )
            sources, missing = centroids.select_slides(
                tmp, ["GTEX-AAAA-0126", "GTEX-BBBB-0126"]
            )
        self.assertEqual([slide_id for slide_id, _ in sources], ["GTEX-AAAA-0126"])
        self.assertEqual(missing, ["GTEX-BBBB-0126"])

    def test_discovers_all_files_when_no_slide_list_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            for slide_id in ("GTEX-AAAA-0126", "GTEX-BBBB-0126"):
                write_geojson(
                    os.path.join(tmp, f"{slide_id}.histoplus.geojson.gz"),
                    [feature("s.00000", "Fibroblasts", 0.5, SQUARE)],
                    compress=True,
                )
            sources, missing = centroids.select_slides(tmp, None)
        self.assertEqual(len(sources), 2)
        self.assertEqual(missing, [])

    def test_reads_slide_ids_from_a_cohort_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cohort.csv")
            with open(path, "w") as handle:
                handle.write('"Tissue Sample ID","label"\n"GTEX-AAAA-0126","healthy"\n')
            self.assertEqual(centroids.read_slide_ids(path), ["GTEX-AAAA-0126"])


class ExtractManyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = os.path.join(self.tmp.name, "GTEX-AAAA-0126.histoplus.geojson.gz")
        write_geojson(
            self.source, [feature("s.00000", "Fibroblasts", 0.5, SQUARE)], compress=True
        )
        self.out_dir = os.path.join(self.tmp.name, "out")

    def test_skips_existing_outputs_unless_overwrite(self):
        sources = [("GTEX-AAAA-0126", self.source)]
        first = centroids.extract_many(sources, self.out_dir, workers=1, overwrite=False)
        second = centroids.extract_many(sources, self.out_dir, workers=1, overwrite=False)
        third = centroids.extract_many(sources, self.out_dir, workers=1, overwrite=True)
        self.assertEqual(first["extracted"], 1)
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(third["extracted"], 1)

    def test_counts_vanished_sources_as_missing(self):
        sources = [
            ("GTEX-AAAA-0126", self.source),
            ("GTEX-CCCC-0126", os.path.join(self.tmp.name, "missing.geojson.gz")),
        ]
        counts = centroids.extract_many(sources, self.out_dir, workers=1, overwrite=False)
        self.assertEqual(counts["extracted"], 1)
        self.assertEqual(counts["missing"], 1)


class MainTests(unittest.TestCase):
    def test_missing_input_directory_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = centroids.main(
                [
                    "--histoplus-dir",
                    os.path.join(tmp, "does-not-exist"),
                    "--out-dir",
                    os.path.join(tmp, "out"),
                ]
            )
        self.assertEqual(exit_code, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync python -m unittest tests.test_centroids -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gtex_meta.centroids'`.

- [ ] **Step 3: Write the implementation**

Create `src/gtex_meta/centroids.py`:

```python
"""Extract cell centroids and class labels from HistoPlus GeoJSON files.

HistoPlus files are a given input. Slides whose file is missing are reported as
warnings and skipped; they never abort the run.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

METHOD_VERSION = "1.0"
SUFFIX = ".histoplus.geojson.gz"


def gdal_path(path: str) -> str:
    """Return a GDAL-readable path, decompressing .gz through /vsigzip/."""
    if path.endswith(".gz"):
        return f"/vsigzip/{path}"
    return path


def read_slide_ids(csv_path: str) -> list[str]:
    """Read slide IDs from a cohort CSV."""
    frame = pd.read_csv(csv_path, dtype=str)
    column = "Tissue Sample ID" if "Tissue Sample ID" in frame.columns else frame.columns[0]
    return [value for value in frame[column].fillna("") if value]


def extract_one(source_path: str, out_path: str) -> dict:
    """Extract centroids from one HistoPlus file and write a compressed npz."""
    slide_id = os.path.basename(source_path).split(".histoplus")[0]
    frame = gpd.read_file(gdal_path(source_path))
    if frame.empty:
        raise ValueError(f"no features in {os.path.basename(source_path)}")

    coords = np.array(
        [(geometry.centroid.x, geometry.centroid.y) for geometry in frame.geometry],
        dtype=np.float32,
    )
    classes = frame["classification"].astype(str)
    class_names = np.array(sorted(classes.unique()))
    lookup = {name: index for index, name in enumerate(class_names)}
    labels = classes.map(lookup).to_numpy(dtype=np.int16)
    prob = (
        frame["prob"].to_numpy(dtype=np.float32)
        if "prob" in frame.columns
        else np.full(len(frame), np.nan, dtype=np.float32)
    )
    provenance = json.dumps(
        {
            "source_file": os.path.basename(source_path),
            "method_version": METHOD_VERSION,
            "n_cells": int(coords.shape[0]),
            "n_classes": int(class_names.size),
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        coords=coords,
        labels=labels,
        prob=prob,
        class_names=class_names,
        provenance=np.array(provenance),
    )
    return {"slide_id": slide_id, "n_cells": int(coords.shape[0]), "path": out_path}


def select_slides(
    histoplus_dir: str, slide_ids: list[str] | None
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (sources, missing) where sources are (slide_id, path) pairs."""
    if slide_ids is None:
        sources = [
            (name.split(".histoplus")[0], os.path.join(histoplus_dir, name))
            for name in sorted(os.listdir(histoplus_dir))
            if name.endswith(SUFFIX)
        ]
        return sources, []

    sources: list[tuple[str, str]] = []
    missing: list[str] = []
    for slide_id in slide_ids:
        path = os.path.join(histoplus_dir, f"{slide_id}{SUFFIX}")
        if os.path.exists(path):
            sources.append((slide_id, path))
        else:
            missing.append(slide_id)
    return sources, missing


def _extract_job(job: tuple[str, str, str, bool]) -> dict:
    """Worker entry point: extract one slide, catching per-file failures."""
    slide_id, source, out_dir, overwrite = job
    if not os.path.exists(source):
        return {"slide_id": slide_id, "status": "missing"}
    out_path = os.path.join(out_dir, f"{slide_id}.npz")
    if os.path.exists(out_path) and not overwrite:
        return {"slide_id": slide_id, "status": "skipped"}
    try:
        extract_one(source, out_path)
        return {"slide_id": slide_id, "status": "extracted"}
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return {"slide_id": slide_id, "status": "failed", "error": str(error)}


def extract_many(
    sources: list[tuple[str, str]], out_dir: str, workers: int, overwrite: bool
) -> dict[str, int]:
    """Extract many slides in parallel, returning status counts."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {"extracted": 0, "skipped": 0, "failed": 0, "missing": 0}
    jobs = [(slide_id, path, out_dir, overwrite) for slide_id, path in sources]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(
            pool.map(_extract_job, jobs), total=len(jobs), unit="slide", desc="Centroids"
        ):
            counts[result["status"]] += 1
            if result["status"] == "failed":
                tqdm.write(f"  {result['slide_id']}: {result['error']}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histoplus-dir", required=True, help="HistoPlus input directory")
    parser.add_argument("--out-dir", required=True, help="Directory for .npz outputs")
    parser.add_argument(
        "--slides",
        default=None,
        help="Optional CSV of slides to process (default: every HistoPlus file)",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--overwrite", action="store_true", help="Redo existing .npz files")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.histoplus_dir):
        print(f"ERROR: no such directory: {args.histoplus_dir}", file=sys.stderr)
        return 1

    slide_ids = read_slide_ids(args.slides) if args.slides else None
    sources, missing = select_slides(args.histoplus_dir, slide_ids)

    if missing:
        print(
            f"WARNING: {len(missing)} requested slides have no HistoPlus file; skipping",
            file=sys.stderr,
        )
        for slide_id in missing[:10]:
            print(f"  missing: {slide_id}", file=sys.stderr)

    counts = extract_many(sources, args.out_dir, args.workers, args.overwrite)
    counts["missing"] += len(missing)
    print(
        "extracted={extracted} skipped={skipped} failed={failed} missing={missing}".format(
            **counts
        )
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync python -m unittest tests.test_centroids -v`

Expected: `OK` (10 tests).

- [ ] **Step 5: Verify against an existing LISC npz**

This confirms our extraction reproduces the reference implementation. It skips cleanly if the local files are absent.

Run:

```bash
uv run --no-sync python -c "
import numpy as np, geopandas as gpd, os
from gtex_meta.centroids import gdal_path
sid = 'GTEX-1JJ6O-0826'
src = os.path.expanduser(f'~/data/GTEX/histoplus/{sid}.histoplus.geojson.gz')
ref = os.path.expanduser(f'~/data/GTEX/pc/{sid}.npz')
if not (os.path.exists(src) and os.path.exists(ref)):
    print('SKIP: local reference files not present')
else:
    gdf = gpd.read_file(gdal_path(src))
    xy = np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry], dtype=np.float32)
    z = np.load(ref, allow_pickle=True)
    ref_class = np.array([str(n) for n in z['class_names']])[z['labels']]
    print('class sequence identical:', bool((ref_class == gdf['classification'].astype(str).to_numpy()).all()))
    print('max coord diff: %.6f' % np.abs(z['coords'] - xy).max())
    print('cells:', len(xy), '| classes:', gdf['classification'].nunique())
"
```

Expected:

```
class sequence identical: True
max coord diff: 0.000973
cells: 35076 | classes: 12
```

- [ ] **Step 6: Commit**

```bash
git add src/gtex_meta/centroids.py tests/test_centroids.py
git commit -m "feat: extract cell centroids and labels from HistoPlus files"
```

---

### Task 5: `consolidate.py` — npz directory to one AnnData

**Files:**
- Create: `src/gtex_meta/consolidate.py`
- Test: `tests/test_consolidate.py`

**Interfaces:**
- Consumes: npz files written by `gtex_meta.centroids` (keys `coords`, `labels`, `prob`, `class_names`, `provenance`) and the cohort CSV written by `gtex_meta.cohort`
- Produces:
  - `META_COLUMNS: list[str]`
  - `load_centroids(path: str) -> dict` with keys `path`, `slide_id`, `coords`, `labels`, `class_names`, `prob`, `has_provenance`
  - `consolidate(entries: list[dict], cohort: pandas.DataFrame) -> anndata.AnnData`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_consolidate.py`:

```python
import json
import os
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from gtex_meta import consolidate

COHORT_COLUMNS = ["Tissue Sample ID", "Subject ID", "Age Bracket", "Sex", "Hardy Scale", "label"]


def write_npz(path, coords, codes, class_names, prob=None, provenance=True):
    arrays = {
        "coords": np.asarray(coords, dtype=np.float32),
        "labels": np.asarray(codes, dtype=np.int16),
        "class_names": np.array(class_names, dtype=object),
        "prob": np.asarray(
            prob if prob is not None else [0.5] * len(coords), dtype=np.float32
        ),
    }
    if provenance:
        arrays["provenance"] = np.array(json.dumps({"method_version": "1.0"}))
    np.savez_compressed(path, **arrays)


def write_cohort(path, rows):
    pd.DataFrame(rows, columns=COHORT_COLUMNS).to_csv(path, index=False)


def cohort_row(slide_id, label="cirrhosis"):
    return [slide_id, "-".join(slide_id.split("-")[:2]), "60-69", "male", "Slow death", label]


class LoadCentroidsTests(unittest.TestCase):
    def test_loads_arrays_and_flags_missing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            legacy = os.path.join(tmp, "GTEX-BBBB-0126.npz")
            write_npz(good, [[1, 2]], [0], ["Fibroblasts"])
            write_npz(legacy, [[3, 4]], [0], ["Fibroblasts"], provenance=False)
            entry = consolidate.load_centroids(good)
            legacy_entry = consolidate.load_centroids(legacy)
        self.assertEqual(entry["slide_id"], "GTEX-AAAA-0126")
        self.assertEqual(entry["coords"].dtype, np.float32)
        self.assertEqual(entry["prob"].dtype, np.float32)
        self.assertTrue(entry["has_provenance"])
        self.assertFalse(legacy_entry["has_provenance"])


class ConsolidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def entry(self, slide_id, coords, codes, class_names):
        path = os.path.join(self.tmp.name, f"{slide_id}.npz")
        write_npz(path, coords, codes, class_names)
        return consolidate.load_centroids(path)

    def cohort(self, slide_ids):
        return pd.DataFrame(
            [cohort_row(slide_id, "cirrhosis" if index == 0 else "healthy")
             for index, slide_id in enumerate(slide_ids)],
            columns=COHORT_COLUMNS,
        ).set_index("Tissue Sample ID")

    def test_remaps_per_file_vocabularies_into_a_shared_var(self):
        first = self.entry(
            "GTEX-AAAA-0126", [[1, 1], [2, 2]], [0, 1], ["Fibroblasts", "Macrophages"]
        )
        second = self.entry("GTEX-BBBB-0126", [[3, 3]], [0], ["Macrophages"])

        adata = consolidate.consolidate(
            [first, second], self.cohort(["GTEX-AAAA-0126", "GTEX-BBBB-0126"])
        )

        self.assertEqual(list(adata.var_names), ["Fibroblasts", "Macrophages"])
        self.assertEqual(
            list(adata.obs["cell_type"].astype(str)),
            ["Fibroblasts", "Macrophages", "Macrophages"],
        )
        self.assertEqual(
            list(adata.obs["slide_id"].astype(str)),
            ["GTEX-AAAA-0126"] * 2 + ["GTEX-BBBB-0126"],
        )
        self.assertEqual(
            list(adata.obs["label"].astype(str)), ["cirrhosis", "cirrhosis", "healthy"]
        )
        self.assertTrue(sp.issparse(adata.X))
        np.testing.assert_array_equal(
            adata.X.toarray(), [[1, 0], [0, 1], [0, 1]]
        )
        np.testing.assert_allclose(adata.obsm["spatial"], [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(adata.obsm["spatial"].dtype, np.float32)
        self.assertIsInstance(adata.obs.index, pd.RangeIndex)
        np.testing.assert_allclose(adata.obs["prob"].to_numpy(), [0.5, 0.5, 0.5])

    def test_keeps_cells_whose_slide_has_no_cohort_row(self):
        entry = self.entry("GTEX-CCCC-0126", [[1, 1]], [0], ["Fibroblasts"])
        adata = consolidate.consolidate([entry], self.cohort(["GTEX-AAAA-0126"]))
        self.assertEqual(adata.n_obs, 1)
        self.assertEqual(adata.obs["label"].astype(str).iloc[0], "")

    def test_rejects_an_empty_entry_list(self):
        with self.assertRaises(ValueError):
            consolidate.consolidate([], self.cohort(["GTEX-AAAA-0126"]))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.centroid_dir = os.path.join(self.tmp.name, "centroids")
        os.makedirs(self.centroid_dir)
        self.cohort_csv = os.path.join(self.tmp.name, "cohort.csv")

    def add_npz(self, slide_id, provenance=True):
        write_npz(
            os.path.join(self.centroid_dir, f"{slide_id}.npz"),
            [[1, 1]],
            [0],
            ["Fibroblasts"],
            provenance=provenance,
        )

    def test_writes_gzipped_h5ad_and_records_provenance(self):
        self.add_npz("GTEX-AAAA-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")

        exit_code = consolidate.main(
            ["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out]
        )
        adata = ad.read_h5ad(out)

        self.assertEqual(exit_code, 0)
        self.assertEqual(adata.n_obs, 1)
        self.assertEqual(adata.uns["n_slides"], 1)
        self.assertIn("Fibroblasts", list(adata.uns["class_names"]))
        self.assertEqual(list(adata.obs["label"].astype(str)), ["cirrhosis"])
        self.assertTrue(os.path.getsize(out) > 0)

    def test_filter_to_cohort_drops_unlisted_slides(self):
        self.add_npz("GTEX-AAAA-0126")
        self.add_npz("GTEX-BBBB-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")

        consolidate.main(
            [
                "--centroids-dir",
                self.centroid_dir,
                "--cohort",
                self.cohort_csv,
                "--out",
                out,
                "--filter-to-cohort",
            ]
        )
        adata = ad.read_h5ad(out)

        self.assertEqual(adata.n_obs, 1)
        self.assertEqual(list(adata.obs["slide_id"].astype(str)), ["GTEX-AAAA-0126"])

    def test_fails_when_no_npz_files_exist(self):
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        exit_code = consolidate.main(
            ["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out]
        )
        self.assertEqual(exit_code, 1)
        self.assertFalse(os.path.exists(out))

    def test_missing_centroids_directory_returns_one(self):
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        exit_code = consolidate.main(
            [
                "--centroids-dir",
                os.path.join(self.tmp.name, "does-not-exist"),
                "--cohort",
                self.cohort_csv,
                "--out",
                os.path.join(self.tmp.name, "cohort.h5ad"),
            ]
        )
        self.assertEqual(exit_code, 1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync python -m unittest tests.test_consolidate -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'gtex_meta.consolidate'`.

- [ ] **Step 3: Write the implementation**

Create `src/gtex_meta/consolidate.py`:

```python
"""Consolidate per-slide centroid npz files into one AnnData object.

Layout decisions (see the design spec for the measurements behind them):
``X`` is a sparse one-hot indicator of ``cell_type``, ``var`` is the sorted
union of the per-slide class vocabularies, ``obsm["spatial"]`` holds float32
centroids, the index is a RangeIndex, and the file is gzip-compressed.
"""

import argparse
import datetime
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

META_COLUMNS = ["Subject ID", "Age Bracket", "Sex", "Hardy Scale", "label"]


def load_centroids(path: str) -> dict:
    """Load one centroid npz, tolerating files from the older LISC pipeline."""
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        coords = data["coords"].astype(np.float32, copy=False)
        entry = {
            "path": path,
            "slide_id": os.path.basename(path)[:-4],
            "coords": coords,
            "labels": data["labels"].astype(np.int16, copy=False),
            "class_names": np.asarray(data["class_names"], dtype=str),
            "prob": (
                data["prob"].astype(np.float32, copy=False)
                if "prob" in keys
                else np.full(len(coords), np.nan, dtype=np.float32)
            ),
            "has_provenance": "provenance" in keys,
        }
    return entry


def consolidate(entries: list[dict], cohort: pd.DataFrame) -> ad.AnnData:
    """Build one AnnData from loaded centroid entries."""
    if not entries:
        raise ValueError("no centroid entries to consolidate")

    class_names = sorted({name for entry in entries for name in entry["class_names"]})
    lookup = {name: index for index, name in enumerate(class_names)}
    total = sum(len(entry["coords"]) for entry in entries)

    coords = np.empty((total, 2), dtype=np.float32)
    prob = np.empty(total, dtype=np.float32)
    codes = np.empty(total, dtype=np.int64)
    cell_types = np.empty(total, dtype=object)
    slide_ids = np.empty(total, dtype=object)

    offset = 0
    for entry in entries:
        n_cells = len(entry["coords"])
        window = slice(offset, offset + n_cells)
        names = entry["class_names"][entry["labels"]]
        coords[window] = entry["coords"]
        prob[window] = entry["prob"]
        cell_types[window] = names
        codes[window] = [lookup[name] for name in names]
        slide_ids[window] = entry["slide_id"]
        offset += n_cells

    obs = pd.DataFrame(
        {
            "slide_id": pd.Categorical(slide_ids),
            "cell_type": pd.Categorical(cell_types, categories=class_names),
            "prob": prob,
        }
    )
    metadata = cohort.reindex(pd.Index(slide_ids, name="Tissue Sample ID"))
    for column in META_COLUMNS:
        if column in metadata.columns:
            obs[column] = pd.Categorical(metadata[column].fillna("").to_numpy())
        else:
            obs[column] = pd.Categorical(np.full(total, ""))

    data = sp.csr_matrix(
        (np.ones(total, dtype=np.float32), (np.arange(total), codes)),
        shape=(total, len(class_names)),
    )
    adata = ad.AnnData(X=data, obs=obs, var=pd.DataFrame(index=pd.Index(class_names)))
    adata.obsm["spatial"] = coords
    adata.uns["n_slides"] = len(entries)
    adata.uns["n_cells"] = int(total)
    adata.uns["class_names"] = class_names
    adata.uns["slide_ids"] = sorted(entry["slide_id"] for entry in entries)
    adata.uns["metadata_columns"] = META_COLUMNS
    adata.uns["created"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return adata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centroids-dir", required=True, help="Directory of .npz files")
    parser.add_argument("--cohort", required=True, help="Cohort CSV to join metadata from")
    parser.add_argument("--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--filter-to-cohort",
        action="store_true",
        help="Exclude npz files whose slide is not in the cohort CSV",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.centroids_dir):
        print(f"ERROR: no such directory: {args.centroids_dir}", file=sys.stderr)
        return 1

    cohort = pd.read_csv(args.cohort, dtype=str).fillna("").set_index("Tissue Sample ID")
    paths = sorted(
        os.path.join(args.centroids_dir, name)
        for name in os.listdir(args.centroids_dir)
        if name.endswith(".npz")
    )
    if not paths:
        print(f"ERROR: no .npz files in {args.centroids_dir}", file=sys.stderr)
        return 1

    entries = [load_centroids(path) for path in paths]
    known = set(cohort.index)
    warnings: list[str] = []

    if args.filter_to_cohort:
        dropped = [entry["slide_id"] for entry in entries if entry["slide_id"] not in known]
        entries = [entry for entry in entries if entry["slide_id"] in known]
        if dropped:
            print(f"filtered out {len(dropped)} slides not in the cohort CSV")
    else:
        unmatched = [entry["slide_id"] for entry in entries if entry["slide_id"] not in known]
        if unmatched:
            warnings.append(
                f"{len(unmatched)} slides have no cohort metadata: {unmatched[:5]}"
            )

    missing_npz = sorted(known - {entry["slide_id"] for entry in entries})
    if missing_npz:
        warnings.append(f"{len(missing_npz)} cohort slides have no .npz: {missing_npz[:5]}")

    legacy = [entry["slide_id"] for entry in entries if not entry["has_provenance"]]
    if legacy:
        warnings.append(
            f"{len(legacy)} files lack provenance (legacy LISC output): {legacy[:5]}"
        )

    if not entries:
        print("ERROR: no entries left after filtering", file=sys.stderr)
        return 1

    adata = consolidate(entries, cohort)
    adata.write_h5ad(args.out, compression="gzip")

    print(f"wrote {adata.n_obs} cells from {adata.uns['n_slides']} slides to {args.out}")
    print(f"  classes: {adata.n_vars} | shape: {adata.shape}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync python -m unittest tests.test_consolidate -v`

Expected: `OK` (8 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`

Expected: `OK` (38 tests: 1 package + 7 portal + 12 cohort + 10 centroids + 8 consolidate).

- [ ] **Step 6: Commit**

```bash
git add src/gtex_meta/consolidate.py tests/test_consolidate.py
git commit -m "feat: consolidate centroid files into a single AnnData"
```

---

### Task 6: SLURM wrappers

**Files:**
- Create: `sbatch/extract_centroids.sbatch`
- Create: `sbatch/consolidate_anndata.sbatch`

**Interfaces:**
- Consumes: the `gtex_meta.centroids` and `gtex_meta.consolidate` entry points
- Produces: two job scripts reading `REPO_DIR`, `HISTOPLUS_DIR`, `CENTROIDS_DIR`, `COHORT_CSV`, `OUT_H5AD`, and `FILTER_TO_COHORT` from the environment

- [ ] **Step 1: Create `sbatch/extract_centroids.sbatch`**

```bash
#!/bin/bash
#SBATCH --job-name=gtex_extract_centroids
#SBATCH --output=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_extract_centroids_%j.out
#SBATCH --error=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_extract_centroids_%j.err
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=localtmp:20
#SBATCH --constraint=ssd
#SBATCH --license=scratch-highio

# Extract cell centroids for the cohort's HistoPlus files.
#
#   sbatch --export=ALL,HISTOPLUS_DIR=/path,CENTROIDS_DIR=/path,COHORT_CSV=/path \
#       sbatch/extract_centroids.sbatch
#
# REPO_DIR defaults to the LISC checkout symlink; the data paths have no
# defaults so no lab path is baked into the Python code.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/lisc/home/user/schindlers/rep/rendeiro/tissuegeometry}"
: "${HISTOPLUS_DIR:?export HISTOPLUS_DIR=/path/to/histoplus}"
: "${CENTROIDS_DIR:?export CENTROIDS_DIR=/path/to/centroids}"
: "${COHORT_CSV:?export COHORT_CSV=/path/to/liver_cohort.csv}"

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:       $(hostname)"
echo "histoplus:  $HISTOPLUS_DIR"
echo "centroids:  $CENTROIDS_DIR"
echo "cohort csv: $COHORT_CSV"
echo "workers:    ${SLURM_CPUS_PER_TASK:-16}"

# The project venv is used directly on purpose: `uv run` re-syncs the
# environment and fails on the optional ../cpyrcolate path dependency.
"$REPO_DIR/.venv/bin/python" -m gtex_meta.centroids \
    --histoplus-dir "$HISTOPLUS_DIR" \
    --out-dir "$CENTROIDS_DIR" \
    --slides "$COHORT_CSV" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

STATUS=$?
echo "npz now:    $(ls -1 "$CENTROIDS_DIR" 2>/dev/null | wc -l)"
exit $STATUS
```

- [ ] **Step 2: Create `sbatch/consolidate_anndata.sbatch`**

```bash
#!/bin/bash
#SBATCH --job-name=gtex_consolidate_anndata
#SBATCH --output=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_consolidate_anndata_%j.out
#SBATCH --error=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_consolidate_anndata_%j.err
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=localtmp:20
#SBATCH --constraint=ssd
#SBATCH --license=scratch-highio

# Consolidate every centroid npz into one AnnData object.
#
#   sbatch --export=ALL,CENTROIDS_DIR=/path,COHORT_CSV=/path,OUT_H5AD=/path \
#       sbatch/consolidate_anndata.sbatch
#
# Add FILTER_TO_COHORT=1 to keep only slides listed in the cohort CSV.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/lisc/home/user/schindlers/rep/rendeiro/tissuegeometry}"
: "${CENTROIDS_DIR:?export CENTROIDS_DIR=/path/to/centroids}"
: "${COHORT_CSV:?export COHORT_CSV=/path/to/liver_cohort.csv}"
: "${OUT_H5AD:?export OUT_H5AD=/path/to/output.h5ad}"

EXTRA_ARGS=()
if [ "${FILTER_TO_COHORT:-0}" = "1" ]; then
    EXTRA_ARGS+=(--filter-to-cohort)
fi

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:       $(hostname)"
echo "centroids:  $CENTROIDS_DIR"
echo "cohort csv: $COHORT_CSV"
echo "output:     $OUT_H5AD"

"$REPO_DIR/.venv/bin/python" -m gtex_meta.consolidate \
    --centroids-dir "$CENTROIDS_DIR" \
    --cohort "$COHORT_CSV" \
    --out "$OUT_H5AD" \
    "${EXTRA_ARGS[@]}"

STATUS=$?
ls -lh "$OUT_H5AD" 2>/dev/null
exit $STATUS
```

- [ ] **Step 3: Make them executable and syntax-check**

Run:

```bash
chmod +x sbatch/extract_centroids.sbatch sbatch/consolidate_anndata.sbatch
bash -n sbatch/extract_centroids.sbatch && bash -n sbatch/consolidate_anndata.sbatch && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 4: Verify directives and the absence of `uv run`**

Run:

```bash
grep -c "#SBATCH" sbatch/extract_centroids.sbatch sbatch/consolidate_anndata.sbatch
grep -n "uv run" sbatch/*.sbatch || echo "no uv run (expected)"
grep -n "PYTHONPATH=src" sbatch/*.sbatch
```

Expected: 11 `#SBATCH` lines per file (`grep -c` prints two numbers), `no uv run (expected)`, and one `PYTHONPATH=src` line per file.

- [ ] **Step 5: Commit**

```bash
git add sbatch/
git commit -m "feat: add SLURM wrappers for extraction and consolidation"
```

---

### Task 7: README documenting every decision

**Files:**
- Modify: `README.md` (full replacement)

**Interfaces:**
- Consumes: the design spec and the measured facts it records
- Produces: a README a new user can follow end to end

- [ ] **Step 1: Replace `README.md` with six sections**

Use exactly these six level-two headings, in this order:

1. `## What this does` — the four stages and their four commands.
2. `## Data provenance` — the two GTEx files, why filtering the sample attributes gives 242 slides while the Portal histology table has 610, the API endpoint, and the fact that no bulk CSV exists.
3. `## Cohort definition` — the labeling rules, the counts (58 cirrhosis, 66 healthy, 486 other, 124 in the healthy/cirrhotic subset), the cirrhosis co-findings, and the limitations: the healthy label is a text heuristic; `SMATSSCR` autolysis exists for only 8 of the 58 cirrhotic slides; `GTEX-1JJ6O-0826` carries the note "liver, not skin" and is flagged for review; an empty note means unknown, never healthy.
4. `## About prob` — one scalar per cell, argmax confidence, no per-class breakdown, measured distribution (median 0.607, 34% below 0.5), and why no default threshold is applied.
5. `## AnnData layout` — sparse one-hot `X`, `cell_type` in `obs`, `RangeIndex`, gzip, with the measured table: 34.5 B/cell gzipped, 1.17 GB for 34M cells, 5.52 GB for 160M cells, and the note that 16 B/cell is the in-memory CSR cost, not the file size.
6. `## Running it` — laptop commands, the sbatch commands with their environment variables, and the `uv run` re-sync caveat on the cluster.

Add a `### Removed from this repository` subsection under "Running it" listing `liver.py`, `downloader.py`, `cohorts.py`, `cohorts_notebook.py`, `meta.py`, `path.py`, `vocab.py`, `main.py`, `path_terms.txt`, `liver_wsis.csv`, and `__marimo__/`, each with a one-line reason.

- [ ] **Step 2: Verify the required headings exist in order**

Run:

```bash
grep -c "^## " README.md
grep -n "^## " README.md
```

Expected: `6`, listing the six headings above in order.

- [ ] **Step 3: Verify the key facts appear**

Run:

```bash
for token in 610 242 58 66 486 124 34.5 5.52 vsigzip; do
  printf "%-8s %s\n" "$token" "$(grep -c -- "$token" README.md)"
done
```

Expected: every count is at least 1.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document pipeline decisions in the README"
```

---

### Task 8: Remove superseded files

**Files:**
- Delete (tracked): `docs/superpowers/plans/2026-08-05-cell-centroids-download.md` (`liver.py` and the two legacy test modules were already removed in Task 1)
- Delete (untracked): `main.py`, `meta.py`, `path.py`, `vocab.py`, `cohorts.py`, `cohorts_notebook.py`, `downloader.py`, `path_terms.txt`, `liver_wsis.csv`, `extract_centroids_liver.sbatch`, `fetch_gtex_histology.py`, `build_liver_cohort.py`, `liver_healthy_vs_cirrhotic.csv`, `liver_hvc_matched_pairs.csv`, `gtex_portal_liver_slides.csv`, `__marimo__/`, `wsi_slides/`, `.superpowers/`, `__pycache__/`, `docs/superpowers/plans/2026-08-04-cohorts-marimo-notebook.md`

**Interfaces:**
- Consumes: the tested pipeline from Tasks 1-7
- Produces: a repository containing only the four-stage pipeline

- [ ] **Step 1: Confirm nothing in the new code references the doomed files**

Run:

```bash
grep -rn -E "liver_wsis|cohorts\.py|downloader|path_terms|marimo" src/ sbatch/ tests/ README.md || echo "no references (expected)"
```

Expected: `no references (expected)`. If anything matches, fix that reference before deleting.

- [ ] **Step 2: Delete the tracked files**

```bash
git rm docs/superpowers/plans/2026-08-05-cell-centroids-download.md
```

- [ ] **Step 3: Delete the untracked files**

```bash
rm -f main.py meta.py path.py vocab.py cohorts.py cohorts_notebook.py downloader.py \
      path_terms.txt liver_wsis.csv extract_centroids_liver.sbatch \
      fetch_gtex_histology.py build_liver_cohort.py \
      liver_healthy_vs_cirrhotic.csv liver_hvc_matched_pairs.csv \
      gtex_portal_liver_slides.csv \
      docs/superpowers/plans/2026-08-04-cohorts-marimo-notebook.md
rm -rf __marimo__ wsi_slides .superpowers __pycache__
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`

Expected: `OK` (38 tests), with only `test_cohort`, `test_centroids`, `test_consolidate`, and `test_package` present.

- [ ] **Step 5: Verify the final repository layout**

Run:

```bash
ls -A
git status --short
find src sbatch tests data -type f | sort
```

Expected: top level contains `README.md`, `pyproject.toml`, `uv.lock`, `.gitignore`, `.python-version`, `.venv`, `data/`, `docs/`, `sbatch/`, `src/`, `tests/`. `find` lists the four modules, `__init__.py`, the two sbatch files, the four test files, and the two cohort CSVs.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove superseded scripts and outputs"
```

---

## Self-Review Notes

- **Spec coverage:** every spec section maps to a task — provenance and the public fetcher (Task 2), labeling rules and the tracked cohort CSV (Task 3), centroid extraction with `prob` and warnings for missing files (Task 4), consolidation and the AnnData layout (Task 5), cluster execution (Task 6), README decisions (Task 7), cleanup (Task 8), dependency trim and package layout (Task 1).
- **Deliberately dropped:** the matched-pair subset (removed by user decision), any HistoPlus download (D1), and all notebooks and plots (non-goals).
- **Type consistency:** `extract_one` writes `coords`, `labels`, `prob`, `class_names`, `provenance`; `load_centroids` reads exactly those keys and uses the presence of `provenance` to flag legacy LISC files. `cohort.build_cohort` writes the `label` column that `consolidate.META_COLUMNS` consumes by that name. `centroids.read_slide_ids` reads the CSV that `cohort.main` writes.
- **Corrections found during planning:** the preservation regex must match `autoly[sz]` — the prototype matched only `autolys` and mislabeled three autolyzed slides as healthy, which is why the documented counts are 66 healthy and 486 other rather than 69 and 483; gzipped GeoJSON requires the `/vsigzip/` prefix with pyogrio; float32 centroids reproduce the LISC output to within 0.000973 px.
