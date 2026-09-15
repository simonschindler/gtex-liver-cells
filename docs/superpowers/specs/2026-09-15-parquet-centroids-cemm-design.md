# Parquet Cell-Type Source on CEMM — Design

Date: 2026-09-15

## Purpose

Switch `gtex-liver-cells` from HistoPlus GeoJSON polygons to the lab's
Parquet cell-type source, and move all cluster execution from LISC to CEMM.

Everything else in the pipeline — the public-data cohort build and the
consolidation into one AnnData — stays as it is.

## Relationship to the previous design

This document supersedes
`2026-09-14-gtex-liver-cohort-design.md` in three places:

- **D1 (input)** — HistoPlus GeoJSON is no longer the input. Parquet replaces
  it; there is no second, coexisting source.
- **Cluster** — execution moves from LISC SLURM to CEMM SLURM, and the LISC
  sbatch files are deleted rather than adapted.
- **Dependencies** — `pyarrow` is added, `geopandas` is dropped.

The following decisions carry over unchanged and are not revisited here:
the 610-slide labeled cohort and its rules (D2, D3), retaining `prob`
(D4), the one-hot `X` AnnData layout (D5), and the tracked-versus-ignored
file policy (D8). The cohort CSV, `cohort.py`, `portal.py`, and
`consolidate.py` are untouched by this change.

## Verified background

All facts below were measured during design, not assumed.

### The new source

Per slide, one directory:
`<root>/<TISSUE_ID>/cell_types.gpd` — Parquet despite the extension
(file magic `PAR1`), with four columns:

| Column | Type | Meaning |
|---|---|---|
| `geometry` | binary | WKB polygon, Arrow extension `geoarrow.wkb` |
| `class` | string | cell-type label |
| `prob` | double | confidence for the predicted class |
| `cell_id` | int64 | per-cell identifier |

The sibling `cell_types_features.h5ad` in the same directory holds 768-d
CellViT embeddings. It is **not** used: the embeddings are unnecessary for
the tutorial and would dominate the object size.

### The data model

- Parquet row count equals the h5ad row count exactly on the one slide where
  both were read (55,867), and there `cell_id` matches in **both set and
  order**, so no join logic would be needed even if embeddings were added
  later. This was checked on `GTEX-15TU5-2426`; the claim is structural, so it
  is expected to hold for every slide, but it has been verified once.
- `library_id` is the constant string `"cell_types"` and `tile_id` is unique
  per cell (0…n-1); neither carries information and neither is carried over.
- Geometry types present are `Polygon` **and** `MultiPolygon`. `shapely`'s
  `.centroid` handles both, so the reader must not assume simple polygons.
- Cell-class vocabularies **vary per slide**, exactly as they did in the
  GeoJSON source: both liver slides measured carry 12 classes, while the skin
  slide carried 13 (adding `Minor Stromal Cell`). Consolidation must therefore
  keep unioning per-slide vocabularies rather than assuming a fixed taxonomy.
- Class composition on two liver slides:

| class | GTEX-14AS3-0126 (349,884 cells) | GTEX-13OVJ-1026 (251,876 cells) |
|---|---|---|
| Epithelial | 245,067 (70.0%) | 59,068 (23.5%) |
| Fibroblasts | 52,335 (15.0%) | 66,880 (26.6%) |
| Cancer cell | 23,110 (6.6%) | 31,814 (12.6%) |
| Apoptotic Body | 5,595 (1.6%) | 42,042 (16.7%) |
| Lymphocytes | 2,316 (0.7%) | 30,641 (12.2%) |
| Muscle Cell | 4,687 (1.3%) | 13,329 (5.3%) |
| Macrophages | 9,143 (2.6%) | 2,554 (1.0%) |
| Neutrophils | 5,511 (1.6%) | 3,043 (1.2%) |
| Endothelial Cell | 1,891 (0.5%) | 1,333 (0.5%) |
| remaining three | <0.1% each | <0.4% each |

- `prob`: on `GTEX-14AS3-0126` min 0.148, median 0.782, max 0.993 with 15.8%
  of cells below 0.5; on `GTEX-13OVJ-1026` median 0.696 with 25.7% below 0.5.

The class labels are CellViT model output, not curation, and the taxonomy is
pan-cancer: it has **no hepatocyte or Kupffer-cell class**, so liver cells are
necessarily mapped onto generic epithelial and stromal labels — `Epithelial`
reaching 70% on one slide is most plausibly hepatocytes under a label that
does not fit them. `Cancer cell` appears at 6.6-12.6% of these GTEx livers
although the donors have no cancer diagnosis. The README must state plainly
that `class` is a model prediction on a mismatched taxonomy, not histology.

### Cell counts versus the old source

An earlier estimate in this project claimed CellViT yields ~10x fewer cells
than HistoPlus. That was wrong: it compared one **skin** slide's CellViT count
against a **liver** slide's HistoPlus count. The corrected paired comparison,
same slide and both segmentations, covers 8 liver slides:

| slide | CellViT | HistoPlus | ratio |
|---|---|---|---|
| GTEX-11ZU8-0126 | 53,535 | 82,261 | 0.651 |
| GTEX-13OVJ-1026 | 251,876 | 360,237 | 0.699 |
| GTEX-1192X-1026 | 390,933 | 554,207 | 0.705 |
| GTEX-13VXU-0926 | 145,604 | 178,995 | 0.813 |
| GTEX-11ZVC-0726 | 184,893 | 225,041 | 0.822 |
| GTEX-14AS3-0126 | 349,884 | 409,900 | 0.854 |
| GTEX-11TT1-1726 | 725,038 | 808,053 | 0.897 |
| GTEX-11DXZ-0126 | 247,768 | 273,422 | 0.906 |

CellViT finds roughly **18% fewer cells** than HistoPlus (median ratio 0.818,
range 0.651-0.906). Mean cells per slide: 293,691 versus 361,514.

Consequences for sizing: ~36M cells for the 124-slide healthy/cirrhotic
subset, ~179M for all 610 labeled slides, and therefore roughly 1.3 GB and
6 GB for the consolidated object at the measured 34.5 B/cell. Parquet file
size averages ~222 bytes/cell (11.9-171.5 MB per slide across the 8 measured).

Caveat: 8 slides is a paired estimate, not a census. It is enough to size
the jobs and kill the earlier "10x" claim, not to promise an exact object
size.

### Performance

On one 349,884-cell slide: reading `geometry`+`class`+`prob` took 1.65 s and
WKB-to-centroid conversion took 2.15 s. Extraction is therefore minutes for
the cohort, and consolidation dominates the runtime.

### The CEMM environment

Probed on `login.int.cemm.at`:

- SLURM 24.05.8. Partitions: `covid`, `develop`, `gpu`, `interactiveq`,
  `longq`, `mediumq`, `shortq`, `tinyq`. Account `lab_rendeiro`.
- EasyBuild modules only — **no conda, no uv**. System `python3` is 3.6.8;
  modules provide `Python-bundle/3.11.3` and `SciPy-bundle`, but nothing for
  `anndata` or `pyarrow`.
- PyPI **is reachable from the login node** (`https://pypi.org/simple/` →
  200), which makes a uv-managed environment viable.
- `$HOME` is `/home/sschindler` on a 2.4 PB filesystem; the
  `lab_rendeiro/projects/histopath` project directory is writable.
- Apptainer/Singularity exist as modules but are not needed.

## Decisions

### P1 — Parquet replaces GeoJSON; no coexistence

`centroids.py` reads the Parquet source only. There is no `--source` switch,
no reader abstraction, and no second fixture. Supporting both would double
the test surface and the documentation for a source that needs 16 GB of
third-party GeoJSON to be useful at all.

### P2 — Embeddings are excluded

Only `cell_types.gpd` is read. `cell_types_features.h5ad` is never opened.
This keeps `X` a sparse one-hot indicator of cell class, which is what the
tutorial needs, and keeps the published object near 1.3 GB instead of tens of
gigabytes.

### P3 — Execution moves to CEMM

LISC is dropped entirely. This affects only the sbatch wrappers and the
documented cluster commands; no Python code is LISC-specific.

### P4 — The environment is uv-managed on CEMM

Install uv into `~/.local/bin` (no root needed, login node has network), and
`uv sync` from `uv.lock` to build `.venv` in the repo checkout. Rationale:
the lockfile stays the single source of environment truth, the same story
works on a laptop and on the cluster, and it avoids hand-pinning
`anndata`/`pyarrow` into an EasyBuild module set that does not have them.
A container was considered and rejected as an unnecessary build step while
the login node has PyPI access.

### P5 — The per-slide npz intermediate is preserved

Extraction still writes one `.npz` per slide with exactly the keys the
current consolidation reads (`coords`, `labels`, `prob`, `class_names`,
`provenance`). This is deliberate:

- `consolidate.py` needs no changes at all.
- Extraction stays parallel and resumable, and a crash mid-cohort does not
  cost the completed slides.
- The intermediate is small: ~14 bytes per cell, so ~4 MB per slide and
  ~500 MB for the cohort.

Writing per-slide h5ad instead was rejected as heavier for no benefit, and
skipping the intermediate was rejected because it would give up resumability
and force one long single-process job.

### P6 — Dependencies: add pyarrow, drop geopandas

`pyarrow` becomes a runtime dependency for Parquet; `shapely` (already
present) parses WKB and computes centroids. `geopandas` is no longer used by
anything and is removed, which also makes the CEMM environment noticeably
lighter. `pandas`, `numpy`, `scipy`, `anndata`, and `tqdm` stay.

### P7 — Outputs live in the topocyte data tree

All products go to
`/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_liver_cells/`,
which already exists, is empty, and is **not** a git checkout, so large
outputs cannot be committed by accident.

Layout:

```
gtex_liver_cells/
├── centroids/<TISSUE_ID>.npz     # per-slide intermediate
├── gtex_liver_cells.h5ad         # the consolidated object
├── liver_cohort.csv              # the cohort actually used
├── provenance.json               # how it was built
└── logs/                         # SLURM stdout/stderr
```

`provenance.json` records the repository commit, UTC creation time, the input
root, requested/extracted/missing slide counts, the class vocabulary, and the
total cell count, so the output directory is self-describing without
duplicating the repository.

### P8 — No transfer script

The Parquet source is read in place on CEMM, so nothing needs to be copied
between clusters and no data passes through a local machine. The earlier
request for a CEMB-to-LISC transfer script is withdrawn by this change.

## Component interfaces

### centroids.py (rewritten reader, same output)

```
python -m gtex_liver_cells.centroids \
    --parquet-dir /nobackup/.../datasets_pq_h5ad \
    --out-dir     /nobackup/.../gtex_liver_cells/centroids \
    --slides      data/liver_cohort.csv \
    --workers     16 [--overwrite]
```

- Input selection: `<parquet-dir>/<TISSUE_ID>/cell_types.gpd`. Without
  `--slides`, every `*/cell_types.gpd` under `--parquet-dir` is processed.
- Per slide: read the three needed Parquet columns, parse each WKB geometry,
  take `.centroid`, and write `<TISSUE_ID>.npz`:

| Key | Type | Meaning |
|---|---|---|
| `coords` | float32 (N, 2) | polygon centroids |
| `labels` | int16 (N,) | index into `class_names` |
| `prob` | float32 (N,) | confidence for the assigned class |
| `class_names` | unicode (C,) | this slide's class vocabulary |
| `provenance` | unicode scalar | JSON: source file, method version, counts |

`METHOD_VERSION` becomes `"2.0"` to distinguish these files from the GeoJSON
era.

### Deleted

- `gdal_path()` and every `/vsigzip/` reference, plus the GeoJSON fixture and
  the tests that exercised them.
- Both LISC sbatch files (`sbatch/extract_centroids.sbatch`,
  `sbatch/consolidate_anndata.sbatch`) — their `--license=scratch-highio`,
  `--gres=localtmp` and `--constraint=ssd` directives do not exist on CEMM and
  would fail immediately.

### Added

- `sbatch/extract_centroids.sbatch` and `sbatch/consolidate_anndata.sbatch` in
  CEMM style: `--account=lab_rendeiro`, `--partition=mediumq`, `--time`,
  `--cpus-per-task`, `--mem`, logs under the output directory, and paths taken
  from the environment (`PARQUET_DIR`, `CENTROIDS_DIR`, `COHORT_CSV`,
  `OUT_DIR`, `REPO_DIR`) so no path is hardcoded in Python.
- An optional `--provenance-out` on `consolidate` to write
  `provenance.json` beside the h5ad; the consolidation job also copies
  `liver_cohort.csv` into the output directory.

## Error handling

| Condition | Behaviour |
|---|---|
| `--parquet-dir` missing | Error, exit non-zero |
| Requested slide has no `cell_types.gpd` | Warn, count as missing, continue |
| Existing npz without `--overwrite` | Skip, count |
| Unreadable or empty Parquet | Count as failed, continue, exit non-zero |
| Geometry that is neither Polygon nor MultiPolygon | Count as failed for that slide, report the type |
| npz without cohort metadata | Warn, still consolidate |

## Testing

Standard-library `unittest`, no new test dependency.

- `test_centroids.py` rewritten around a Parquet fixture built in-test with
  pyarrow: three cells, two simple polygons and one MultiPolygon, asserting
  `coords`, `labels`, `prob`, `class_names`, and centroid values; plus the
  missing-slide warning path and the `<root>/<TISSUE_ID>/cell_types.gpd`
  convention.
- `test_consolidate.py` unchanged except that its fixture writer must emit the
  same npz keys (it already does).
- `test_cohort.py` and `test_portal.py` unchanged.

A verification step, not a test, compares one extracted slide's cell count
against the Parquet row count to confirm nothing is dropped.

## README updates

- "What this does" — the input is the Parquet cell-type source, not GeoJSON.
- "Data provenance" — add the Parquet source, its `class`/`prob` semantics,
  and the label caveat: CellViT's pan-cancer taxonomy has no hepatocyte class
  and reports Cancer cell at 6.6-12.6% of these GTEx livers.
- "Running it" — CEMM commands, `uv sync` setup, the two sbatch invocations,
  and the output layout.
- Remove the `/vsigzip/` note; it no longer applies.

## Risks and open items

- **Unconfirmed:** the repository checkout path on CEMM. This document assumes
  `/home/sschindler/rep/gtex-liver-cells`; if it differs, only the sbatch
  defaults change.
- Cell counts are a paired estimate over 8 slides, not a census of the cohort.
- Cell classes are model predictions with no curation, and `prob` is unfiltered
  by design; both are documented rather than silently corrected.
- `mediumq` (2-day limit) is assumed sufficient; extraction is minutes, but
  consolidation of all 610 slides would need ~6 GB of output and should be
  checked against the partition's memory and time limits if that scope is used.
