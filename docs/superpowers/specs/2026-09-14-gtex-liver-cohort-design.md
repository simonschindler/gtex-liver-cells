# GTEx Liver Cohort and Cell-Centroid Pipeline — Design

Date: 2026-09-14

## Purpose

Turn this repository into a self-contained, explainable pipeline that

1. fetches the public GTEx histology metadata,
2. builds a liver slide cohort with an explicit healthy/cirrhotic label and
   writes it to CSV,
3. extracts per-slide cell centroids and cell-class labels from HistoPlus
   GeoJSON files, and
4. consolidates all centroid files into a single AnnData object on disk.

Every decision is documented in the README, with the evidence behind it.
Nothing in the pipeline depends on the LISC `tissuegeometry` repository.

## Non-goals

- Downloading whole-slide images (the old `downloader.py` / IDC path).
- Downloading HistoPlus GeoJSON files from CEMB. These are a given input.
- Interactive notebooks, cohort plots, or a marimo notebook.
- Any modeling, network construction, or topology analysis.

## Verified background

These facts were established by inspecting the actual files and endpoints
during design, and they justify the decisions in the next section.

### Two different GTEx documents describe liver slides

| Source | Rows | Liver slides |
|---|---|---|
| `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt` | 22,951 | 242 |
| GTEx Portal histology table (`GTEx Portal.csv`) | 25,713 | 610 |

The sample-attributes file lists only samples that carry assay annotations.
Filtering it on `SMTS == "Liver"` yields 242 slides (251 rows, deduplicated by
`TISSUE_ID`). The remaining 368 liver slides in the Portal table do not appear
in the sample-attributes file under any tissue, so they are histology-only
slides with no molecular annotation. The 242-slide set is a strict subset of
the 610-slide set.

The lab's `GTEx_Meta.csv` is byte-identical to the CEMB `GTEx Portal.csv`
(md5 `603ca443d8acab8b45b4fb7902a8aa27`), confirming a single origin.

### The public source is the Portal histology API

`https://gtexportal.org/api/v2/histology/image` is paginated JSON and returns
25,713 slides, 610 of them liver, with fields matching the Portal CSV exactly:
`histologyImageId`, `tissueSiteDetail`, `subjectId`, `sex`, `ageBracket`,
`hardyScale`, `pathologyNotes`, `pathologyNotesCategories`, plus a `hide` flag.

Verified equivalence against the CEMB CSV for liver: identical slide IDs
(610/610), identical notes (610/610), identical category sets (610/610).

There is no bulk CSV. The public GCS bucket (`adult-gtex`) contains only the
annotation files under `annotations/v8/metadata-files/`, and the Portal file
catalog (`/api/v2/dataset/fileList`) lists no histology or image entries. The
API is the only public route, which is why the repository ships a fetcher.

### What the data actually contains

- 610 liver slides, each from a distinct donor, so slide-level and donor-level
  splits are equivalent for this cohort.
- Categories per slide: 0 for 101 slides, 1 for 308, 2 for 158, 3 for 30,
  4 for 11, 5 for 2.
- Curated categories are absent for 101 slides. Those slides are *not* healthy:
  27 describe a finding in free text ("<10% macrovesicular fat", "mild central
  degeneration", "thick fibrous trabeculae"), 4 carry preservation warnings,
  and 7 have no note at all.
- Exactly 7 slides have no note, and they are the same 7 slides that have no
  HistoPlus file. All 7 fall in the `other` label.
- 31 slides are flagged `hide` in the Portal; exactly one is liver
  (`GTEX-1269W-1826`), and it falls in `other` regardless.
- HistoPlus coverage: 603 of 610 slides. Both the healthy (69) and cirrhotic
  (58) groups are fully covered.

### HistoPlus GeoJSON and the centroid files

- Each feature has exactly three properties: `cell_id`, `classification`, and
  `prob`. Verified across 9,000 feature blocks from three slides.
- `prob` is a single scalar per cell — the confidence for the predicted class,
  consistent with an argmax softmax. There is no per-class breakdown and no
  companion file carrying one. Measured distribution: min 0.139, median 0.607,
  max 0.995; 34% of calls are below 0.5 and 11% clear 0.9.
- The LISC-produced npz files store `coords` (float64), `labels` (int64), and
  `class_names`, discarding both `prob` and `cell_id`.
- Class vocabularies differ per file because a `LabelEncoder` is fit per slide.
  Across the 66 local files there are three distinct vocabularies: 12 classes
  are universal, 14 are in the union (`Minor Stromal Cell` and `Mitotic
  Figures` are missing from some). Integer labels are therefore not comparable
  across files and must be remapped through `class_names`.
- Measured cell counts: 18,501,178 cells across 66 files; per slide, minimum
  35,076, median 265,670, maximum 808,053. Extrapolated: roughly 34M cells for
  the 127-slide healthy/cirrhotic subset and 160M for all 603 available slides.

## Decisions

### D1 — HistoPlus files are a given input

The repository does not download HistoPlus files. A stage that needs them
takes a directory or a slide list and emits a warning for every requested slide
whose GeoJSON is absent, then continues. Missing files never abort a run.

### D2 — The canonical cohort is all 610 liver slides, labeled

One CSV with one row per liver slide and a `label` of `healthy`, `cirrhosis`,
or `other`, plus `label_reason`, `quality_flag`, and `has_finding_text`.
Downstream subsets (healthy vs cirrhosis, or graded tiers) are derived by
filtering this file, so there is a single source of truth. There is no matched
subset artifact; matching is a downstream analysis choice.

### D3 — Labeling rules

- `cirrhosis`: the curated categories contain `cirrhosis` (58 slides).
- `healthy`: either the curator assigned `no_abnormalities` or
  `clean_specimens` (6 slides), or the slide has no curated category, a
  non-empty note, no finding term after negation stripping, and no preservation
  warning (63 slides).
- `other`: everything else (483 slides).

An empty note is treated as unknown, never as healthy. The finding and
preservation term lists are explicit constants in the code, and the reason for
every label is written to the CSV so the classification is auditable.

Resulting counts: 58 cirrhosis, 69 healthy, 483 other. Cirrhosis is
heterogeneous (21 steatotic, 18 fibrotic, 10 hepatitis, 8 nodularity, 8
congested, 6 inflamed, plus rarer findings); only 8 slides carry cirrhosis
alone. This is recorded as a property of the cohort rather than used to filter.

Known limitations, to be documented in the README:

- The healthy label is a documented text heuristic, not a pathologist's call.
- Autolysis score (`SMATSSCR`) exists only for the 242 sample-attributes
  slides, covering just 8 of the 58 cirrhotic ones, so tissue-degradation QC is
  not available cohort-wide.
- `GTEX-1JJ6O-0826` carries the note "liver, not skin", a tissue-label
  correction; it is labeled healthy by the rules and flagged for review in the
  README.

### D4 — Retain `prob`, do not filter on it by default

Extraction stores `prob` as float32 per cell. No default threshold is applied,
because dropping 34% of cells silently would be a hidden analysis decision.
Any filtering is an explicit downstream choice. Enabling a threshold requires
re-exporting the full softmax upstream, which is out of scope.

### D5 — AnnData layout: one-hot `X`, integer index, gzip

Measured by building real objects from three slides (1,107,515 cells, 14
classes) and writing them to disk:

| Layout | gzip B/cell | raw B/cell | 34M cells | 160M cells | write 160M |
|---|---|---|---|---|---|
| `X=None`, string index | 49.2 | 71.0 | 1.67 GB | 7.87 GB | 71 s |
| one-hot f32, string index | 51.0 | 83.1 | 1.73 GB | 8.16 GB | 80 s |
| **one-hot f32, integer index** | **34.5** | 64.1 | **1.17 GB** | **5.52 GB** | 76 s |
| `X=None`, integer index | 32.7 | 55.1 | 1.11 GB | 5.23 GB | 64 s |

The chosen layout is sparse one-hot `X` (CSR, float32) with `cell_type` also in
`obs`, a `RangeIndex`, and gzip compression. Rationale:

- On disk the one-hot costs only ~1.8 B/cell after gzip, because the CSR
  indices are a stride-1 run and the data are all ones. The 16 B/cell figure is
  the in-memory CSR cost, which matters for job RSS but not for file size.
- The global string cell index is the larger cost (~16.5 B/cell), hence the
  `RangeIndex`; cell identity is `(slide_id, row offset)`.
- `X=None` would save 6% but breaks tools that assume a matrix.

### D6 — Recompute centroid npz rather than reuse the LISC files

Existing LISC npz files lack `prob` and use float64/int64. Recomputing gives one
homogeneous dataset with a single method version. Extraction should therefore
write to a fresh output directory. When the output directory already contains
files, they are skipped unless `--overwrite` is passed, so that re-running an
interrupted job does not redo completed work. To make mixed provenance visible
rather than silent, consolidation warns about any npz that lacks `prob` or
`provenance`, which identifies files produced by the older LISC pipeline.

### D7 — geopandas is the extraction backend

The LISC stage uses `geopandas.read_file` and `geometry.centroid`; we use the
same approach so coordinates are comparable with existing outputs. geopandas is
already present (via pyogrio; fiona is not required). A stdlib-only streaming
parser was considered and rejected as unnecessary complexity for now.

### D8 — What is tracked in git

Tracked: source, tests, sbatch files, README, design and plan documents, and the
two small CSVs under `data/` — `liver_slides.csv` (the 610-row liver subset of
the Portal table) and `liver_cohort.csv` (the labeled cohort). Ignored:
`.venv/`, `__pycache__/`, the full `gtex_portal_slides.csv` fetch (25,713 rows,
regenerable from the API), centroid npz files, and `.h5ad` outputs.

## Repository layout

```
gtex_meta/
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── liver_slides.csv          # tracked: liver subset of the Portal table
│   └── liver_cohort.csv          # tracked: labeled cohort
├── docs/superpowers/specs/2026-09-14-gtex-liver-cohort-design.md
├── sbatch/
│   ├── extract_centroids.sbatch
│   └── consolidate_anndata.sbatch
├── src/gtex_meta/
│   ├── __init__.py
│   ├── portal.py
│   ├── cohort.py
│   ├── centroids.py
│   └── consolidate.py
└── tests/
    ├── test_cohort.py
    ├── test_centroids.py
    └── test_consolidate.py
```

Dependencies: `numpy`, `pandas`, `scipy`, `anndata`, `geopandas`, `shapely`,
`tqdm`. Removed: `idc-index`, `lazyslide`, `marimo`, `matplotlib`, `seaborn`,
`pydicom`. Python stays at 3.14 per `.python-version`.

Modules are run as `uv run --no-sync python -m gtex_meta.<stage>` from the
repository root, or `.venv/bin/python -m ...` with `PYTHONPATH=src`. Using the
existing environment matters on the cluster: job 5802838 failed because
`uv run` re-synced and could not resolve the optional `../cpyrcolate` path
dependency.

## Component interfaces

### portal.py — fetch the public slide table

```
python -m gtex_meta.portal --out data/gtex_portal_slides.csv     # all 25,713 slides
python -m gtex_meta.portal --tissue Liver --out data/liver_slides.csv
```

Pages `https://gtexportal.org/api/v2/histology/image` at 1000 items per page
using only the standard library, retrying transient failures with backoff.
Writes the eight Portal columns in their original order plus a ninth `Hidden`
column, with all fields quoted so the file diffs cleanly against the lab's
`GTEx Portal.csv`. Without `--tissue` all 25,713 slides are written; with
`--tissue Liver` it produces the 610-row `data/liver_slides.csv` that the rest
of the pipeline consumes.

### cohort.py — label the cohort

```
python -m gtex_meta.cohort --slides data/liver_slides.csv --out data/liver_cohort.csv
```

Applies D3 and writes the nine Portal columns plus `label`, `label_reason`,
`quality_flag`, and `has_finding_text`. Prints label counts and the cirrhosis
co-finding breakdown. It asserts that every row is liver tissue rather than
silently labeling another organ.

### centroids.py — HistoPlus GeoJSON to per-slide npz

```
python -m gtex_meta.centroids --histoplus-dir DIR --out-dir DIR \
    [--slides data/liver_cohort.csv] [--workers N] [--overwrite]
```

Selects slides from `--slides` when given, otherwise every
`*.histoplus.geojson.gz` in `--histoplus-dir`. For each slide it writes
`<slide>.npz` containing:

| Key | Type | Meaning |
|---|---|---|
| `coords` | float32 (N, 2) | polygon centroids |
| `labels` | int16 (N,) | index into `class_names` |
| `prob` | float32 (N,) | model confidence for the assigned class |
| `class_names` | unicode (C,) | class vocabulary for this slide |
| `provenance` | unicode scalar | JSON with source file, method version, counts |

Files are processed in parallel with `ProcessPoolExecutor` and `tqdm`. Existing
outputs are skipped unless `--overwrite`. A missing GeoJSON is a warning and a
counted skip, not an error. A corrupt file is a counted failure; the run
continues and exits non-zero if any file failed.

### consolidate.py — npz to a single AnnData

```
python -m gtex_meta.consolidate --centroids-dir DIR --cohort data/liver_cohort.csv \
    --out data/liver_cohort.h5ad [--filter-to-cohort]
```

Reads every npz in `--centroids-dir`, remaps per-file labels into the sorted
union of class names, and writes one `.h5ad`:

- `obs`: `slide_id` (category), `cell_type` (category), `subject_id`
  (category), `label` (category), `Age Bracket`, `Sex`, `Hardy Scale`, `prob`
  (float32); index is a `RangeIndex`.
- `var`: the sorted union of class names.
- `X`: CSR one-hot float32, shape `(n_cells, n_classes)`.
- `obsm["spatial"]`: float32 centroids.
- `uns`: slide count, cell count, class vocabulary, per-slide source files,
  parameter values, and a generation timestamp.
- Gzip compression.

Metadata is joined from `--cohort`. Warnings are emitted for npz files with no
cohort row, cohort slides with no npz, and any class name outside the union
observed in the inputs. The observed vocabulary is always recorded in `uns`
rather than assumed. With `--filter-to-cohort`, npz files whose slide is not in
the cohort CSV are excluded instead of warned about, which is how all 603
slides are narrowed to the 127-slide healthy/cirrhotic subset.

## Error handling summary

| Condition | Behaviour |
|---|---|
| Input directory or CSV missing | Error, exit non-zero |
| Requested slide has no GeoJSON | Warn, count, continue |
| Existing npz without `--overwrite` | Skip, count |
| Corrupt or empty GeoJSON | Count as failure, continue, exit non-zero |
| npz without cohort metadata | Warn, still consolidate |
| Hard failure with zero output | Error, exit non-zero |

## Testing

`uv run python -m unittest discover -s tests -v`, standard library only.

- `test_cohort.py`: one fixture row per labeling branch — curated cirrhosis,
  curated clean, uncategorized with no finding, uncategorized with a finding,
  uncategorized with empty notes, and with a preservation warning.
- `test_centroids.py`: a three-polygon GeoJSON fixture, asserting geometry
  order, `coords`, `labels`, `prob`, `class_names`, and that a missing slide is
  reported without raising.
- `test_consolidate.py`: two small npz files with *different* class
  vocabularies, asserting the union `var`, correct label remapping, the one-hot
  contents, the `obs` metadata join, and gzip output.

## Cluster execution

Both sbatch files follow the LISC house style: absolute `#SBATCH --output` and
`--error` paths under `logs/`, mail to `simon.schindler@univie.ac.at`,
`--time`, `--cpus-per-task`, `--mem=128G`, `--gres=localtmp:20`,
`--constraint=ssd`, `--license=scratch-highio`.

`extract_centroids.sbatch` defaults to 16 CPUs and processes the cohort's
HistoPlus files. `consolidate_anndata.sbatch` defaults to 8 CPUs and 128G.

Paths are supplied through environment variables so no checkout path is
hardcoded in the Python code:

| Variable | Purpose |
|---|---|
| `REPO_DIR` | repository root, defaulted to `/lisc/home/user/schindlers/rep/rendeiro/tissuegeometry` |
| `HISTOPLUS_DIR` | given HistoPlus input directory |
| `CENTROIDS_DIR` | npz output directory |
| `COHORT_CSV` | cohort CSV consumed by consolidation |
| `OUT_H5AD` | AnnData output path |

Only `REPO_DIR` has a default; the four data paths are cluster-specific and must
be exported, so no lab path is hardcoded in the Python code. Both scripts set
`PYTHONPATH=src` and call the project venv directly, avoiding `uv run` re-sync.

## Cleanup plan

Deleted once the new stages pass their tests:

| Path | Reason |
|---|---|
| `main.py`, `meta.py`, `path.py`, `vocab.py` | scratch exploration |
| `cohorts.py`, `cohorts_notebook.py`, `__marimo__/` | superseded by `cohort.py` |
| `downloader.py`, `wsi_slides/` | IDC WSI download, out of scope |
| `liver.py`, `tests/test_liver.py`, `tests/test_import_safety.py` | superseded; its rsync logic is replaced by D1 |
| `path_terms.txt`, `liver_wsis.csv` | intermediates of the old approach |
| `liver_healthy_vs_cirrhotic.csv`, `liver_hvc_matched_pairs.csv` | replaced by the new cohort CSV |
| `extract_centroids_liver.sbatch` | replaced by `sbatch/` versions |
| `docs/superpowers/plans/2026-08-04-*.md`, `2026-08-05-*.md` | completed or obsolete plans |
| `__pycache__/`, `.superpowers/` | build cruft |

Kept and rewritten: `README.md`, `pyproject.toml`, `uv.lock`, `.gitignore`,
`.python-version`. `fetch_gtex_histology.py` becomes `src/gtex_meta/portal.py`;
`build_liver_cohort.py` becomes `src/gtex_meta/cohort.py` with the matched-pair
logic removed.

Deletion happens after the new code is in place and tested, so the repository
is never left without a working path.

## README outline

1. What the pipeline does and the four commands.
2. Data provenance: the two GTEx files, the Portal histology API, and why the
   cohort has 610 slides rather than the 242 assay-annotated ones.
3. Cohort definition: the labeling rules, the counts, the cirrhosis
   co-findings, and the limitations listed under D3.
4. `prob` semantics and why no default threshold is applied.
5. AnnData layout decisions with the measured size table.
6. How to run each stage, on a laptop and on the cluster.
7. What was removed from the repository and why.

## Risks

- The healthy label is heuristic. Mitigation: reasons written per row, terms
  kept as explicit constants, and the limitation documented.
- Consolidating all 603 slides approaches 5.5 GB on disk and needs several GB
  of RAM while building; the 128G request covers it, but the node must have
  enough free space in the output filesystem.
- The Portal API could change shape or rate-limit. Mitigation: the fetched
  table is written to disk once and reused, and the fetcher retries with
  backoff.
- Class vocabularies could differ from the 14 observed. Mitigation: the union
  is computed from the inputs and recorded in `uns` so drift is visible.
