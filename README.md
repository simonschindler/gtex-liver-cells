# gtex-liver-cells

A self-contained pipeline that builds a cell-resolved GTEx liver
histopathology dataset: a labeled liver slide cohort plus cell centroids,
cell-class labels, and coordinates consolidated into one AnnData object.

The resulting object is what downstream analyses consume — for example the
spatial-topology workflows in [topocyte](https://github.com/rendeirolab/topocyte),
which loads it through the sibling `topocyte_data` repository.

## What this does

Four stages, each a standalone command reading and writing explicit paths.

1. **`portal`** fetches the public GTEx Portal histology table — 25,713 slides
   across all tissues.
2. **`cohort`** labels the 610 liver slides `healthy`, `cirrhosis`, or `other`
   and writes `data/liver_cohort.csv`.
3. **`centroids`** extracts cell centroids and class labels from the lab's
   Parquet cell-type source — one `<TISSUE_ID>/cell_types.gpd` file per slide —
   into one `.npz` per slide. Those files are a given input.
4. **`consolidate`** merges every `.npz` into one gzipped AnnData object.

```bash
uv run --no-sync python -m gtex_liver_cells.portal --tissue Liver --out data/liver_slides.csv
uv run --no-sync python -m gtex_liver_cells.cohort --slides data/liver_slides.csv --out data/liver_cohort.csv
uv run --no-sync python -m gtex_liver_cells.centroids --parquet-dir /path/to/datasets_pq_h5ad \
    --out-dir data/centroids --slides data/liver_cohort.csv --workers 16
uv run --no-sync python -m gtex_liver_cells.consolidate --centroids-dir data/centroids \
    --cohort data/liver_cohort.csv --out data/liver_cohort.h5ad
```

## Data provenance

Two different GTEx documents describe liver slides, and they disagree on how
many there are.

| Source | Rows | Liver slides |
|---|---|---|
| `GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt` | 22,951 | 242 |
| GTEx Portal histology table | 25,713 | 610 |

The sample-attributes file lists only samples carrying assay annotations, so
filtering it on `SMTS == "Liver"` yields 242 slides. The remaining 368 liver
slides in the Portal table do not appear in that file under any tissue — they
are histology-only slides with no molecular annotation. The 242-slide set is a
strict subset of the 610-slide set, and this pipeline uses the larger one.

The public source is the GTEx Portal histology API:

```
https://gtexportal.org/api/v2/histology/image?page=0&itemsPerPage=1000
```

It returns paginated JSON with exactly the columns the lab's `GTEx Portal.csv`
carries. Verified against that file for liver: identical slide IDs, notes, and
category sets, 610 out of 610. **There is no bulk CSV** — the public GCS bucket
contains only the annotation files, and the Portal file catalog lists no
histology entries — which is why this repository ships a fetcher instead of a
download link.

Two details worth knowing: 7 of the 610 liver slides have no pathology note and
no HistoPlus file in the older source; extraction now warns about any requested
slide whose `cell_types.gpd` is absent and counts it as missing. 31 slides are
flagged `Hidden` in the Portal, exactly one of which is liver
(`GTEX-1269W-1826`), and it falls in `other` regardless.

### The cell-type source

Cell segmentation and cell-type labels come from the lab's Parquet cell-type
source on CEMM, not from the public Portal. Per slide there is one directory
`<TISSUE_ID>/` holding `cell_types.gpd` — Parquet despite the extension (file
magic `PAR1`) — with four columns:

| Column | Type | Meaning |
|---|---|---|
| `geometry` | binary | WKB cell polygon, Arrow extension `geoarrow.wkb` |
| `class` | string | CellViT cell-type label |
| `prob` | double | Confidence for the predicted class |
| `cell_id` | int64 | Per-cell identifier |

Only `geometry`, `class` and `prob` are read. The sibling
`cell_types_features.h5ad` (768-d CellViT embeddings) is deliberately **not**
used: embeddings are unnecessary for the tutorial and would dominate the object
size. `cell_id` matches the h5ad row order exactly on the one slide where both
were read, so it is available as a join key if embeddings are ever added, while
`library_id` (constant `"cell_types"`) and `tile_id` (0…n-1 per slide) carry no
information and are never stored.

The class vocabulary varies per slide — both measured liver slides carry 12
classes, while a skin slide carried 13 (adding `Minor Stromal Cell`) — so
consolidation unions the per-slide vocabularies instead of assuming a fixed
taxonomy.

Two caveats belong with any downstream claim:

- **The labels are model output on a mismatched taxonomy.** The CellViT
  taxonomy is pan-cancer and has no hepatocyte or Kupffer-cell class, so liver
  cells are necessarily mapped onto generic epithelial and stromal labels;
  `Epithelial` reaching 70% of `GTEX-14AS3-0126` is most plausibly hepatocytes
  under a label that does not fit them. `Cancer cell` is reported for 6.6-12.6%
  of these GTEx livers although the donors have no cancer diagnosis.
- **Cell counts are smaller than the older HistoPlus source.** Paired over 8
  liver slides, CellViT finds 0.651-0.906x as many cells (median 0.818; 293,691
  versus 361,514 cells per slide on average), so the earlier "10x fewer cells"
  estimate was wrong — it compared a skin slide against a liver slide. Expect
  roughly 36M cells for the 124-slide healthy/cirrhotic subset and 179M for all
  610 labeled slides.

## Cohort definition

`cohort` labels every slide from the curated pathology categories and, for
uncategorized slides, the free-text pathology note:

- **`cirrhosis`** — the curated categories contain `cirrhosis`. 58 slides.
- **`healthy`** — either the curator assigned `no_abnormalities` or
  `clean_specimens` (6 slides), or the slide has no curated category, a
  non-empty note, no finding term after negation stripping, and no preservation
  warning (60 slides). Total 66.
- **`other`** — everything else. 486 slides.

That gives a 124-slide healthy/cirrhotic subset. An empty note is treated as
unknown, never as healthy. Every row carries a `label_reason` so the
classification is auditable, and preservation warnings cover autolysis in
either spelling (`autolys` / `autolyz`), poor preservation, and poor fixation.

Cirrhosis is heterogeneous: 21 slides are also steatotic, 18 fibrotic, 10
hepatitic, 8 nodular, 8 congested, and 6 inflamed, while only 8 carry
cirrhosis alone. That is recorded as a property of the cohort rather than used
to filter it.

Known limitations:

- The healthy label is a documented text heuristic, not a pathologist's call.
  The 60 inferred controls deserve review.
- Autolysis score (`SMATSSCR`) exists only for the 242 assay-annotated slides,
  covering just 8 of the 58 cirrhotic ones, so tissue-degradation QC is not
  available cohort-wide.
- `GTEX-1JJ6O-0826` carries the note "liver, not skin", a tissue-label
  correction. It is labeled healthy and is worth excluding or reviewing.

## About prob

Each cell carries a `class` label and a single scalar `prob` — the confidence
for the predicted class, consistent with an argmax softmax. There is no
per-class breakdown and no companion file carrying one, so full probabilities,
soft labels, or prediction entropy cannot be recovered without re-exporting
from the upstream model.

Measured over real slides: min 0.148, median 0.782, max 0.993 with 15.8% of
cells below 0.5 on `GTEX-14AS3-0126`; median 0.696 with 25.7% below 0.5 on
`GTEX-13OVJ-1026`. Extraction stores `prob` as a float32 `obs` column and
**applies no threshold by default**, because silently dropping a quarter of the
cells would be a hidden analysis decision. Filter downstream if you want one.

## AnnData layout

The consolidated object uses a sparse one-hot `X` over the union of the
per-slide class vocabularies (12 classes on each measured liver slide; the
union is what lands in `var`), `cell_type` in `obs`, float32 centroids in
`obsm["spatial"]`, no per-cell identifier, and gzip compression.

Measured by building real objects from three slides (1,107,515 cells) and
writing them to disk:

| Layout | gzip B/cell | raw B/cell | 34M cells | 160M cells | write 160M |
|---|---|---|---|---|---|
| `X=None`, string index | 49.2 | 71.0 | 1.67 GB | 7.87 GB | 71 s |
| one-hot f32, string index | 51.0 | 83.1 | 1.73 GB | 8.16 GB | 80 s |
| **one-hot f32, default index** | **34.5** | 64.1 | **1.17 GB** | **5.52 GB** | 76 s |
| `X=None`, default index | 32.7 | 55.1 | 1.11 GB | 5.23 GB | 64 s |

Two things fall out of those numbers. The one-hot costs only ~1.8 B/cell after
gzip, because its CSR indices are a stride-1 run and the data are all ones —
the 16 B/cell figure is the in-memory cost, which matters for job RSS, not for
file size. The index matters more: a global per-cell identifier costs ~16.5
B/cell, so none is stored. anndata assigns its own sequential `obs_names`
regardless (`"0"`, `"1"`, …), but sequential digits compress to almost nothing,
and cell identity is recoverable as `(slide_id, row offset)`.

## Running it

On a laptop, the portal and cohort stages and the full test suite run without a
cluster:

```bash
uv sync
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m gtex_liver_cells.portal --tissue Liver --out data/liver_slides.csv
uv run --no-sync python -m gtex_liver_cells.cohort --slides data/liver_slides.csv --out data/liver_cohort.csv
```

### CEMM

CEMM has no conda, no `uv` in its EasyBuild module set, and a system `python3`
of 3.6.8, but its login node reaches PyPI. Install uv into `~/.local/bin` once
and let `uv sync` build `.venv` in the checkout from `uv.lock`, which keeps the
lockfile the single source of environment truth on both laptop and cluster:

```bash
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
cd "$REPO_DIR" && uv sync
```

Everything runs in place: the Parquet source is read on CEMM, so no data is
transferred between clusters and nothing passes through a local machine. Set
the paths once, then submit both wrappers from the output directory so SLURM
writes its logs into `logs/`:

```bash
export REPO_DIR="$HOME/rep/gtex-liver-cells"
export OUT_DIR=/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_liver_cells
export PARQUET_DIR=/path/to/datasets_pq_h5ad      # one <TISSUE_ID>/cell_types.gpd per slide
export CENTROIDS_DIR="$OUT_DIR/centroids"
export COHORT_CSV="$REPO_DIR/data/liver_cohort.csv"

mkdir -p "$OUT_DIR/centroids" "$OUT_DIR/logs" && cd "$OUT_DIR"

sbatch --export=ALL,PARQUET_DIR=$PARQUET_DIR,CENTROIDS_DIR=$CENTROIDS_DIR,COHORT_CSV=$COHORT_CSV \
    "$REPO_DIR/sbatch/extract_centroids.sbatch"

sbatch --export=ALL,CENTROIDS_DIR=$CENTROIDS_DIR,COHORT_CSV=$COHORT_CSV,OUT_DIR=$OUT_DIR,FILTER_TO_COHORT=1 \
    "$REPO_DIR/sbatch/consolidate_anndata.sbatch"
```

`FILTER_TO_COHORT=1` keeps only slides listed in the cohort CSV, which is how
all 603 available slides narrow to the 124-slide healthy/cirrhotic subset. Both
wrappers take their paths from the environment, so `REPO_DIR`, `PARQUET_DIR`,
`CENTROIDS_DIR`, `COHORT_CSV` and `OUT_DIR` have to be exported in the
submitting shell; the wrappers default only `REPO_DIR`, to
`/home/sschindler/rep/gtex-liver-cells`.

The wrappers call the project venv directly rather than `uv run`, on purpose:
`uv run` re-syncs the environment on every invocation, which is not what a
batch job should spend its time on.

### Outputs

Everything lands in `$OUT_DIR`:

```
gtex_liver_cells/
├── centroids/<TISSUE_ID>.npz     # per-slide intermediate, ~14 B/cell (~4 MB per slide)
├── gtex_liver_cells.h5ad         # the consolidated object, ~1.3 GB for 36M cells
├── liver_cohort.csv              # the cohort actually used
├── provenance.json               # commit, UTC time, inputs, slide counts, classes, cells
└── logs/                         # SLURM stdout/stderr
```

`provenance.json` is written by the consolidation job, which also copies the
cohort CSV next to the object, so the directory is self-describing without
duplicating the repository.

### Removed from this repository

| Removed | Why |
|---|---|
| `liver.py` | rsync downloader; HistoPlus files are now a given input |
| `downloader.py` | IDC whole-slide-image download, out of scope |
| `cohorts.py`, `cohorts_notebook.py`, `__marimo__/` | superseded by `gtex_liver_cells.cohort` |
| `meta.py`, `path.py`, `vocab.py`, `main.py` | scratch exploration |
| `path_terms.txt`, `liver_wsis.csv` | intermediates of the earlier 242-slide approach |

## License

The code is MIT-licensed (`LICENSE`). The data files this pipeline produces —
`data/liver_slides.csv`, `data/liver_cohort.csv`, and any AnnData or centroid
files — are CC BY 4.0 (`LICENSE-DATA`), with the GTEx Consortium to be credited
as the source of the underlying annotations. Note that the cell-class labels
are model output with per-cell confidences in `prob`, not curated ground truth.
