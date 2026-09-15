# gtex-liver-cells

A self-contained pipeline that builds a cell-resolved GTEx liver
histopathology dataset: cell centroids, cell-class labels, and coordinates for
the liver slides of the GTEx Portal histology cohort, consolidated into one
AnnData object.

The resulting object is what downstream analyses consume — for example the
spatial-topology workflows in
[topocyte](https://github.com/rendeirolab/topocyte), which load it through the
sibling `topocyte_data` repository.

## The dataset

The consolidated object holds **126,219,124 cells across 594 slides** in **14
cell classes**, as one gzip-compressed `.h5ad` of about **4.7 GB**.

| Slot | Contents |
|---|---|
| `X` | sparse one-hot (CSR) indicator of `cell_type`, shape `(126219124, 14)` |
| `obs` | `slide_id`, `cell_type`, `prob`, plus `Subject ID`, `Age Bracket`, `Sex`, `Hardy Scale`, `label` |
| `obsm["spatial"]` | float32 `(N, 2)` cell centroids, in slide pixels |
| `var` | the 14 class names |
| `uns` | `n_slides`, `n_cells`, `class_names`, `slide_ids`, `metadata_columns`, `created` |

```python
import anndata as ad

adata = ad.read_h5ad("gtex_liver_cells.h5ad", backed="r")  # backed avoids loading X
adata.shape                       # (126219124, 14)
adata.obs[["slide_id", "cell_type", "prob", "label"]].head()
adata.obsm["spatial"]             # float32 centroids
```

There is no per-cell identifier: `obs_names` is anndata's sequential `"0"`,
`"1"`, …, and a cell is addressed as `(slide_id, row offset within that
slide)`. `prob` is the model confidence for the assigned `cell_type`, stored
unfiltered (see [Findings](#findings)).

## Usage

### Set up on CEMM

CEMM has no conda and no `uv` in its EasyBuild module set, but its login node
reaches PyPI. `scripts/setup_cemm.sh` installs uv if needed, creates the
project venv, and syncs it from `uv.lock`:

```bash
export REPO_DIR="$HOME/rep/gtex-liver-cells"
cd "$REPO_DIR"
LIVERCELLS_VENV="$REPO_DIR/.venv" bash scripts/setup_cemm.sh
```

Set `LIVERCELLS_VENV` to `$REPO_DIR/.venv` on purpose: the sbatch wrappers run
`"$REPO_DIR/.venv/bin/python"`. Without the override the script builds a
throwaway venv under `${TMPDIR}` that the jobs would not use.

### Build the full cohort object

`scripts/step_1.sh` extracts one centroid `.npz` per slide from the Parquet
cell-type source; `scripts/step_2.sh` consolidates every `.npz` into one
AnnData object. Run them in order:

```bash
bash scripts/step_1.sh    # extract centroids (SLURM)
bash scripts/step_2.sh    # consolidate the full cohort (SLURM)
```

Both submit the wrappers in `sbatch/`, which take every path from the
environment (`PARQUET_DIR`, `CENTROIDS_DIR`, `COHORT_CSV`, `OUT_DIR`,
`REPO_DIR`) and default only `REPO_DIR`. Output goes to
`/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_liver_cells`
(override with `OUT_DIR`). The result is 594 slides and 126,219,124 cells.

### Build the healthy-vs-cirrhosis subset

`scripts/step_3.sh` keeps only the `healthy` and `cirrhosis` rows of the
labeled cohort and consolidates those slides into their own object. It reuses
the centroids from step 1 and writes to a sibling directory
(`${OUT_DIR}_healthy_cirrhosis`):

```bash
bash scripts/step_3.sh
```

That subset is **123 slides and 30,836,186 cells** (66 healthy, 57 cirrhosis).
The labeled cohort has 124 healthy/cirrhotic slides, but one of them
(`GTEX-1I1HK-1126`, cirrhosis) has no `cell_types.gpd` in the Parquet source.

### Run stages on a laptop

The portal and cohort stages and the full test suite run without a cluster:

```bash
uv sync
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m gtex_liver_cells.portal --tissue Liver --out data/liver_slides.csv
uv run --no-sync python -m gtex_liver_cells.cohort --slides data/liver_slides.csv --out data/liver_cohort.csv
```

### Outputs

```
gtex_liver_cells/
├── centroids/<TISSUE_ID>.npz     # per-slide intermediate, ~14 B/cell
├── gtex_liver_cells.h5ad         # the consolidated object (~4.7 GB for the full cohort)
├── liver_cohort.csv              # the cohort actually used
├── provenance.json               # commit, UTC time, inputs, slide counts, classes, cells
└── logs/                         # SLURM stdout/stderr
```

`provenance.json` is written by the consolidation job, which also copies the
cohort CSV next to the object, so the output directory is self-describing
without duplicating the repository.

## The data

### Cohort

`cohort` labels all 610 liver slides of the GTEx Portal histology table as
`cirrhosis` (58), `healthy` (66), or `other` (486). It uses the curated
pathology categories first and the free-text note only for uncategorized
slides; an empty note is treated as unknown, never healthy. The 124
healthy/cirrhotic slides form the analysis subset. The healthy label is a
documented text heuristic, not a pathologist's call.

The slide table comes from the GTEx Portal histology API, which `portal`
fetches (there is no bulk CSV).

### Cell-type source

Cell segmentation and cell-type labels come from the lab's Parquet source on
CEMM, one `<TISSUE_ID>/cell_types.gpd` per slide — Parquet despite the
extension (file magic `PAR1`):

| Column | Type | Meaning |
|---|---|---|
| `geometry` | binary | WKB cell polygon, Arrow extension `geoarrow.wkb` |
| `class` | string | CellViT cell-type label |
| `prob` | double | confidence for the predicted class |
| `cell_id` | int64 | per-cell identifier |

Only `geometry`, `class` and `prob` are read. The sibling
`cell_types_features.h5ad` (768-d embeddings) is deliberately not used;
`cell_id`, `library_id` and `tile_id` are not carried into the output. The
class vocabulary varies per slide, so consolidation unions the per-slide
vocabularies — 14 classes across this cohort.

## Findings

### The production run

Built on CEMM from commit `adaac66`; recorded in `provenance.json` (created
2026-09-15).

- 610 slides requested, **594 extracted**, 594 consolidated. The other 16 have
  no `cell_types.gpd` (15 `other`, 1 cirrhosis: `GTEX-1I1HK-1126`).
- **126,219,124 cells** in 14 classes; object 5,003,259,637 bytes (~4.7 GB).
- md5 `b77dbf1a53b0a83b4ee2f2f3e65a5de1`.
- Extraction was clean: `extracted=594 skipped=0 failed=0 missing=16` — no
  slide failed.

Cell composition of the full object:

| class | cells | share |
|---|---:|---:|
| Epithelial | 72,230,725 | 57.23% |
| Cancer cell | 17,403,079 | 13.79% |
| Fibroblasts | 16,771,623 | 13.29% |
| Apoptotic Body | 9,270,568 | 7.34% |
| Lymphocytes | 3,301,465 | 2.62% |
| Neutrophils | 2,530,170 | 2.00% |
| Macrophages | 2,126,493 | 1.68% |
| Muscle Cell | 1,734,764 | 1.37% |
| Endothelial Cell | 346,925 | 0.27% |
| Red blood cell | 258,822 | 0.21% |
| Plasmocytes | 210,599 | 0.17% |
| Eosinophils | 33,601 | 0.03% |
| Mitotic Figures | 234 | <0.01% |
| Minor Stromal Cell | 56 | <0.01% |

### Confidence (`prob`)

`prob` is a single scalar per cell — the confidence for the assigned class —
with no per-class breakdown and no companion file. Object-wide it runs min
0.105, median 0.709, max 0.996, with no NaN. Extraction stores it unfiltered;
filtering is an analysis decision, not a silent default. On the design slides,
`GTEX-14AS3-0126` had median 0.782 with 15.8% of cells below 0.5, and
`GTEX-13OVJ-1026` median 0.696 with 25.7% below 0.5.

### Labels are model output on a mismatched taxonomy

`class` is CellViT output on a pan-cancer taxonomy with **no hepatocyte or
Kupffer-cell class**, so liver cells are necessarily mapped onto generic
epithelial and stromal labels. `Epithelial` is 57.2% of the object and is most
plausibly hepatocytes under a label that does not fit them; `Cancer cell` is
13.8% of these GTEx livers although the donors have no cancer diagnosis. Treat
the labels as model predictions, not histology.

### Cell counts vs the older HistoPlus source

Paired over 8 liver slides, CellViT finds 0.651-0.906x as many cells (median
0.818; 293,691 vs 361,514 cells per slide on average). An earlier "10x fewer
cells" claim was wrong — it compared a skin slide against a liver slide.

### Verification

- Every one of the 594 `.npz` files matches the object per-slide, with zero
  mismatches; both total 126,219,124 cells.
- `GTEX-14AS3-0126` carries 349,884 cells in the Parquet source, in its `.npz`,
  and in the object — an exact match, with a 12-class vocabulary.
- `X` is a true one-hot (all data `1.0`, every row exactly one entry), and
  `obsm["spatial"]` is finite float32.

## History

Removed from this repository:

| Removed | Why |
|---|---|
| `liver.py` | rsync downloader; the Parquet source is a given input |
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
