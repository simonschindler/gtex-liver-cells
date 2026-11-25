# One AnnData per GTEx Tissue — Design

Date: 2026-11-25

## Purpose

The repository builds one cell-resolved AnnData object for the GTEx liver
cohort. The user wants the same object for **every GTEx tissue**, one object
per tissue, so that each organ's cells can be looked at in the browser viewer
in `topocyte-analyses` (`gtex-liver-viewer/`). Viewing is the first use.

The viewer is being adapted in parallel, on its own branch of
`topocyte-analyses`, against the output layout fixed in this document. That
layout is therefore a contract: changing it means changing the viewer too.

## Relationship to the earlier designs

This builds on `2026-09-14-gtex-liver-cohort-design.md` and
`2026-09-15-parquet-centroids-cemm-design.md`. Carried over unchanged and not
revisited: the Parquet cell-type source and its reader (P1, P2), the per-slide
npz intermediate (P5), the AnnData layout (D5), `prob` stored unfiltered (D4),
and CEMM execution from the uv-managed `.venv` (P3, P4).

The liver pipeline keeps working exactly as it does today:
`scripts/step_1.sh` to `step_3.sh`, `sbatch/extract_centroids.sbatch`,
`sbatch/consolidate_anndata.sbatch` and `cohort.py`. The existing liver object
in `topocyte_data/gtex_liver_cells/` is not touched by anything in this design.

## Verified background

Measured during design, not assumed:

- **The Portal table.** `portal.py` without `--tissue`, run 2026-11-25:
  25,713 slides with unique `Tissue Sample ID`s and 41 distinct `Tissue`
  (`tissueSiteDetail`) strings that are **40 tissues**: `"Esophagus - Mucosa "`
  (3 slides, trailing space) is the same tissue as `"Esophagus - Mucosa"`
  (960 slides). No other column has leading or trailing whitespace. The API
  has no coarser organ field than `tissueSiteDetail`.
- **Tissue sizes.** From 38 slides (Cervix - Ectocervix) to 1,001 (Skin - Sun
  Exposed (Lower leg), Muscle - Skeletal); liver has 610. Fifteen tissues
  have more than 850 slides.
- **Slugs.** The slug rule below maps the 40 tissue names to 40 distinct
  slugs, from `adipose_subcutaneous` to `vagina`.
- **Liver as the yardstick.** The liver production run found 594 of 610
  slides in the Parquet source and gave 126,219,124 cells, a 4.7 GB h5ad and
  ~14 B/cell of npz. If every tissue were as dense as liver, all 40 would be
  about 5 billion cells: ~200 GB of h5ad and ~75 GB of npz. Real densities
  differ by tissue (adipose is sparse, lymphoid tissue dense) and are unknown
  until the run.
- **The viewer's reads.** `gtex-liver-viewer/serve.py` opens the h5ad with
  h5py, not anndata, and reads `obs/slide_id` and `obs/cell_type` as
  categoricals (`codes`, `categories`), `obsm/spatial` as level-0 pixel
  centroids, and each slide's cells as one contiguous block of rows. Donor
  fields come from a CSV beside the h5ad. Everything else it uses (density
  deciles, pathology groups, row ranges) comes from liver-only study files.
- **Memory, measured** on synthetic data (10M cells in 500 slides, the locked
  anndata 0.13.2 and pandas 3.0.5, gzip write). Today's `consolidate` peaks at
  about **330 B/cell** with its five metadata columns: it builds a Python
  string per cell for `cell_type`, an object array of slide IDs, and, through
  `cohort.reindex` on N slide IDs, an object column per metadata field per
  cell. Building every column from integer codes instead (T5) peaks at about
  **210 B/cell** with eight metadata columns, plus ~16 B/cell of loaded npz.
  Most of what remains, ~110 B/cell, is anndata itself turning the default
  index into one Python string per cell inside `AnnData()`; pre-building the
  index as Arrow strings does not avoid it (also measured). Liver's 126M cells
  at ~330 B/cell is ~42 GB, consistent with its 128G run succeeding.
- **The liver object's encoding** (read with h5py): categoricals are groups of
  `codes` and `categories` (`slide_id` codes int16 for 594 slides, `cell_type`
  int8), `prob` float32, the index a `nullable-string-array`, and `X` CSR with
  int32 `indices` and `indptr` and float32 `data`, all gzip.

Checked by the user on CEMM, 2026-11-25:

- **One Parquet root for every tissue.**
  `/nobackup/lab_rendeiro/projects/histopath/processed/histopathology/datasets_pq_h5ad/`
  holds 25,429 `GTEX-*` slide folders against 25,713 Portal slides, so all
  tissues share liver's root and layout (`<TISSUE_ID>/cell_types.gpd`). How
  many folders hold a `cell_types.gpd` is counted per tissue at launch.
- **`mediumq`**: 11 nodes, at least 24 CPUs and about 1.5 TB of memory each
  (`sinfo -o "%P %m %c %D"`: 1,546,900 MB).
- **`/nobackup`**: 6.4 PB, 4.9 PB free, so ~300 GB of output needs no cleanup
  step.

## Goals

- One AnnData per GTEx tissue site, 40 in total, each in its own
  self-contained folder that can be copied to a laptop on its own.
- The same AnnData layout as the liver object, so the viewer's reading code
  carries over.
- `obs` metadata is the Portal table's columns; nothing liver-specific.
- One sbatch array script and one launcher script; everything else in Python,
  tested.
- A single organ can be rerun on its own; an interrupted run resumes without
  redoing finished slides.
- Every organ records which of its slides had no source and which failed.

## Non-goals

- Healthy/disease labels outside liver. `cohort.py` stays liver-only.
- One combined object across organs (a later `anndata.concat` if needed).
- Changes to the viewer; they happen on the `topocyte-analyses` branch.
- Renaming the repository or the `gtex_liver_cells` package.
- Embeddings (`cell_types_features.h5ad`) or any `prob` threshold.
- Replacing the existing liver object or the liver scripts.

## Anti-goals

What would make this a failure even if it technically ran:

- A silently partial object: a missing or failed slide that shows up nowhere
  but in a scrolled-off log.
- Any change to what the liver pipeline produces, or a failing existing test.
- Layout drift from the contract below, which the viewer branch codes against.
- All 40 tasks' extraction workers reading the shared filesystem at once.

## Approaches

Three ways to run 40 organs on SLURM were weighed; the user picked the first.

| | Approach | Trade-off |
|---|---|---|
| **A (picked)** | **One array job, one task per organ**: each task extracts its organ's slides and then consolidates them. | One new sbatch file and one submit; rerunning an organ is `--array=<i>`. Every task gets the same CPUs and memory although consolidation uses one core, and memory is sized for the largest organ. |
| B | Two chained arrays, extraction then consolidation, task *i* of the second waiting for task *i* of the first (`--dependency=aftercorr`). | Right-sized resources per stage, but two sbatch files, dependency wiring, and two submits per rerun. |
| C | A launcher loop submitting the existing two wrappers per organ (80 jobs, `afterok` pairs). | No new array logic, but both wrappers hard-code liver names (`gtex_liver_cells.h5ad`, `liver_cohort.csv`) and need edits anyway; 80 loose jobs are awkward to watch or cancel. |

Dropped before the comparison: one job looping over all organs in series
(days of runtime, one failure stops everything, `mediumq` caps jobs at two
days).

## Output layout (the contract with the viewer)

```
<ROOT>/                                  cluster: /nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_tissues
├── gtex_portal_slides.csv               the Portal table this run used (fetched once)
├── tissues.csv                          manifest: index, slug, tissue, n_slides
├── logs/                                SLURM stdout/stderr, gtex_tissue_<job>_<task>.{out,err}
└── <slug>/                              one per tissue
    ├── portal_slides.csv                input: every Portal row of this tissue
    ├── gtex_<slug>_cells.h5ad           the object
    ├── slides.csv                       one row per slide in the object, with row ranges
    ├── provenance.json                  how it was built, coverage
    └── centroids/<TISSUE_ID>.npz        per-slide intermediate
```

`<ROOT>` is a new sibling of `topocyte_data/gtex_liver_cells/`, so liver's
new-layout object (`gtex_tissues/liver/gtex_liver_cells.h5ad`) never collides
with the existing one.

**Slug.** The tissue name with surrounding whitespace removed, lowercased,
each run of characters outside `[a-z0-9]` replaced by `_`, and leading or
trailing `_` trimmed: `Liver` → `liver`, `Skin - Sun Exposed (Lower leg)` →
`skin_sun_exposed_lower_leg`, `Adipose - Visceral (Omentum)` →
`adipose_visceral_omentum`.

**`gtex_<slug>_cells.h5ad`**, written by anndata with gzip, in the liver
object's encoding:

| Slot | Contents |
|---|---|
| `obs/slide_id` | categorical; categories are the object's slide IDs in ascending order; each slide's cells form one contiguous block of rows, the blocks in ascending slide-ID order |
| `obs/cell_type` | categorical; categories are the sorted union of the organ's per-slide class vocabularies |
| `obs/prob` | float32 confidence for the assigned class |
| `obs/<Portal column>` | one categorical each: `Tissue`, `Subject ID`, `Sex`, `Age Bracket`, `Hardy Scale`, `Pathology Categories`, `Pathology Notes`, `Hidden`; no `label` |
| `obsm/spatial` | float32 `(N, 2)` centroids in level-0 slide pixels |
| `X` | CSR one-hot float32 indicator of `cell_type`; `var` index is the class names |
| `uns` | `n_slides`, `n_cells`, `class_names`, `slide_ids`, `metadata_columns`, `created` |

`obs_names` stay anndata's default sequential `"0"`, `"1"`, … (D5).

Integer widths are not part of the contract: pandas picks a categorical's
codes dtype from its category count (int8 for up to 126 categories, int16
above, as for liver's 594 slides), so an organ with 126 or fewer slides gets
int8 `obs/slide_id/codes`. Readers take whatever dtype h5py returns. The
viewer branch has been told.

**`slides.csv`**: one row per slide in the h5ad, in h5ad row order, with the
columns of `portal_slides.csv` — `Tissue Sample ID`, `Tissue`, `Subject ID`,
`Sex`, `Age Bracket`, `Hardy Scale`, `Pathology Categories`, `Pathology
Notes`, `Hidden` — followed by `n_cells`, `row_start` and `row_end` (end
exclusive). The first `row_start` is 0, each next `row_start` is the previous
`row_end`, and the last `row_end` is the object's `n_obs`. Slides with no cell
data are not listed; `provenance.json` lists them.

**`provenance.json`**: today's fields (`created`, `repository_commit`,
`centroids_dir`, `cohort_csv`, `slides_requested`, `slides_extracted`,
`slides_used`, `slides_missing`, `n_cells`, `class_names`), plus
`missing_slide_ids` (every requested slide with no npz), and, from the
per-organ build, `tissue`, `slug`, `parquet_dir`, `no_source_slide_ids`
(no `cell_types.gpd`) and `failed_slide_ids` (a source that failed to
extract). `missing_slide_ids` is the union of the last two.

**`tissues.csv`**: one row per tissue, sorted by slug, with columns `index`
(0-based, the SLURM array index), `slug`, `tissue` (whitespace-stripped name),
`n_slides` (Portal slides for that tissue) and `n_with_source` (those with a
`cell_types.gpd` under the Parquet root when `prepare` ran). The viewer does
not read it.

## Decisions

### T1 — An organ is a Portal tissue site

`tissueSiteDetail` is the only tissue field the Portal has, and it is what
`portal.py` already writes as `Tissue`. Skin - Sun Exposed and Skin - Not Sun
Exposed stay separate; merging objects later is a cheap concat, while
splitting a merged one means rebuilding. Names are whitespace-stripped before
grouping, which folds the three `"Esophagus - Mucosa "` slides into
Esophagus - Mucosa (963 slides).

### T2 — Liver is rebuilt in the new layout

All 40 tissues, liver included, are built the same way, so the viewer can
treat every organ alike. The new liver object has the Portal columns and no
`label`; the existing labeled liver object stays where it is for the studies
that use it.

### T3 — Portal metadata only

`obs` carries the eight Portal columns listed above (the user: "just carry
over the portal metadata"). They are slide-level values stored per cell as
categoricals, so they cost one small integer code per cell.

### T4 — One Python entry point per stage of the organ run

A new module, `gtex_liver_cells.tissues`, has two subcommands:

```
python -m gtex_liver_cells.tissues prepare --slides <ROOT>/gtex_portal_slides.csv --root <ROOT> \
    --parquet-dir <PARQUET_DIR>
python -m gtex_liver_cells.tissues array --root <ROOT>
python -m gtex_liver_cells.tissues build --root <ROOT> (--index I | --slug S) \
    --parquet-dir <PARQUET_DIR> --workers N [--overwrite]
```

- **`prepare`** reads the full Portal table, strips `Tissue`, refuses
  duplicate slide IDs and slug collisions (error naming them), writes
  `<ROOT>/<slug>/portal_slides.csv` for every tissue and `<ROOT>/tissues.csv`,
  and prints a per-tissue coverage table: Portal slides against slides with a
  `cell_types.gpd` under `--parquet-dir`. It overwrites those files on every
  run and never touches `centroids/`, the h5ad, `slides.csv` or
  `provenance.json`. Counting the sources is 25,713 existence checks on the
  shared filesystem, run once on the login node.
- **`array`** prints the SLURM array spec of every tissue with at least one
  source, comma-separated (`0,1,2,4,...`), and nothing if none has one.
- **`build`** looks up one tissue in `tissues.csv` and runs the whole organ:
  extraction of its `portal_slides.csv` into `<slug>/centroids/` with the
  existing `centroids` functions (existing npz are skipped unless
  `--overwrite`), then consolidation of those npz, filtered to the organ's
  slides, into the h5ad, `slides.csv` and `provenance.json`.

Keeping the orchestration in Python rather than bash makes it testable end to
end on synthetic Parquet fixtures, and keeps the sbatch script thin.

### T5 — Consolidation builds `obs` and `X` from integer codes

`consolidate` keeps its output identical but stops creating per-cell Python
objects:

- Entries are ordered by slide ID explicitly, so the `slide_id` categories are
  ascending and the row blocks follow the same order by construction.
- `slide_id` codes are each entry's index repeated over its cells;
  `cell_type` codes come from remapping each slide's integer labels through a
  per-slide lookup into the union vocabulary; each metadata column is a
  per-slide value table whose codes are taken per cell. All become
  `pd.Categorical.from_codes`, with categories in the same sorted order the
  current `pd.Categorical(values)` produces.
- `X` is built directly as CSR from `indptr = arange(N + 1)`,
  `indices = cell_type codes` and `data = 1.0`, with int32 `indices` and
  `indptr` while N fits, as the liver object has (checked: both int32, data
  float32, gzip).
- The loaded npz stay in memory until the object is built; the measured peak
  below includes them.

Measured peak: about 225 B/cell including the loaded npz, against about 330
today, so the 256G request (T9) holds an organ of roughly 1.1B cells. A
1,000-slide tissue at liver density (~210M cells) needs ~50 GB.

The existing behaviours stay: a missing metadata column becomes a single `""`
category; a slide without a cohort row gets `""`; legacy npz without `prob`
get NaN.

### T6 — consolidate gains two options

- `--metadata-columns COL [COL ...]`: the cohort columns joined into `obs`.
  The default is today's `META_COLUMNS`, so the liver steps are unchanged.
- `--slides-out PATH`: write the per-slide table described above, from every
  cohort column of the slides used plus `n_cells`, `row_start`, `row_end`.

`provenance.json` gains `missing_slide_ids` everywhere; `build` adds its
organ fields (see the layout).

### T7 — Writes are atomic per file

The h5ad is written to a temporary name in the same folder and renamed into
place, then `slides.csv`, then `provenance.json`. A task that dies mid-write
never leaves a truncated object under the final name, and a present
`slides.csv` means its h5ad is complete.

### T8 — Exit status

`build` exits 0 when the object was written and no slide failed. It exits 1
when any slide failed to extract (the object is still written from the
others, and the failures are listed in `provenance.json` and the log), when no
slide of the organ has a source, or on any consolidation error. A slide with no
`cell_types.gpd` is not a failure: it is warned about and listed, as in the
liver run. So a FAILED array task means "look at this organ", not necessarily
"no object".

### T9 — The sbatch script

`sbatch/build_tissues.sbatch`, in the house style of the two existing
wrappers:

- `--account=lab_rendeiro`, `--partition=mediumq`, `--time=12:00:00`,
  `--cpus-per-task=8`, `--mem=256G`, mail to the existing address with
  `--mail-type=ALL` (SLURM mails per array, not per task).
- `--output=logs/gtex_tissue_%A_%a.out`, `--error=logs/gtex_tissue_%A_%a.err`,
  relative to the submit directory, which is `<ROOT>`.
- Takes `PARQUET_DIR` and `ROOT` from the environment, required; `REPO_DIR`
  defaults to the CEMM checkout as in the other wrappers, and `PYTHON` to
  `$REPO_DIR/.venv/bin/python` (the script tests point it at the test
  interpreter).
- Runs `"$REPO_DIR/.venv/bin/python" -m gtex_liver_cells.tissues build --root
  "$ROOT" --index "$SLURM_ARRAY_TASK_ID" --parquet-dir "$PARQUET_DIR"
  --workers "${SLURM_CPUS_PER_TASK:-8}"` with `PYTHONPATH=src`, and prints
  node, tissue and paths first.

### T10 — The launcher

`scripts/all_tissues.sh`, alongside `step_1.sh` to `step_3.sh`:

1. Sets `REPO_DIR`, `PARQUET_DIR` (the liver run's root) and
   `ROOT=/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_tissues`,
   and creates `<ROOT>/logs`.
2. Fetches the Portal table into `<ROOT>/gtex_portal_slides.csv` with
   `portal.py` if that file is absent. This runs on the login node, which has
   internet (compute nodes may not). The cached file pins the run's input;
   deleting it fetches a fresh table next time.
3. Runs `tissues prepare`, whose coverage table is the first look at whether
   every tissue's slides are under the Parquet root.
4. Takes the array from `ARRAY` if set, else from `tissues array` (every
   tissue with a source), names the tissues left out for having none, and
   stops without submitting if the array is empty.
5. `cd "$ROOT"` and `sbatch --array="$ARRAY%${MAX_PARALLEL:-8}"
   --export=ALL,PARQUET_DIR=...,ROOT=... sbatch/build_tissues.sbatch`.

`ARRAY=17 bash scripts/all_tissues.sh` reruns one organ; `tissues.csv` maps
slugs to indices. `MAX_PARALLEL` (default 8) caps concurrent tasks to spare
the shared filesystem.

A tissue that is only partly covered still builds and its task exits 0, so the
coverage table and each organ's `provenance.json` are where a gap shows, not
`sacct`.

## Error handling

| Condition | Behaviour |
|---|---|
| Portal table missing or unreadable in `prepare` | Error, exit non-zero |
| Duplicate slide ID or two tissues with one slug | Error naming them, nothing written |
| `--index`/`--slug` not in `tissues.csv` | Error, exit non-zero |
| `--parquet-dir` missing | Error, exit non-zero (as `centroids`) |
| Slide without `cell_types.gpd` | Warn, list in `no_source_slide_ids`, continue |
| Slide whose source fails to extract | Report, list in `failed_slide_ids`, build the object from the rest, exit 1 |
| No slide of the organ has a source | Left out of the launcher's default array and named; if built anyway, no object, exit 1 with the count |
| Existing npz without `--overwrite` | Skip (resume) |
| Crash during the h5ad write | Final name never holds a partial file |

## Resources and scale

- Per task: 8 CPUs, 256G, 12 h on `mediumq`; at most 8 tasks at once.
  - **8 CPUs** because extraction is minutes even at 8 workers and
    consolidation is single-threaded. On 24-CPU nodes, 8 CPUs fit three tasks
    per node where 16 would fit one.
  - **256G** holds ~1.1B cells per organ (measured ~225 B/cell, T5), about 5×
    liver's density on a 1,000-slide tissue, and is a sixth of a node.
  - An organ that needs more can be resubmitted alone with a larger `--mem`;
    nodes have ~1.5 TB.
- Extraction measured ~4 s per 350k-cell slide (the 2026-09-15 design), so a
  1,000-slide tissue is about an hour of CPU, under ten minutes across 8
  workers; consolidation and the gzip write dominate.
- Disk: ~200 GB of h5ad and ~75 GB of npz at liver density (estimate). The
  npz can be deleted once an organ's object is checked; they only make
  re-consolidation fast.

## Testing

Standard-library `unittest`, as now, with synthetic data only.

- **`tests/test_tissues.py`** (new): the slug rule, including the trailing
  space and the examples above; `prepare` splitting a small Portal table into
  per-tissue files, folding padded names, the manifest's order, indices and
  `n_with_source`, refusing a slug collision and a duplicate slide ID, and
  leaving an existing `centroids/` and h5ad untouched; `array` leaving out
  tissues without sources; `build` end to end on Parquet fixtures (the
  existing `write_source` style) for two organs: object written with the
  Portal columns and no `label`, `slides.csv` row ranges matching the h5ad's
  `obs/slide_id/codes` read with h5py as the viewer reads them,
  `no_source_slide_ids` and `failed_slide_ids` recorded, exit status per T8,
  `--slug` and `--index` agreeing, and a second run skipping existing npz.
- **`tests/test_consolidate.py`**: the existing tests pass unchanged; new ones
  for `--metadata-columns`, `--slides-out`, ascending contiguous slide blocks,
  `missing_slide_ids`, CSR index dtype, and categories equal to what
  `pd.Categorical(values)` gives for the same values.
- **Scripts**: `bash -n` on the new sbatch and launcher; the sbatch run under
  plain `bash` with `SLURM_ARRAY_TASK_ID=0` against a synthetic `<ROOT>`; the
  launcher run with a stub `sbatch` on `PATH` that records its arguments, to
  check the default array (tissues without sources left out), the `ARRAY`
  override, the `%` cap and the exported variables.

## README

A new "Build every tissue" section: what the run produces, the launcher
command, rerunning one organ, the output layout with the `slides.csv`
contract, resources and the disk estimate, and that the liver object and
steps 1-3 are unchanged. The opening paragraph and the outputs section mention
the per-tissue objects; the liver-specific findings stay as they are.

## Left to the builder

- Function names and decomposition inside `tissues.py` and `consolidate.py`.
- Whether `build` calls the consolidation through a function or through
  `consolidate.main` with an argument list.
- Log wording and progress output.

## Risks and open items

- 284 Portal slides have no folder under the Parquet root, and some folders
  may lack a `cell_types.gpd` (as 16 liver slides did). The coverage table at
  launch and each organ's `provenance.json` show where.
- Cell density per tissue is unknown. 256G covers ~1.1B cells per organ
  (measured on synthetic data); the per-organ resubmit with more memory is
  the fallback.
- About 300 GB of output in total on `/nobackup`, which has 4.9 PB free.
- The Portal table can change between fetches; the cached copy in `<ROOT>`
  pins each run, and `provenance.json` records its path.
- `gtexportal.org` is assumed reachable from the CEMM login node, as PyPI is;
  otherwise the table is fetched elsewhere and copied into `<ROOT>`.
