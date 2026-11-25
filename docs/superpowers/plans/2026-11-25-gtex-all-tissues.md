# One AnnData per GTEx Tissue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one cell-resolved AnnData per GTEx tissue site (40) on CEMM with one SLURM array job, in a per-organ folder layout the topocyte-analyses viewer reads.

**Architecture:** `consolidate.py` is reworked to build `obs` and `X` from integer codes and gains a metadata-column option, a per-slide table and atomic writes, behind a `run()` function. A new module `gtex_liver_cells.tissues` splits the Portal table per tissue (`prepare`), prints the array of tissues with data (`array`) and runs one organ end to end (`build`: existing `centroids` extraction, then `consolidate.run`). A thin array sbatch calls `build` per task; a launcher fetches, prepares and submits.

**Tech Stack:** Python 3.14, uv, numpy, pandas 3, scipy, anndata 0.13, pyarrow, shapely, h5py (via anndata), standard-library unittest, bash, SLURM.

**Spec:** `docs/superpowers/specs/2026-11-25-gtex-all-tissues-design.md`

## Global Constraints

- Python `>=3.14`; no new runtime or dev dependencies.
- Tests: standard-library `unittest`, run with `uv run --no-sync python -m unittest discover -s tests -v` from the repository root (single files with `-p "<file>"`, so `tests/` is on `sys.path` and test modules can import each other's fixtures); `uv run --no-sync ruff check src tests` stays clean (line length 100).
- The liver pipeline is unchanged: `consolidate` with default options writes the same object as before (same `obs` columns `slide_id`, `cell_type`, `prob`, then `META_COLUMNS`; same `uns` keys); `centroids.py`, `cohort.py`, `portal.py`, `scripts/step_1.sh`–`step_3.sh` and both existing sbatch files are not edited.
- No lab path in Python. Cluster paths live only in `sbatch/build_tissues.sbatch` and `scripts/all_tissues.sh`.
- Layout contract (spec, "Output layout"): `<ROOT>/tissues.csv` columns `index, slug, tissue, n_slides, n_with_source`; per tissue `<ROOT>/<slug>/portal_slides.csv`, `gtex_<slug>_cells.h5ad`, `slides.csv`, `provenance.json`, `centroids/<TISSUE_ID>.npz`.
- `slides.csv` columns: the cohort CSV's columns (for tissues, the nine Portal columns `Tissue Sample ID, Tissue, Subject ID, Sex, Age Bracket, Hardy Scale, Pathology Categories, Pathology Notes, Hidden`) then `n_cells, row_start, row_end` (end exclusive), one row per slide in h5ad row order.
- Tissue `obs` metadata columns, in order: `Tissue, Subject ID, Sex, Age Bracket, Hardy Scale, Pathology Categories, Pathology Notes, Hidden`. No `label`.
- Slug: strip, lowercase, runs of `[^a-z0-9]` → `_`, trim `_`.
- sbatch resources: `--partition=mediumq --account=lab_rendeiro --time=12:00:00 --cpus-per-task=8 --mem=256G --mail-type=ALL`, logs `logs/gtex_tissue_%A_%a.{out,err}`; launcher default `MAX_PARALLEL=8`.
- Commit after every task, message in the repository's style (`feat:`, `refactor:`, `docs:`, `test:`), ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.

## Review Focus

1. **Slides fed in listing order, not sorted** — `consolidate` must still emit ascending slide categories with contiguous blocks; pinned in Task 1 (`test_orders_slides_ascending_in_contiguous_blocks`).
2. **Organs with 126 or fewer slides** get int8 `obs/slide_id/codes`; `slides.csv` ranges must still match the codes read with h5py; pinned in Task 2 (`test_small_objects_get_int8_slide_codes_and_matching_ranges`).
3. **A leftover `<h5ad>.tmp` from a killed task** must not break the next run or survive it; pinned in Task 2 (`test_writes_through_a_temporary_file`).
4. **Empty Portal values** (68% of slides have no `Pathology Categories`) must become a `""` category, not NaN; pinned in Task 1 (`test_joins_the_requested_metadata_columns`).
5. **Re-running `prepare` over a finished run** must not touch `centroids/`, the h5ad or `slides.csv`; pinned in Task 3 (`test_leaves_existing_outputs_alone`).

---

### Task 0: Environment and baseline

**Files:** none.

- [ ] **Step 1: Create the worktree venv**

Run: `uv sync`
Expected: `.venv/` created in the repository root (gitignored), exit 0.

- [ ] **Step 2: Run the existing suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`
Expected: all tests PASS (record the count; later tasks only add tests).

- [ ] **Step 3: Commit the spec and this plan**

```bash
git add docs/superpowers/specs/2026-11-25-gtex-all-tissues-design.md docs/superpowers/plans/2026-11-25-gtex-all-tissues.md
git commit -m "docs: design and plan one AnnData per GTEx tissue"
```

---

### Task 1: Build `obs` and `X` from integer codes

**Files:**
- Modify: `src/gtex_liver_cells/consolidate.py` (`consolidate`, module docstring)
- Test: `tests/test_consolidate.py` (class `ConsolidateTests`)

**Interfaces:**
- Produces: `consolidate(entries: list[dict], cohort: pd.DataFrame, metadata_columns: Sequence[str] = META_COLUMNS) -> ad.AnnData`. `cohort` is indexed by `Tissue Sample ID`. Does not mutate `entries`.

- [ ] **Step 1: Write the failing tests** (use the class's `entry()` and `cohort()` helpers)

```python
def test_orders_slides_ascending_in_contiguous_blocks(self):
    later = self.entry("GTEX-BBBB-0126", [[3, 3], [4, 4]], [0, 0], ["Fibroblasts"])
    earlier = self.entry("GTEX-AAAA-0126", [[1, 1]], [0], ["Fibroblasts"])
    adata = consolidate.consolidate(
        [later, earlier], self.cohort(["GTEX-AAAA-0126", "GTEX-BBBB-0126"])
    )
    self.assertEqual(
        list(adata.obs["slide_id"].cat.categories), ["GTEX-AAAA-0126", "GTEX-BBBB-0126"]
    )
    self.assertEqual(list(adata.obs["slide_id"].cat.codes), [0, 1, 1])
    np.testing.assert_allclose(adata.obsm["spatial"], [[1, 1], [3, 3], [4, 4]])
    self.assertEqual(list(adata.uns["slide_ids"]), ["GTEX-AAAA-0126", "GTEX-BBBB-0126"])

def test_joins_the_requested_metadata_columns(self):
    first = self.entry("GTEX-AAAA-0126", [[1, 1]], [0], ["Fibroblasts"])
    second = self.entry("GTEX-BBBB-0126", [[2, 2]], [0], ["Fibroblasts"])
    cohort = pd.DataFrame(
        {
            "Tissue Sample ID": ["GTEX-AAAA-0126", "GTEX-BBBB-0126"],
            "Tissue": ["Liver", "Liver"],
            "Pathology Categories": ["", "fibrosis"],
        }
    ).set_index("Tissue Sample ID")
    adata = consolidate.consolidate(
        [first, second], cohort, metadata_columns=["Tissue", "Pathology Categories"]
    )
    self.assertEqual(
        list(adata.obs.columns),
        ["slide_id", "cell_type", "prob", "Tissue", "Pathology Categories"],
    )
    self.assertEqual(list(adata.obs["Pathology Categories"].astype(str)), ["", "fibrosis"])
    self.assertEqual(list(adata.obs["Pathology Categories"].cat.categories), ["", "fibrosis"])
    self.assertEqual(list(adata.uns["metadata_columns"]), ["Tissue", "Pathology Categories"])

def test_categories_match_pandas_factorization(self):
    first = self.entry("GTEX-BBBB-0126", [[1, 1], [2, 2]], [1, 0], ["Fibroblasts", "Macrophages"])
    second = self.entry("GTEX-AAAA-0126", [[3, 3]], [0], ["Epithelial"])
    adata = consolidate.consolidate(
        [first, second], self.cohort(["GTEX-BBBB-0126", "GTEX-AAAA-0126"])
    )
    for column in ("slide_id", "cell_type", "label", "Sex"):
        values = adata.obs[column].astype(str).to_numpy()
        self.assertEqual(
            list(adata.obs[column].cat.categories), list(pd.Categorical(values).categories)
        )

def test_a_missing_metadata_column_is_one_empty_category(self):
    entry = self.entry("GTEX-AAAA-0126", [[1, 1]], [0], ["Fibroblasts"])
    adata = consolidate.consolidate(
        [entry], self.cohort(["GTEX-AAAA-0126"]), metadata_columns=["Pathology Notes"]
    )
    self.assertEqual(list(adata.obs["Pathology Notes"].cat.categories), [""])

def test_x_is_a_one_hot_csr_with_int32_indices(self):
    entry = self.entry("GTEX-AAAA-0126", [[1, 1], [2, 2]], [0, 1], ["Fibroblasts", "Macrophages"])
    adata = consolidate.consolidate([entry], self.cohort(["GTEX-AAAA-0126"]))
    self.assertTrue(sp.isspmatrix_csr(adata.X) or isinstance(adata.X, sp.csr_array))
    self.assertEqual(adata.X.dtype, np.float32)
    self.assertEqual(adata.X.indices.dtype, np.int32)
    self.assertEqual(adata.X.indptr.dtype, np.int32)
    np.testing.assert_array_equal(adata.X.toarray(), [[1, 0], [0, 1]])
```

- [ ] **Step 2: Run them to see the new ones fail**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_consolidate.py" -v`
Expected: the five new tests FAIL or ERROR (`metadata_columns` unexpected keyword; listing-order categories); existing tests PASS.

- [ ] **Step 3: Rewrite `consolidate` in `src/gtex_liver_cells/consolidate.py`**

Approach (spec T5): sort a copy of `entries` by `slide_id`; `class_names` = sorted union; per entry a remap array (its `class_names` → union index) applied to its int `labels` gives int32 type codes; `slide_codes = np.repeat(np.arange(n_slides), counts)` in the smallest int dtype that holds `n_slides`; every categorical via `pd.Categorical.from_codes` (categories in sorted order: slide IDs as sorted, class names as sorted, metadata values as `sorted(set(values))`). Per-slide metadata values come from `cohort[column].reindex(slide_ids)`, NaN → `""`; a column absent from `cohort` gives all `""`. `X = sp.csr_matrix((ones float32, type_codes int32, arange(N + 1) int32), shape=(N, n_classes))`, using int64 for both index arrays only when `N >= 2**31 - 1`. No per-cell Python objects. `uns` keys unchanged; `uns["metadata_columns"] = list(metadata_columns)`.

- [ ] **Step 4: Run the consolidate tests**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_consolidate.py" -v`
Expected: all PASS, including the existing `test_remaps_per_file_vocabularies_into_a_shared_var` and `test_keeps_cells_whose_slide_has_no_cohort_row`.

- [ ] **Step 5: Commit**

```bash
git add src/gtex_liver_cells/consolidate.py tests/test_consolidate.py
git commit -m "refactor: build obs and X from integer codes in consolidate"
```

---

### Task 2: `run()`, metadata and slide-table options, atomic writes

**Files:**
- Modify: `src/gtex_liver_cells/consolidate.py` (`build_provenance`, new `ConsolidateError`, `slide_table`, `write_h5ad_atomic`, `run`, `main`)
- Test: `tests/test_consolidate.py` (classes `MainTests`, `BuildProvenanceTests`, new `RunTests`)

**Interfaces:**
- Consumes: `consolidate(...)` from Task 1.
- Produces:
  - `class ConsolidateError(Exception)` — raised by `run` for: missing centroids directory, no `.npz` files, no entries left after filtering. `main` turns it into `ERROR: <message>` on stderr and exit 1 (the existing exit-1 tests keep passing).
  - `slide_table(adata: ad.AnnData, cohort: pd.DataFrame) -> pd.DataFrame` — `Tissue Sample ID` plus every `cohort` column for each slide category in order (missing values `""`), then `n_cells`, `row_start`, `row_end` from the `slide_id` codes.
  - `write_h5ad_atomic(adata: ad.AnnData, path: str) -> None` — gzip-writes `f"{path}.tmp"`, then `os.replace` onto `path`.
  - `run(centroids_dir: str, cohort_csv: str, out: str, *, provenance_out: str | None = None, filter_to_cohort: bool = False, metadata_columns: Sequence[str] = META_COLUMNS, slides_out: str | None = None, extra_provenance: dict | None = None) -> dict` — returns `{"n_obs": int, "n_slides": int, "missing_slide_ids": list[str]}`; writes the h5ad, then `slides_out`, then `provenance_out` (with `extra_provenance` merged in); prints what `main` prints today.
  - `build_provenance(...)` keeps its signature and adds `"missing_slide_ids": sorted(missing)`.
  - CLI: `--metadata-columns COL [COL ...]` (default `META_COLUMNS`), `--slides-out PATH`.

- [ ] **Step 1: Write the failing tests**

In `BuildProvenanceTests.test_records_inputs_and_counts` add:

```python
self.assertEqual(provenance["missing_slide_ids"], ["GTEX-BBBB-0126", "GTEX-CCCC-0126"])
```

In `MainTests` add (use `write_npz`, `write_cohort`, `cohort_row`, `COHORT_COLUMNS`; `import h5py` at the top):

```python
def test_slides_out_lists_row_ranges_per_slide(self):
    write_npz(os.path.join(self.centroid_dir, "GTEX-BBBB-0126.npz"), [[1, 1], [2, 2]], [0, 0], ["Fibroblasts"])
    self.add_npz("GTEX-AAAA-0126")
    write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126"), cohort_row("GTEX-BBBB-0126")])
    out = os.path.join(self.tmp.name, "cohort.h5ad")
    slides = os.path.join(self.tmp.name, "slides.csv")
    exit_code = consolidate.main(["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out, "--slides-out", slides])
    table = pd.read_csv(slides, dtype=str, keep_default_na=False)
    self.assertEqual(exit_code, 0)
    self.assertEqual(list(table.columns), COHORT_COLUMNS + ["n_cells", "row_start", "row_end"])
    self.assertEqual(list(table["Tissue Sample ID"]), ["GTEX-AAAA-0126", "GTEX-BBBB-0126"])
    self.assertEqual(list(table["n_cells"]), ["1", "2"])
    self.assertEqual(list(table["row_start"]), ["0", "1"])
    self.assertEqual(list(table["row_end"]), ["1", "3"])

def test_small_objects_get_int8_slide_codes_and_matching_ranges(self):
    for slide_id in ("GTEX-AAAA-0126", "GTEX-BBBB-0126", "GTEX-CCCC-0126"):
        self.add_npz(slide_id)
    write_cohort(self.cohort_csv, [cohort_row(s) for s in ("GTEX-AAAA-0126", "GTEX-BBBB-0126", "GTEX-CCCC-0126")])
    out = os.path.join(self.tmp.name, "cohort.h5ad")
    slides = os.path.join(self.tmp.name, "slides.csv")
    consolidate.main(["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out, "--slides-out", slides])
    table = pd.read_csv(slides)
    with h5py.File(out, "r") as handle:
        codes = handle["obs/slide_id/codes"][:]
    self.assertEqual(codes.dtype, np.int8)
    for position, row in table.iterrows():
        self.assertTrue((codes[row["row_start"]:row["row_end"]] == position).all())
    self.assertEqual(table["row_end"].iloc[-1], len(codes))

def test_metadata_columns_flag_replaces_the_default(self):
    self.add_npz("GTEX-AAAA-0126")
    write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
    out = os.path.join(self.tmp.name, "cohort.h5ad")
    consolidate.main(["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out, "--metadata-columns", "Subject ID", "Sex"])
    adata = ad.read_h5ad(out)
    self.assertEqual(list(adata.obs.columns), ["slide_id", "cell_type", "prob", "Subject ID", "Sex"])

def test_writes_through_a_temporary_file(self):
    self.add_npz("GTEX-AAAA-0126")
    write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
    out = os.path.join(self.tmp.name, "cohort.h5ad")
    with open(f"{out}.tmp", "wb") as handle:
        handle.write(b"left over by a killed task")
    exit_code = consolidate.main(["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out])
    self.assertEqual(exit_code, 0)
    self.assertEqual(ad.read_h5ad(out).n_obs, 1)
    self.assertFalse(os.path.exists(f"{out}.tmp"))

def test_provenance_lists_missing_slide_ids(self):
    self.add_npz("GTEX-AAAA-0126")
    write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126"), cohort_row("GTEX-CCCC-0126")])
    out = os.path.join(self.tmp.name, "cohort.h5ad")
    provenance_path = os.path.join(self.tmp.name, "provenance.json")
    consolidate.main(["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out, "--provenance-out", provenance_path])
    with open(provenance_path, encoding="utf-8") as handle:
        self.assertEqual(json.load(handle)["missing_slide_ids"], ["GTEX-CCCC-0126"])
```

New class `RunTests` (same `setUp` as `MainTests`):

```python
def test_merges_extra_provenance_and_returns_a_summary(self):
    # one npz + cohort row; run(..., provenance_out=p, extra_provenance={"tissue": "Liver"})
    self.assertEqual(summary, {"n_obs": 1, "n_slides": 1, "missing_slide_ids": []})
    self.assertEqual(provenance["tissue"], "Liver")

def test_raises_when_no_npz_exist(self):
    with self.assertRaises(consolidate.ConsolidateError):
        consolidate.run(self.centroid_dir, self.cohort_csv, os.path.join(self.tmp.name, "x.h5ad"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_consolidate.py" -v`
Expected: the new tests FAIL/ERROR (`--slides-out` unrecognized, `run` missing, no `missing_slide_ids`).

- [ ] **Step 3: Implement** `ConsolidateError`, `slide_table`, `write_h5ad_atomic`, `run` and the two CLI options in `src/gtex_liver_cells/consolidate.py`; reduce `main` to argument parsing plus `run` inside `try/except ConsolidateError`. `slide_table` takes counts with `np.bincount(codes, minlength=n_slides)` and `row_end = cumsum(counts)`. `run` reads the cohort with `pd.read_csv(cohort_csv, dtype=str).fillna("").set_index("Tissue Sample ID")`, as `main` does today.

- [ ] **Step 4: Run the whole suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gtex_liver_cells/consolidate.py tests/test_consolidate.py
git commit -m "feat: add run, a slides table and atomic writes to consolidate"
```

---

### Task 3: `tissues prepare` and `tissues array`

**Files:**
- Create: `src/gtex_liver_cells/tissues.py`
- Test: `tests/test_tissues.py`

**Interfaces:**
- Consumes: `portal.COLUMNS` (the nine Portal columns), `portal.write_csv(rows, path)` (fixtures), `centroids.source_path(parquet_dir, slide_id)`.
- Produces (all in `gtex_liver_cells.tissues`):
  - `PORTAL_COLUMNS = portal.COLUMNS`; `METADATA_COLUMNS = PORTAL_COLUMNS[1:]`; `MANIFEST_COLUMNS = ["index", "slug", "tissue", "n_slides", "n_with_source"]`; file-name constants `MANIFEST = "tissues.csv"`, `PORTAL_SLIDES = "portal_slides.csv"`, `SLIDES = "slides.csv"`, `PROVENANCE = "provenance.json"`, `CENTROIDS = "centroids"`.
  - `slugify(name: str) -> str`
  - `object_filename(slug: str) -> str` → `f"gtex_{slug}_cells.h5ad"`
  - `split_tissues(slides: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]` — `(slug, tissue, rows)` sorted by slug, `Tissue` stripped in `rows`; `ValueError` naming a duplicate slide ID, an empty slug, a missing Portal column, or two tissues sharing a slug.
  - `count_sources(parquet_dir: str, slide_ids: Iterable[str]) -> int`
  - `prepare(slides_csv: str, root: str, parquet_dir: str) -> pd.DataFrame` — writes `<root>/<slug>/portal_slides.csv` (columns `PORTAL_COLUMNS`) and `<root>/tissues.csv`, prints the coverage table and each tissue without sources, returns the manifest.
  - `read_manifest(root: str) -> pd.DataFrame`
  - `array_spec(manifest: pd.DataFrame) -> str` — comma-separated `index` values with `n_with_source > 0`, `""` if none.
  - `main(argv: list[str] | None = None) -> int` with subcommands `prepare --slides CSV --root DIR --parquet-dir DIR` and `array --root DIR` (prints the spec); exit 1 with `ERROR:` for a missing `--parquet-dir` or slides CSV and for `ValueError` from `split_tissues`.

- [ ] **Step 1: Write the failing tests**

Fixture helper at module level, reused by Tasks 4 and 5:

```python
from test_centroids import SQUARE, TWO_SQUARES, slide_source, write_source

def portal_row(slide_id, tissue):
    return [slide_id, tissue, "-".join(slide_id.split("-")[:2]), "female", "50-59",
            "Slow death", "", "2 pieces", "false"]

def make_fixture(tmp):
    """Return (portal_csv, parquet_dir): liver A, B; esophagus C (padded name), C2; skin E (no source)."""
```

`make_fixture` writes, with `portal.write_csv`: `GTEX-AAAA-0126` and `GTEX-BBBB-0126` (`Liver`), `GTEX-CCCC-0126` (`Esophagus - Mucosa `, trailing space) and `GTEX-CCCD-0126` (`Esophagus - Mucosa`), `GTEX-EEEE-0126` (`Skin - Sun Exposed (Lower leg)`); and Parquet sources for A (two cells: `SQUARE`, `TWO_SQUARES`), B (one cell) and C (one cell) only.

```python
class SlugifyTests(unittest.TestCase):
    def test_examples(self):
        for name, slug in [
            ("Liver", "liver"),
            ("Skin - Sun Exposed (Lower leg)", "skin_sun_exposed_lower_leg"),
            ("Adipose - Visceral (Omentum)", "adipose_visceral_omentum"),
            ("Esophagus - Mucosa ", "esophagus_mucosa"),
        ]:
            self.assertEqual(tissues.slugify(name), slug)
        self.assertEqual(tissues.object_filename("liver"), "gtex_liver_cells.h5ad")

class SplitTissuesTests(unittest.TestCase):
    def test_groups_by_stripped_tissue_sorted_by_slug(self):
        # frame from portal_row for A, B (Liver), C ("Esophagus - Mucosa "), C2 ("Esophagus - Mucosa")
        self.assertEqual([(slug, tissue, len(rows)) for slug, tissue, rows in groups],
                         [("esophagus_mucosa", "Esophagus - Mucosa", 2), ("liver", "Liver", 2)])
        self.assertEqual(set(groups[0][2]["Tissue"]), {"Esophagus - Mucosa"})
    def test_refuses_a_duplicate_slide_id(self):  # assertRaisesRegex(ValueError, "GTEX-AAAA-0126")
    def test_refuses_two_tissues_with_one_slug(self):  # "Brain - Cortex" and "Brain Cortex": assertRaisesRegex(ValueError, "brain_cortex")

class PrepareTests(unittest.TestCase):
    def test_writes_per_tissue_slide_tables_and_the_manifest(self):
        manifest = tissues.prepare(portal_csv, root, parquet_dir)
        self.assertEqual(list(manifest.columns), tissues.MANIFEST_COLUMNS)
        self.assertEqual(list(manifest["slug"]), ["esophagus_mucosa", "liver", "skin_sun_exposed_lower_leg"])
        self.assertEqual(list(manifest["index"]), [0, 1, 2])
        self.assertEqual(list(manifest["n_slides"]), [2, 2, 1])
        self.assertEqual(list(manifest["n_with_source"]), [1, 2, 0])
        pd.testing.assert_frame_equal(tissues.read_manifest(root), manifest)
        liver = pd.read_csv(os.path.join(root, "liver", "portal_slides.csv"), dtype=str, keep_default_na=False)
        self.assertEqual(list(liver.columns), tissues.PORTAL_COLUMNS)
        self.assertEqual(list(liver["Tissue Sample ID"]), ["GTEX-AAAA-0126", "GTEX-BBBB-0126"])
    def test_leaves_existing_outputs_alone(self):
        # after a first prepare, write b"keep" to <root>/liver/centroids/GTEX-AAAA-0126.npz,
        # <root>/liver/gtex_liver_cells.h5ad and <root>/liver/slides.csv; prepare again; bytes unchanged
    def test_array_spec_leaves_out_tissues_without_sources(self):
        self.assertEqual(tissues.array_spec(manifest), "0,1")
    def test_array_spec_is_empty_without_sources(self):
        # prepare against an empty parquet dir
        self.assertEqual(tissues.array_spec(manifest), "")
    def test_main_prepare_then_array(self):
        self.assertEqual(tissues.main(["prepare", "--slides", portal_csv, "--root", root, "--parquet-dir", parquet_dir]), 0)
        # capture stdout (contextlib.redirect_stdout) of main(["array", "--root", root])
        self.assertEqual(captured.getvalue().strip(), "0,1")
    def test_main_prepare_rejects_a_missing_parquet_dir(self):
        self.assertEqual(tissues.main(["prepare", "--slides", portal_csv, "--root", root, "--parquet-dir", missing]), 1)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_tissues.py" -v`
Expected: ERROR, `cannot import name 'tissues'`.

- [ ] **Step 3: Implement** the Task 3 interface in `src/gtex_liver_cells/tissues.py`, with a module docstring describing the layout (spec, "Output layout"). Read the Portal CSV with `pd.read_csv(path, dtype=str, keep_default_na=False)`; write CSVs with `index=False`. The coverage table prints `index`, `slug`, `n_with_source/n_slides` per tissue and a total line.

- [ ] **Step 4: Run them to see them pass**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_tissues.py" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gtex_liver_cells/tissues.py tests/test_tissues.py
git commit -m "feat: split the Portal table per tissue with a coverage check"
```

---

### Task 4: `tissues build`

**Files:**
- Modify: `src/gtex_liver_cells/tissues.py`
- Test: `tests/test_tissues.py` (new class `BuildTests`)

**Interfaces:**
- Consumes: Task 3's constants, `read_manifest`, `make_fixture`; `centroids.read_slide_ids`, `centroids.select_slides`, `centroids.extract_many`; `consolidate.run`, `consolidate.ConsolidateError`.
- Produces:
  - `find_tissue(manifest: pd.DataFrame, *, index: int | None = None, slug: str | None = None) -> pd.Series` — the manifest row; `ValueError` if absent.
  - `build_tissue(root: str, slug: str, tissue: str, parquet_dir: str, workers: int, overwrite: bool = False) -> int` — exit status per spec T8: 1 with no object when no slide has a source; else extraction into `<root>/<slug>/centroids/`, then `consolidate.run(centroids_dir, portal_csv, <root>/<slug>/gtex_<slug>_cells.h5ad, provenance_out=..., filter_to_cohort=True, metadata_columns=METADATA_COLUMNS, slides_out=..., extra_provenance={"tissue", "slug", "parquet_dir" (absolute), "no_source_slide_ids", "failed_slide_ids"})`; 1 if any slide failed or `ConsolidateError`, else 0. A failed slide is one with a source but no `<id>.npz` after extraction.
  - CLI: `build --root DIR (--index I | --slug S) --parquet-dir DIR [--workers N] [--overwrite]` (mutually exclusive, one required; `--workers` default `os.cpu_count()`); exit 1 with `ERROR:` for an unknown index or slug or a missing `--parquet-dir`.

- [ ] **Step 1: Write the failing tests**

`setUp`: `make_fixture`, then add a liver slide `GTEX-BBBC-0126` whose source holds a `Point` (extraction fails) and a liver slide `GTEX-DDDD-0126` with no source, both as Portal rows, then `tissues.prepare(...)`. Run builds with `workers=1`.

```python
def test_builds_the_object_slides_table_and_provenance(self):
    self.assertEqual(tissues.main(["build", "--root", root, "--slug", "esophagus_mucosa", "--parquet-dir", parquet_dir, "--workers", "1"]), 0)
    organ = os.path.join(root, "esophagus_mucosa")
    adata = ad.read_h5ad(os.path.join(organ, "gtex_esophagus_mucosa_cells.h5ad"))
    self.assertEqual(list(adata.obs.columns), ["slide_id", "cell_type", "prob"] + tissues.METADATA_COLUMNS)
    self.assertEqual(set(adata.obs["Tissue"].astype(str)), {"Esophagus - Mucosa"})
    table = pd.read_csv(os.path.join(organ, "slides.csv"), dtype=str, keep_default_na=False)
    self.assertEqual(list(table.columns), tissues.PORTAL_COLUMNS + ["n_cells", "row_start", "row_end"])
    self.assertEqual(list(table["Tissue Sample ID"]), ["GTEX-CCCC-0126"])
    provenance = json.load(open(os.path.join(organ, "provenance.json"), encoding="utf-8"))
    self.assertEqual((provenance["tissue"], provenance["slug"]), ("Esophagus - Mucosa", "esophagus_mucosa"))
    self.assertEqual(provenance["no_source_slide_ids"], ["GTEX-CCCD-0126"])
    self.assertEqual(provenance["failed_slide_ids"], [])

def test_records_missing_and_failed_slides_and_exits_one(self):
    self.assertEqual(tissues.build_tissue(root, "liver", "Liver", parquet_dir, workers=1), 1)
    # object holds A and B only; provenance: failed ["GTEX-BBBC-0126"], no_source ["GTEX-DDDD-0126"],
    # missing_slide_ids ["GTEX-BBBC-0126", "GTEX-DDDD-0126"]; slides.csv lists A, B

def test_row_ranges_match_the_codes_the_viewer_reads(self):
    # after building liver: h5py obs/slide_id/{codes,categories}; for each slides.csv row i,
    # categories[i] == its Tissue Sample ID and codes[row_start:row_end] are all i; last row_end == len(codes)

def test_index_and_slug_select_the_same_tissue(self):
    manifest = tissues.read_manifest(root)
    self.assertEqual(tissues.find_tissue(manifest, index=1)["slug"], "liver")
    self.assertEqual(tissues.find_tissue(manifest, slug="liver")["index"], 1)

def test_rerun_skips_existing_npz(self):
    # build esophagus twice; os.stat(<its npz>).st_mtime_ns is unchanged by the second run

def test_a_tissue_without_sources_exits_one_without_an_object(self):
    self.assertEqual(tissues.build_tissue(root, "skin_sun_exposed_lower_leg", "Skin - Sun Exposed (Lower leg)", parquet_dir, workers=1), 1)
    self.assertFalse(os.path.exists(os.path.join(root, "skin_sun_exposed_lower_leg", "gtex_skin_sun_exposed_lower_leg_cells.h5ad")))

def test_unknown_slug_or_index_exits_one(self):
    self.assertEqual(tissues.main(["build", "--root", root, "--slug", "nope", "--parquet-dir", parquet_dir]), 1)
    self.assertEqual(tissues.main(["build", "--root", root, "--index", "99", "--parquet-dir", parquet_dir]), 1)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_tissues.py" -k BuildTests -v`
Expected: ERROR (`build_tissue`/`find_tissue` missing, `build` subcommand unknown).

- [ ] **Step 3: Implement** the Task 4 interface in `src/gtex_liver_cells/tissues.py`. `build_tissue` prints the tissue, slug, and counts (`Portal`, `with source`, `extracted/skipped/failed`) before consolidating.

- [ ] **Step 4: Run the whole suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gtex_liver_cells/tissues.py tests/test_tissues.py
git commit -m "feat: build one tissue's object from extraction to slides table"
```

---

### Task 5: The array sbatch and the launcher

**Files:**
- Create: `sbatch/build_tissues.sbatch`, `scripts/all_tissues.sh`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `python -m gtex_liver_cells.tissues prepare|array|build` (Tasks 3, 4); `python -m gtex_liver_cells.portal --out`.
- Produces:
  - `sbatch/build_tissues.sbatch`: the Global Constraints' `#SBATCH` lines and `--job-name=gtex_tissue_anndata`; `set -uo pipefail`; requires `PARQUET_DIR`, `ROOT`, `SLURM_ARRAY_TASK_ID` (`: "${VAR:?...}"`); `REPO_DIR` defaults to `/home/sschindler/rep/gtex-liver-cells`, `PYTHON` to `$REPO_DIR/.venv/bin/python`; `cd "$REPO_DIR"`, `export PYTHONPATH=src`; echoes node, task, paths, workers; runs `"$PYTHON" -m gtex_liver_cells.tissues build --root "$ROOT" --index "$SLURM_ARRAY_TASK_ID" --parquet-dir "$PARQUET_DIR" --workers "${SLURM_CPUS_PER_TASK:-8}"` as its last command. Header comment: purpose and the submit command, in the style of the existing wrappers.
  - `scripts/all_tissues.sh`: `set -euo pipefail`; defaults overridable from the environment: `REPO_DIR=$HOME/rep/gtex-liver-cells`, `PARQUET_DIR=/nobackup/lab_rendeiro/projects/histopath/processed/histopathology/datasets_pq_h5ad/`, `ROOT=/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_tissues`, `PYTHON=$REPO_DIR/.venv/bin/python`, `MAX_PARALLEL=8`; steps per spec T10: `mkdir -p "$ROOT/logs"`; fetch `"$ROOT/gtex_portal_slides.csv"` with `portal` only if missing or empty; `tissues prepare`; `ARRAY` from the environment or `tissues array`; empty → `ERROR` on stderr, exit 1, nothing submitted; `cd "$ROOT"`; `sbatch --array="$ARRAY%$MAX_PARALLEL" --export=ALL,PARQUET_DIR=$PARQUET_DIR,ROOT=$ROOT,REPO_DIR=$REPO_DIR,PYTHON=$PYTHON "$REPO_DIR/sbatch/build_tissues.sbatch"`. Header comment: what it does, `ARRAY` and `MAX_PARALLEL`.

- [ ] **Step 1: Write the failing tests** (`subprocess.run(["bash", script], env=..., capture_output=True, text=True)`; `REPO = Path(__file__).resolve().parents[1]`; env always sets `REPO_DIR=str(REPO)`, `PYTHON=sys.executable`, `ROOT`, `PARQUET_DIR` from `make_fixture`)

```python
def test_scripts_parse(self):
    for script in ("sbatch/build_tissues.sbatch", "scripts/all_tissues.sh"):
        self.assertEqual(subprocess.run(["bash", "-n", str(REPO / script)]).returncode, 0)

def test_sbatch_builds_one_task_under_plain_bash(self):
    # tissues.prepare(...) first; env SLURM_ARRAY_TASK_ID="0", SLURM_CPUS_PER_TASK="1"
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertTrue(os.path.exists(os.path.join(root, "esophagus_mucosa", "gtex_esophagus_mucosa_cells.h5ad")))

def test_sbatch_requires_an_array_task_id(self):
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("SLURM_ARRAY_TASK_ID", result.stderr)

def test_launcher_submits_tissues_with_sources(self):
    # stub dir on PATH first, holding an executable `sbatch` that writes "$@" one per line to $STUB_LOG;
    # copy the fixture's portal CSV to <root>/gtex_portal_slides.csv so nothing is fetched
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--array=0,1%8", args)
    self.assertTrue(args[-1].endswith("sbatch/build_tissues.sbatch"))
    self.assertTrue(any(a.startswith("--export=ALL,PARQUET_DIR=") and f"ROOT={root}" in a for a in args))
    self.assertTrue(os.path.isdir(os.path.join(root, "logs")))

def test_launcher_honours_array_and_max_parallel(self):
    # env ARRAY="1", MAX_PARALLEL="2"
    self.assertIn("--array=1%2", args)

def test_launcher_submits_nothing_without_sources(self):
    # PARQUET_DIR = an empty directory
    self.assertEqual(result.returncode, 1)
    self.assertFalse(os.path.exists(stub_log))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --no-sync python -m unittest discover -s tests -p "test_scripts.py" -v`
Expected: FAIL (scripts do not exist; `bash -n` returns 127/2).

- [ ] **Step 3: Write** `sbatch/build_tissues.sbatch` and `scripts/all_tissues.sh` to the interface above.

- [ ] **Step 4: Run the whole suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add sbatch/build_tissues.sbatch scripts/all_tissues.sh tests/test_scripts.py
git commit -m "feat: add the per-tissue array sbatch and its launcher"
```

---

### Task 6: README and final checks

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update `README.md`** per spec "README": a "Build every tissue" subsection under Usage (the launcher command, `ARRAY=<i>` reruns, `MAX_PARALLEL`, the coverage table, resources 8 CPUs/256G/12 h, disk ~300 GB at liver density); the per-tissue tree with `slides.csv`'s columns and row-range meaning under Outputs; one sentence in the opening that the pipeline also builds one object per GTEx tissue; the laptop section gains the `tissues` tests in the unittest command (unchanged command) and nothing else. Liver sections stay.

- [ ] **Step 2: Lint**

Run: `uv run --no-sync ruff check src tests`
Expected: `All checks passed!`

- [ ] **Step 3: Full suite**

Run: `uv run --no-sync python -m unittest discover -s tests -v`
Expected: all PASS; count = Task 0 baseline + the tests added here.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document building one object per GTEx tissue"
```
