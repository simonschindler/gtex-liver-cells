import json
import os
import tempfile
import unittest

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from gtex_liver_cells import consolidate

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
            [
                cohort_row(slide_id, "cirrhosis" if index == 0 else "healthy")
                for index, slide_id in enumerate(slide_ids)
            ],
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
        np.testing.assert_array_equal(adata.X.toarray(), [[1, 0], [0, 1], [0, 1]])
        np.testing.assert_allclose(adata.obsm["spatial"], [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(adata.obsm["spatial"].dtype, np.float32)
        # anndata stringifies obs_names even for a RangeIndex; keeping the
        # default sequential names is what keeps the h5ad small, because
        # sequential digits compress to almost nothing. What matters is that
        # we never pay for a long per-cell identifier.
        self.assertEqual(list(adata.obs_names), ["0", "1", "2"])
        np.testing.assert_allclose(adata.obs["prob"].to_numpy(), [0.5, 0.5, 0.5])

    def test_keeps_cells_whose_slide_has_no_cohort_row(self):
        entry = self.entry("GTEX-CCCC-0126", [[1, 1]], [0], ["Fibroblasts"])
        adata = consolidate.consolidate([entry], self.cohort(["GTEX-AAAA-0126"]))
        self.assertEqual(adata.n_obs, 1)
        self.assertEqual(adata.obs["label"].astype(str).iloc[0], "")

    def test_rejects_an_empty_entry_list(self):
        with self.assertRaises(ValueError):
            consolidate.consolidate([], self.cohort(["GTEX-AAAA-0126"]))

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
        self.assertEqual(
            list(adata.obs["Pathology Categories"].cat.categories), ["", "fibrosis"]
        )
        self.assertEqual(list(adata.uns["metadata_columns"]), ["Tissue", "Pathology Categories"])

    def test_categories_match_pandas_factorization(self):
        first = self.entry(
            "GTEX-BBBB-0126", [[1, 1], [2, 2]], [1, 0], ["Fibroblasts", "Macrophages"]
        )
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
        entry = self.entry(
            "GTEX-AAAA-0126", [[1, 1], [2, 2]], [0, 1], ["Fibroblasts", "Macrophages"]
        )

        adata = consolidate.consolidate([entry], self.cohort(["GTEX-AAAA-0126"]))

        self.assertEqual(adata.X.format, "csr")
        self.assertEqual(adata.X.dtype, np.float32)
        self.assertEqual(adata.X.indices.dtype, np.int32)
        self.assertEqual(adata.X.indptr.dtype, np.int32)
        np.testing.assert_array_equal(adata.X.toarray(), [[1, 0], [0, 1]])


class RepositoryCommitTests(unittest.TestCase):
    def test_returns_unknown_outside_a_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(consolidate.repository_commit(tmp), "unknown")

    def test_returns_the_checkout_commit(self):
        commit = consolidate.repository_commit(consolidate.REPO_DIR)
        self.assertRegex(commit, r"^([0-9a-f]{40}|unknown)$")


class BuildProvenanceTests(unittest.TestCase):
    def test_records_inputs_and_counts(self):
        provenance = consolidate.build_provenance(
            centroids_dir="/tmp/centroids",
            cohort_csv="/tmp/liver_cohort.csv",
            requested=610,
            extracted=603,
            used=124,
            missing=["GTEX-BBBB-0126", "GTEX-CCCC-0126"],
            class_names=["Apoptotic Body", "Epithelial"],
            n_cells=1234,
            repo_dir=consolidate.REPO_DIR,
        )
        self.assertEqual(provenance["slides_requested"], 610)
        self.assertEqual(provenance["slides_extracted"], 603)
        self.assertEqual(provenance["slides_used"], 124)
        self.assertEqual(provenance["slides_missing"], 2)
        self.assertEqual(provenance["n_cells"], 1234)
        self.assertEqual(provenance["class_names"], ["Apoptotic Body", "Epithelial"])
        self.assertEqual(provenance["centroids_dir"], "/tmp/centroids")
        self.assertEqual(provenance["cohort_csv"], "/tmp/liver_cohort.csv")
        self.assertRegex(provenance["repository_commit"], r"^([0-9a-f]{40}|unknown)$")
        self.assertRegex(provenance["created"], r"^\d{4}-\d{2}-\d{2}T")
        self.assertEqual(provenance["missing_slide_ids"], ["GTEX-BBBB-0126", "GTEX-CCCC-0126"])


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

    def test_writes_provenance_json_when_asked(self):
        self.add_npz("GTEX-AAAA-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        provenance_path = os.path.join(self.tmp.name, "provenance.json")

        exit_code = consolidate.main(
            [
                "--centroids-dir",
                self.centroid_dir,
                "--cohort",
                self.cohort_csv,
                "--out",
                out,
                "--provenance-out",
                provenance_path,
            ]
        )
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual(provenance["slides_requested"], 1)
        self.assertEqual(provenance["slides_extracted"], 1)
        self.assertEqual(provenance["slides_used"], 1)
        self.assertEqual(provenance["slides_missing"], 0)
        self.assertEqual(provenance["n_cells"], 1)
        self.assertEqual(provenance["class_names"], ["Fibroblasts"])
        self.assertEqual(provenance["centroids_dir"], os.path.abspath(self.centroid_dir))

    def test_provenance_separates_extracted_from_used(self):
        self.add_npz("GTEX-AAAA-0126")
        self.add_npz("GTEX-BBBB-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        provenance_path = os.path.join(self.tmp.name, "provenance.json")

        consolidate.main(
            [
                "--centroids-dir",
                self.centroid_dir,
                "--cohort",
                self.cohort_csv,
                "--out",
                out,
                "--provenance-out",
                provenance_path,
                "--filter-to-cohort",
            ]
        )
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)

        self.assertEqual(provenance["slides_requested"], 1)
        self.assertEqual(provenance["slides_extracted"], 2)
        self.assertEqual(provenance["slides_used"], 1)
        self.assertEqual(provenance["slides_missing"], 0)

    def run_main(self, out, *extra):
        return consolidate.main(
            ["--centroids-dir", self.centroid_dir, "--cohort", self.cohort_csv, "--out", out, *extra]
        )

    def test_slides_out_lists_row_ranges_per_slide(self):
        write_npz(
            os.path.join(self.centroid_dir, "GTEX-BBBB-0126.npz"),
            [[1, 1], [2, 2]],
            [0, 0],
            ["Fibroblasts"],
        )
        self.add_npz("GTEX-AAAA-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126"), cohort_row("GTEX-BBBB-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        slides = os.path.join(self.tmp.name, "slides.csv")

        exit_code = self.run_main(out, "--slides-out", slides)
        table = pd.read_csv(slides, dtype=str, keep_default_na=False)

        self.assertEqual(exit_code, 0)
        self.assertEqual(list(table.columns), COHORT_COLUMNS + ["n_cells", "row_start", "row_end"])
        self.assertEqual(list(table["Tissue Sample ID"]), ["GTEX-AAAA-0126", "GTEX-BBBB-0126"])
        self.assertEqual(list(table["n_cells"]), ["1", "2"])
        self.assertEqual(list(table["row_start"]), ["0", "1"])
        self.assertEqual(list(table["row_end"]), ["1", "3"])

    def test_small_objects_get_int8_slide_codes_and_matching_ranges(self):
        slide_ids = ["GTEX-AAAA-0126", "GTEX-BBBB-0126", "GTEX-CCCC-0126"]
        for slide_id in slide_ids:
            self.add_npz(slide_id)
        write_cohort(self.cohort_csv, [cohort_row(slide_id) for slide_id in slide_ids])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        slides = os.path.join(self.tmp.name, "slides.csv")

        self.run_main(out, "--slides-out", slides)
        table = pd.read_csv(slides)
        with h5py.File(out, "r") as handle:
            codes = handle["obs/slide_id/codes"][:]

        self.assertEqual(codes.dtype, np.int8)
        for position, row in table.iterrows():
            self.assertTrue((codes[row["row_start"] : row["row_end"]] == position).all())
        self.assertEqual(table["row_end"].iloc[-1], len(codes))

    def test_metadata_columns_flag_replaces_the_default(self):
        self.add_npz("GTEX-AAAA-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")

        self.run_main(out, "--metadata-columns", "Subject ID", "Sex")
        adata = ad.read_h5ad(out)

        self.assertEqual(
            list(adata.obs.columns), ["slide_id", "cell_type", "prob", "Subject ID", "Sex"]
        )

    def test_writes_through_a_temporary_file(self):
        self.add_npz("GTEX-AAAA-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        with open(f"{out}.tmp", "wb") as handle:
            handle.write(b"left over by a killed task")

        exit_code = self.run_main(out)

        self.assertEqual(exit_code, 0)
        self.assertEqual(ad.read_h5ad(out).n_obs, 1)
        self.assertFalse(os.path.exists(f"{out}.tmp"))

    def test_provenance_lists_missing_slide_ids(self):
        self.add_npz("GTEX-AAAA-0126")
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126"), cohort_row("GTEX-CCCC-0126")])
        out = os.path.join(self.tmp.name, "cohort.h5ad")
        provenance_path = os.path.join(self.tmp.name, "provenance.json")

        self.run_main(out, "--provenance-out", provenance_path)
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)

        self.assertEqual(provenance["missing_slide_ids"], ["GTEX-CCCC-0126"])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.centroid_dir = os.path.join(self.tmp.name, "centroids")
        os.makedirs(self.centroid_dir)
        self.cohort_csv = os.path.join(self.tmp.name, "cohort.csv")

    def test_merges_extra_provenance_and_returns_a_summary(self):
        write_npz(
            os.path.join(self.centroid_dir, "GTEX-AAAA-0126.npz"), [[1, 1]], [0], ["Fibroblasts"]
        )
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        provenance_path = os.path.join(self.tmp.name, "provenance.json")

        summary = consolidate.run(
            self.centroid_dir,
            self.cohort_csv,
            os.path.join(self.tmp.name, "cohort.h5ad"),
            provenance_out=provenance_path,
            extra_provenance={"tissue": "Liver"},
        )
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)

        self.assertEqual(summary, {"n_obs": 1, "n_slides": 1, "missing_slide_ids": []})
        self.assertEqual(provenance["tissue"], "Liver")

    def test_raises_when_no_npz_exist(self):
        write_cohort(self.cohort_csv, [cohort_row("GTEX-AAAA-0126")])
        with self.assertRaises(consolidate.ConsolidateError):
            consolidate.run(
                self.centroid_dir, self.cohort_csv, os.path.join(self.tmp.name, "x.h5ad")
            )
