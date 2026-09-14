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
