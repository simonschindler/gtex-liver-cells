import gzip
import json
import os
import tempfile
import unittest

import numpy as np

from gtex_liver_cells import centroids

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
