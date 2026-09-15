import json
import os
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon

from gtex_liver_cells import centroids

# Simplest geometries with hand-checkable centroids.
SQUARE = Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])  # centroid (1, 1)
TRIANGLE = Polygon([[0, 0], [2, 0], [0, 2], [0, 0]])  # centroid (2/3, 2/3)
# Two equal 1x1 squares side by side: area-weighted centroid (1, 0.5).
TWO_SQUARES = MultiPolygon(
    [
        Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
        Polygon([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]),
    ]
)


def write_source(path, rows):
    """Write a cell_types.gpd fixture; each row is (geometry, class, prob)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "geometry": pa.array(
                    [shapely.to_wkb(geometry) for geometry, _, _ in rows],
                    type=pa.binary(),
                ),
                "class": pa.array([label for _, label, _ in rows], type=pa.string()),
                "prob": pa.array([prob for _, _, prob in rows], type=pa.float64()),
                "cell_id": pa.array(range(len(rows)), type=pa.int64()),
            }
        ),
        path,
    )


def slide_source(root, slide_id):
    """Return the convention path `<root>/<TISSUE_ID>/cell_types.gpd`."""
    return os.path.join(root, slide_id, "cell_types.gpd")


class WkbBytesTests(unittest.TestCase):
    def test_returns_raw_bytes_for_a_plain_binary_column(self):
        column = pa.chunked_array([pa.array([b"abc", b"def"], type=pa.binary())])
        self.assertEqual(list(centroids.wkb_bytes(column)), [b"abc", b"def"])

    def test_converts_large_binary_to_binary(self):
        column = pa.chunked_array([pa.array([b"abc"], type=pa.large_binary())])
        self.assertEqual(list(centroids.wkb_bytes(column)), [b"abc"])

    def test_rejects_a_non_binary_column(self):
        column = pa.chunked_array([pa.array(["abc"], type=pa.string())])
        with self.assertRaises(ValueError):
            centroids.wkb_bytes(column)


class ExtractOneTests(unittest.TestCase):
    def test_writes_coords_labels_prob_and_class_names(self):
        rows = [
            (SQUARE, "Fibroblasts", 0.9),
            (TRIANGLE, "Apoptotic Body", 0.2),
            (TWO_SQUARES, "Fibroblasts", 0.7),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = slide_source(tmp, "GTEX-AAAA-0126")
            out = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            write_source(source, rows)

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
        self.assertEqual(prob.dtype, np.float32)
        np.testing.assert_allclose(prob, [0.9, 0.2, 0.7], rtol=1e-6)
        # The MultiPolygon must give the area-weighted centroid, not the
        # centroid of one of its parts.
        np.testing.assert_allclose(
            coords, [[1.0, 1.0], [2 / 3, 2 / 3], [1.0, 0.5]], atol=1e-4
        )
        self.assertEqual(provenance["method_version"], "2.0")
        self.assertEqual(provenance["method_version"], centroids.METHOD_VERSION)
        self.assertEqual(provenance["n_cells"], 3)
        self.assertEqual(provenance["n_classes"], 2)
        self.assertEqual(provenance["source_file"], "GTEX-AAAA-0126/cell_types.gpd")

    def test_rejects_empty_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = slide_source(tmp, "GTEX-AAAA-0126")
            out = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            write_source(source, [])
            with self.assertRaises(ValueError):
                centroids.extract_one(source, out)

    def test_rejects_unsupported_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = slide_source(tmp, "GTEX-AAAA-0126")
            out = os.path.join(tmp, "GTEX-AAAA-0126.npz")
            write_source(source, [(Point(0, 0), "Fibroblasts", 0.5)])
            with self.assertRaisesRegex(ValueError, "POINT"):
                centroids.extract_one(source, out)


class SelectSlidesTests(unittest.TestCase):
    def test_reports_missing_slides_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_source(
                slide_source(tmp, "GTEX-AAAA-0126"), [(SQUARE, "Fibroblasts", 0.5)]
            )
            sources, missing = centroids.select_slides(
                tmp, ["GTEX-AAAA-0126", "GTEX-BBBB-0126"]
            )
            expected_path = slide_source(tmp, "GTEX-AAAA-0126")
        self.assertEqual([slide_id for slide_id, _ in sources], ["GTEX-AAAA-0126"])
        self.assertEqual(sources[0][1], expected_path)
        self.assertEqual(missing, ["GTEX-BBBB-0126"])

    def test_discovers_slides_when_no_list_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            for slide_id in ("GTEX-AAAA-0126", "GTEX-BBBB-0126"):
                write_source(
                    slide_source(tmp, slide_id), [(SQUARE, "Fibroblasts", 0.5)]
                )
            # A slide directory without cell_types.gpd and a stray file must
            # both be ignored.
            os.makedirs(os.path.join(tmp, "GTEX-CCCC-0126"))
            with open(os.path.join(tmp, "README"), "w", encoding="utf-8") as handle:
                handle.write("not a slide\n")
            sources, missing = centroids.select_slides(tmp, None)
            names = [(slide_id, os.path.basename(path)) for slide_id, path in sources]
        self.assertEqual(
            names,
            [
                ("GTEX-AAAA-0126", "cell_types.gpd"),
                ("GTEX-BBBB-0126", "cell_types.gpd"),
            ],
        )
        self.assertEqual(missing, [])

    def test_reads_slide_ids_from_a_cohort_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cohort.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('"Tissue Sample ID","label"\n"GTEX-AAAA-0126","healthy"\n')
            self.assertEqual(centroids.read_slide_ids(path), ["GTEX-AAAA-0126"])


class ExtractManyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = slide_source(self.tmp.name, "GTEX-AAAA-0126")
        write_source(self.source, [(SQUARE, "Fibroblasts", 0.5)])
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
            ("GTEX-CCCC-0126", slide_source(self.tmp.name, "GTEX-CCCC-0126")),
        ]
        counts = centroids.extract_many(sources, self.out_dir, workers=1, overwrite=False)
        self.assertEqual(counts["extracted"], 1)
        self.assertEqual(counts["missing"], 1)

    def test_counts_unsupported_geometry_as_failed(self):
        source = slide_source(self.tmp.name, "GTEX-DDDD-0126")
        write_source(source, [(Point(0, 0), "Fibroblasts", 0.5)])
        counts = centroids.extract_many(
            [("GTEX-DDDD-0126", source)], self.out_dir, workers=1, overwrite=False
        )
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["extracted"], 0)


class MainTests(unittest.TestCase):
    def test_missing_input_directory_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = centroids.main(
                [
                    "--parquet-dir",
                    os.path.join(tmp, "does-not-exist"),
                    "--out-dir",
                    os.path.join(tmp, "out"),
                ]
            )
        self.assertEqual(exit_code, 1)

    def test_extracts_requested_slides_from_the_convention_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "datasets_pq_h5ad")
            out_dir = os.path.join(tmp, "centroids")
            csv_path = os.path.join(tmp, "cohort.csv")
            write_source(
                slide_source(root, "GTEX-AAAA-0126"), [(SQUARE, "Fibroblasts", 0.5)]
            )
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write('"Tissue Sample ID","label"\n"GTEX-AAAA-0126","healthy"\n')

            exit_code = centroids.main(
                [
                    "--parquet-dir",
                    root,
                    "--out-dir",
                    out_dir,
                    "--slides",
                    csv_path,
                    "--workers",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "GTEX-AAAA-0126.npz")))
