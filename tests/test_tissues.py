import contextlib
import io
import os
import tempfile
import unittest

import pandas as pd
from test_centroids import SQUARE, TWO_SQUARES, slide_source, write_source

from gtex_liver_cells import portal, tissues

# (slide_id, Portal tissue, cells); cells None means the slide has no
# cell_types.gpd. The padded "Esophagus - Mucosa " is how the Portal spells
# three real slides.
FIXTURE_SLIDES = [
    ("GTEX-AAAA-0126", "Liver", [(SQUARE, "Fibroblasts", 0.9), (TWO_SQUARES, "Epithelial", 0.6)]),
    ("GTEX-BBBB-0126", "Liver", [(SQUARE, "Macrophages", 0.7)]),
    ("GTEX-CCCC-0126", "Esophagus - Mucosa ", [(SQUARE, "Epithelial", 0.8)]),
    ("GTEX-CCCD-0126", "Esophagus - Mucosa", None),
    ("GTEX-EEEE-0126", "Skin - Sun Exposed (Lower leg)", None),
]


def portal_row(slide_id, tissue):
    """One Portal table row in `portal.COLUMNS` order."""
    subject = "-".join(slide_id.split("-")[:2])
    return [slide_id, tissue, subject, "female", "50-59", "Slow death", "", "2 pieces", "false"]


def make_fixture(tmp, extra=()):
    """Write a Portal table and a Parquet root for FIXTURE_SLIDES plus `extra`.

    Returns (portal_csv, parquet_dir).
    """
    parquet_dir = os.path.join(tmp, "datasets_pq_h5ad")
    os.makedirs(parquet_dir, exist_ok=True)
    slides = [*FIXTURE_SLIDES, *extra]
    for slide_id, _, cells in slides:
        if cells is not None:
            write_source(slide_source(parquet_dir, slide_id), cells)
    portal_csv = os.path.join(tmp, "gtex_portal_slides.csv")
    portal.write_csv([portal_row(slide_id, tissue) for slide_id, tissue, _ in slides], portal_csv)
    return portal_csv, parquet_dir


def quietly(function, *args, **kwargs):
    """Call `function`, returning (result, captured stdout)."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        result = function(*args, **kwargs)
    return result, stdout.getvalue()


def portal_frame(rows):
    return pd.DataFrame(rows, columns=tissues.PORTAL_COLUMNS)


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
        slides = portal_frame(
            [
                portal_row("GTEX-AAAA-0126", "Liver"),
                portal_row("GTEX-BBBB-0126", "Liver"),
                portal_row("GTEX-CCCC-0126", "Esophagus - Mucosa "),
                portal_row("GTEX-CCCD-0126", "Esophagus - Mucosa"),
            ]
        )

        groups = tissues.split_tissues(slides)

        self.assertEqual(
            [(slug, tissue, len(rows)) for slug, tissue, rows in groups],
            [("esophagus_mucosa", "Esophagus - Mucosa", 2), ("liver", "Liver", 2)],
        )
        self.assertEqual(set(groups[0][2]["Tissue"]), {"Esophagus - Mucosa"})

    def test_refuses_a_duplicate_slide_id(self):
        slides = portal_frame(
            [portal_row("GTEX-AAAA-0126", "Liver"), portal_row("GTEX-AAAA-0126", "Lung")]
        )
        with self.assertRaisesRegex(ValueError, "GTEX-AAAA-0126"):
            tissues.split_tissues(slides)

    def test_refuses_two_tissues_with_one_slug(self):
        slides = portal_frame(
            [
                portal_row("GTEX-AAAA-0126", "Brain - Cortex"),
                portal_row("GTEX-BBBB-0126", "Brain Cortex"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "brain_cortex"):
            tissues.split_tissues(slides)

    def test_refuses_a_tissue_without_a_name(self):
        slides = portal_frame([portal_row("GTEX-AAAA-0126", " - ")])
        with self.assertRaisesRegex(ValueError, "GTEX-AAAA-0126"):
            tissues.split_tissues(slides)

    def test_refuses_a_table_without_the_portal_columns(self):
        slides = portal_frame([portal_row("GTEX-AAAA-0126", "Liver")]).drop(columns="Sex")
        with self.assertRaisesRegex(ValueError, "Sex"):
            tissues.split_tissues(slides)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.portal_csv, self.parquet_dir = make_fixture(self.tmp.name)
        self.root = os.path.join(self.tmp.name, "gtex_tissues")

    def prepare(self, parquet_dir=None):
        manifest, _ = quietly(
            tissues.prepare, self.portal_csv, self.root, parquet_dir or self.parquet_dir
        )
        return manifest

    def test_writes_per_tissue_slide_tables_and_the_manifest(self):
        manifest = self.prepare()

        self.assertEqual(list(manifest.columns), tissues.MANIFEST_COLUMNS)
        self.assertEqual(
            list(manifest["slug"]), ["esophagus_mucosa", "liver", "skin_sun_exposed_lower_leg"]
        )
        self.assertEqual(list(manifest["index"]), [0, 1, 2])
        self.assertEqual(list(manifest["n_slides"]), [2, 2, 1])
        self.assertEqual(list(manifest["n_with_source"]), [1, 2, 0])
        pd.testing.assert_frame_equal(tissues.read_manifest(self.root), manifest)
        liver = pd.read_csv(
            os.path.join(self.root, "liver", "portal_slides.csv"),
            dtype=str,
            keep_default_na=False,
        )
        self.assertEqual(list(liver.columns), tissues.PORTAL_COLUMNS)
        self.assertEqual(list(liver["Tissue Sample ID"]), ["GTEX-AAAA-0126", "GTEX-BBBB-0126"])
        esophagus = pd.read_csv(
            os.path.join(self.root, "esophagus_mucosa", "portal_slides.csv"), dtype=str
        )
        self.assertEqual(set(esophagus["Tissue"]), {"Esophagus - Mucosa"})

    def test_leaves_existing_outputs_alone(self):
        self.prepare()
        organ = os.path.join(self.root, "liver")
        kept = [
            os.path.join(organ, "centroids", "GTEX-AAAA-0126.npz"),
            os.path.join(organ, "gtex_liver_cells.h5ad"),
            os.path.join(organ, "slides.csv"),
            os.path.join(organ, "provenance.json"),
        ]
        os.makedirs(os.path.join(organ, "centroids"))
        for path in kept:
            with open(path, "wb") as handle:
                handle.write(b"keep")

        self.prepare()

        for path in kept:
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"keep", path)

    def test_array_spec_leaves_out_tissues_without_sources(self):
        self.assertEqual(tissues.array_spec(self.prepare()), "0,1")

    def test_array_spec_is_empty_without_sources(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        self.assertEqual(tissues.array_spec(self.prepare(empty)), "")

    def test_main_prepare_then_array(self):
        exit_code, _ = quietly(
            tissues.main,
            [
                "prepare",
                "--slides",
                self.portal_csv,
                "--root",
                self.root,
                "--parquet-dir",
                self.parquet_dir,
            ],
        )
        array_exit, printed = quietly(tissues.main, ["array", "--root", self.root])

        self.assertEqual(exit_code, 0)
        self.assertEqual(array_exit, 0)
        self.assertEqual(printed.strip(), "0,1")

    def test_main_prepare_rejects_a_missing_parquet_dir(self):
        missing = os.path.join(self.tmp.name, "does-not-exist")
        exit_code, _ = quietly(
            tissues.main,
            ["prepare", "--slides", self.portal_csv, "--root", self.root, "--parquet-dir", missing],
        )
        self.assertEqual(exit_code, 1)
        self.assertFalse(os.path.exists(os.path.join(self.root, "tissues.csv")))

    def test_main_prepare_rejects_a_missing_slides_csv(self):
        missing = os.path.join(self.tmp.name, "no-such-table.csv")
        exit_code, _ = quietly(
            tissues.main,
            ["prepare", "--slides", missing, "--root", self.root, "--parquet-dir", self.parquet_dir],
        )
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
