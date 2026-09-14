import csv
import os
import tempfile
import unittest
from unittest import mock

from gtex_meta import portal


def record(slide_id, tissue="Liver", hidden=False, notes=None, categories=None):
    return {
        "histologyImageId": slide_id,
        "tissueSiteDetail": tissue,
        "subjectId": "-".join(slide_id.split("-")[:2]),
        "sex": "male",
        "ageBracket": "60-69",
        "hardyScale": "Slow death",
        "pathologyNotes": notes,
        "pathologyNotesCategories": categories or {},
        "hide": hidden,
    }


class ToRowTests(unittest.TestCase):
    def test_maps_fields_and_joins_true_categories(self):
        row = portal.to_row(
            record(
                "GTEX-AAAA-0126",
                notes="2 pieces; congestion",
                categories={"congestion": True, "steatosis": False},
            )
        )
        self.assertEqual(
            row,
            [
                "GTEX-AAAA-0126",
                "Liver",
                "GTEX-AAAA",
                "male",
                "60-69",
                "Slow death",
                "congestion",
                "2 pieces; congestion",
                "false",
            ],
        )

    def test_marks_hidden_slides(self):
        row = portal.to_row(record("GTEX-AAAA-0126", hidden=True))
        self.assertEqual(row[-1], "true")

    def test_missing_note_and_categories_become_empty_strings(self):
        row = portal.to_row(record("GTEX-AAAA-0126"))
        self.assertEqual(row[6], "")
        self.assertEqual(row[7], "")


class PaginationTests(unittest.TestCase):
    def test_iter_slides_follows_paging_info(self):
        pages = [
            {
                "data": [record("GTEX-AAAA-0126"), record("GTEX-AAA1-0126")],
                "paging_info": {"numberOfPages": 2, "page": 0},
            },
            {
                "data": [record("GTEX-AAA2-0126")],
                "paging_info": {"numberOfPages": 2, "page": 1},
            },
        ]
        with mock.patch.object(portal, "fetch_page", side_effect=pages) as fake:
            slides = list(portal.iter_slides(per_page=2))
        self.assertEqual(len(slides), 3)
        self.assertEqual([call.args[0] for call in fake.call_args_list], [0, 1])

    def test_fetch_slides_filters_by_tissue(self):
        pages = [
            {
                "data": [record("GTEX-AAAA-0126"), record("GTEX-AAAB-0126", tissue="Skin")],
                "paging_info": {"numberOfPages": 1, "page": 0},
            }
        ]
        with mock.patch.object(portal, "fetch_page", side_effect=pages):
            rows = portal.fetch_slides("Liver")
        self.assertEqual([row[0] for row in rows], ["GTEX-AAAA-0126"])


class WriteCsvTests(unittest.TestCase):
    def test_writes_header_and_quotes_every_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "slides.csv")
            portal.write_csv([portal.to_row(record("GTEX-AAAA-0126"))], path)
            with open(path, newline="") as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines[0], ",".join(f'"{name}"' for name in portal.COLUMNS))
        self.assertTrue(lines[1].startswith('"GTEX-AAAA-0126"'))
        self.assertEqual(len(lines), 2)

    def test_main_writes_requested_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "liver.csv")
            rows = [portal.to_row(record("GTEX-AAAA-0126"))]
            with mock.patch.object(portal, "fetch_slides", return_value=rows):
                exit_code = portal.main(["--tissue", "Liver", "--out", path])
            with open(path, newline="") as handle:
                written = list(csv.reader(handle))
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(written), 2)
