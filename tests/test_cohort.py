import unittest

import pandas as pd

from gtex_meta import cohort


def slide(slide_id="GTEX-AAAA-0126", categories="", notes=""):
    return pd.Series(
        {
            "Tissue Sample ID": slide_id,
            "Tissue": "Liver",
            "Pathology Categories": categories,
            "Pathology Notes": notes,
        }
    )


class SplitCategoriesTests(unittest.TestCase):
    def test_splits_on_commas_and_ignores_blanks(self):
        self.assertEqual(
            cohort.split_categories("fibrosis, inflammation"), {"fibrosis", "inflammation"}
        )
        self.assertEqual(cohort.split_categories(""), set())


class HasFindingTests(unittest.TestCase):
    def test_detects_positive_finding(self):
        self.assertTrue(cohort.has_finding("2 pieces; <10% macrovesicular fat"))

    def test_ignores_negated_findings(self):
        self.assertFalse(cohort.has_finding("2 pieces; not congested; no fat"))
        self.assertFalse(cohort.has_finding("2 pieces, no significant steatosis/congestion"))

    def test_detects_findings_that_are_easy_to_miss(self):
        self.assertTrue(cohort.has_finding("several bile duct hamartomas occupy 5%"))
        self.assertTrue(cohort.has_finding("Mild central hypoxia"))
        self.assertTrue(cohort.has_finding("central sinusoidal dilatation; focal fragmentation"))


class LabelRowTests(unittest.TestCase):
    def test_curated_cirrhosis(self):
        label, reason = cohort.label_row(
            slide(categories="cirrhosis, steatosis", notes="cirrhosis")
        )
        self.assertEqual(label, "cirrhosis")
        self.assertEqual(reason, "curated cirrhosis")

    def test_curated_clean(self):
        label, reason = cohort.label_row(
            slide(categories="no_abnormalities", notes="no abnormalities")
        )
        self.assertEqual(label, "healthy")
        self.assertEqual(reason, "curated no_abnormalities")

    def test_uncategorized_note_without_finding_is_healthy(self):
        label, reason = cohort.label_row(slide(notes="2 pieces; not congested; no fat"))
        self.assertEqual(label, "healthy")
        self.assertIn("no finding", reason)

    def test_uncategorized_note_with_finding_is_other(self):
        label, reason = cohort.label_row(slide(notes="2 pieces; ~ 5% macrovesicular fat."))
        self.assertEqual(label, "other")
        self.assertIn("finding", reason)

    def test_empty_note_is_unknown_not_healthy(self):
        label, reason = cohort.label_row(slide(notes=""))
        self.assertEqual(label, "other")
        self.assertIn("unknown", reason)

    def test_autolyzed_slide_is_other(self):
        for note in (
            "2 alquots, ~8x7mm, badly autolyzed",
            "2 pieces ~7x5mm. No steatosis, moderately autolyzed",
            "2 pieces; focally moderate autolysis",
        ):
            label, reason = cohort.label_row(slide(notes=note))
            self.assertEqual(label, "other", note)
            self.assertIn("preservation", reason)


class BuildCohortTests(unittest.TestCase):
    def test_adds_label_columns(self):
        slides = pd.DataFrame(
            [
                {
                    **slide(categories="cirrhosis", notes="cirrhosis").to_dict(),
                    "Tissue Sample ID": "GTEX-A-0126",
                },
                {
                    **slide(notes="2 pieces").to_dict(),
                    "Tissue Sample ID": "GTEX-B-0126",
                },
            ]
        )
        result = cohort.build_cohort(slides)
        self.assertEqual(list(result["label"]), ["cirrhosis", "healthy"])
        for column in ("label_reason", "quality_flag", "has_finding_text"):
            self.assertIn(column, result.columns)

    def test_rejects_non_liver_rows(self):
        slides = pd.DataFrame([{**slide().to_dict(), "Tissue": "Skin"}])
        with self.assertRaises(ValueError):
            cohort.build_cohort(slides)
