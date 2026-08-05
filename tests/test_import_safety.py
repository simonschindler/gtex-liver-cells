import importlib
import unittest
from unittest import mock


class ImportSafetyTests(unittest.TestCase):
    def test_import_does_not_fetch_metadata(self):
        with mock.patch(
            "pandas.read_csv",
            side_effect=AssertionError("metadata fetched at import time"),
        ):
            module = importlib.import_module("liver")
        self.assertTrue(hasattr(module, "main"))
