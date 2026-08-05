import importlib
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd


def load_liver():
    return importlib.import_module("liver")


class MainTests(unittest.TestCase):
    def test_main_runs_histoplus_download(self):
        df_samples = pd.DataFrame(
            {"SAMPID": ["GTEX-1117F-0626-SM-5GZZY"], "SMTS": ["Liver"]}
        )
        df_subjects = pd.DataFrame(
            {"SUBJID": ["GTEX-1117F"], "AGE": ["60-69"], "SEX": [1]}
        )
        with mock.patch(
            "pandas.read_csv", side_effect=[df_samples, df_subjects]
        ):
            with tempfile.TemporaryDirectory() as tmp:
                old_cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    liver = load_liver()
                    with mock.patch.object(
                        liver, "download_histoplus"
                    ) as mock_histo:
                        liver.main()
                finally:
                    os.chdir(old_cwd)
        self.assertEqual(mock_histo.call_args.args[0], ["GTEX-1117F-0626"])
