import importlib
import os
import shutil
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


class CheckSshTests(unittest.TestCase):
    def test_check_ssh_success(self):
        liver = load_liver()
        with mock.patch.object(liver.subprocess, "run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="ok\n", stderr="")
            self.assertTrue(liver.check_ssh("user@example.org"))
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], "ssh")
        self.assertEqual(cmd[-2], "user@example.org")
        self.assertEqual(cmd[-1], "echo ok")

    def test_check_ssh_failure(self):
        liver = load_liver()
        with mock.patch.object(liver.subprocess, "run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=255, stdout="", stderr="denied")
            self.assertFalse(liver.check_ssh("user@example.org"))


class DownloadFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.real_tmp = os.path.realpath(self.tmpdir.name)

    def test_downloads_missing_and_skips_existing(self):
        liver = load_liver()
        ids = ["GTEX-14AS3-0126", "GTEX-1117F-0626"]
        existing = os.path.join(self.real_tmp, "GTEX-1117F-0626.npz")
        open(existing, "w").close()

        with mock.patch.object(liver, "check_ssh", return_value=True), \
             mock.patch.object(liver.os, "makedirs"), \
             mock.patch.object(
                 liver.os.path, "exists", side_effect=lambda p: p == existing
             ), \
             mock.patch.object(liver.subprocess, "run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            counts = liver.download_files(
                ids,
                remote_host="user@example.org",
                remote_base="/remote/base",
                local_dir=self.tmpdir.name,
                suffix=".npz",
            )

        self.assertEqual(counts, {"downloaded": 1, "skipped": 1, "failed": 0})
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], "rsync")
        self.assertEqual(cmd[-2], "user@example.org:/remote/base/GTEX-14AS3-0126.npz")
        self.assertEqual(cmd[-1], os.path.join(self.real_tmp, "GTEX-14AS3-0126.npz"))

    def test_failed_rsync_is_counted(self):
        liver = load_liver()
        with mock.patch.object(liver, "check_ssh", return_value=True), \
             mock.patch.object(liver.os, "makedirs"), \
             mock.patch.object(liver.os.path, "exists", return_value=False), \
             mock.patch.object(liver.subprocess, "run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=23, stderr="error")
            counts = liver.download_files(
                ["GTEX-14AS3-0126"],
                "user@example.org",
                "/remote/base",
                self.tmpdir.name,
                ".npz",
            )
        self.assertEqual(counts, {"downloaded": 0, "skipped": 0, "failed": 1})

    def test_raises_when_ssh_check_fails(self):
        liver = load_liver()
        with mock.patch.object(liver, "check_ssh", return_value=False), \
             mock.patch.object(liver.os, "makedirs"):
            with self.assertRaises(ConnectionError):
                liver.download_files(
                    [], "user@example.org", "/remote/base", self.tmpdir.name, ".npz"
                )

    def test_resolves_local_dir_symlink(self):
        liver = load_liver()
        real_dir = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, real_dir)
        link_dir = os.path.join(self.tmpdir.name, "link")
        os.symlink(real_dir, link_dir)

        with mock.patch.object(liver, "check_ssh", return_value=True), \
             mock.patch.object(liver.os.path, "exists", return_value=False), \
             mock.patch.object(liver.subprocess, "run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stderr="")
            liver.download_files(
                ["GTEX-14AS3-0126"],
                "user@example.org",
                "/remote/base",
                link_dir,
                ".npz",
            )

        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[-1], os.path.join(real_dir, "GTEX-14AS3-0126.npz"))


class HistoplusWrapperTests(unittest.TestCase):
    def test_histoplus_delegates_to_download_files(self):
        liver = load_liver()
        with mock.patch.object(
            liver,
            "download_files",
            return_value={"downloaded": 0, "skipped": 1, "failed": 0},
        ) as mock_df:
            liver.download_histoplus(["GTEX-13VXT-0626"])
        self.assertEqual(mock_df.call_args.args[0], ["GTEX-13VXT-0626"])
        self.assertEqual(mock_df.call_args.kwargs["remote_host"], liver.REMOTE_HOST)
        self.assertEqual(mock_df.call_args.kwargs["remote_base"], liver.REMOTE_BASE)
        self.assertEqual(mock_df.call_args.kwargs["local_dir"], liver.LOCAL_DIR)
        self.assertEqual(mock_df.call_args.kwargs["suffix"], ".histoplus.geojson.gz")
