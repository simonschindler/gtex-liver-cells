"""The per-tissue sbatch script and its launcher, run under plain bash.

SLURM is not needed: the sbatch script's #SBATCH lines are comments to bash,
and the launcher finds a stub `sbatch` first on PATH that records its
arguments instead of submitting anything.
"""

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_tissues import make_fixture, quietly

from gtex_liver_cells import tissues

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "sbatch" / "build_tissues.sbatch"
LAUNCHER = REPO / "scripts" / "all_tissues.sh"

STUB_SBATCH = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_LOG"
echo "Submitted batch job 1"
"""


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.portal_csv, self.parquet_dir = make_fixture(self.tmp.name)
        self.root = os.path.join(self.tmp.name, "gtex_tissues")
        self.stub_dir = os.path.join(self.tmp.name, "bin")
        self.stub_log = os.path.join(self.tmp.name, "sbatch-args.txt")
        os.makedirs(self.stub_dir)
        stub = os.path.join(self.stub_dir, "sbatch")
        with open(stub, "w", encoding="utf-8") as handle:
            handle.write(STUB_SBATCH)
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR)

    def env(self, **overrides):
        env = dict(os.environ)
        for name in ("ARRAY", "MAX_PARALLEL", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK"):
            env.pop(name, None)
        env.update(
            REPO_DIR=str(REPO),
            PYTHON=sys.executable,
            ROOT=self.root,
            PARQUET_DIR=self.parquet_dir,
            STUB_LOG=self.stub_log,
            PATH=f"{self.stub_dir}{os.pathsep}{env.get('PATH', '')}",
        )
        env.update(overrides)
        return env

    def bash(self, script, **overrides):
        return subprocess.run(
            ["bash", str(script)],
            env=self.env(**overrides),
            capture_output=True,
            text=True,
            check=False,
        )

    def submitted_args(self):
        with open(self.stub_log, encoding="utf-8") as handle:
            return handle.read().splitlines()

    def stage_portal_table(self):
        """Put the Portal table where the launcher looks, so it fetches nothing."""
        os.makedirs(self.root, exist_ok=True)
        shutil.copy(self.portal_csv, os.path.join(self.root, "gtex_portal_slides.csv"))

    def test_scripts_parse(self):
        for script in (SBATCH, LAUNCHER):
            self.assertEqual(subprocess.run(["bash", "-n", str(script)], check=False).returncode, 0, script)

    def test_sbatch_builds_one_task_under_plain_bash(self):
        quietly(tissues.prepare, self.portal_csv, self.root, self.parquet_dir)

        result = self.bash(SBATCH, SLURM_ARRAY_TASK_ID="0", SLURM_CPUS_PER_TASK="1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.root, "esophagus_mucosa", "gtex_esophagus_mucosa_cells.h5ad")
            )
        )

    def test_sbatch_requires_an_array_task_id(self):
        result = self.bash(SBATCH)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SLURM_ARRAY_TASK_ID", result.stderr)

    def test_launcher_submits_tissues_with_sources(self):
        self.stage_portal_table()

        result = self.bash(LAUNCHER)

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.submitted_args()
        self.assertIn("--array=0,1%8", args)
        self.assertTrue(args[-1].endswith("sbatch/build_tissues.sbatch"), args)
        self.assertTrue(
            any(
                arg.startswith("--export=ALL,PARQUET_DIR=") and f"ROOT={self.root}" in arg
                for arg in args
            ),
            args,
        )
        self.assertTrue(os.path.isdir(os.path.join(self.root, "logs")))

    def test_launcher_honours_array_and_max_parallel(self):
        self.stage_portal_table()

        result = self.bash(LAUNCHER, ARRAY="1", MAX_PARALLEL="2")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--array=1%2", self.submitted_args())

    def test_launcher_submits_nothing_without_sources(self):
        self.stage_portal_table()
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)

        result = self.bash(LAUNCHER, PARQUET_DIR=empty)

        self.assertEqual(result.returncode, 1)
        self.assertIn("ERROR", result.stderr)
        self.assertFalse(os.path.exists(self.stub_log))


if __name__ == "__main__":
    unittest.main()
