# Cell Centroids Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `liver.py` so that, after the existing HistoPlus download, it also rsyncs per-tissue `.npz` cell-centroid files from the LISC transfer host into `~/data/GTEX/pc`, using the same SSH check, skip-if-present, rsync, and progress-bar behavior.

**Architecture:** Generalize the existing download loop into one parameterized helper (`download_files`) that takes remote host, remote base, local dir, and suffix, then wrap it with per-dataset functions for HistoPlus and cell centroids. Move the module-level metadata fetch into a `main()` function so the module can be imported by tests without network access. Each dataset gets its own SSH connectivity check before downloading.

**Tech Stack:** Python 3.14, stdlib `unittest`, `subprocess` + `rsync` over SSH, `tqdm` (already installed), `pandas` (existing).

## Global Constraints

- Remote endpoint for cell centroids, verbatim: `schindlers@transfer01.lisc.univie.ac.at:/lisc/data/scratch/menche/schindlers/tissuegeometry/data/cell_centroids/GTEX-14AS3-0126.npz`
- Local destination, verbatim: `~/data/GTEX/pc`
- File naming: `<TISSUE_ID>.npz`; tissue IDs come from `liver_wsis.csv` (one per line, no header).
- Keep existing HistoPlus behavior: skip files that already exist locally, `rsync -az` with the current SSH options, `tqdm` progress bar, per-file failure reporting, final summary.
- Do not add dependencies: tests use stdlib `unittest`; `tqdm` and `pandas` are already used.
- Do not change the GTEx metadata fetch or `liver_wsis.csv` generation logic; only relocate it into `main()`.
- Independence: an SSH failure for one dataset must not stop the other dataset. `main()` attempts both downloads and only exits nonzero after both have been tried.
- Resolve symlinks: `download_files` must call `os.path.realpath(local_dir)` before creating the directory or checking local files, so symlinked destinations like `~/data/GTEX/pc` resolve to their real path.
- ASCII only in code.
- Commands run from `/Users/simon/Desktop/gtex_meta`; use `.venv/bin/python`.

## File Structure

- Modify: `liver.py` - add `main()`, generic `download_files()`, `download_cell_centroids()`, per-host SSH check; keep `download_histoplus()` as a thin wrapper.
- Create: `tests/test_import_safety.py` - proves importing `liver` has no network side effects.
- Create: `tests/test_liver.py` - unit tests for `check_ssh`, `download_files`, both wrappers, and `main()`.
- No other files change.

---

### Task 1: Make `liver.py` import-safe with a `main()` function

**Files:**
- Modify: `liver.py:1-32` (move metadata fetch), `liver.py:104-109` (replace `__main__` block)
- Create: `tests/test_import_safety.py`
- Create: `tests/test_liver.py` (MainTests only in this task)

**Interfaces:**
- Consumes: nothing new.
- Produces: `main() -> None` - fetches GTEx metadata, writes `liver_wsis.csv`, reads the IDs back, and calls `download_histoplus(ids)`. The module must be importable with no network side effects.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_import_safety.py`:

```python
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
```

Create `tests/test_liver.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: FAIL. `test_import_does_not_fetch_metadata` raises `AssertionError` because the module-level `pd.read_csv` calls run during import; `test_main_runs_histoplus_download` fails because `main` does not exist yet.

- [ ] **Step 3: Move the metadata fetch into `main()`**

In `liver.py`, replace the module-level metadata block (currently lines 12-32) and the old `__main__` block (currently lines 104-109) with:

```python
def main() -> None:
    print("Loading GTEx metadata files...")
    df_samples = pd.read_csv(sample_attr_url, sep="\t")
    df_subjects = pd.read_csv(subject_pheno_url, sep="\t")

    df_samples["SUBJID"] = df_samples["SAMPID"].apply(
        lambda x: "-".join(x.split("-")[:2])
    )
    df_samples["TISSUE_ID"] = df_samples["SAMPID"].apply(
        lambda x: x.split("-SM-")[0] if "-SM-" in x else x
    )

    liver_all = df_samples[df_samples["SMTS"] == "Liver"].copy()
    liver_wsi = liver_all.drop_duplicates(subset=["TISSUE_ID"]).copy()

    liver_wsi = liver_wsi.merge(
        df_subjects[["SUBJID", "AGE", "SEX"]], on="SUBJID", how="left"
    )

    liver_wsi["TISSUE_ID"].to_csv("liver_wsis.csv", header=False, index=False)

    with open("liver_wsis.csv") as fh:
        ids = [line.strip() for line in fh if line.strip()]
    print(f"Found {len(ids)} tissue IDs in liver_wsis.csv\n")

    download_histoplus(ids)


if __name__ == "__main__":
    main()
```

`download_histoplus` and all constants above it stay exactly as they are for now.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: 2 PASS. Import is side-effect-free (no metadata fetch), and `main()` writes/reads `liver_wsis.csv` in the temp cwd then calls `download_histoplus(["GTEX-1117F-0626"])`.

- [ ] **Step 5: Commit**

```bash
git add liver.py tests/test_import_safety.py tests/test_liver.py
git commit -m "refactor: make liver.py import-safe with main()"
```

---

### Task 2: Generalize the download loop into `download_files`

**Files:**
- Modify: `liver.py:34-101` (download section)
- Test: `tests/test_liver.py`

**Interfaces:**
- Consumes: import-safe module from Task 1.
- Produces:
  - `check_ssh(remote_host: str) -> bool`
  - `download_files(tissue_ids: list[str], remote_host: str, remote_base: str, local_dir: str, suffix: str, desc: str = "Downloading") -> dict[str, int]` - returns `{"downloaded": int, "skipped": int, "failed": int}`; raises `ConnectionError` when the SSH check fails.
  - `download_histoplus(tissue_ids: list[str]) -> dict[str, int]` - now a thin wrapper over `download_files`.

- [ ] **Step 1: Write the failing tests**

Add `import shutil` to the imports at the top of `tests/test_liver.py`, then append:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: FAIL. `check_ssh` currently takes no arguments (`TypeError`), `download_files` does not exist (`AttributeError`), and `download_histoplus` does not call `download_files`.

- [ ] **Step 3: Replace the download section**

In `liver.py`, replace everything from the download comment banner through the end of the current `download_histoplus` function (currently lines 34-101) with:

```python
# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

REMOTE_HOST = "sschindler@login.int.cemm.at"
REMOTE_BASE = "/nobackup/lab_rendeiro/projects/histopath/data/gtex/HistoPlus"
LOCAL_DIR = os.path.expanduser("~/data/GTEX/histoplus")

SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
]
RSYNC_SSH = (
    "ssh -o ConnectTimeout=30 -o BatchMode=yes "
    "-o StrictHostKeyChecking=accept-new"
)


def check_ssh(remote_host: str) -> bool:
    """Test that SSH to remote_host works without a password prompt."""
    print(f"Checking SSH connection to {remote_host} ...", end=" ", flush=True)
    result = subprocess.run(
        ["ssh", *SSH_OPTS, remote_host, "echo ok"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and "ok" in result.stdout:
        print("OK")
        return True
    print("FAILED")
    print(f"  stderr: {result.stderr.strip()}")
    return False


def download_files(
    tissue_ids: list[str],
    remote_host: str,
    remote_base: str,
    local_dir: str,
    suffix: str,
    desc: str = "Downloading",
) -> dict[str, int]:
    """Download <tissue_id><suffix> files from remote_host:remote_base.

    Skips files that already exist locally. Returns counts for
    downloaded, skipped, and failed files.
    """
    local_dir = os.path.realpath(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if not check_ssh(remote_host):
        raise ConnectionError(f"SSH connection to {remote_host} failed")

    total = len(tissue_ids)
    done = skipped = failed = 0

    pbar = tqdm(tissue_ids, unit="file", desc=desc)
    for tid in pbar:
        local_path = os.path.join(local_dir, f"{tid}{suffix}")
        if os.path.exists(local_path):
            pbar.set_postfix_str(f"{tid} (skipped)")
            skipped += 1
            continue

        remote_path = f"{remote_host}:{remote_base}/{tid}{suffix}"
        pbar.set_postfix_str(f"{tid}")

        result = subprocess.run(
            ["rsync", "-az", "-e", RSYNC_SSH, remote_path, local_path],
            capture_output=True, text=True,
        )

        if result.returncode == 0:
            done += 1
        else:
            pbar.set_postfix_str(f"{tid} FAILED")
            tqdm.write(f"  {tid}: {result.stderr.strip()}")
            failed += 1

    pbar.set_postfix_str(f"done={done} skip={skipped} fail={failed}")
    pbar.close()
    print(f"\nDone: {done} downloaded, {skipped} skipped, {failed} failed "
          f"(out of {total})")
    return {"downloaded": done, "skipped": skipped, "failed": failed}


def download_histoplus(tissue_ids: list[str]) -> dict[str, int]:
    """Download .histoplus.geojson.gz files from the CEMB host."""
    return download_files(
        tissue_ids,
        remote_host=REMOTE_HOST,
        remote_base=REMOTE_BASE,
        local_dir=LOCAL_DIR,
        suffix=".histoplus.geojson.gz",
        desc="HistoPlus",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: ALL PASS (2 from Task 1 plus the 7 new ones).

- [ ] **Step 5: Commit**

```bash
git add liver.py tests/test_liver.py
git commit -m "refactor: generalize rsync download loop"
```

---

### Task 3: Add the cell-centroid endpoint and wrapper

**Files:**
- Modify: `liver.py` (add constants after `LOCAL_DIR`, add wrapper after `download_histoplus`)
- Test: `tests/test_liver.py`

**Interfaces:**
- Consumes: `download_files` from Task 2.
- Produces: `download_cell_centroids(tissue_ids: list[str]) -> dict[str, int]`, plus constants `CENTROID_HOST`, `CENTROID_REMOTE_BASE`, `CENTROID_LOCAL_DIR`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_liver.py`:

```python
class CellCentroidsWrapperTests(unittest.TestCase):
    def test_cell_centroids_uses_correct_endpoint(self):
        liver = load_liver()
        with mock.patch.object(
            liver,
            "download_files",
            return_value={"downloaded": 0, "skipped": 1, "failed": 0},
        ) as mock_df:
            liver.download_cell_centroids(["GTEX-14AS3-0126"])
        self.assertEqual(mock_df.call_args.args[0], ["GTEX-14AS3-0126"])
        self.assertEqual(
            mock_df.call_args.kwargs["remote_host"],
            "schindlers@transfer01.lisc.univie.ac.at",
        )
        self.assertEqual(
            mock_df.call_args.kwargs["remote_base"],
            "/lisc/data/scratch/menche/schindlers/tissuegeometry/data/cell_centroids",
        )
        self.assertEqual(
            mock_df.call_args.kwargs["local_dir"],
            os.path.expanduser("~/data/GTEX/pc"),
        )
        self.assertEqual(mock_df.call_args.kwargs["suffix"], ".npz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_liver.py -v`

Expected: FAIL with `AttributeError: module 'liver' has no attribute 'download_cell_centroids'`.

- [ ] **Step 3: Add the constants and wrapper**

In `liver.py`, directly after the `LOCAL_DIR` line, add:

```python
CENTROID_HOST = "schindlers@transfer01.lisc.univie.ac.at"
CENTROID_REMOTE_BASE = (
    "/lisc/data/scratch/menche/schindlers/tissuegeometry/data/cell_centroids"
)
CENTROID_LOCAL_DIR = os.path.expanduser("~/data/GTEX/pc")
```

Directly after `download_histoplus`, add:

```python
def download_cell_centroids(tissue_ids: list[str]) -> dict[str, int]:
    """Download .npz cell-centroid files from the LISC transfer host."""
    return download_files(
        tissue_ids,
        remote_host=CENTROID_HOST,
        remote_base=CENTROID_REMOTE_BASE,
        local_dir=CENTROID_LOCAL_DIR,
        suffix=".npz",
        desc="Cell centroids",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add liver.py tests/test_liver.py
git commit -m "feat: download cell centroid npz files"
```

---

### Task 4: Run both downloads from `main()`

**Files:**
- Modify: `liver.py` (`main()` body)
- Test: `tests/test_liver.py` (MainTests)

**Interfaces:**
- Consumes: `download_histoplus` and `download_cell_centroids` from Tasks 2-3.
- Produces: updated `main()` that runs both downloads, records `ConnectionError`s per dataset, still attempts the remaining dataset, and exits nonzero afterward if any SSH check failed.

- [ ] **Step 1: Update the failing test**

Replace `test_main_runs_histoplus_download` in `tests/test_liver.py` with:

```python
    def test_main_runs_both_downloads(self):
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
                    ) as mock_histo, mock.patch.object(
                        liver, "download_cell_centroids"
                    ) as mock_centro:
                        liver.main()
                finally:
                    os.chdir(old_cwd)
        self.assertEqual(mock_histo.call_args.args[0], ["GTEX-1117F-0626"])
        self.assertEqual(mock_centro.call_args.args[0], ["GTEX-1117F-0626"])

    def test_main_tries_second_host_after_first_fails(self):
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
                        liver,
                        "download_histoplus",
                        side_effect=ConnectionError("host unreachable"),
                    ), mock.patch.object(
                        liver, "download_cell_centroids"
                    ) as mock_centro, mock.patch.object(
                        liver.sys, "exit"
                    ) as mock_exit:
                        liver.main()
                finally:
                    os.chdir(old_cwd)
        mock_centro.assert_called_once_with(["GTEX-1117F-0626"])
        mock_exit.assert_called_once_with(1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/test_liver.py -v`

Expected: FAIL. `test_main_runs_both_downloads` fails because `main()` never calls `download_cell_centroids` (`mock_centro.call_args` is `None`); `test_main_tries_second_host_after_first_fails` fails because the current code calls `sys.exit(1)` immediately when `download_histoplus` raises, so `mock_centro` is never called.

- [ ] **Step 3: Update `main()` to run both downloads**

Replace the final two lines of `main()` (`download_histoplus(ids)` and its preceding blank line) with:

```python
    failures = 0
    for download in (download_histoplus, download_cell_centroids):
        try:
            download(ids)
        except ConnectionError as exc:
            print(f"Skipping {download.__name__}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        print(
            f"{failures} download(s) skipped because SSH connection failed",
            file=sys.stderr,
        )
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add liver.py tests/test_liver.py
git commit -m "feat: run cell centroid download from main"
```

---

### Task 5: Full verification and smoke test

**Files:**
- Verify: `liver.py`, `tests/`

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: confirmation that the script works end to end against the real hosts.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: ALL PASS (1 import-safety test, 2 main tests, 2 SSH tests, 4 download-file tests, 1 HistoPlus wrapper test, 1 centroids wrapper test).

- [ ] **Step 2: Syntax check**

Run: `.venv/bin/python -m py_compile liver.py`

Expected: exit code 0, no output.

- [ ] **Step 3: Smoke test against the real hosts**

Run: `.venv/bin/python liver.py`

Expected sequence:
1. Metadata loads, `liver_wsis.csv` is regenerated, "Found 231 tissue IDs" is printed.
2. "Checking SSH connection to sschindler@login.int.cemm.at ... OK".
3. HistoPlus progress bar runs; the 3 already-downloaded files are skipped.
4. "Checking SSH connection to schindlers@transfer01.lisc.univie.ac.at ... OK".
5. "Cell centroids" progress bar runs for the 231 `.npz` files.
6. Both summary lines print with `failed 0`.

Verify the output:

```bash
ls ~/data/GTEX/pc/ | wc -l
ls -l ~/data/GTEX/pc/GTEX-14AS3-0126.npz
```

Expected: 231 files; the named file exists with nonzero size.

- [ ] **Step 4: Commit any verification fixes**

If the smoke test revealed a bug, fix it, rerun Steps 1-3, then:

```bash
git add liver.py tests/
git commit -m "fix: verify cell centroid download end to end"
```

If no fixes were needed, this step is a no-op.
