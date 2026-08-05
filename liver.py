import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os
import sys
from tqdm import tqdm

# 1. Dataset URLs
sample_attr_url = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
subject_pheno_url = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

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

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

REMOTE_HOST = "sschindler@login.int.cemm.at"
REMOTE_BASE = "/nobackup/lab_rendeiro/projects/histopath/data/gtex/HistoPlus"
LOCAL_DIR = os.path.expanduser("~/data/GTEX/histoplus")

CENTROID_HOST = "schindlers@transfer01.lisc.univie.ac.at"
CENTROID_REMOTE_BASE = (
    "/lisc/data/scratch/menche/schindlers/tissuegeometry/data/cell_centroids"
)
CENTROID_LOCAL_DIR = os.path.expanduser("~/data/GTEX/pc")

SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
]
RSYNC_SSH = (
    "ssh -o ConnectTimeout=30 -o BatchMode=yes "
    "-o StrictHostKeyChecking=accept-new"
)
RSYNC_TIMEOUT = 300


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
    follow_symlinks: bool = False,
    timeout: float = RSYNC_TIMEOUT,
) -> dict[str, int]:
    """Download <tissue_id><suffix> files from remote_host:remote_base.

    Skips files that already exist locally. Each rsync call is killed after
    `timeout` seconds and counted as failed, so a stalled transfer cannot
    hang the whole batch. Returns counts for downloaded, skipped, and
    failed files.
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

        rsync_flags = "-azL" if follow_symlinks else "-az"
        try:
            result = subprocess.run(
                ["rsync", rsync_flags, "-e", RSYNC_SSH, remote_path, local_path],
                capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            pbar.set_postfix_str(f"{tid} FAILED")
            tqdm.write(f"  {tid}: timed out after {timeout:.0f}s")
            failed += 1
            continue

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


def download_cell_centroids(tissue_ids: list[str]) -> dict[str, int]:
    """Download .npz cell-centroid files from the LISC transfer host."""
    return download_files(
        tissue_ids,
        remote_host=CENTROID_HOST,
        remote_base=CENTROID_REMOTE_BASE,
        local_dir=CENTROID_LOCAL_DIR,
        suffix=".npz",
        desc="Cell centroids",
        follow_symlinks=True,
    )


if __name__ == "__main__":
    main()
