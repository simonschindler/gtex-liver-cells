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

    download_histoplus(ids)

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


if __name__ == "__main__":
    main()
