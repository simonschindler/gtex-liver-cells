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

print("Loading GTEx metadata files...")
df_samples = pd.read_csv(sample_attr_url, sep="\t")
df_subjects = pd.read_csv(subject_pheno_url, sep="\t")

# 2. Extract Donor ID (SUBJID) and Tissue Slide ID (TISSUE_ID)
df_samples["SUBJID"] = df_samples["SAMPID"].apply(lambda x: "-".join(x.split("-")[:2]))
df_samples["TISSUE_ID"] = df_samples["SAMPID"].apply(
    lambda x: x.split("-SM-")[0] if "-SM-" in x else x
)

# 3. Filter for Liver tissue specimens and deduplicate by Tissue ID
liver_all = df_samples[df_samples["SMTS"] == "Liver"].copy()
liver_wsi = liver_all.drop_duplicates(subset=["TISSUE_ID"]).copy()

# 4. Merge Subject Phenotypes (AGE, SEX)
liver_wsi = liver_wsi.merge(
    df_subjects[["SUBJID", "AGE", "SEX"]], on="SUBJID", how="left"
)

liver_wsi["TISSUE_ID"].to_csv("liver_wsis.csv", header=False, index=False)

# ---------------------------------------------------------------------------
# Download HistoPlus geojson files from CEMB
# ---------------------------------------------------------------------------

REMOTE_HOST = "sschindler@login.int.cemm.at"
REMOTE_BASE = "/nobackup/lab_rendeiro/projects/histopath/data/gtex/HistoPlus"
LOCAL_DIR = os.path.expanduser("~/data/GTEX/histoplus")


def check_ssh() -> bool:
    """Test that SSH to the remote host works without a password prompt."""
    print(f"Checking SSH connection to {REMOTE_HOST} ...", end=" ", flush=True)
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10",
         "-o", "BatchMode=yes",
         "-o", "StrictHostKeyChecking=accept-new",
         REMOTE_HOST, "echo ok"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and "ok" in result.stdout:
        print("OK")
        return True
    print("FAILED")
    print(f"  stderr: {result.stderr.strip()}")
    return False


def download_histoplus(tissue_ids: list[str]) -> None:
    """Download .histoplus.geojson.gz files for each tissue id, skipping
    those already present locally."""
    os.makedirs(LOCAL_DIR, exist_ok=True)

    if not check_ssh():
        print("Aborting: SSH connection could not be established.", file=sys.stderr)
        sys.exit(1)

    total = len(tissue_ids)
    done = skipped = failed = 0

    pbar = tqdm(tissue_ids, unit="file", desc="Downloading")
    for tid in pbar:
        local_path = os.path.join(LOCAL_DIR, f"{tid}.histoplus.geojson.gz")
        if os.path.exists(local_path):
            pbar.set_postfix_str(f"{tid} (skipped)")
            skipped += 1
            continue

        remote_path = f"{REMOTE_HOST}:{REMOTE_BASE}/{tid}.histoplus.geojson.gz"
        pbar.set_postfix_str(f"{tid}")

        result = subprocess.run(
            ["rsync", "-az",
             "-e", "ssh -o ConnectTimeout=30 -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
             remote_path, local_path],
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


if __name__ == "__main__":
    # Read tissue IDs generated above and start the download
    with open("liver_wsis.csv") as fh:
        ids = [line.strip() for line in fh if line.strip()]
    print(f"Found {len(ids)} tissue IDs in liver_wsis.csv\n")
    download_histoplus(ids)
