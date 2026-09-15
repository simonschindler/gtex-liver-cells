#!/usr/bin/env bash
set -euo pipefail

export REPO_DIR="$HOME/rep/gtex-liver-cells"
export FULL_OUT_DIR=/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_liver_cells
export OUT_DIR="${FULL_OUT_DIR}_healthy_cirrhosis"
export CENTROIDS_DIR="$FULL_OUT_DIR/centroids"
export FULL_COHORT_CSV="$REPO_DIR/data/liver_cohort.csv"
export COHORT_CSV="$OUT_DIR/cohort_healthy_cirrhosis.csv"

mkdir -p "$OUT_DIR/logs"

# Keep only the healthy and cirrhosis rows of the labeled cohort, then
# consolidate those slides into their own object. Reuses the centroids from
# the full run (scripts/step_1.sh must have finished first).
"$REPO_DIR/.venv/bin/python" - "$FULL_COHORT_CSV" "$COHORT_CSV" <<'PY'
import csv
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = [row for row in reader if row["label"] in ("healthy", "cirrhosis")]
    fields = reader.fieldnames
with open(dst, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} slides to {dst}")
PY

cd "$OUT_DIR"

sbatch --export=ALL,CENTROIDS_DIR=$CENTROIDS_DIR,COHORT_CSV=$COHORT_CSV,OUT_DIR=$OUT_DIR,FILTER_TO_COHORT=1 \
    "$REPO_DIR/sbatch/consolidate_anndata.sbatch"
