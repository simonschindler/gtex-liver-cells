#!/bin/bash
#SBATCH --job-name=gtex_extract_centroids
#SBATCH --account=lab_rendeiro
#SBATCH --partition=mediumq
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --output=logs/gtex_extract_centroids_%j.out
#SBATCH --error=logs/gtex_extract_centroids_%j.err

# Extract cell centroids and class labels from the Parquet cell-type source.
#
# Submit from the output directory so that `logs/` receives the SLURM output:
#
#   mkdir -p "$OUT_DIR/logs" && cd "$OUT_DIR"
#   sbatch --export=ALL,PARQUET_DIR=$PARQUET_DIR,CENTROIDS_DIR=$CENTROIDS_DIR,COHORT_CSV=$COHORT_CSV \
#       "$REPO_DIR/sbatch/extract_centroids.sbatch"
#
# REPO_DIR defaults to the CEMM checkout; the data paths have no defaults so no
# lab path is baked into the Python code.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/sschindler/rep/gtex-liver-cells}"
: "${PARQUET_DIR:?export PARQUET_DIR=/path/to/datasets_pq_h5ad}"
: "${CENTROIDS_DIR:?export CENTROIDS_DIR=/path/to/centroids}"
: "${COHORT_CSV:?export COHORT_CSV=/path/to/liver_cohort.csv}"

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:       $(hostname)"
echo "parquet:    $PARQUET_DIR"
echo "centroids:  $CENTROIDS_DIR"
echo "cohort csv: $COHORT_CSV"
echo "workers:    ${SLURM_CPUS_PER_TASK:-16}"

# The project venv is used directly on purpose: `uv run` re-syncs the
# environment on every invocation, which is not what a batch job should pay.
"$REPO_DIR/.venv/bin/python" -m gtex_liver_cells.centroids \
    --parquet-dir "$PARQUET_DIR" \
    --out-dir "$CENTROIDS_DIR" \
    --slides "$COHORT_CSV" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

STATUS=$?
echo "npz now:    $(ls -1 "$CENTROIDS_DIR" 2>/dev/null | wc -l)"
exit $STATUS
