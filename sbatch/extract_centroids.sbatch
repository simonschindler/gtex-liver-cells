#!/bin/bash
#SBATCH --job-name=gtex_extract_centroids
#SBATCH --output=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_extract_centroids_%j.out
#SBATCH --error=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_extract_centroids_%j.err
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=localtmp:20
#SBATCH --constraint=ssd
#SBATCH --license=scratch-highio

# Extract cell centroids for the cohort's HistoPlus files.
#
#   sbatch --export=ALL,HISTOPLUS_DIR=/path,CENTROIDS_DIR=/path,COHORT_CSV=/path \
#       sbatch/extract_centroids.sbatch
#
# REPO_DIR defaults to the LISC checkout symlink; the data paths have no
# defaults so no lab path is baked into the Python code.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/lisc/home/user/schindlers/rep/rendeiro/tissuegeometry}"
: "${HISTOPLUS_DIR:?export HISTOPLUS_DIR=/path/to/histoplus}"
: "${CENTROIDS_DIR:?export CENTROIDS_DIR=/path/to/centroids}"
: "${COHORT_CSV:?export COHORT_CSV=/path/to/liver_cohort.csv}"

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:       $(hostname)"
echo "histoplus:  $HISTOPLUS_DIR"
echo "centroids:  $CENTROIDS_DIR"
echo "cohort csv: $COHORT_CSV"
echo "workers:    ${SLURM_CPUS_PER_TASK:-16}"

# The project venv is used directly on purpose: `uv run` re-syncs the
# environment and fails on the optional ../cpyrcolate path dependency.
"$REPO_DIR/.venv/bin/python" -m gtex_liver_cells.centroids \
    --histoplus-dir "$HISTOPLUS_DIR" \
    --out-dir "$CENTROIDS_DIR" \
    --slides "$COHORT_CSV" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

STATUS=$?
echo "npz now:    $(ls -1 "$CENTROIDS_DIR" 2>/dev/null | wc -l)"
exit $STATUS
