#!/bin/bash
#SBATCH --job-name=gtex_consolidate_anndata
#SBATCH --output=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_consolidate_anndata_%j.out
#SBATCH --error=/lisc/data/scratch/menche/schindlers/tissuegeometry/logs/gtex_consolidate_anndata_%j.err
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=localtmp:20
#SBATCH --constraint=ssd
#SBATCH --license=scratch-highio

# Consolidate every centroid npz into one AnnData object.
#
#   sbatch --export=ALL,CENTROIDS_DIR=/path,COHORT_CSV=/path,OUT_H5AD=/path \
#       sbatch/consolidate_anndata.sbatch
#
# Add FILTER_TO_COHORT=1 to keep only slides listed in the cohort CSV.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/lisc/home/user/schindlers/rep/rendeiro/tissuegeometry}"
: "${CENTROIDS_DIR:?export CENTROIDS_DIR=/path/to/centroids}"
: "${COHORT_CSV:?export COHORT_CSV=/path/to/liver_cohort.csv}"
: "${OUT_H5AD:?export OUT_H5AD=/path/to/output.h5ad}"

EXTRA_ARGS=()
if [ "${FILTER_TO_COHORT:-0}" = "1" ]; then
    EXTRA_ARGS+=(--filter-to-cohort)
fi

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:       $(hostname)"
echo "centroids:  $CENTROIDS_DIR"
echo "cohort csv: $COHORT_CSV"
echo "output:     $OUT_H5AD"

"$REPO_DIR/.venv/bin/python" -m gtex_meta.consolidate \
    --centroids-dir "$CENTROIDS_DIR" \
    --cohort "$COHORT_CSV" \
    --out "$OUT_H5AD" \
    "${EXTRA_ARGS[@]}"

STATUS=$?
ls -lh "$OUT_H5AD" 2>/dev/null
exit $STATUS
