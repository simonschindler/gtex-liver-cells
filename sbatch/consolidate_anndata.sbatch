#!/bin/bash
#SBATCH --job-name=gtex_consolidate_anndata
#SBATCH --account=lab_rendeiro
#SBATCH --partition=mediumq
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --output=logs/gtex_consolidate_anndata_%j.out
#SBATCH --error=logs/gtex_consolidate_anndata_%j.err

# Consolidate every centroid npz into one AnnData object, write provenance.json
# beside it, and copy the cohort CSV into the output directory.
#
# Submit from the output directory so that `logs/` receives the SLURM output:
#
#   mkdir -p "$OUT_DIR/logs" && cd "$OUT_DIR"
#   sbatch --export=ALL,CENTROIDS_DIR=$CENTROIDS_DIR,COHORT_CSV=$COHORT_CSV,OUT_DIR=$OUT_DIR,FILTER_TO_COHORT=1 \
#       "$REPO_DIR/sbatch/consolidate_anndata.sbatch"
#
# Add FILTER_TO_COHORT=1 to keep only slides listed in the cohort CSV.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/sschindler/rep/gtex-liver-cells}"
: "${CENTROIDS_DIR:?export CENTROIDS_DIR=/path/to/centroids}"
: "${COHORT_CSV:?export COHORT_CSV=/path/to/liver_cohort.csv}"
: "${OUT_DIR:?export OUT_DIR=/path/to/gtex_liver_cells}"

H5AD="$OUT_DIR/gtex_liver_cells.h5ad"

EXTRA_ARGS=()
if [ "${FILTER_TO_COHORT:-0}" = "1" ]; then
    EXTRA_ARGS+=(--filter-to-cohort)
fi

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:       $(hostname)"
echo "centroids:  $CENTROIDS_DIR"
echo "cohort csv: $COHORT_CSV"
echo "output:     $H5AD"

"$REPO_DIR/.venv/bin/python" -m gtex_liver_cells.consolidate \
    --centroids-dir "$CENTROIDS_DIR" \
    --cohort "$COHORT_CSV" \
    --out "$H5AD" \
    --provenance-out "$OUT_DIR/provenance.json" \
    "${EXTRA_ARGS[@]}"

STATUS=$?
cp -f "$COHORT_CSV" "$OUT_DIR/liver_cohort.csv"
ls -lh "$H5AD" 2>/dev/null
exit $STATUS
