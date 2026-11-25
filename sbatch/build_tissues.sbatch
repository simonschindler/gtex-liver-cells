#!/bin/bash
#SBATCH --job-name=gtex_tissue_anndata
#SBATCH --account=lab_rendeiro
#SBATCH --partition=mediumq
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-user=simon.schindler@univie.ac.at
#SBATCH --mail-type=ALL
#SBATCH --output=logs/gtex_tissue_%A_%a.out
#SBATCH --error=logs/gtex_tissue_%A_%a.err

# Build one GTEx tissue's AnnData per array task: extract the tissue's cell
# centroids from the Parquet source, then consolidate them into
# $ROOT/<slug>/gtex_<slug>_cells.h5ad with slides.csv and provenance.json.
# The array index is the tissue's `index` in $ROOT/tissues.csv, which
# `gtex_liver_cells.tissues prepare` writes.
#
# scripts/all_tissues.sh prepares $ROOT and submits this from $ROOT, so that
# `logs/` receives the SLURM output:
#
#   cd "$ROOT"
#   sbatch --array=0,1,2%8 \
#       --export=ALL,PARQUET_DIR=$PARQUET_DIR,ROOT=$ROOT,REPO_DIR=$REPO_DIR,PYTHON=$PYTHON \
#       "$REPO_DIR/sbatch/build_tissues.sbatch"
#
# The task exits 1 when a slide failed to extract (the object is still built
# from the others) or when the tissue has no cell data; see provenance.json.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/sschindler/rep/gtex-liver-cells}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
: "${PARQUET_DIR:?export PARQUET_DIR=/path/to/datasets_pq_h5ad}"
: "${ROOT:?export ROOT=/path/to/gtex_tissues}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array job: sbatch --array=<indices from tissues.csv>}"

cd "$REPO_DIR" || exit 1
export PYTHONPATH=src

echo "node:     $(hostname)"
echo "task:     $SLURM_ARRAY_TASK_ID"
echo "parquet:  $PARQUET_DIR"
echo "root:     $ROOT"
echo "workers:  ${SLURM_CPUS_PER_TASK:-8}"

# The project venv is used directly on purpose: `uv run` re-syncs the
# environment on every invocation, which is not what a batch job should pay.
"$PYTHON" -m gtex_liver_cells.tissues build \
    --root "$ROOT" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --parquet-dir "$PARQUET_DIR" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
