#!/usr/bin/env bash
set -euo pipefail

# Build one AnnData per GTEx tissue on CEMM.
#
# Fetches the GTEx Portal slide table into $ROOT once (on the login node, which
# has internet), splits it per tissue with a coverage check, and submits one
# array task per tissue that has cell data (sbatch/build_tissues.sbatch).
#
#   bash scripts/all_tissues.sh                    # every tissue with cell data
#   ARRAY=17 bash scripts/all_tissues.sh           # rerun tissue 17 (see $ROOT/tissues.csv)
#   MAX_PARALLEL=4 bash scripts/all_tissues.sh     # at most 4 tasks at once (default 8)
#
# Delete $ROOT/gtex_portal_slides.csv to fetch a fresh Portal table. Every path
# below can be overridden from the environment.

export REPO_DIR="${REPO_DIR:-$HOME/rep/gtex-liver-cells}"
export PARQUET_DIR="${PARQUET_DIR:-/nobackup/lab_rendeiro/projects/histopath/processed/histopathology/datasets_pq_h5ad/}"
export ROOT="${ROOT:-/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_tissues}"
export PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
export PYTHONPATH="$REPO_DIR/src"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
PORTAL_CSV="$ROOT/gtex_portal_slides.csv"

mkdir -p "$ROOT/logs"

if [ ! -s "$PORTAL_CSV" ]; then
    "$PYTHON" -m gtex_liver_cells.portal --out "$PORTAL_CSV"
fi

"$PYTHON" -m gtex_liver_cells.tissues prepare \
    --slides "$PORTAL_CSV" \
    --root "$ROOT" \
    --parquet-dir "$PARQUET_DIR"

if [ -z "${ARRAY:-}" ]; then
    ARRAY="$("$PYTHON" -m gtex_liver_cells.tissues array --root "$ROOT")"
fi
if [ -z "$ARRAY" ]; then
    echo "ERROR: no tissue has a cell_types.gpd under $PARQUET_DIR; nothing submitted" >&2
    exit 1
fi

echo "submitting array $ARRAY%$MAX_PARALLEL from $ROOT"
cd "$ROOT"
sbatch --array="$ARRAY%$MAX_PARALLEL" \
    --export="ALL,PARQUET_DIR=$PARQUET_DIR,ROOT=$ROOT,REPO_DIR=$REPO_DIR,PYTHON=$PYTHON" \
    "$REPO_DIR/sbatch/build_tissues.sbatch"
