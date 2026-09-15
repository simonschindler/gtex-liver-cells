export REPO_DIR="$HOME/rep/gtex-liver-cells"
export OUT_DIR=/nobackup/lab_rendeiro/projects/topocyte/topocyte_data/gtex_liver_cells
export PARQUET_DIR=/nobackup/lab_rendeiro/projects/histopath/processed/histopathology/datasets_pq_h5ad/
export CENTROIDS_DIR="$OUT_DIR/centroids"
export COHORT_CSV="$REPO_DIR/data/liver_cohort.csv"

mkdir -p "$OUT_DIR/centroids" "$OUT_DIR/logs" && cd "$OUT_DIR"

sbatch --export=ALL,CENTROIDS_DIR=$CENTROIDS_DIR,COHORT_CSV=$COHORT_CSV,OUT_DIR=$OUT_DIR,FILTER_TO_COHORT=1 \
	    "$REPO_DIR/sbatch/consolidate_anndata.sbatch"
