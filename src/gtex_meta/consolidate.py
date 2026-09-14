"""Consolidate per-slide centroid npz files into one AnnData object.

Layout decisions (see the design spec for the measurements behind them):
``X`` is a sparse one-hot indicator of ``cell_type``, ``var`` is the sorted
union of the per-slide class vocabularies, ``obsm["spatial"]`` holds float32
centroids, the index is a RangeIndex, and the file is gzip-compressed.
"""

import argparse
import datetime
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

META_COLUMNS = ["Subject ID", "Age Bracket", "Sex", "Hardy Scale", "label"]


def load_centroids(path: str) -> dict:
    """Load one centroid npz, tolerating files from the older LISC pipeline."""
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        coords = data["coords"].astype(np.float32, copy=False)
        entry = {
            "path": path,
            "slide_id": os.path.basename(path)[:-4],
            "coords": coords,
            "labels": data["labels"].astype(np.int16, copy=False),
            "class_names": np.asarray(data["class_names"], dtype=str),
            "prob": (
                data["prob"].astype(np.float32, copy=False)
                if "prob" in keys
                else np.full(len(coords), np.nan, dtype=np.float32)
            ),
            "has_provenance": "provenance" in keys,
        }
    return entry


def consolidate(entries: list[dict], cohort: pd.DataFrame) -> ad.AnnData:
    """Build one AnnData from loaded centroid entries."""
    if not entries:
        raise ValueError("no centroid entries to consolidate")

    class_names = sorted({name for entry in entries for name in entry["class_names"]})
    lookup = {name: index for index, name in enumerate(class_names)}
    total = sum(len(entry["coords"]) for entry in entries)

    coords = np.empty((total, 2), dtype=np.float32)
    prob = np.empty(total, dtype=np.float32)
    codes = np.empty(total, dtype=np.int64)
    cell_types = np.empty(total, dtype=object)
    slide_ids = np.empty(total, dtype=object)

    offset = 0
    for entry in entries:
        n_cells = len(entry["coords"])
        window = slice(offset, offset + n_cells)
        names = entry["class_names"][entry["labels"]]
        coords[window] = entry["coords"]
        prob[window] = entry["prob"]
        cell_types[window] = names
        codes[window] = [lookup[name] for name in names]
        slide_ids[window] = entry["slide_id"]
        offset += n_cells

    obs = pd.DataFrame(
        {
            "slide_id": pd.Categorical(slide_ids),
            "cell_type": pd.Categorical(cell_types, categories=class_names),
            "prob": prob,
        }
    )
    metadata = cohort.reindex(pd.Index(slide_ids, name="Tissue Sample ID"))
    for column in META_COLUMNS:
        if column in metadata.columns:
            obs[column] = pd.Categorical(metadata[column].fillna("").to_numpy())
        else:
            obs[column] = pd.Categorical(np.full(total, ""))

    data = sp.csr_matrix(
        (np.ones(total, dtype=np.float32), (np.arange(total), codes)),
        shape=(total, len(class_names)),
    )
    adata = ad.AnnData(X=data, obs=obs, var=pd.DataFrame(index=pd.Index(class_names)))
    adata.obsm["spatial"] = coords
    adata.uns["n_slides"] = len(entries)
    adata.uns["n_cells"] = int(total)
    adata.uns["class_names"] = class_names
    adata.uns["slide_ids"] = sorted(entry["slide_id"] for entry in entries)
    adata.uns["metadata_columns"] = META_COLUMNS
    adata.uns["created"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return adata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centroids-dir", required=True, help="Directory of .npz files")
    parser.add_argument("--cohort", required=True, help="Cohort CSV to join metadata from")
    parser.add_argument("--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--filter-to-cohort",
        action="store_true",
        help="Exclude npz files whose slide is not in the cohort CSV",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.centroids_dir):
        print(f"ERROR: no such directory: {args.centroids_dir}", file=sys.stderr)
        return 1

    cohort = pd.read_csv(args.cohort, dtype=str).fillna("").set_index("Tissue Sample ID")
    paths = sorted(
        os.path.join(args.centroids_dir, name)
        for name in os.listdir(args.centroids_dir)
        if name.endswith(".npz")
    )
    if not paths:
        print(f"ERROR: no .npz files in {args.centroids_dir}", file=sys.stderr)
        return 1

    entries = [load_centroids(path) for path in paths]
    known = set(cohort.index)
    warnings: list[str] = []

    if args.filter_to_cohort:
        dropped = [entry["slide_id"] for entry in entries if entry["slide_id"] not in known]
        entries = [entry for entry in entries if entry["slide_id"] in known]
        if dropped:
            print(f"filtered out {len(dropped)} slides not in the cohort CSV")
    else:
        unmatched = [entry["slide_id"] for entry in entries if entry["slide_id"] not in known]
        if unmatched:
            warnings.append(
                f"{len(unmatched)} slides have no cohort metadata: {unmatched[:5]}"
            )

    missing_npz = sorted(known - {entry["slide_id"] for entry in entries})
    if missing_npz:
        warnings.append(f"{len(missing_npz)} cohort slides have no .npz: {missing_npz[:5]}")

    legacy = [entry["slide_id"] for entry in entries if not entry["has_provenance"]]
    if legacy:
        warnings.append(
            f"{len(legacy)} files lack provenance (legacy LISC output): {legacy[:5]}"
        )

    if not entries:
        print("ERROR: no entries left after filtering", file=sys.stderr)
        return 1

    adata = consolidate(entries, cohort)
    adata.write_h5ad(args.out, compression="gzip")

    print(f"wrote {adata.n_obs} cells from {adata.uns['n_slides']} slides to {args.out}")
    print(f"  classes: {adata.n_vars} | shape: {adata.shape}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
