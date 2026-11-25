"""Consolidate per-slide centroid npz files into one AnnData object.

Layout decisions (see the design specs for the measurements behind them):
``X`` is a sparse one-hot indicator of ``cell_type``, ``var`` is the sorted
union of the per-slide class vocabularies, ``obsm["spatial"]`` holds float32
centroids, the index is a RangeIndex, and the file is gzip-compressed. Slides
are stored in ascending ID order, each as one contiguous block of rows, and
every ``obs`` column is built from integer codes rather than per-cell Python
objects, which keeps the build's memory near what anndata itself needs.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

META_COLUMNS = ["Subject ID", "Age Bracket", "Sex", "Hardy Scale", "label"]
SLIDE_RANGE_COLUMNS = ["n_cells", "row_start", "row_end"]

# Repository root: this file lives at <repo>/src/gtex_liver_cells/consolidate.py.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConsolidateError(Exception):
    """An input problem that leaves nothing to consolidate."""


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


def repository_commit(repo_dir: str) -> str:
    """Return the HEAD commit of repo_dir, or "unknown" if it is not a checkout."""
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        return "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def build_provenance(
    *,
    centroids_dir: str,
    cohort_csv: str,
    requested: int,
    extracted: int,
    used: int,
    missing: list[str],
    class_names: list[str],
    n_cells: int,
    repo_dir: str,
) -> dict:
    """Describe one consolidation run, for provenance.json beside the h5ad."""
    return {
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "repository_commit": repository_commit(repo_dir),
        "centroids_dir": os.path.abspath(centroids_dir),
        "cohort_csv": os.path.abspath(cohort_csv),
        "slides_requested": requested,
        "slides_extracted": extracted,
        "slides_used": used,
        "slides_missing": len(missing),
        "missing_slide_ids": sorted(missing),
        "n_cells": n_cells,
        "class_names": list(class_names),
    }


def _smallest_int(n_values: int) -> type[np.signedinteger]:
    """The narrowest signed integer type that holds codes 0..n_values-1."""
    for dtype in (np.int8, np.int16, np.int32):
        if n_values <= np.iinfo(dtype).max:
            return dtype
    return np.int64


def _per_slide_categorical(values: list[str], slide_codes: np.ndarray) -> pd.Categorical:
    """Spread one value per slide to every cell of that slide, as a categorical.

    The categories are the sorted distinct values, which is the order
    ``pd.Categorical(values)`` would give, so the stored object does not change.
    """
    categories = sorted(set(values))
    lookup = {value: code for code, value in enumerate(categories)}
    dtype = _smallest_int(len(categories))
    slide_level = np.array([lookup[value] for value in values], dtype=dtype)
    return pd.Categorical.from_codes(slide_level[slide_codes], categories=categories)


def consolidate(
    entries: list[dict], cohort: pd.DataFrame, metadata_columns: Sequence[str] = META_COLUMNS
) -> ad.AnnData:
    """Build one AnnData from loaded centroid entries.

    Every column is assembled from integer codes, never from one Python object
    per cell: slides are ordered by ID (so each slide is one contiguous block of
    rows and the ``slide_id`` categories ascend), each slide's labels are
    remapped into the union vocabulary, and each metadata column is looked up
    once per slide. ``cohort`` is indexed by ``Tissue Sample ID``; a column it
    lacks, or a slide it lacks, becomes ``""``.
    """
    if not entries:
        raise ValueError("no centroid entries to consolidate")

    entries = sorted(entries, key=lambda entry: entry["slide_id"])
    slide_ids = [entry["slide_id"] for entry in entries]
    class_names = sorted({str(name) for entry in entries for name in entry["class_names"]})
    lookup = {name: index for index, name in enumerate(class_names)}
    counts = np.array([len(entry["coords"]) for entry in entries], dtype=np.int64)
    total = int(counts.sum())

    coords = np.empty((total, 2), dtype=np.float32)
    prob = np.empty(total, dtype=np.float32)
    type_codes = np.empty(total, dtype=np.int32)
    offset = 0
    for entry, n_cells in zip(entries, counts):
        window = slice(offset, offset + int(n_cells))
        remap = np.array([lookup[str(name)] for name in entry["class_names"]], dtype=np.int32)
        coords[window] = entry["coords"]
        prob[window] = entry["prob"]
        type_codes[window] = remap[entry["labels"]]
        offset += int(n_cells)

    slide_codes = np.repeat(np.arange(len(entries), dtype=_smallest_int(len(entries))), counts)
    obs = pd.DataFrame(
        {
            "slide_id": pd.Categorical.from_codes(slide_codes, categories=slide_ids),
            "cell_type": pd.Categorical.from_codes(type_codes, categories=class_names),
            "prob": prob,
        }
    )
    for column in metadata_columns:
        if column in cohort.columns:
            values = [str(value) for value in cohort[column].reindex(slide_ids).fillna("")]
        else:
            values = [""] * len(slide_ids)
        obs[column] = _per_slide_categorical(values, slide_codes)

    index_dtype = np.int32 if total < np.iinfo(np.int32).max else np.int64
    data = sp.csr_matrix(
        (
            np.ones(total, dtype=np.float32),
            type_codes.astype(index_dtype, copy=False),
            np.arange(total + 1, dtype=index_dtype),
        ),
        shape=(total, len(class_names)),
    )
    adata = ad.AnnData(X=data, obs=obs, var=pd.DataFrame(index=pd.Index(class_names)))
    adata.obsm["spatial"] = coords
    adata.uns["n_slides"] = len(entries)
    adata.uns["n_cells"] = total
    adata.uns["class_names"] = class_names
    adata.uns["slide_ids"] = slide_ids
    adata.uns["metadata_columns"] = list(metadata_columns)
    adata.uns["created"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return adata


def slide_table(adata: ad.AnnData, cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per slide of ``adata``, in row order, with its cohort row and row range.

    The columns are ``Tissue Sample ID`` and every ``cohort`` column (``""``
    where the cohort has no row for the slide), then ``n_cells``, ``row_start``
    and ``row_end`` (end exclusive). The ranges rely on each slide being one
    contiguous block of rows in category order, which ``consolidate`` ensures.
    """
    slide_ids = list(adata.obs["slide_id"].cat.categories)
    codes = adata.obs["slide_id"].cat.codes.to_numpy()
    counts = np.bincount(codes, minlength=len(slide_ids))
    ends = np.cumsum(counts)
    table = cohort.reindex(pd.Index(slide_ids, name="Tissue Sample ID")).fillna("").reset_index()
    table["n_cells"] = counts
    table["row_start"] = ends - counts
    table["row_end"] = ends
    return table


def write_h5ad_atomic(adata: ad.AnnData, path: str) -> None:
    """Write ``path`` gzip-compressed through ``<path>.tmp``, so it is never left partial."""
    partial = f"{path}.tmp"
    adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, path)


def run(
    centroids_dir: str,
    cohort_csv: str,
    out: str,
    *,
    provenance_out: str | None = None,
    filter_to_cohort: bool = False,
    metadata_columns: Sequence[str] = META_COLUMNS,
    slides_out: str | None = None,
    extra_provenance: dict | None = None,
) -> dict:
    """Consolidate every npz in ``centroids_dir`` into ``out`` and its side files.

    Writes the h5ad first (atomically), then ``slides_out``, then
    ``provenance_out`` with ``extra_provenance`` merged in, so a present side
    file means its h5ad is complete. Raises ``ConsolidateError`` when the
    directory is missing, holds no npz, or nothing is left after filtering.
    Returns the cell and slide counts and the cohort slides that had no npz.
    """
    if not os.path.isdir(centroids_dir):
        raise ConsolidateError(f"no such directory: {centroids_dir}")

    cohort = pd.read_csv(cohort_csv, dtype=str).fillna("").set_index("Tissue Sample ID")
    paths = sorted(
        os.path.join(centroids_dir, name)
        for name in os.listdir(centroids_dir)
        if name.endswith(".npz")
    )
    if not paths:
        raise ConsolidateError(f"no .npz files in {centroids_dir}")

    n_extracted = len(paths)

    entries = [load_centroids(path) for path in paths]
    known = set(cohort.index)
    warnings: list[str] = []

    if filter_to_cohort:
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
        raise ConsolidateError("no entries left after filtering")

    adata = consolidate(entries, cohort, metadata_columns)
    write_h5ad_atomic(adata, out)

    if slides_out:
        os.makedirs(os.path.dirname(os.path.abspath(slides_out)), exist_ok=True)
        slide_table(adata, cohort).to_csv(slides_out, index=False)
        print(f"wrote the slide table to {slides_out}")

    if provenance_out:
        provenance = build_provenance(
            centroids_dir=centroids_dir,
            cohort_csv=cohort_csv,
            requested=len(cohort),
            extracted=n_extracted,
            used=len(entries),
            missing=missing_npz,
            class_names=list(adata.var_names),
            n_cells=int(adata.n_obs),
            repo_dir=REPO_DIR,
        )
        provenance.update(extra_provenance or {})
        os.makedirs(os.path.dirname(os.path.abspath(provenance_out)), exist_ok=True)
        with open(provenance_out, "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote provenance to {provenance_out}")

    print(f"wrote {adata.n_obs} cells from {adata.uns['n_slides']} slides to {out}")
    print(f"  classes: {adata.n_vars} | shape: {adata.shape}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return {
        "n_obs": int(adata.n_obs),
        "n_slides": int(adata.uns["n_slides"]),
        "missing_slide_ids": missing_npz,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centroids-dir", required=True, help="Directory of .npz files")
    parser.add_argument("--cohort", required=True, help="Cohort CSV to join metadata from")
    parser.add_argument("--out", required=True, help="Output .h5ad path")
    parser.add_argument(
        "--provenance-out",
        default=None,
        help="Optional path for a provenance.json describing this run",
    )
    parser.add_argument(
        "--filter-to-cohort",
        action="store_true",
        help="Exclude npz files whose slide is not in the cohort CSV",
    )
    parser.add_argument(
        "--metadata-columns",
        nargs="+",
        default=META_COLUMNS,
        metavar="COLUMN",
        help="Cohort columns to join into obs (default: %(default)s)",
    )
    parser.add_argument(
        "--slides-out",
        default=None,
        help="Optional path for a per-slide table with each slide's row range",
    )
    args = parser.parse_args(argv)

    try:
        run(
            args.centroids_dir,
            args.cohort,
            args.out,
            provenance_out=args.provenance_out,
            filter_to_cohort=args.filter_to_cohort,
            metadata_columns=args.metadata_columns,
            slides_out=args.slides_out,
        )
    except ConsolidateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
