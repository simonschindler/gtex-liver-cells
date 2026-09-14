"""Extract cell centroids and class labels from HistoPlus GeoJSON files.

HistoPlus files are a given input. Slides whose file is missing are reported as
warnings and skipped; they never abort the run.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

METHOD_VERSION = "1.0"
SUFFIX = ".histoplus.geojson.gz"


def gdal_path(path: str) -> str:
    """Return a GDAL-readable path, decompressing .gz through /vsigzip/."""
    if path.endswith(".gz"):
        return f"/vsigzip/{path}"
    return path


def read_slide_ids(csv_path: str) -> list[str]:
    """Read slide IDs from a cohort CSV."""
    frame = pd.read_csv(csv_path, dtype=str)
    column = "Tissue Sample ID" if "Tissue Sample ID" in frame.columns else frame.columns[0]
    return [value for value in frame[column].fillna("") if value]


def extract_one(source_path: str, out_path: str) -> dict:
    """Extract centroids from one HistoPlus file and write a compressed npz."""
    slide_id = os.path.basename(source_path).split(".histoplus")[0]
    frame = gpd.read_file(gdal_path(source_path))
    if frame.empty:
        raise ValueError(f"no features in {os.path.basename(source_path)}")

    coords = np.array(
        [(geometry.centroid.x, geometry.centroid.y) for geometry in frame.geometry],
        dtype=np.float32,
    )
    classes = frame["classification"].astype(str)
    class_names = np.array(sorted(classes.unique()))
    lookup = {name: index for index, name in enumerate(class_names)}
    labels = classes.map(lookup).to_numpy(dtype=np.int16)
    prob = (
        frame["prob"].to_numpy(dtype=np.float32)
        if "prob" in frame.columns
        else np.full(len(frame), np.nan, dtype=np.float32)
    )
    provenance = json.dumps(
        {
            "source_file": os.path.basename(source_path),
            "method_version": METHOD_VERSION,
            "n_cells": int(coords.shape[0]),
            "n_classes": int(class_names.size),
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        coords=coords,
        labels=labels,
        prob=prob,
        class_names=class_names,
        provenance=np.array(provenance),
    )
    return {"slide_id": slide_id, "n_cells": int(coords.shape[0]), "path": out_path}


def select_slides(
    histoplus_dir: str, slide_ids: list[str] | None
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (sources, missing) where sources are (slide_id, path) pairs."""
    if slide_ids is None:
        sources = [
            (name.split(".histoplus")[0], os.path.join(histoplus_dir, name))
            for name in sorted(os.listdir(histoplus_dir))
            if name.endswith(SUFFIX)
        ]
        return sources, []

    sources: list[tuple[str, str]] = []
    missing: list[str] = []
    for slide_id in slide_ids:
        path = os.path.join(histoplus_dir, f"{slide_id}{SUFFIX}")
        if os.path.exists(path):
            sources.append((slide_id, path))
        else:
            missing.append(slide_id)
    return sources, missing


def _extract_job(job: tuple[str, str, str, bool]) -> dict:
    """Worker entry point: extract one slide, catching per-file failures."""
    slide_id, source, out_dir, overwrite = job
    if not os.path.exists(source):
        return {"slide_id": slide_id, "status": "missing"}
    out_path = os.path.join(out_dir, f"{slide_id}.npz")
    if os.path.exists(out_path) and not overwrite:
        return {"slide_id": slide_id, "status": "skipped"}
    try:
        extract_one(source, out_path)
        return {"slide_id": slide_id, "status": "extracted"}
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return {"slide_id": slide_id, "status": "failed", "error": str(error)}


def extract_many(
    sources: list[tuple[str, str]], out_dir: str, workers: int, overwrite: bool
) -> dict[str, int]:
    """Extract many slides in parallel, returning status counts."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {"extracted": 0, "skipped": 0, "failed": 0, "missing": 0}
    jobs = [(slide_id, path, out_dir, overwrite) for slide_id, path in sources]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(
            pool.map(_extract_job, jobs), total=len(jobs), unit="slide", desc="Centroids"
        ):
            counts[result["status"]] += 1
            if result["status"] == "failed":
                tqdm.write(f"  {result['slide_id']}: {result['error']}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histoplus-dir", required=True, help="HistoPlus input directory")
    parser.add_argument("--out-dir", required=True, help="Directory for .npz outputs")
    parser.add_argument(
        "--slides",
        default=None,
        help="Optional CSV of slides to process (default: every HistoPlus file)",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--overwrite", action="store_true", help="Redo existing .npz files")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.histoplus_dir):
        print(f"ERROR: no such directory: {args.histoplus_dir}", file=sys.stderr)
        return 1

    slide_ids = read_slide_ids(args.slides) if args.slides else None
    sources, missing = select_slides(args.histoplus_dir, slide_ids)

    if missing:
        print(
            f"WARNING: {len(missing)} requested slides have no HistoPlus file; skipping",
            file=sys.stderr,
        )
        for slide_id in missing[:10]:
            print(f"  missing: {slide_id}", file=sys.stderr)

    counts = extract_many(sources, args.out_dir, args.workers, args.overwrite)
    counts["missing"] += len(missing)
    print(
        "extracted={extracted} skipped={skipped} failed={failed} missing={missing}".format(
            **counts
        )
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
