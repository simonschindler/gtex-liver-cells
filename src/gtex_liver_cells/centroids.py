"""Extract cell centroids and class labels from the Parquet cell-type source.

Per slide the source is one file ``<parquet_dir>/<TISSUE_ID>/cell_types.gpd`` —
Parquet despite the extension — with ``geometry`` (binary WKB under the Arrow
extension ``geoarrow.wkb``), ``class``, ``prob`` and ``cell_id`` columns. Only
geometry, class and confidence are read; the sibling
``cell_types_features.h5ad`` embeddings are deliberately not used. Slides whose
file is missing are reported as warnings and skipped; they never abort the run.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from shapely import GeometryType
from tqdm import tqdm

METHOD_VERSION = "2.0"
SOURCE_FILENAME = "cell_types.gpd"
GEOMETRY_COLUMN = "geometry"
CLASS_COLUMN = "class"
PROB_COLUMN = "prob"

# Cell outlines arrive as Polygon or MultiPolygon and shapely's centroid handles
# both, so the reader must not assume simple polygons. Anything else — a Point,
# a LineString, a null geometry — fails that slide with the type named.
SUPPORTED_GEOMETRY_IDS = {int(GeometryType.POLYGON), int(GeometryType.MULTIPOLYGON)}


def source_path(parquet_dir: str, slide_id: str) -> str:
    """Return the cell_types.gpd path for one slide."""
    return os.path.join(parquet_dir, slide_id, SOURCE_FILENAME)


def slide_id_from_source(source: str) -> str:
    """Return the slide ID of a `<parquet_dir>/<TISSUE_ID>/cell_types.gpd` path."""
    return os.path.basename(os.path.dirname(os.path.abspath(source)))


def read_slide_ids(csv_path: str) -> list[str]:
    """Read slide IDs from a cohort CSV."""
    frame = pd.read_csv(csv_path, dtype=str)
    column = "Tissue Sample ID" if "Tissue Sample ID" in frame.columns else frame.columns[0]
    return [value for value in frame[column].fillna("") if value]


def wkb_bytes(column: pa.ChunkedArray) -> np.ndarray:
    """Return the raw WKB bytes held by a Parquet geometry column.

    Parquet files with a WKB geometry column store it as the Arrow extension
    type ``geoarrow.wkb``. With no geoarrow extension package installed pyarrow
    downgrades that to its storage type (``binary``), which is what we read;
    when an extension package *is* installed pyarrow returns the extension
    array instead, so unwrap it. ``large_binary`` storage is normalised too.
    """
    array = column.combine_chunks()
    if isinstance(array.type, pa.ExtensionType):
        array = array.storage
    if pa.types.is_large_binary(array.type):
        array = array.cast(pa.binary())
    if not pa.types.is_binary(array.type):
        raise ValueError(f"geometry column has type {array.type}, expected binary WKB")
    return array.to_numpy(zero_copy_only=False)


def centroids_from_wkb(wkb: np.ndarray, source: str) -> np.ndarray:
    """Parse WKB polygons and return their float32 (N, 2) centroids."""
    geometries = shapely.from_wkb(wkb)
    type_ids = np.unique(shapely.get_type_id(geometries))
    unsupported = [
        GeometryType(int(type_id)).name
        for type_id in type_ids
        if int(type_id) not in SUPPORTED_GEOMETRY_IDS
    ]
    if unsupported:
        raise ValueError(f"{source}: unsupported geometry types: {', '.join(unsupported)}")
    points = shapely.centroid(geometries)
    return np.column_stack([shapely.get_x(points), shapely.get_y(points)]).astype(np.float32)


def extract_one(source: str, out_path: str) -> dict:
    """Extract centroids from one cell_types.gpd file and write a compressed npz."""
    slide_id = slide_id_from_source(source)
    table = pq.read_table(source, columns=[GEOMETRY_COLUMN, CLASS_COLUMN, PROB_COLUMN])
    if table.num_rows == 0:
        raise ValueError(f"no cells in {source}")

    coords = centroids_from_wkb(wkb_bytes(table.column(GEOMETRY_COLUMN)), source)
    classes = table.column(CLASS_COLUMN).to_pylist()
    prob = table.column(PROB_COLUMN).to_numpy(zero_copy_only=False).astype(np.float32)
    class_names = np.array(sorted(set(classes)))
    lookup = {name: index for index, name in enumerate(class_names)}
    labels = np.array([lookup[name] for name in classes], dtype=np.int16)
    provenance = json.dumps(
        {
            "source_file": f"{slide_id}/{SOURCE_FILENAME}",
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
    parquet_dir: str, slide_ids: list[str] | None
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (sources, missing) where sources are (slide_id, path) pairs."""
    if slide_ids is None:
        sources = [
            (name, source_path(parquet_dir, name))
            for name in sorted(os.listdir(parquet_dir))
            if os.path.isfile(source_path(parquet_dir, name))
        ]
        return sources, []

    sources: list[tuple[str, str]] = []
    missing: list[str] = []
    for slide_id in slide_ids:
        path = source_path(parquet_dir, slide_id)
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
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Directory holding <TISSUE_ID>/cell_types.gpd per slide",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for .npz outputs")
    parser.add_argument(
        "--slides",
        default=None,
        help="Optional CSV of slides to process (default: every cell_types.gpd)",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--overwrite", action="store_true", help="Redo existing .npz files")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.parquet_dir):
        print(f"ERROR: no such directory: {args.parquet_dir}", file=sys.stderr)
        return 1

    slide_ids = read_slide_ids(args.slides) if args.slides else None
    sources, missing = select_slides(args.parquet_dir, slide_ids)

    if missing:
        print(
            f"WARNING: {len(missing)} requested slides have no {SOURCE_FILENAME}; skipping",
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
