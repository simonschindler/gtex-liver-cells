"""Build one cell-resolved AnnData per GTEx tissue.

The GTEx Portal histology table (``portal.py`` without ``--tissue``) is split
by tissue site (``Tissue``, whitespace-stripped) under one output root:

    <ROOT>/
    ├── gtex_portal_slides.csv      the Portal table the run used
    ├── tissues.csv                 index, slug, tissue, n_slides, n_with_source
    ├── logs/                       SLURM output
    └── <slug>/
        ├── portal_slides.csv       every Portal row of this tissue
        ├── gtex_<slug>_cells.h5ad  the object
        ├── slides.csv              one row per slide in the object, with its row range
        ├── provenance.json         how it was built, and which slides are missing
        └── centroids/              the per-slide npz intermediate

``prepare`` writes the per-tissue tables and the manifest and reports how many
of each tissue's slides have a ``cell_types.gpd``; ``array`` prints the SLURM
array of the tissues that have any; ``build`` runs one tissue, extraction and
then consolidation. A tissue's ``index`` in ``tissues.csv`` is its SLURM array
index.

``build`` exits 0 only when the object was written and no slide failed. A slide
whose source fails to extract is left out of the object, listed in
``provenance.json``, and makes the exit status 1; a slide with no
``cell_types.gpd`` is listed but is not a failure.
"""

import argparse
import os
import re
import sys
from collections.abc import Iterable

import pandas as pd

from gtex_liver_cells import centroids, consolidate, portal

PORTAL_COLUMNS = list(portal.COLUMNS)
METADATA_COLUMNS = PORTAL_COLUMNS[1:]
ID_COLUMN = PORTAL_COLUMNS[0]
MANIFEST_COLUMNS = ["index", "slug", "tissue", "n_slides", "n_with_source"]

MANIFEST = "tissues.csv"
PORTAL_SLIDES = "portal_slides.csv"
SLIDES = "slides.csv"
PROVENANCE = "provenance.json"
CENTROIDS = "centroids"


def slugify(name: str) -> str:
    """Lowercase ``name`` and join its alphanumeric runs with ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def object_filename(slug: str) -> str:
    """The h5ad name of one tissue's object."""
    return f"gtex_{slug}_cells.h5ad"


def split_tissues(slides: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Group the Portal table by tissue, as ``(slug, tissue, rows)`` sorted by slug.

    Tissue names are whitespace-stripped first, which folds the Portal's
    ``"Esophagus - Mucosa "`` into ``"Esophagus - Mucosa"``. Raises
    ``ValueError`` for a missing Portal column, a duplicate slide ID, a tissue
    name that gives an empty slug, or two tissue names that give one slug.
    """
    absent = [column for column in PORTAL_COLUMNS if column not in slides.columns]
    if absent:
        raise ValueError(f"the slide table lacks the Portal columns {absent}")

    duplicated = sorted(set(slides.loc[slides[ID_COLUMN].duplicated(), ID_COLUMN]))
    if duplicated:
        raise ValueError(f"{len(duplicated)} duplicate slide IDs: {duplicated[:5]}")

    table = slides[PORTAL_COLUMNS].copy()
    table["Tissue"] = table["Tissue"].str.strip()
    slugs = table["Tissue"].map(slugify)

    nameless = list(table.loc[slugs == "", ID_COLUMN])
    if nameless:
        raise ValueError(f"{len(nameless)} slides have no usable tissue name: {nameless[:5]}")

    names_by_slug: dict[str, list[str]] = {}
    for name in table["Tissue"].unique():
        names_by_slug.setdefault(slugify(name), []).append(name)
    clashes = {slug: sorted(names) for slug, names in names_by_slug.items() if len(names) > 1}
    if clashes:
        raise ValueError(f"tissue names that share a slug: {clashes}")

    return [
        (slug, names[0], table[table["Tissue"] == names[0]].reset_index(drop=True))
        for slug, names in sorted(names_by_slug.items())
    ]


def count_sources(parquet_dir: str, slide_ids: Iterable[str]) -> int:
    """How many of ``slide_ids`` have a ``cell_types.gpd`` under ``parquet_dir``."""
    return sum(
        os.path.exists(centroids.source_path(parquet_dir, slide_id)) for slide_id in slide_ids
    )


def print_coverage(manifest: pd.DataFrame, parquet_dir: str) -> None:
    """Print each tissue's slides with a source against its Portal slides."""
    print(f"{'index':>5}  {'with source':>16}  tissue")
    for index, slug, n_slides, n_with_source in zip(
        manifest["index"], manifest["slug"], manifest["n_slides"], manifest["n_with_source"]
    ):
        print(f"{index:>5}  {n_with_source:>6} of {n_slides:<6}  {slug}")
    print(
        f"total: {manifest['n_with_source'].sum()} of {manifest['n_slides'].sum()} slides "
        f"have a cell_types.gpd, across {len(manifest)} tissues"
    )
    empty = list(manifest.loc[manifest["n_with_source"] == 0, "slug"])
    if empty:
        print(f"no cell data under {parquet_dir} for {len(empty)} tissues: {', '.join(empty)}")


def prepare(slides_csv: str, root: str, parquet_dir: str) -> pd.DataFrame:
    """Write ``<root>/<slug>/portal_slides.csv`` per tissue and ``<root>/tissues.csv``.

    Only those two kinds of file are written, so a finished run's centroids,
    objects, slide tables and provenance are left as they are. Returns the
    manifest.
    """
    slides = pd.read_csv(slides_csv, dtype=str, keep_default_na=False)
    rows = []
    for index, (slug, tissue, table) in enumerate(split_tissues(slides)):
        organ = os.path.join(root, slug)
        os.makedirs(organ, exist_ok=True)
        table.to_csv(os.path.join(organ, PORTAL_SLIDES), index=False)
        rows.append(
            {
                "index": index,
                "slug": slug,
                "tissue": tissue,
                "n_slides": len(table),
                "n_with_source": count_sources(parquet_dir, table[ID_COLUMN]),
            }
        )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(os.path.join(root, MANIFEST), index=False)
    print_coverage(manifest, parquet_dir)
    return manifest


def read_manifest(root: str) -> pd.DataFrame:
    """Read ``<root>/tissues.csv``."""
    return pd.read_csv(
        os.path.join(root, MANIFEST), dtype={"slug": str, "tissue": str}, keep_default_na=False
    )


def array_spec(manifest: pd.DataFrame) -> str:
    """The SLURM array of every tissue with a source, e.g. ``"0,1,3"``; ``""`` if none."""
    return ",".join(str(index) for index in manifest.loc[manifest["n_with_source"] > 0, "index"])


def find_tissue(
    manifest: pd.DataFrame, *, index: int | None = None, slug: str | None = None
) -> pd.Series:
    """The manifest row with this ``index`` or this ``slug``; ``ValueError`` if absent."""
    if (index is None) == (slug is None):
        raise ValueError("give exactly one of index and slug")
    if index is not None:
        rows = manifest[manifest["index"] == index]
        wanted = f"index {index}"
    else:
        rows = manifest[manifest["slug"] == slug]
        wanted = f"slug {slug!r}"
    if rows.empty:
        raise ValueError(f"no tissue with {wanted} in {MANIFEST}")
    return rows.iloc[0]


def build_tissue(
    root: str, slug: str, tissue: str, parquet_dir: str, workers: int, overwrite: bool = False
) -> int:
    """Extract one tissue's slides and consolidate them into its object.

    Writes ``<root>/<slug>/centroids/``, the h5ad, ``slides.csv`` and
    ``provenance.json``. Returns the exit status described in the module
    docstring.
    """
    organ = os.path.join(root, slug)
    portal_csv = os.path.join(organ, PORTAL_SLIDES)
    centroids_dir = os.path.join(organ, CENTROIDS)

    slide_ids = centroids.read_slide_ids(portal_csv)
    sources, no_source = centroids.select_slides(parquet_dir, slide_ids)
    print(f"tissue:     {tissue} ({slug})")
    print(f"slides:     {len(slide_ids)} in the Portal, {len(sources)} with a cell_types.gpd")
    if not sources:
        print(
            f"ERROR: none of the {len(slide_ids)} slides has a cell_types.gpd under {parquet_dir}",
            file=sys.stderr,
        )
        return 1

    counts = centroids.extract_many(sources, centroids_dir, workers, overwrite)
    print(
        "extraction: extracted={extracted} skipped={skipped} failed={failed} "
        "missing={missing}".format(**counts)
    )
    failed = sorted(
        slide_id
        for slide_id, _ in sources
        if not os.path.exists(os.path.join(centroids_dir, f"{slide_id}.npz"))
    )

    try:
        consolidate.run(
            centroids_dir,
            portal_csv,
            os.path.join(organ, object_filename(slug)),
            provenance_out=os.path.join(organ, PROVENANCE),
            filter_to_cohort=True,
            metadata_columns=METADATA_COLUMNS,
            slides_out=os.path.join(organ, SLIDES),
            extra_provenance={
                "tissue": tissue,
                "slug": slug,
                "parquet_dir": os.path.abspath(parquet_dir),
                "no_source_slide_ids": sorted(no_source),
                "failed_slide_ids": failed,
            },
        )
    except consolidate.ConsolidateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if failed:
        print(
            f"ERROR: {len(failed)} slides failed to extract and are not in the object: "
            f"{failed[:5]}",
            file=sys.stderr,
        )
        return 1
    return 0


def _load_manifest(root: str) -> pd.DataFrame | None:
    """The manifest under ``root``, or None after printing why it is missing."""
    path = os.path.join(root, MANIFEST)
    if not os.path.isfile(path):
        print(f"ERROR: no manifest at {path}; run prepare first", file=sys.stderr)
        return None
    return read_manifest(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare", help="Split the Portal table per tissue and write tissues.csv"
    )
    prepare_parser.add_argument(
        "--slides", required=True, help="The full Portal slide table (portal.py, no --tissue)"
    )
    prepare_parser.add_argument("--root", required=True, help="Output root, one folder per tissue")
    prepare_parser.add_argument(
        "--parquet-dir", required=True, help="Directory holding <TISSUE_ID>/cell_types.gpd"
    )

    array_parser = commands.add_parser(
        "array", help="Print the SLURM array of the tissues that have cell data"
    )
    array_parser.add_argument("--root", required=True, help="Output root holding tissues.csv")

    build_parser = commands.add_parser("build", help="Extract and consolidate one tissue")
    build_parser.add_argument("--root", required=True, help="Output root holding tissues.csv")
    which = build_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--index", type=int, help="The tissue's index in tissues.csv")
    which.add_argument("--slug", help="The tissue's slug in tissues.csv")
    build_parser.add_argument(
        "--parquet-dir", required=True, help="Directory holding <TISSUE_ID>/cell_types.gpd"
    )
    build_parser.add_argument("--workers", type=int, default=os.cpu_count())
    build_parser.add_argument("--overwrite", action="store_true", help="Redo existing .npz files")

    args = parser.parse_args(argv)

    if args.command == "prepare":
        if not os.path.isdir(args.parquet_dir):
            print(f"ERROR: no such directory: {args.parquet_dir}", file=sys.stderr)
            return 1
        if not os.path.isfile(args.slides):
            print(f"ERROR: no such file: {args.slides}", file=sys.stderr)
            return 1
        try:
            prepare(args.slides, args.root, args.parquet_dir)
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        return 0

    if args.command == "build" and not os.path.isdir(args.parquet_dir):
        print(f"ERROR: no such directory: {args.parquet_dir}", file=sys.stderr)
        return 1

    manifest = _load_manifest(args.root)
    if manifest is None:
        return 1

    if args.command == "array":
        print(array_spec(manifest))
        return 0

    try:
        row = find_tissue(manifest, index=args.index, slug=args.slug)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return build_tissue(
        args.root, row["slug"], row["tissue"], args.parquet_dir, args.workers, args.overwrite
    )


if __name__ == "__main__":
    sys.exit(main())
