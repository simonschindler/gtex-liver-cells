"""Fetch the public GTEx Portal histology slide table.

The Portal histology API is the public source behind the lab's
``GTEx Portal.csv``: 25,713 slides across 40+ tissues, 610 of them liver.
"""

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://gtexportal.org/api/v2/histology/image"
PAGE_SIZE = 1000

COLUMNS = [
    "Tissue Sample ID",
    "Tissue",
    "Subject ID",
    "Sex",
    "Age Bracket",
    "Hardy Scale",
    "Pathology Categories",
    "Pathology Notes",
    "Hidden",
]


def fetch_page(page: int, per_page: int = PAGE_SIZE, retries: int = 3) -> dict:
    """Fetch one page of histology slides, retrying transient failures."""
    query = urllib.parse.urlencode({"page": page, "itemsPerPage": per_page})
    url = f"{API}?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def iter_slides(per_page: int = PAGE_SIZE):
    """Yield every slide record, following the API pagination."""
    page = 0
    while True:
        payload = fetch_page(page, per_page)
        yield from payload["data"]
        page += 1
        if page >= payload["paging_info"]["numberOfPages"]:
            return


def to_row(record: dict) -> list[str]:
    """Convert one API record into the nine output CSV fields."""
    categories = record.get("pathologyNotesCategories") or {}
    return [
        record.get("histologyImageId") or "",
        record.get("tissueSiteDetail") or "",
        record.get("subjectId") or "",
        record.get("sex") or "",
        record.get("ageBracket") or "",
        record.get("hardyScale") or "",
        ", ".join(name for name, flag in categories.items() if flag),
        record.get("pathologyNotes") or "",
        "true" if record.get("hide") else "false",
    ]


def fetch_slides(tissue: str | None = None) -> list[list[str]]:
    """Fetch all slides, optionally restricted to one tissue."""
    rows = []
    for record in iter_slides():
        if tissue and record.get("tissueSiteDetail") != tissue:
            continue
        rows.append(to_row(record))
    return rows


def write_csv(rows: list[list[str]], path: str) -> None:
    """Write rows using the Portal column order, quoting every field."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--tissue",
        default=None,
        help="Only fetch this tissue, e.g. 'Liver' (default: all slides)",
    )
    args = parser.parse_args(argv)

    rows = fetch_slides(args.tissue)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} slides to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
