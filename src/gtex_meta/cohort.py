"""Label the GTEx liver cohort from the Portal histology table.

Rules
-----
cirrhosis
    The curated ``Pathology Categories`` contain ``cirrhosis``.
healthy
    Either the pathologist curated ``no_abnormalities`` / ``clean_specimens``,
    or the slide has no curated category *and* its note is non-empty, contains
    no positive finding, and carries no preservation warning.
other
    Everything else. An empty note is unknown, never healthy.
"""

import argparse
import re
import sys

import pandas as pd

# Findings that disqualify a slide from the control group even when the curator
# assigned no category. Kept wide on purpose: the uncategorized slides do
# contain real findings in their free text.
FINDING_RE = re.compile(
    r"(?i)("
    r"steato|fatty|\bfat\b|congest|sinusoidal|fibro|cirrh|necro|inflamm|hepatit"
    r"|lymphocy|lymphoid|infiltrat|sclero|nodul|atroph|hemorrhag|hyalin|ischem"
    r"|hypox|scar|hyperplas|pigment|balloon|degener|apoptos|granulom|cholang"
    r"|vacuol|amyloid|hamartom|pallor|dilat|cord cell|abscess|metaplas|calcif"
    r"|eosinophil|neutrophil"
    r")"
)

# Tissue-quality problems: not disease, but they confound histology features.
# Both spellings of autolysis count; an earlier prototype matched only
# "autolys" and mislabeled three autolyzed slides as healthy.
QUALITY_RE = re.compile(
    r"(?i)(autoly[sz]|poorly preserved|poor preservation|poor fixation|bad fixation)"
)

# Phrases that negate a finding, e.g. "no fat", "not congested".
NEGATION_RE = re.compile(
    r"(?i)\b(no|not|without|free of|negative for)\b[^.;,]{0,30}"
    r"(fat|steato|congest|fibro|cirrh|necro|inflamm|hepatit|infiltrat|sclero"
    r"|nodul|atroph|hemorrhag|hyalin|ischem|scar|hyperplas|pigment|balloon"
    r"|degener|lesion|abnormal)"
)

CLEAN_CATEGORIES = {"no_abnormalities", "clean_specimens"}


def split_categories(value) -> set[str]:
    """Split a comma-separated category string into a set of names."""
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def has_finding(note: str) -> bool:
    """True if the note describes a finding, after removing negated phrases."""
    stripped = NEGATION_RE.sub(" ", str(note or ""))
    return bool(FINDING_RE.search(stripped))


def label_row(row: pd.Series) -> tuple[str, str]:
    """Return the (label, reason) for one slide."""
    categories = split_categories(row["Pathology Categories"])
    note = str(row["Pathology Notes"] or "").strip()

    if "cirrhosis" in categories:
        return "cirrhosis", "curated cirrhosis"
    if categories & CLEAN_CATEGORIES:
        curated = "/".join(sorted(categories & CLEAN_CATEGORIES))
        return "healthy", f"curated {curated}"
    if not categories:
        if not note:
            return "other", "uncategorized, no note (unknown)"
        if QUALITY_RE.search(note):
            return "other", "uncategorized, preservation warning"
        if has_finding(note):
            return "other", "uncategorized, finding in note"
        return "healthy", "uncategorized, note reports no finding"
    return "other", "curated non-cirrhotic pathology"


def build_cohort(slides: pd.DataFrame) -> pd.DataFrame:
    """Return the slide table with label columns appended."""
    tissues = set(slides["Tissue"].astype(str).unique())
    if tissues != {"Liver"}:
        raise ValueError(f"expected liver-only input, found tissues: {sorted(tissues)}")

    result = slides.copy()
    labels, reasons = zip(*(label_row(row) for _, row in result.iterrows()))
    result["label"] = labels
    result["label_reason"] = reasons
    result["quality_flag"] = result["Pathology Notes"].apply(
        lambda note: "preservation" if QUALITY_RE.search(str(note or "")) else ""
    )
    result["has_finding_text"] = result["Pathology Notes"].apply(has_finding)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", required=True, help="Input slide table CSV")
    parser.add_argument("--out", required=True, help="Output cohort CSV")
    args = parser.parse_args(argv)

    slides = pd.read_csv(args.slides, dtype=str).fillna("")
    labelled = build_cohort(slides)
    labelled.to_csv(args.out, index=False)

    counts = labelled["label"].value_counts()
    print(f"wrote {len(labelled)} slides to {args.out}")
    for name in ("healthy", "cirrhosis", "other"):
        print(f"  {name:10} {counts.get(name, 0)}")

    cirrhosis = labelled[labelled["label"] == "cirrhosis"]
    co_findings: dict[str, int] = {}
    for value in cirrhosis["Pathology Categories"]:
        for category in split_categories(value) - {"cirrhosis"}:
            co_findings[category] = co_findings.get(category, 0) + 1
    for category, count in sorted(co_findings.items(), key=lambda item: -item[1]):
        print(f"  {category:15} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
