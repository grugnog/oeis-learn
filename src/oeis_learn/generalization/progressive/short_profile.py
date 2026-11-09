"""Dedicated four-visible/nine-total finite task; authoritative terms never padded."""

from oeis_learn.synthesis.contracts import canonical_integer

PROFILE = "combinatorial-short/v1"
VISIBLE = 4
TOTAL = 9


def validate_terms(row):
    if (
        set(row) != {"indices", "values"}
        or row["indices"] != list(map(str, range(TOTAL)))
        or type(row["values"]) is not list
        or len(row["values"]) != TOTAL
    ):
        raise ValueError("exact n=0..8 short task required")
    for v in row["values"]:
        canonical_integer(v)
    return row


def visible(row):
    validate_terms(row)
    return dict(indices=row["indices"][:VISIBLE], values=row["values"][:VISIBLE])


def collate(rows):
    """One exact-length bucket, so input padding is unnecessary and prohibited."""
    if not 1 <= len(rows) <= 64:
        raise ValueError("bounded four-input bucket required")
    for r in rows:
        if (
            set(r) != {"indices", "values"}
            or r["indices"] != ["0", "1", "2", "3"]
            or len(r["values"]) != 4
        ):
            raise ValueError("four visible terms required")
        for v in r["values"]:
            canonical_integer(v)
    return dict(values=[list(r["values"]) for r in rows], position_mask=[[True] * 4 for _ in rows])
