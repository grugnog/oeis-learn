"""Read-only source census. Completeness and signed bounds have separate denominators."""

from collections import Counter
import re

HORIZONS = (20, 50, 100, 120)


def bit_width(value):
    if type(value) is not int:
        raise ValueError("source values must be exact integers")
    return value.bit_length() + 1 if value >= 0 else (~value).bit_length() + 1


def classify_growth_profile(terms):
    """Finite full-series descriptor; makes no asymptotic inference."""
    if len(terms) < 4:
        return "insufficient_terms"
    magnitudes = [abs(v) for v in terms]
    if len(set(magnitudes)) == 1:
        return "constant_magnitude"
    if all(a <= b for a, b in zip(magnitudes, magnitudes[1:])):
        return "nondecreasing_magnitude"
    if all(a >= b for a, b in zip(magnitudes, magnitudes[1:])):
        return "nonincreasing_magnitude"
    return "nonmonotone_magnitude"


def analyze_sequence_cohort(records, horizons=HORIZONS):
    from numpy import percentile

    for row in records:
        if not isinstance(row.get("terms"), list) or any(type(v) is not int for v in row["terms"]):
            raise ValueError("invalid exact source terms")
    result = {"total_sequences": len(records), "horizons": {}}
    for h in horizons:
        if type(h) is not int or h not in HORIZONS:
            raise ValueError("unsupported census horizon")
        complete = [r["terms"][:h] for r in records if len(r["terms"]) >= h]
        widths = [max(map(bit_width, terms)) for terms in complete]
        counts = {
            f"signed{bits}": sum(
                all(-(1 << (bits - 1)) <= v < (1 << (bits - 1)) for v in terms)
                for terms in complete
            )
            for bits in (64, 128, 256, 512)
        }
        result["horizons"][f"horizon_{h}"] = {
            "available_prefixes": sum(bool(r["terms"]) for r in records),
            "available_prefix_in_range": sum(
                bool(r["terms"]) and all(-(1 << 255) <= v < (1 << 255) for v in r["terms"][:h])
                for r in records
            ),
            "sequences_evaluated": len(complete),
            "sequences_with_full_terms": len(complete),
            "coverage_counts": counts,
            "coverage_pct": {
                k: v / len(complete) * 100 if complete else None for k, v in counts.items()
            },
            "percentiles": {f"p{p}": float(percentile(widths, p)) for p in (50, 75, 90, 95, 99)}
            | {"max": max(widths)}
            if widths
            else {},
        }
    result["growth_profile_counts"] = dict(
        Counter(classify_growth_profile(r["terms"]) for r in records)
    )
    result["growth_policy"] = "finite_full_series_descriptor/v1"
    result["source_metadata"] = {
        "offsets": dict(
            Counter(
                str(r.get("offset")) if r.get("offset") is not None else "unavailable"
                for r in records
            )
        ),
        "finite": sum(
            "fini"
            in keywords(r.get("source_keywords") if "source_keywords" in r else r.get("tags"))
            for r in records
        ),
        "table": sum(
            "tabl"
            in keywords(r.get("source_keywords") if "source_keywords" in r else r.get("tags"))
            for r in records
        ),
        "keywords": dict(
            Counter(
                k
                for r in records
                for k in keywords(
                    r.get("source_keywords") if "source_keywords" in r else r.get("tags")
                )
            )
        ),
        "missing_keywords": sum(
            (r.get("source_keywords") if "source_keywords" in r else r.get("tags")) is None
            for r in records
        ),
    }
    return result


def keywords(tags):
    return set(re.findall(r"[a-z]+", tags.lower())) if isinstance(tags, str) else set()
