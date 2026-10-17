"""Standard-library controls and admitted views for bounded memorization runs."""

from collections import defaultdict
import re

from .common import digest


def controls(value=None):
    result = dict(
        max_updates=3000,
        budget_seconds=3600,
        validation_every_updates=100,
        diagnostic_seconds=180,
        view="shortest",
    )
    if value is not None:
        if not isinstance(value, dict) or set(value) != set(result):
            raise ValueError("closed learnability controls required")
        result.update(value)
    for key, low, high in (
        ("max_updates", 1, 100000),
        ("budget_seconds", 600, 43200),
        ("validation_every_updates", 1, 2000),
        ("diagnostic_seconds", 10, 3000),
    ):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key}: integer in {low}..{high} required")
    if result["budget_seconds"] < result["diagnostic_seconds"] + 180:
        raise ValueError("learnability budget must include diagnostic and cleanup reserves")
    if result["view"] not in {"shortest", "diverse"}:
        raise ValueError("learnability view must be shortest or diverse")
    return result


def gate(report, *, detailed=False):
    training = report["splits"]["training"]
    checks = dict(
        training_denominator=training["denominator"] == 32,
        training_accuracy=training["exact"] >= 31,
        conditioning_coverage=training["shuffled_denominator"] == 32,
        conditioning_effect=training["exact"] - training["shuffled_exact"] >= 4,
        teacher_forced_cache=report["cache_parity"]["passed"],
    )
    if detailed:
        parity = training.get("generation_parity", {})
        checks.update(
            evaluation_coverage=training["coverage_complete"],
            free_generation_cache=parity.get("passed", False)
            and parity.get("requested") == 32
            and parity.get("evaluated") == 32,
        )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        failures=[key for key, passed in checks.items() if not passed],
    )


def select_view(rows, view):
    """Select only existing admitted training rows; never invent supervision."""
    ordered = sorted(rows, key=lambda r: (len(r["tokens"]), r["program_id"]))
    if view == "shortest":
        selected = ordered[:32]
        if len(selected) != 32:
            raise ValueError("learnability requires 32 admitted training examples")
        return selected
    if view != "diverse":
        raise ValueError("unknown learnability view")
    from oeis_learn.decoder.program_codec import decode_body

    groups = defaultdict(list)
    for row in ordered:
        if len(row["tokens"]) > 128:
            continue
        source = decode_body(row["tokens"])
        shape = re.sub(r"(?<!\S)-?\d+(?!\S)", "#", source)
        groups[digest(shape)].append(row)
    selected, prefixes, offsets, affine = [], set(), 0, 0
    # Round-robin syntax shapes; no one shape or offset template may dominate.
    for index in range(min(8, max(map(len, groups.values()), default=0))):
        for _, candidates in sorted(groups.items()):
            if index >= len(candidates):
                continue
            row = candidates[index]
            values = list(map(int, row["values"]))
            prefix = digest(values)
            differences = [b - a for a, b in zip(values, values[1:])]
            is_affine = len(set(differences)) <= 1
            is_offset = differences == [1] * (len(values) - 1)
            if prefix in prefixes or (is_offset and offsets >= 8) or (is_affine and affine >= 16):
                continue
            selected.append(row)
            prefixes.add(prefix)
            offsets += is_offset
            affine += is_affine
            if len(selected) == 32:
                return selected
    raise ValueError(
        "diverse diagnostic needs 32 distinct admitted short prompts across >=4 shapes, "
        "including >=16 non-affine behaviors; prepare more connected examples"
    )
