"""Deterministic within-structure holdout for new bootstrap runs only."""

from collections import defaultdict
from .common import digest


def parameter_view(examples):
    """Reserve one distinct behavior per eligible structure before any updates.

    Keep at least two training variants, cap the holdout at 1/8 of the original
    training view (and 128 rows), and never touch structural development. An
    empty split is reported as unavailable, not as evidence of transfer.
    """
    groups = defaultdict(list)
    for row in examples["training"]:
        if row.get("structure"):
            groups[row["structure"]].append(row)
    candidates = []
    for rows in groups.values():
        if len({digest(r["values"]) for r in rows}) >= 3:
            candidates.append(min(rows, key=lambda r: r["program_id"]))
    stages = defaultdict(list)
    for row in sorted(candidates, key=lambda r: r["program_id"]):
        stages[row.get("stage", 0)].append(row)
    ordered = [
        rows[i]
        for i in range(max(map(len, stages.values()), default=0))
        for _, rows in sorted(stages.items())
        if i < len(rows)
    ]
    held = ordered[: min(128, len(examples["training"]) // 8)]
    prefixes = {digest(r["values"]) for r in held}
    return dict(
        examples,
        training=[r for r in examples["training"] if digest(r["values"]) not in prefixes],
        parameter_development=held,
    )


MIXTURE = dict(
    version="connected-training-mixture/v1",
    stages=4,
    max_prefix_constant_percent=5,
    max_identity_prefixes=1,
)


def behavior(row):
    values = list(map(int, row["values"]))
    if len(set(values)) == 1:
        return "prefix_constant"
    if values == list(range(len(values))):
        return "identity_prefix"
    return "other"


def training_mixture(training, count):
    """Choose shortest admitted representatives under explicit stage/behavior caps.

    Classification uses twenty visible terms; it is not a semantic proof over
    the complete domain. Archive evidence is retained, never simplified away.
    """
    result, stages, constants, identities = {}, defaultdict(int), 0, 0
    for key, row in sorted(
        training.items(), key=lambda pair: (len(pair[1]["tokens"]), pair[1]["program_id"])
    ):
        stage = row["stage"]
        kind = behavior(row)
        if stages[stage] >= (count + 3 - stage) // 4:
            continue
        if kind == "prefix_constant" and constants >= count * 5 // 100:
            continue
        if kind == "identity_prefix" and identities >= 1:
            continue
        result[key] = row
        stages[stage] += 1
        constants += kind == "prefix_constant"
        identities += kind == "identity_prefix"
    return result
