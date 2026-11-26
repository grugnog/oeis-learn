"""Case-weighted, training-only pairwise ranker for one frozen eligible bank."""

import math
import re
from collections import Counter, defaultdict

from .contracts import digest, seal


def features(values, body):
    ns = list(map(int, values))
    if len(ns) != 20:
        raise ValueError("selector sees exactly the visible prefix")
    ds = [b - a for a, b in zip(ns, ns[1:])]
    case = [
        sum((a < 0) != (b < 0) for a, b in zip(ns, ns[1:])) / 19,
        sum(n % 2 for n in ns) / 20,
        sum(n == 0 for n in ds) / 19,
        sum(n.bit_length() for n in ds) / 19,
    ]
    atoms = re.findall(r"\(|\)|[^\s()]+", body)
    counts = Counter(atoms)
    depth = peak = 0
    for atom in atoms:
        depth += (atom == "(") - (atom == ")")
        peak = max(depth, peak)
    program = [
        len(atoms),
        peak,
        counts["select"] + counts["if"],
        counts["add"],
        counts["mul"],
        counts["mod"],
        max((int(a).bit_length() for a in atoms if a.lstrip("-").isdigit()), default=0),
    ]
    return [*program, *(v * n for v in case for n in program)]


def fit_ranker(examples):
    groups = defaultdict(list)
    samples = []
    for row in examples:
        a, b = (features(row["values"], row[k]) for k in ("positive", "negative"))
        groups[row["case"]].append((a, b))
        samples.extend((a, b))
    if not groups:
        raise ValueError("no verified training pairs for selector")
    # Fit feature scales on training rows; centering cancels in pair differences.
    scales = []
    for col in zip(*samples):
        mean = sum(col) / len(col)
        scales.append(math.sqrt(sum((v - mean) ** 2 for v in col) / len(col)) or 1.0)
    pairs = [
        (tuple((x - y) / s for x, y, s in zip(a, b, scales)), 1 / (len(groups) * len(group)))
        for group in groups.values()
        for a, b in group
    ]
    weights = [0.0] * len(scales)
    step = 1 / (1 + sum(w * sum(x * x for x in delta) / 4 for delta, w in pairs))
    for _ in range(256):
        grad = weights.copy()  # unit L2 regularization
        for delta, weight in pairs:
            margin = sum(v * x for v, x in zip(weights, delta))
            factor = weight / (1 + math.exp(max(-700, min(700, margin))))
            grad = [v - factor * x for v, x in zip(grad, delta)]
        weights = [v - step * g for v, g in zip(weights, grad)]
    return seal(
        dict(
            version="round028-pairwise-selector/v1",
            weights=weights,
            scales=scales,
            training=digest(examples),
            cases=len(groups),
            pairs=len(pairs),
        )
    )


def choose(candidates, values, model):
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope

    if model != seal(model):
        raise ValueError("selector model changed")

    def key(envelope):
        selected = validate_selection_envelope(envelope)
        body = selected["source"]
        prediction = sum(
            w * x / s
            for w, x, s in zip(
                model["weights"], features(values, body), model["scales"], strict=True
            )
        )
        if selected["selection"] == "visible-only/v1":
            return (-prediction, len(body.encode()), selected["cost"], selected["source_sha256"])
        return (
            -prediction,
            len(body.encode()),
            selected["source_sha256"],
            selected["attempt_ordinal"],
            selected["job_id"],
        )

    return min(candidates, key=key) if candidates else None


def shuffled_values(roster, prompts, seed):
    """Match different visible values within strata, keeping each recipient's own indices."""
    by_stratum = defaultdict(list)
    for row in roster:
        by_stratum[row["stratum"]].append(row["task"])
    result = {}
    for ids in by_stratum.values():
        ids.sort(key=lambda k: (tuple(prompts[k]["values"]), digest([seed, k])))
        largest = max(Counter(tuple(prompts[k]["values"]) for k in ids).values())
        for i, key in enumerate(ids):
            donor = ids[(i + largest) % len(ids)]
            values = prompts[donor]["values"]
            result[key] = values if donor != key and values != prompts[key]["values"] else None
    return result
