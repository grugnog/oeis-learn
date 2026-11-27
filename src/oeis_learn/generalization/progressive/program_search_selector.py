"""Case-weighted, training-only pairwise ranker for one frozen eligible bank."""

import math
import re
from collections import Counter, defaultdict
from fractions import Fraction

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


def fit_ranker(examples, *, seed=0, stream=None):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("bounded selector training seed required")
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
            version="round028-pairwise-selector/v2",
            weights=weights,
            scales=scales,
            training=digest(examples),
            seed=seed,
            stream=stream or digest([seed, sorted(groups)]),
            fit={"loss":"case_weighted_pairwise_logistic", "l2":1, "iterations":256,
                 "scales":"training_population_sd", "tie":"frozen_visible_policy"},
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


# Positions in the anonymous joint distribution. Model/template components, when
# present, follow these six and do not participate in registered spending.
SELECTOR_ARMS = ("incumbent", "learned", "shuffled", "live_incumbent", "live_learned", "reference")
COMPONENTS = ("model_component", "template_component")


def selector_spending(cells, *, comparable_cost, repeat_funded=False, replay_funded=False):
    """Reconstruct the full-budget, fixed-bank and conditioning contrasts from family counts."""
    if any(type(v) is not bool for v in (comparable_cost, repeat_funded, replay_funded)):
        raise ValueError("explicit cost and funding dispositions required")
    comparisons = {"end_to_end": (5, 4), "same_bank": (0, 1), "shuffled": (2, 1)}
    grouped = defaultdict(list)
    seen = set()
    for cell in cells:
        key = (cell["stratum"], cell["group"])
        if (set(cell) != {"stratum", "group", "counts"} or key in seen or key[0] not in
                {"composition", "unseen_structure", "parameter", "oeis", "retention"}
                or type(key[1]) is not int or key[1] < 0 or not cell["counts"]):
            raise ValueError("invalid selector family")
        seen.add(key)
        for code, count in cell["counts"].items():
            if len(code) not in (6, 8) or set(code) - set("SFU") or type(count) is not int or count <= 0:
                raise ValueError("invalid selector outcomes")
        grouped[key[0]].append(cell["counts"])

    def effects(counts, baseline, candidate):
        n = sum(counts.values())
        gain = sum(v * ((k[candidate] == "S") - (k[baseline] == "S")) for k, v in counts.items())
        missing = [sum(v for k, v in counts.items() if k[i] == "U") for i in (baseline, candidate)]
        return tuple(Fraction(v, n) for v in (gain, gain - missing[0], gain + missing[1]))

    contrasts = {}
    for name, (baseline, candidate) in comparisons.items():
        strata = {}
        for stratum, families in grouped.items():
            vectors = [effects(counts, baseline, candidate) for counts in families]
            strata[stratum] = [sum(col) / len(col) for col in zip(*vectors)]
        primary = [strata[s] for s in ("composition", "unseen_structure") if s in strata]
        weighted = [float(sum(col) / 2) for col in zip(*primary)] if len(primary) == 2 else [None] * 3
        contrasts[name] = dict(zip(("gain", "lower", "upper"), weighted))
        contrasts[name]["strata"] = {s: list(map(float, v)) for s, v in strata.items()}
    retention = sum((Counter(c) for c in grouped["retention"]), Counter())
    loss = None
    if retention:
        _, low, high = effects(retention, 5, 4)
        loss = [-float(high), -float(low)]
    if not comparable_cost:
        action = "INCOMPARABLE"
    elif any(c["gain"] is None for c in contrasts.values()) or loss is None:
        action = "INSUFFICIENT_PANEL"
    elif any(c["upper"] <= 0 for c in contrasts.values()) or loss[0] > 0.02:
        action = "RETAIN_REFERENCE"
    elif any(c["lower"] <= 0 for c in contrasts.values()) or loss[1] > 0.02:
        action = "REPLAY" if replay_funded else "UNRESOLVED_SPENDING_CONTRAST"
    else:
        action = "RUN_RESERVED_REPLICATION" if repeat_funded else "REPLICATION_UNFUNDED"
    return dict(contrasts=contrasts, retention_loss=loss, action=action)


def selector_counts(cells, arms):
    totals = {arm: Counter() for arm in arms}
    pairs = {}
    for challenger, baseline in (("learned","incumbent"), ("learned","shuffled"),
                                  ("live_learned","live_incumbent"), ("live_learned","reference")):
        a, b = arms.index(challenger), arms.index(baseline)
        rows = []
        for cell in cells:
            joint = Counter()
            for code, n in cell["counts"].items():
                joint[code[a] + code[b]] += n
            rows.append(dict(stratum=cell["stratum"], group=cell["group"], counts=dict(joint)))
        pairs[challenger + "_minus_" + baseline] = rows
    for cell in cells:
        for code, n in cell["counts"].items():
            for arm, outcome in zip(arms, code, strict=True):
                totals[arm][outcome] += n
    return {a:dict(c) for a,c in totals.items()}, pairs


def verify_selector(path):
    from .program_search_outcome import read_canonical
    from .program_search_rehearsal_report import encode
    import json
    raw = read_canonical(path, ("selector.json",))["selector.json"]
    report = json.loads(raw)
    fields = {"version","design","registration","selections","cases","arms","pairs","cells","spending",
              "elapsed_us","search_us","costs","gpu_dispatches","identity"}
    design = report["design"]
    arms = tuple(design["arms"])
    if (set(report) != fields or report["version"] != "round028-selector-report/v1" or report != seal(report)
            or raw != encode(report) or design != seal(design) or set(design) !=
            {"version","registration","roster","shuffle","model","reference","arms","seed","profile","sizes","identity"}
            or design["version"] != "round028-selector-design/v1" or design["registration"] != report["registration"]
            or arms not in (SELECTOR_ARMS, SELECTOR_ARMS + COMPONENTS)
            or design["reference"] not in {"retained_D","trial05","template","polynomial","linear_recurrence"}
            or type(design["seed"]) is not int or not 0 <= design["seed"] < 2**32
            or design["profile"] != {"attempts":8,"portfolio_us":60000000,"fit_us":4000000,"tokens":1024}
            or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", s) for s in
                   (design["roster"], design["model"], design["shuffle"], report["registration"], report["selections"]))):
        raise ValueError("closed selector design and report required")
    cells = report["cells"]
    spending = selector_spending(cells, comparable_cost=True)
    sizes = [dict(stratum=c["stratum"], group=c["group"], cases=sum(c["counts"].values())) for c in cells]
    totals, pairs = selector_counts(cells, arms)
    costs = report["costs"]
    if (sizes != design["sizes"] or not sizes or sum(c["cases"] for c in sizes) != report["cases"]
            or report["spending"] != spending or report["arms"] != totals or report["pairs"] != pairs
            or set(costs) != {"bank","live_incumbent","live_learned","reference","score"}
            or any(type(n) is not int or n < 0 for n in (*costs.values(), report["search_us"]))
            or report["elapsed_us"] != sum(costs.values()) or report["gpu_dispatches"] != 0):
        raise ValueError("selector counts, costs or spending differ")
    return dict(status="VERIFIED", scope="archive_only_selector", report=report["identity"],
                design=design["identity"], cases=report["cases"], next_action=spending["action"],
                elapsed_us=report["elapsed_us"], final_assessment="SEALED")
