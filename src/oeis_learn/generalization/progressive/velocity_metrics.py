"""Paired descriptive measurements in percentage points for fixed round029 rosters."""

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from math import isfinite

from .contracts import _check, schema, seal


def case_outcome_receipt(case, *, outcome_ref, choice_ref, scorer_ref, truth_ref, status, inference_ref=None):
    """Bind a private score (or unresolved opportunity) independently of its use at a look."""
    hashes = schema("common")["$defs"]["hash"]
    for ref in (outcome_ref, scorer_ref, truth_ref):
        _check(hashes, ref, {})
    if choice_ref is not None:
        _check(hashes, choice_ref, {})
    if not isinstance(case, str) or not case or status not in {"solve", "miss", "unresolved"}:
        raise ValueError("invalid case outcome receipt")
    if status == "solve" and choice_ref is None:
        raise ValueError("a verified solve requires a sealed choice")
    scope = {}
    if inference_ref is not None:
        _check(hashes, inference_ref, {})
        scope["inference_ref"] = inference_ref
    return seal(dict(version="round029-case-outcome/v2" if scope else "round029-case-outcome/v1", **scope, case=case, outcome_ref=outcome_ref,
                     choice_ref=choice_ref, scorer_ref=scorer_ref, truth_ref=truth_ref, status=status))


def _comparison_studies(views, registrations):
    from .campaign import validate_velocity_registration
    from .graph import velocity_study_looks

    requested = {p["look"]["registration"] for p in views}
    if not registrations and len(requested) == 1:
        return
    studies = {r["identity"]: validate_velocity_registration(r) for r in registrations}
    if set(studies) != requested or (len(studies) > 1 and
            (len(studies) != 2 or not any(r["parent_ref"] in studies for r in studies.values()))):
        raise ValueError("comparison requires one study or its explicitly registered child")
    graphs = {k: velocity_study_looks(r) for k, r in studies.items()}
    for probe in views:
        look = probe["look"]
        graph = graphs[look["registration"]]
        if look not in graph.get(look["method"], {}).get(look["profile"], []):
            raise ValueError("look outside the linked pre-score registrations")


def _frozen_rows(rows, looks, names, registrations):
    from oeis_learn.pipeline_validation.progressive_evaluation import checked_probe_model, probe_inference
    from .graph import velocity_look

    if not names or not all(name in looks for name in names):
        raise ValueError("registered look required for every contrast coordinate")
    if any(looks[n].get("version") != "round029-probe/v2" for n in names):
        raise ValueError("registered probe required on the frozen roster")
    _comparison_studies([looks[n] for n in names], registrations)
    members = None
    coordinates = []
    inferences, by_coordinate = {}, {}
    for name in names:
        probe = looks[name]
        look = probe["look"]
        if (probe != seal(probe) or probe["mode"] not in ("true", "shuffled")
                or set(probe["donors"]) != {r["task"] for r in look["members"]}
                or set(probe["donors"].values()) != set(probe["donors"])):
            raise ValueError("invalid registered probe or roster")
        _check(schema("common")["$defs"]["hash"], probe["recipients"], {})
        checked_probe_model(look, probe["model"])
        coordinate = (look["registration"], look["method"], look["seed"], look["look"], look["profile"], probe["mode"])
        if by_coordinate.setdefault(coordinate, probe["identity"]) != probe["identity"]:
            raise ValueError("conflicting probes at one registered coordinate")
        inferences[name] = probe_inference(probe)
        # Reopen the actual registered graph record, including its order and profile.
        development = [r for r in look["members"] if r["stratum"] != "retention"]
        retention = [r for r in look["members"] if r["stratum"] == "retention"]
        coord = {field: look[field] for field in ("method", "seed", "look", "profile")}
        _check(schema("common")["$defs"]["hash"], look["registration"], {})
        expected = velocity_look(registration=look["registration"], **coord, development=development,
                                 retention=retention, extension=True)
        if type(coord["seed"]) is not int or coord["seed"] < 0 or expected != look:
            raise ValueError("invalid frozen look or roster")
        if coordinates and coord["profile"] != coordinates[0]["profile"]:
            raise ValueError("compared looks have different inference profiles")
        if members is not None and members != look["members"]:
            raise ValueError("compared looks have different frozen rosters")
        members = look["members"]
        coordinates.append(dict(coord, registration=look["registration"], input_mode=probe["mode"]))
    roster = {r["task"]: (r["stratum"], r["group"]) for r in members}
    remaining = set(roster)
    families = defaultdict(list)
    known = {}
    for row in rows:
        case = row["case"]
        if case not in remaining or roster[case] != (row["stratum"], row["family"]):
            raise ValueError("case does not match the unique frozen roster")
        remaining.remove(case)
        same_inference = {}
        scoring_scope = None
        for name in names:
            obs = row["observations"].get(name)
            if not isinstance(obs, dict) or type(obs.get("closed")) is not bool or type(obs.get("complete")) is not bool:
                raise ValueError("explicit case observation and completion required on frozen roster")
            receipt = obs.get("receipt")
            evidence_fields = ("outcome_ref", "choice_ref", "truth_ref", "scorer_ref", "inference_ref")
            if obs.get("probe_ref") != looks[name]["identity"] or (receipt or {}).get("inference_ref") != inferences[name]:
                raise ValueError("outcome receipt belongs to another probe or coordinate")
            if not isinstance(receipt, dict) or not all(k in receipt for k in evidence_fields):
                raise ValueError("sealed per-case outcome receipt required")
            evidence = {k: receipt[k] for k in evidence_fields}
            binding = case_outcome_receipt(case, status=obs["status"], **evidence)
            if receipt != binding or known.setdefault(receipt["outcome_ref"], binding["identity"]) != binding["identity"]:
                raise ValueError("inconsistent case, choice, scorer or truth on outcome receipt")
            scope = receipt["truth_ref"], receipt["scorer_ref"]
            if scoring_scope is not None and scope != scoring_scope:
                raise ValueError("compared outcome receipts use different scorer or truth")
            scoring_scope = scope
            observed = receipt["identity"], obs["complete"], obs["closed"]
            if same_inference.setdefault(inferences[name], observed) != observed:
                raise ValueError("conflicting case evidence at the same inference opportunity")
        families[roster[case]].append(row)
    if remaining:
        raise ValueError("missing case on frozen roster")
    return families, coordinates


def contrast(rows, coefficients, *, stratum, looks, registrations=()):
    """Compare only the complete authenticated rosters of the registered looks."""
    names = sorted(coefficients)
    all_families, coordinates = _frozen_rows(rows, looks, names, registrations)
    return family_contrast(all_families, coordinates, coefficients, stratum)


def family_contrast(all_families, coordinates, coefficients, stratum):
    """Reduce validated private rows or the same joint patterns from a copied public report."""
    names = sorted(coefficients)
    strata = ("composition", "unseen_structure") if stratum == "structural" else (stratum,)
    families = {k: v for k, v in sorted(all_families.items()) if k[0] in strata}
    support = Counter(s for s, _ in families)
    denominator = sum(len(members) for members in families.values())
    supported = all(support[s] for s in strata)
    unknown = defaultdict(Fraction)
    paired = Counter(
        {
            k: 0
            for k in (
                "gains",
                "losses",
                "both",
                "neither",
                "unresolved_overlay",
                "first_solve_families",
                "last_solve_families",
            )
        }
    )
    pair = sorted(coefficients, key=coefficients.get)
    binary = [coefficients[k] for k in pair] == [-1, 1]
    observed = Fraction()
    counts = {name: Counter() for name in coefficients}
    joint = []
    for (s, family), members in families.items():
        patterns = Counter()
        weight = Fraction(100, len(strata) * support[s] * len(members))
        solved = Counter()
        for row in members:
            obs = row["observations"]
            classes = {}
            pattern = []
            for name in names:
                coefficient = coefficients[name]
                receipt = obs[name]["receipt"]["identity"]
                equality = classes.setdefault(receipt, len(classes))
                pattern.append((obs[name]["status"], obs[name]["complete"], obs[name]["closed"], equality))
                status = obs[name]["status"]
                counts[name][status] += 1
                counts[name]["complete"] += obs[name]["complete"]
                counts[name]["closed"] += obs[name]["closed"]
                if status == "solve":
                    observed += weight * coefficient
                    solved[name] += 1
                elif status == "unresolved":
                    unknown[receipt] += weight * coefficient
            patterns[tuple(pattern)] += 1
            if binary:
                a, b = [obs[k]["status"] for k in pair]
                paired["unresolved_overlay"] += "unresolved" in (a, b)
                key = {
                    (False, True): "gains",
                    (True, False): "losses",
                    (True, True): "both",
                    (False, False): "neither",
                }[a == "solve", b == "solve"]
                paired[key] += 1
        cells = []
        for pattern, count in sorted(patterns.items()):
            outcomes = [dict(status=status, complete=complete, closed=closed, equality=equality)
                        for status, complete, closed, equality in pattern]
            cells.append(dict(count=count, outcomes=outcomes))
        joint.append(dict(stratum=s, family=family, frozen_size=len(members), patterns=cells))
        if binary:
            paired["first_solve_families"] += not solved[pair[0]] and bool(solved[pair[1]])
            paired["last_solve_families"] += bool(solved[pair[0]]) and not solved[pair[1]]
    return dict(
        observed_pp=float(observed) if supported else None,
        lower_pp=float(observed + sum(min(0, c) for c in unknown.values())) if supported else None,
        upper_pp=float(observed + sum(max(0, c) for c in unknown.values())) if supported else None,
        **({} if supported else {"reason": "missing stratum"}),
        denominator=denominator,
        family_counts=dict(support),
        coordinates=coordinates,
        families=joint,
        outcomes={k: dict(v) for k, v in counts.items()},
        **(paired if binary else {}),
    )


def rates(delta_pp, tokens, charged_us):
    def rate(den, scale):
        if den is None or delta_pp is None or den == 0:
            return None
        if den < 0 or not isfinite(den):
            raise ValueError("invalid additional cost")
        return delta_pp * scale / den

    return dict(pp_per_million=rate(tokens, 1_000_000), pp_per_hour=rate(charged_us, 3_600_000_000))


def affordable_look(looks, axis, budget, *, overlap):
    if not overlap[0] <= budget <= overlap[1]:
        return None
    eligible = [
        row
        for row in looks
        if row["verified"] and row.get(axis) is not None and row[axis] <= budget
    ]

    def coordinate(row):
        tokens = row.get("useful_tokens", 0)
        return row[axis], row.get("look", tokens), tokens

    return max(eligible, key=coordinate, default=None)


def factorial(levels):
    def block(ids):
        a, b, c, d = [levels.get(k) for k in ids]
        if any(v is None for v in (a, b, c, d)):
            return None
        return dict(
            first=[b - a, d - c],
            second=[c - a, d - b],
            first_mean=((b - a) + (d - c)) / 2,
            second_mean=((c - a) + (d - b)) / 2,
            interaction=(d - c) - (b - a),
        )

    return dict(
        composition_matching=block(("00", "01", "02", "03")),
        batch_decay=block(("00", "04", "05", "06")),
    )


def sampling_interval(rows, coefficients, *, stratum, seed, looks, scale=1, draws=10000, registrations=()):
    """Resample each linked family once per draw across all arms, looks and strata."""
    result = contrast(rows, coefficients, stratum=stratum, looks=looks, registrations=registrations)
    return _sample_contrast(result, coefficients, stratum, seed, draws, scale)


@lru_cache(maxsize=32)
def _supported_draws(masks, full, seed, draws):
    import random

    if all(m == full for m in masks):
        return draws
    rng = random.Random(seed)
    supported = 0
    for _ in range(draws):
        mask = 0
        for m in rng.choices(masks, k=len(masks)):
            mask |= m
        supported += mask == full
    return supported


def _sample_contrast(result, coefficients, stratum, seed, draws, scale=1):
    import random

    rng = random.Random(seed)
    strata = ("composition", "unseen_structure") if stratum == "structural" else (stratum,)
    coefficients_in_order = [coefficients[name] for name in sorted(coefficients)]
    by_group = defaultdict(dict)
    for family in result["families"]:
        net_solves = sum(pattern["count"] * sum(c for c, outcome in zip(coefficients_in_order, pattern["outcomes"])
                                                if outcome["status"] == "solve")
                         for pattern in family["patterns"])
        by_group[family["family"]][family["stratum"]] = 100 * net_solves / family["frozen_size"]
    contributions = [by_group[g] for g in sorted(by_group)]
    if not all(any(s in row for row in contributions) for s in strata):
        return dict(lower=None, upper=None, draws=draws, supported_draws=0, seed=seed)
    if len(contributions) == 1:
        value = scale * sum(contributions[0][s] for s in strata) / len(strata)
        return dict(lower=value, upper=value, draws=draws, supported_draws=draws, seed=seed)
    constants = [{r[s] for r in contributions if s in r} for s in strata]
    if all(len(v) == 1 and next(iter(v)).is_integer() for v in constants):
        masks = tuple(sum(1 << i for i, s in enumerate(strata) if s in row) for row in contributions)
        supported = _supported_draws(masks, (1 << len(strata)) - 1, seed, draws)
        value = scale * sum(next(iter(v)) for v in constants) / len(strata) if supported else None
        return dict(lower=value, upper=value, draws=draws, supported_draws=supported, seed=seed)
    values = []
    for _ in range(draws):
        totals, counts = Counter(), Counter()
        for entry in rng.choices(contributions, k=len(contributions)):
            for s, value in entry.items():
                totals[s] += value
                counts[s] += 1
        if all(counts[s] for s in strata):
            values.append(scale * sum(totals[s] / counts[s] for s in strata) / len(strata))
    values.sort()
    return dict(
        lower=values[int(0.025 * (len(values) - 1))] if values else None,
        upper=values[int(0.975 * (len(values) - 1))] if values else None,
        draws=draws,
        supported_draws=len(values),
        seed=seed,
    )


def _score_rows(measurements):
    """Align the selected-choice scorer's private outputs on the first frozen roster."""
    first = next(iter(measurements.values()))["probe"]["look"]
    return [
        dict(case=row["task"], stratum=row["stratum"], family=row["group"],
             observations={name: m["scores"]["outcomes"][row["task"]]["equal_budget"]
                           for name, m in measurements.items()})
        for row in first["members"]
    ]


def measured_look(probe, scores, *, cut=None, work=()):
    """Bind a scored probe to its worker's whole-update cut and cumulative operation receipts."""
    from .campaign import velocity_policy
    from .program_search_costing import workstation_accounting
    from .trials import integer
    from .velocity_trials import crossed

    look = probe["look"]
    if (scores != seal(scores) or scores.get("version") != "discovery-scoring/v1"
            or scores["denominator"] != len(look["members"])
            or set(scores["outcomes"]) != {row["task"] for row in look["members"]}):
        raise ValueError("score differs from the registered probe roster")
    measurement = dict(version="round029-measured-look/v1", probe=probe, scores=scores,
                       cut=None, work=list(work))
    learner = look["method"] in {r["id"] for r in velocity_policy()["recipes"]}
    if learner:
        if cut is None or cut["target"] != look["look"] or cut["model"] != probe["model"]:
            raise ValueError("scored learner weights must match the original worker cut")
        fields = ("useful_tokens", "charged_tokens", "discarded_tokens", "updates", "last_update_tokens", "overshoot")
        exposure = {k: integer(cut[k]) for k in fields}
        useful = exposure["useful_tokens"]
        if (not crossed(exposure, look["look"]) or useful - look["look"] != exposure["overshoot"]
                or useful + exposure["discarded_tokens"] != exposure["charged_tokens"]
                or exposure["last_update_tokens"] > useful
                or (look["look"] == 0 and useful != 0)
                or (look["look"] > 0 and not exposure["updates"])):
            raise ValueError("scored look does not cross the registered whole-update boundary")
        measurement["cut"] = dict(exposure, target=cut["target"], model=cut["model"])
    elif cut is not None:
        raise ValueError("a fixed reference has no training cut")
    _frozen_rows(_score_rows({"look": measurement}), {"look": probe}, ["look"], ())
    workstation_accounting(measurement["work"])
    return seal(measurement)


FACTOR_BLOCKS = {
    "composition_matching": ("00", "01", "02", "03"),
    "batch_decay": ("00", "04", "05", "06"),
}


def _factor_effects(coordinates, estimate):
    """Each effect is one paired contrast across exactly the coordinates that it uses."""
    def block(methods):
        def effect(weights):
            terms = {}
            for method, weight in zip(methods, weights, strict=True):
                if not weight:
                    continue
                if not coordinates.get(method):
                    return None
                for coordinate, coefficient in coordinates[method].items():
                    terms[coordinate] = terms.get(coordinate, 0) + coefficient * weight
            return estimate(terms)

        return dict(first=[effect((-1, 1, 0, 0)), effect((0, 0, -1, 1))],
                    second=[effect((-1, 0, 1, 0)), effect((0, -1, 0, 1))],
                    first_mean=effect((-.5, .5, -.5, .5)), second_mean=effect((-.5, -.5, .5, .5)),
                    interaction=effect((1, -1, -1, 1)))

    return {name: block(methods) for name, methods in FACTOR_BLOCKS.items()}


def study_curve(looks, measurements, *, stratum, seed, draws=10_000, budgets=None, paired=None):
    """Reduce one registered seed's common series and separate full endpoints."""
    from .cohorts import STRATA
    from .program_search_costing import workstation_accounting
    from .trials import integer

    if stratum not in (*STRATA, "structural", "retention"):
        raise ValueError("unregistered curve stratum")
    integer(draws, 1)
    integer(seed)
    measured = {ref: item for ref, item in measurements.items() if "probe" in item}
    estimates = {}

    def estimate(terms):
        key = tuple(sorted(terms.items()))
        if key not in estimates:
            subset = {ref: measured[ref] for ref in terms}
            probes = {ref: m["probe"] for ref, m in subset.items()}
            observed = (contrast(_score_rows(subset), terms, stratum=stratum, looks=probes)
                        if paired is None else paired(terms, stratum))
            turnover = {k: observed[k] for k in ("gains", "losses", "both", "neither", "unresolved_overlay",
                                                 "first_solve_families", "last_solve_families") if k in observed}
            estimates[key] = dict(
                **turnover, observed=observed["observed_pp"], lower=observed["lower_pp"], upper=observed["upper_pp"],
                denominator=observed["denominator"], family_counts=observed["family_counts"],
                sampling=_sample_contrast(observed, terms, stratum, seed, draws),
            )
        return estimates[key]

    def point(look):
        ref = look["identity"]
        row = dict(look=look["look"], reference=ref, verified=ref in measured)
        if ref not in measured:
            return dict(row, reason=measurements.get(ref, {}).get("reason", "unmeasured"))
        m = measured[ref]
        if m["probe"]["look"] != look or m["probe"]["mode"] != "true":
            raise ValueError("primary curve requires its registered true-conditioning probe")
        costs = workstation_accounting(m["work"])["by_beneficiary"].get(look["method"], {})
        row.update(charged_us=costs.get("production_us"), historical_us=costs.get("historical_us"),
                   incremental_us=costs.get("incremental_us"))
        if m["cut"] is None:
            search_work = [record for record in m["work"] if record["role"] == "search"]
            search_cost = workstation_accounting(search_work)["by_beneficiary"].get(look["method"], {})
            row["inference_us"] = search_cost.get("production_us")
        else:
            row.update({k: m["cut"][k] for k in ("useful_tokens", "charged_tokens", "discarded_tokens",
                                                "updates", "last_update_tokens", "overshoot")})
            training = [w for w in m["work"] if w["role"] == "train"]
            if training and all("issued_tokens" in w for w in training):
                row["issued_tokens"] = sum(integer(w["issued_tokens"]) for w in training)
                row["unknown_execution_tokens"] = integer(row["issued_tokens"] - row["charged_tokens"])
        row["yield"] = estimate({ref: 1})
        return row

    curves, endpoints, levels, rate_terms = {}, {}, {}, {}
    for method, profiles in looks.items():
        if profiles.get("full"):
            endpoints[method] = point(profiles["full"][0])
        if not profiles.get("common"):
            continue
        points = [point(look) for look in profiles["common"]]
        intervals = []
        for row in points:
            frame = levels.setdefault(row["look"], {})
            if row["verified"]:
                frame[method] = {row["reference"]: 1}
        for a, b in zip(points, points[1:]):
            interval = dict(start=a["look"], end=b["look"], change=None, rates=rates(None, None, None),
                            sampling={}, useful_tokens=None, charged_us=None)
            slopes = rate_terms.setdefault(b["look"], {"pp_per_million": {}, "pp_per_hour": {}})
            if a["verified"] and b["verified"]:
                for name in ("useful_tokens", "charged_tokens", "discarded_tokens", "updates", "charged_us"):
                    if a[name] is not None and b[name] is not None and b[name] < a[name]:
                        raise ValueError("cumulative curve cost or exposure decreased between cuts")
                tokens = b["useful_tokens"] - a["useful_tokens"]
                elapsed = None if a["charged_us"] is None or b["charged_us"] is None else b["charged_us"] - a["charged_us"]
                terms = {a["reference"]: -1, b["reference"]: 1}
                change = estimate(terms)
                interval.update(change=change, rates=rates(change["observed"], tokens, elapsed),
                                useful_tokens=tokens, charged_us=elapsed)
                for axis, scale in rates(1, tokens, elapsed).items():
                    if scale is not None:
                        slopes[axis][method] = {ref: w * scale for ref, w in terms.items()}
                        interval["sampling"][axis] = {
                            k: None if change["sampling"][k] is None else change["sampling"][k] * scale
                            for k in ("lower", "upper")
                        }
            intervals.append(interval)
        curves[method] = dict(looks=points, intervals=intervals)

    comparisons = {}
    for axis, requested in (budgets or {}).items():
        if axis not in {"useful_tokens", "charged_us"}:
            raise ValueError("budget axis requires useful tokens or charged workstation time")
        eligible = {}
        for method, curve in curves.items():
            known = [r for r in curve["looks"] if r["verified"] and r.get(axis) is not None]
            if any(r["look"] > 0 for r in known):
                eligible[method] = known
        overlap = None
        if len(eligible) >= 2:
            lo = max(min(r[axis] for r in group) for group in eligible.values())
            hi = min(max(r[axis] for r in group) for group in eligible.values())
            if lo <= hi:
                overlap = [lo, hi]
        comparisons[axis] = []
        for budget in requested:
            if type(budget) not in (int, float) or not isfinite(budget) or budget < 0:
                raise ValueError("finite nonnegative comparison budget required")
            selected = {
                method: affordable_look(known, axis, budget, overlap=overlap) if overlap else None
                for method, known in eligible.items()
            }
            comparisons[axis].append(dict(
                budget=budget, overlap=overlap, methods=sorted(eligible),
                looks={method: row["look"] if row else None for method, row in selected.items()},
                points={method: row for method, row in selected.items() if row is not None},
            ))
    return dict(stratum=stratum, curves=curves, endpoints=endpoints, budgets=comparisons,
                effects={str(k): _factor_effects(v, estimate) for k, v in levels.items()},
                rate_effects={str(k): {axis: _factor_effects(v, estimate) for axis, v in axes.items()}
                              for k, axes in rate_terms.items()})
