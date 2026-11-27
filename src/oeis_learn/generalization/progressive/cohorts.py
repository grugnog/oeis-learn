"""Frozen outcome-blind census, group separation and private/public publication."""

import json
from .contracts import validate, verify_refs, atomic_write, digest
from .exposure import exposed, groups as related_groups
from oeis_learn.pipeline_validation.identity import seal

STRATA = ("parameter", "composition", "unseen_structure", "oeis")
DEFAULT_PROFILE = dict(
    id="bigint-20-100/v1",
    visible=20,
    total=100,
    attempts=8,
    portfolio_us=60_000_000,
    fit_us=4_000_000,
    token_cap=1024,
    schedule="drain",
)


def validate_profile(profile):
    if profile != DEFAULT_PROFILE or any(
        type(profile[k]) is not type(v) for k, v in DEFAULT_PROFILE.items()
    ):
        raise ValueError("fixed base evaluation profile required")
    return profile


def _integer(value):
    from oeis_learn.synthesis.contracts import canonical_integer

    return canonical_integer(value)


def validate_prompt(prompt):
    if (
        not isinstance(prompt, dict)
        or set(prompt) != {"id", "indices", "values"}
        or not isinstance(prompt["id"], str)
        or not prompt["id"]
        or len(prompt["id"]) > 1024
    ):
        raise ValueError("closed visible-only prompt")
    if (
        not isinstance(prompt["indices"], list)
        or not isinstance(prompt["values"], list)
        or len(prompt["indices"]) != 20
        or len(prompt["values"]) != 20
    ):
        raise ValueError("20 visible terms required")
    indices = [_integer(x) for x in prompt["indices"]]
    for v in prompt["values"]:
        _integer(v)
    if indices != list(range(indices[0], indices[0] + 20)):
        raise ValueError("exact consecutive source indices required")
    return prompt


def validate_cohort(cohort, *, exposure, other_cohorts=()):
    c = validate("cohort", cohort)
    p = c["payload"]
    e = validate("exposure", exposure)
    validate_profile(p["profile"])
    if p["exposure"] != e["identity"]:
        raise ValueError("exposure cut differs")
    if len({r["id"] for r in p["rows"]}) != len(p["rows"]):
        raise ValueError("duplicate cohort task")
    if any(r["stratum"] not in p["strata"] for r in p["rows"]):
        raise ValueError("undeclared stratum")
    if p["role"] == "training" and set(p["strata"]) - {"reward", "retention"}:
        raise ValueError("evaluation tasks cannot be training")
    if p["role"] != "training" and "reward" in p["strata"]:
        raise ValueError("reward probes are training-only")
    for r in p["rows"]:
        for group in r["groups"]:
            if p["role"] in {"confirmation", "assessment"} and exposed(
                e,
                group,
                endpoint=r["stratum"],
                role=p["role"],
                structure_sharing=p["group_policy"] == "parameter_structure_sharing",
            ):
                raise ValueError("exposed group cannot become fresh")
    comparisons = [validate("cohort", other) for other in other_cohorts]
    for other in comparisons:
        q = other["payload"]
        if q["panel"] == p["panel"] and other["identity"] != c["identity"]:
            raise ValueError("panel name conflicts")
        if (
            p["role"] == "confirmation"
            and q["role"] == "confirmation"
            and p["decision"] != q["decision"]
            and p["panel"] == q["panel"]
        ):
            raise ValueError("independent decisions need distinct panels")
        if q["role"] != p["role"] or q["panel"] != p["panel"]:
            for a in p["rows"]:
                for b in q["rows"]:
                    if _linked(a, b, e, p["group_policy"]):
                        raise ValueError("cross-panel related group leakage")
    return c


def _linked(a, b, exposure, policy):
    same = set(a["groups"]) & set(b["groups"])
    if same:
        return True
    if any(
        related_groups(exposure["payload"]["events"], g, endpoint=a["stratum"]) & set(b["groups"])
        for g in a["groups"]
    ):
        return True
    if a["behavior"] == b["behavior"]:
        return True
    if policy == "parameter_structure_sharing" and a["stratum"] in {"parameter","retention"} and b["stratum"] in {"parameter","retention"}:
        return False
    return a["structure"] == b["structure"] or (
        a["stratum"] == "composition" and a["composition"] == b["composition"]
    )


def census(cohort, *, exposure, other_cohorts=()):
    c = validate_cohort(cohort, exposure=exposure, other_cohorts=other_cohorts)
    p = c["payload"]
    from .complexity import summarize
    difficulty = {}
    strata = {}
    for stratum in p["strata"]:
        rows = [r for r in p["rows"] if r["stratum"] == stratum]
        difficulty[stratum] = summarize(rows)
        # Connected related rows count as one independent group, not one/group key.
        clusters = []
        for row in rows:
            overlaps = [
                i
                for i, cluster in enumerate(clusters)
                if any(_linked(row, old, exposure, p["group_policy"]) for old in cluster)
            ]
            combined = [row]
            for index in reversed(overlaps):
                combined.extend(clusters.pop(index))
            clusters.append(combined)
        strata[stratum] = dict(
            tasks=len(rows),
            groups=len(clusters),
            task_deficit=max(0, p["required_per_stratum"] - len(rows)),
            group_deficit=max(0, p["minimum_groups"] - len(clusters)),
        )
    complete = all(v["task_deficit"] == 0 and v["group_deficit"] == 0 for v in strata.values())
    return seal(
        dict(
            version="discovery-census/v1",
            cohort=c["identity"],
            status="COMPLETE" if complete else "INCOMPLETE",
            strata=strata,
            difficulty=difficulty,
            selection="fixed_roster_no_reseed",
            detector_limits=p["detector_limits"],
        )
    )


def publish(root, output, cohort, *, exposure, other_cohorts=()):
    c = validate_cohort(cohort, exposure=exposure, other_cohorts=other_cohorts)
    from .indexed_history import open_indexes, check_panel
    with open_indexes(root,[exposure]) as indexes:
        check_panel(root,c,indexes)
    return _publish_validated(root,output,c,exposure=exposure,other_cohorts=other_cohorts)


def publish_many(root, publications, *, exposure):
    """Check the shared original index once for an outcome-blind panel roster.

    This is a custodian batch operation, not a way to supply unchecked indexes.
    All cross-panel relations are checked before publishing the first channel.
    """
    from .indexed_history import open_indexes, check_panel
    panels=[c for _,c in publications]
    if len({str(path) for path,_ in publications})!=len(publications):
        raise ValueError('distinct cohort publication destinations required')
    with open_indexes(root,[exposure]) as indexes:
        for panel in panels:
            validate_cohort(panel,exposure=exposure,other_cohorts=panels)
            check_panel(root,panel,indexes)
    return [_publish_validated(root,path,c,exposure=exposure,other_cohorts=panels)
            for path,c in publications]


def _publish_validated(root, output, c, *, exposure, other_cohorts):
    result = census(c, exposure=exposure, other_cohorts=other_cohorts)
    from oeis_learn.pipeline_validation.identity import confined

    target = confined(root, output, exists=False)
    if target.exists():
        raise ValueError("fresh cohort publication required")
    rows = c["payload"]["rows"]
    refs = [ref for r in rows for ref in (r["visible"], r["truth"])]
    blobs = verify_refs(root, refs, max_bytes=512 << 20)
    for index, row in enumerate(rows):
        prompt = json.loads(blobs[2 * index])
        truth = json.loads(blobs[2 * index + 1])
        validate_prompt(prompt)
        if (
            set(truth) != {"id", "indices", "values"}
            or truth["id"] != row["id"]
            or prompt["id"] != row["id"]
        ):
            raise ValueError("truth identity mismatch")
        expected = [str(i) for i in range(_integer(row["offset"]), _integer(row["offset"]) + 100)]
        if (
            truth["indices"] != expected
            or len(truth["values"]) != 100
            or prompt["indices"] != expected[:20]
            or prompt["values"] != truth["values"][:20]
        ):
            raise ValueError("no silent horizon/source offset change")
        for v in truth["values"]:
            _integer(v)
    target.mkdir(parents=True)
    atomic_write(target / "census.json", result)
    # A deficit publishes availability only; it cannot produce a scoreable roster.
    if result["status"] != "COMPLETE":
        return result
    for index, row in enumerate(rows):
        public = seal(
            dict(
                version="discovery-channel/v1",
                kind="visible_prompt",
                role="generator",
                data=json.loads(blobs[2 * index]),
            )
        )
        private = seal(
            dict(
                version="discovery-channel/v1",
                kind=c["payload"]["role"] + "_truth",
                role="assessment" if c["payload"]["role"] == "assessment" else "scorer",
                data=json.loads(blobs[2 * index + 1]),
            )
        )
        name = digest(row["id"]).split(":")[1] + ".json"
        atomic_write(target / "public" / name, public)
        atomic_write(target / "private" / name, private)
    (target / "private").chmod(0o700)
    atomic_write(target / "cohort.json", c)
    return result


def validate_frozen_panels(root, panels, exposures):
    """Validate original cuts and detectable relations before any scoring.

    Detector augmentation is private and ephemeral: it never rewrites the frozen
    producer. All supplied history is retained, including access after a panel's
    original census. Independent/forked registries cannot erase one another.
    """
    from .exposure import detect, validate_extension

    from .indexed_history import open_indexes, check_panel
    with open_indexes(root,exposures.values()) as indexes:
        for panel in panels:
            check_panel(root,panel,indexes)

    for exposure in exposures.values():
        previous = exposure["payload"]["previous"]
        if previous is not None:
            if previous not in exposures:
                raise ValueError("missing original exposure cut ancestry")
            validate_extension(exposures[previous], exposure)
    augmented = []
    for panel in panels:
        p = panel["payload"]
        if p["exposure"] not in exposures:
            raise ValueError("missing original exposure cut")
        validate_cohort(panel, exposure=exposures[p["exposure"]], other_cohorts=panels)
        rows = []
        for row in p["rows"]:
            prompt, truth = [
                json.loads(raw)
                for raw in verify_refs(root, [row["visible"], row["truth"]], max_bytes=512 << 20)
            ]
            validate_prompt(prompt)
            expected = [
                str(i) for i in range(_integer(row["offset"]), _integer(row["offset"]) + 100)
            ]
            if (
                set(truth) != {"id", "indices", "values"}
                or truth["id"] != row["id"]
                or prompt["id"] != row["id"]
                or truth["indices"] != expected
                or len(truth["values"]) != 100
                or prompt["indices"] != expected[:20]
                or prompt["values"] != truth["values"][:20]
            ):
                raise ValueError("frozen cohort source horizon/identity differs")
            detected = detect(
                [
                    dict(
                        truth,
                        structure=row["structure"],
                        composition=row["composition"],
                        relations=[row["source"] + ":" + row["id"]],
                    )
                ]
            )[0]["groups"]
            # Parameter sharing permits structure alone, never identical/shifted
            # finite values, source ancestry or declared relation groups.
            if p["group_policy"] == "parameter_structure_sharing" and row["stratum"] in {"parameter","retention"}:
                detected = [g for g in detected if not g.startswith(("structure:", "composition:"))]
            effective = dict(row, groups=sorted(set(row["groups"]) | set(detected)))
            if p["role"] in {"confirmation", "assessment"} and any(
                exposed(
                    e,
                    g,
                    endpoint=row["stratum"],
                    role=p["role"],
                    structure_sharing=p["group_policy"] == "parameter_structure_sharing",
                )
                for e in exposures.values()
                for g in effective["groups"]
            ):
                raise ValueError("exposed group cannot become fresh")
            rows.append(effective)
        augmented.append(seal(dict(panel, payload=dict(p, rows=rows))))
    for panel in augmented:
        p = panel["payload"]
        for other in augmented:
            q = other["payload"]
            if p["role"] != q["role"] or p["panel"] != q["panel"]:
                if any(
                    _linked(a, b, e, p["group_policy"])
                    for a in p["rows"]
                    for b in q["rows"]
                    for e in exposures.values()
                ):
                    raise ValueError("cross-panel detectable related group leakage")
