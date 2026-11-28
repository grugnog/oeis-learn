"""Publish and independently reduce anonymous joint family counts for the exploration wave."""

from collections import defaultdict
import os
import re

from oeis_learn.generalization.progressive import velocity_donation_report as public
from oeis_learn.generalization.progressive import velocity_metrics as metrics
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.program_search_costing import workstation_accounting
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_report import public_family_cells
from . import progressive_velocity_trial_preparation as preparation
from .identity import confined
from .preparation_recovery import retain

ARCHIVE = public.REPORT
MEMBER = "comparative-exploration.json"
VERSION = "round029-comparative-report/v1"
EXPOSURE = (
    "useful_tokens charged_tokens discarded_tokens updates last_update_tokens overshoot".split()
)
WORK = "origin receipt role beneficiaries state host boot start_us end_us charged_us worker_cpu_us lease_us covered_us busy_us"


def sufficient(accepted, measured, key):
    expected = {r["roster"] for r in accepted["schedule"]}
    if set(measured) != expected:
        raise ValueError("every supported opportunity needs its verified score")
    costs, points, panels = {}, {}, {}
    for row in accepted["schedule"]:
        ref = row["roster"]
        m = measured[ref]
        if m != seal(m) or m["probe"]["look"]["identity"] != ref:
            raise ValueError("measurement differs from its accepted look")
        work = []
        for span in m["work"]:
            public.closed(span, WORK)
            if costs.setdefault(span["receipt"], span) != span:
                raise ValueError("conflicting shared role accounting")
            work.append(span["receipt"])
        points[ref] = dict(
            score=m["scores"]["identity"],
            measurement=m["identity"],
            work=work,
            cut={k: m["cut"][k] for k in EXPOSURE} if m["cut"] is not None else None,
        )
    for profile in ("common", "full"):
        subset = {ref: m for ref, m in measured.items() if m["probe"]["look"]["profile"] == profile}
        names = sorted(subset)
        probes = {ref: subset[ref]["probe"] for ref in names}
        rows = metrics._score_rows(subset)
        families, coordinates = metrics._frozen_rows(rows, probes, names, ())
        cells = []
        for s in (*STRATA, "retention"):
            result = metrics.family_contrast(families, coordinates, dict.fromkeys(names, 1), s)
            cells.extend(public_family_cells(result, key=key)["families"])
        panels[profile] = dict(rosters=names, families=cells)
    return dict(points=points, costs=costs, panels=panels)


def expand(panel, refs, size):
    public.closed(panel, "families rosters")
    if panel["rosters"] != refs:
        raise ValueError("public comparison coordinates differ from the pre-score roster")
    rows = {}
    for family in panel["families"]:
        public.closed(family, "stratum family frozen_size patterns")
        s, label = family["stratum"], family["family"]
        if (
            s not in (*STRATA, "retention")
            or not re.fullmatch("[a-f0-9]{64}", label)
            or (s, label) in rows
        ):
            raise ValueError("invalid anonymous family or stratum")
        entries = []
        for part in family["patterns"]:
            public.closed(part, "count outcomes")
            count = integer(part["count"], 1)
            if len(part["outcomes"]) != len(refs) or len(entries) + count > size:
                raise ValueError("invalid public joint pattern")
            groups = {}
            for o in part["outcomes"]:
                public.closed(o, "status complete closed equality")
                eq = integer(o["equality"])
                val = o["status"], o["complete"], o["closed"]
                if (
                    eq > len(groups)
                    or groups.setdefault(eq, val) != val
                    or val[0] not in ("solve", "miss", "unresolved")
                    or any(type(v) is not bool for v in val[1:])
                ):
                    raise ValueError("invalid public outcome equality or closure")
            for _ in range(count):
                i = len(entries)
                entries.append(
                    dict(
                        observations={
                            ref: dict(o, receipt=dict(identity=(s, label, i, o["equality"])))
                            for ref, o in zip(refs, part["outcomes"], strict=True)
                        }
                    )
                )
        if len(entries) != integer(family["frozen_size"], 1):
            raise ValueError("joint counts differ from the frozen family denominator")
        rows[s, label] = entries
    if sum(map(len, rows.values())) != size:
        raise ValueError("public panel denominator differs from pre-score commitment")
    return rows


def reduce(accepted, data, *, draws=10_000):
    public.closed(data, "points costs panels")
    points, costs = data["points"], data["costs"]
    graph, measurements, indexes, coords = {}, {}, {}, {}
    scheduled = {r["roster"]: r for r in accepted["schedule"]}
    if points.keys() != scheduled.keys() or set(data["panels"]) != {"common", "full"}:
        raise ValueError("public report lacks a registered opportunity")
    used = set()
    for k, v in costs.items():
        public.closed(v, WORK)
        public.hashes(k, v["host"])
        if (
            v["receipt"] != k
            or v["origin"] != "incremental"
            or v["role"] not in {"train", "search", "score"}
            or not set(v["beneficiaries"])
            <= set(accepted["trials"]) | {"D", "template", "polynomial", "linear_recurrence"}
            or not re.fullmatch("[0-9a-fA-F-]{36}", v["boot"])
        ):
            raise ValueError("invalid public workstation operation")
    workstation_accounting(costs.values())
    for ref, row in scheduled.items():
        m = points[ref]
        public.closed(m, "score measurement work cut")
        public.hashes(m["score"], m["measurement"])
        if len(set(m["work"])) != len(m["work"]) or not set(m["work"]) <= costs.keys():
            raise ValueError("missing or duplicate workstation receipt")
        used.update(m["work"])
        method, profile, target = row["method"], row["profile"], row["look"]
        cut = m["cut"]
        if method in accepted["supported"]:
            public.closed(cut, " ".join(EXPOSURE))
            for k in EXPOSURE:
                integer(cut[k])
            from oeis_learn.generalization.progressive.velocity_trials import crossed

            if (
                not crossed(cut, target)
                or cut["useful_tokens"] - target != cut["overshoot"]
                or cut["charged_tokens"] != cut["useful_tokens"] + cut["discarded_tokens"]
            ):
                raise ValueError("public whole-update crossing differs")
        elif cut is not None:
            raise ValueError("fixed comparator has a training trajectory")
        look = dict(identity=ref, method=method, look=target, profile=profile)
        graph.setdefault(method, {}).setdefault(profile, []).append(look)
        coords[ref] = dict(
            registration=accepted["registration"],
            method=method,
            look=target,
            profile=profile,
            seed=accepted["seed"],
            input_mode="true",
        )
        measurements[ref] = dict(
            probe=dict(look=look, mode="true"), cut=cut, work=[costs[k] for k in m["work"]]
        )
    if used != costs.keys():
        raise ValueError("orphan public workstation operation")
    for method in accepted["omitted"]:
        graph[method] = {}
    for profile, panel in data["panels"].items():
        refs = sorted(k for k, r in scheduled.items() if r["profile"] == profile)
        sizes = {scheduled[k]["members"] for k in refs}
        if len(sizes) != 1:
            raise ValueError("unmatched public opportunities")
        indexes[profile] = expand(panel, refs, sizes.pop())

    def paired(terms, stratum):
        profiles = {scheduled[k]["profile"] for k in terms}
        if len(profiles) != 1:
            raise ValueError("mixed common and full opportunities")
        return metrics.family_contrast(
            indexes[profiles.pop()], [coords[k] for k in sorted(terms)], terms, stratum
        )

    curves = {
        s: metrics.study_curve(
            graph,
            measurements,
            stratum=s,
            seed=accepted["seed"],
            draws=draws,
            budgets=dict(
                useful_tokens=[1_000_000, 4_000_000],
                charged_us=sorted(
                    {
                        workstation_accounting(m["work"])["by_beneficiary"]
                        .get(m["probe"]["look"]["method"], {})
                        .get("production_us")
                        for m in measurements.values()
                    }
                    - {None}
                ),
            ),
            paired=paired,
        )
        for s in (*STRATA, "structural", "retention")
    }
    direct = {}
    for method in accepted["supported"]:
        arm = graph[method]["full"][0]["identity"]
        direct[method] = {}
        for comparator in ("D", "00"):
            if comparator == method:
                continue
            other = graph[comparator]["full"][0]["identity"]
            terms = {arm: 1, other: -1}

            def estimate(s):
                v = paired(terms, s)
                return {
                    **{
                        k: v[k]
                        for k in (
                            "observed_pp",
                            "lower_pp",
                            "upper_pp",
                            "denominator",
                            "gains",
                            "losses",
                            "both",
                            "neither",
                            "unresolved_overlay",
                        )
                    },
                    "sampling": metrics._sample_contrast(v, terms, s, accepted["seed"], draws),
                }

            direct[method][comparator] = {
                s: estimate(s) for s in ("oeis", "structural", "retention")
            }
    negative = all(
        direct[m]["D"][s]["upper_pp"] is not None and direct[m]["D"][s]["upper_pp"] <= 0
        for m in accepted["supported"]
        for s in ("oeis", "structural")
    )
    decision = dict(
        verdict="retain_D" if negative else "inconclusive",
        incumbent="D",
        assessment="sealed",
        confirmation="separate_grant_required",
        comparator="D",
        reason="no_positive_supported_upper_bound" if negative else "independent_evidence_required",
        excluded=accepted["omitted"],
    )
    return dict(curves=curves, direct=direct, decision=decision)


def publication(area, grant, accepted, graph, observed, *, readonly):
    key_path = confined(area, "family-key.json", exists=readonly)
    if not key_path.exists():
        retain(key_path, dict(key=os.urandom(32).hex()))
    data = sufficient(accepted, observed, bytes.fromhex(load(key_path)["key"]))
    value = seal(
        dict(
            version=VERSION,
            accepted=accepted["identity"],
            grant=grant["identity"],
            registration=accepted["registration"],
            statistics=data,
            **reduce(accepted, data),
        )
    )
    path = area / "share" / ARCHIVE
    if not readonly:
        public.install(path, MEMBER, value)
    if public.read_member(path, MEMBER) != value:
        raise ValueError("comparative publication differs from private role evidence")
    return value


def verify(path, qualification):
    q = public.read_member(qualification, preparation.MEMBER)
    if (
        q != seal(q)
        or q["version"] != preparation.VERSION
        or q["assessment"] != "sealed"
        or q["research_authority"] is not None
    ):
        raise ValueError("independent supported-wave commitment required")
    value = public.read_member(path, MEMBER)
    public.closed(
        value, "version accepted grant registration statistics curves direct decision", sealed=True
    )
    public.hashes(value["grant"])
    if (
        value["version"] != VERSION
        or value["accepted"] != q["identity"]
        or value["registration"] != q["registration"]
        or any(value[k] != v for k, v in reduce(q, value["statistics"]).items())
    ):
        raise ValueError("comparative decision or statistics differ from independent replay")
    return value
