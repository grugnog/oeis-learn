"""Publish and independently reduce anonymous joint family counts for the exploration wave."""

import json
import re

from oeis_learn.generalization.progressive import velocity_donation_report as public
from oeis_learn.generalization.progressive import velocity_metrics as metrics
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import _atomic_write, digest, load, seal
from oeis_learn.generalization.progressive.budgets import KEYS, vector
from oeis_learn.generalization.progressive.program_search_outcome import (
    archive_bytes,
    read_canonical,
)
from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode
from oeis_learn.generalization.progressive.velocity_records import grant_terms, validate_grant
from .identity import confined
from oeis_learn.generalization.progressive.program_search_costing import workstation_accounting
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_report import public_family_cells
from . import progressive_velocity_trial_preparation as preparation

ARCHIVE = public.REPORT
MEMBER = "comparative-exploration.json"
ACCOUNT = "comparative-accounting.json"
SCIENCE = "comparative-observations.zip"
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
    shortlist = []
    for method in accepted["supported"]:
        if method == "00":
            continue
        contrasts = direct[method]["00"]
        direct_oeis, structural = contrasts["oeis"], contrasts["structural"]
        expected = direct_oeis["observed_pp"]
        if (
            expected is None
            or max(expected, structural["observed_pp"] or 0, direct_oeis["upper_pp"] or 0) <= 0
        ):
            continue
        arm = graph[method]["full"][0]["identity"]
        ordinary = graph["00"]["full"][0]["identity"]
        terms = {arm: 1, ordinary: -1}
        groups = indexes["full"]
        leave_one = []
        for group in {label for s, label in groups if s == "oeis"}:
            subset = {k: v for k, v in groups.items() if k[1] != group}
            change = metrics.family_contrast(subset, [], terms, "oeis")["observed_pp"]
            if change is not None:
                leave_one.append(change)
        sensitive = (
            direct_oeis["lower_pp"] <= 0 < direct_oeis["upper_pp"]
            or bool(leave_one)
            and min(leave_one) <= 0 < max(leave_one)
        )
        followup = (
            "resolve_missing"
            if sensitive
            else "oeis_measurement" if expected <= 0 else "confirmation"
        )
        shortlist.append(
            dict(
                method=method,
                comparator="00",
                direct_oeis_pp=expected,
                structural_pp=structural["observed_pp"],
                leave_one_family_out=[min(leave_one), max(leave_one)] if leave_one else None,
                proposed_branch=followup,
                marginal_us=(accepted["followup_us"] if followup == "oeis_measurement" else None),
            )
        )
    shortlist.sort(
        key=lambda r: (
            -r["direct_oeis_pp"],
            r["marginal_us"] is None,
            r["marginal_us"] or 0,
            r["method"],
        )
    )
    decision.update(
        shortlist=shortlist,
        protected_followup_us=accepted["followup_us"],
        spending_action="review_unfunded_branch" if shortlist else "retain_incumbent",
        next_dispatch=None,
    )
    policy = velocity_policy()
    missing = {
        m: [
            dict(profile=p, look=t, disposition="excluded_before_scoring", reason_ref=reason)
            for p in ("common", "full")
            for t in policy["profiles"][p]["looks"]
        ]
        for m, reason in accepted["omitted"].items()
    }
    decision["missing_looks"] = missing
    return dict(curves=curves, direct=direct, decision=decision)


def publication(area, grant, accepted, observed, *, marker, key, readonly):
    data = sufficient(accepted, observed, bytes.fromhex(key))
    value = seal(
        dict(
            version=VERSION,
            accepted=accepted["identity"],
            grant=grant["identity"],
            registration=accepted["registration"],
            controller=marker,
            statistics=data,
            **reduce(accepted, data),
        )
    )
    path = area / "share" / SCIENCE
    if not readonly:
        public.install(path, MEMBER, value)
    if public.read_member(path, MEMBER) != value:
        raise ValueError("comparative publication differs from private role evidence")
    return value


def checked_science(value, qualification):
    q = public.read_member(qualification, preparation.MEMBER)
    if (
        q != seal(q)
        or q["version"] != preparation.VERSION
        or q["assessment"] != "sealed"
        or q["research_authority"] is not None
    ):
        raise ValueError("independent supported-wave commitment required")
    public.closed(
        value,
        "version accepted grant registration controller statistics curves direct decision",
        sealed=True,
    )
    public.hashes(value["grant"])
    pinned = value["controller"]
    public.closed(
        pinned, "version accepted grant controller driver family_labels operations", sealed=True
    )
    public.hashes(pinned["driver"], pinned["family_labels"], pinned["operations"])
    if (
        pinned["version"] != "round029-comparative-wave/v1"
        or pinned["grant"] != value["grant"]
        or pinned["accepted"] != q["identity"]
        or not re.fullmatch("[a-f0-9]{40}", pinned["controller"])
    ):
        raise ValueError("comparative controller differs from the accepted wave")
    if (
        value["version"] != VERSION
        or value["accepted"] != q["identity"]
        or value["registration"] != q["registration"]
        or any(value[k] != v for k, v in reduce(q, value["statistics"]).items())
    ):
        raise ValueError("comparative decision or statistics differ from independent replay")
    return value


def accounting(area, grant, report, balance):
    if balance["pending"]:
        raise ValueError("unsettled comparative work")
    dispatches = []
    for key in sorted(balance["committed"]):
        logical, verb, ordinal = key.split("/")
        if verb != "dispatch":
            raise ValueError("unexpected comparative journal operation")
        block = confined(area, "operations/" + logical[7:])
        registered = load(confined(block, "input.json"))
        window = balance["windows"][key]
        work = dict(balance["reserved_work"][key], elapsed_us=window["charged_us"])
        dispatches.append(
            dict(
                receipt=digest([grant["identity"], key]),
                role=registered["coordinates"]["role"],
                attempt=int(ordinal),
                work=work,
                terminated=window["end_us"] is not None,
            )
        )
    return seal(
        dict(
            version="round029-comparative-accounting/v1",
            grant=grant,
            report=report,
            dispatches=dispatches,
            consumed=balance["consumed"],
            remaining=balance["remaining"],
        )
    )


def check_account(account, science, accepted):
    public.closed(account, "version grant report dispatches consumed remaining", sealed=True)
    grant = account["grant"]
    pinned = science["controller"]
    if grant.get("controller_ref") != digest(
        [
            pinned["version"],
            accepted["identity"],
            pinned["controller"],
            pinned["driver"],
            pinned["family_labels"],
            pinned["operations"],
        ]
    ):
        raise ValueError(
            "controller or frozen statistical labels differ from the research authority"
        )
    authorization = seal(
        dict(version="round029-authorization/v1", terms=digest(grant_terms(grant)))
    )
    validate_grant(
        grant, authorization, root=accepted["owner"], parent=accepted["identity"], action="run"
    )
    if (
        account["version"] != "round029-comparative-accounting/v1"
        or account["report"] != science["identity"]
        or grant["identity"] != science["grant"]
        or grant["allowed_operations"] != ["run"]
        or grant["holdback_us"] != accepted["followup_us"]
        or account["remaining"]["elapsed_us"] < grant["holdback_us"]
    ):
        raise ValueError("comparative accounting or protected reserve differs")
    vector(account["consumed"])
    vector(account["remaining"])
    totals = dict.fromkeys(KEYS, 0)
    seen, stages, expected = set(), set(), science["statistics"]["costs"]
    for dispatch in account["dispatches"]:
        public.closed(dispatch, "receipt role attempt work terminated")
        receipt, role = dispatch["receipt"], dispatch["role"]
        public.hashes(receipt)
        integer(dispatch["attempt"])
        if (
            receipt in seen
            or role not in {"prepare", "train", "search", "score", "publish"}
            or type(dispatch["terminated"]) is not bool
        ):
            raise ValueError("duplicate or invalid comparative dispatch")
        seen.add(receipt)
        stages.add(role)
        work = vector(dispatch["work"])
        if role in {"train", "search", "score"} and (
            receipt not in expected
            or expected[receipt]["role"] != role
            or expected[receipt]["charged_us"] != work["elapsed_us"]
        ):
            raise ValueError("public look cost differs from settled grant debit")
        for k in KEYS:
            totals[k] += work[k]
    if (
        not expected.keys() <= seen
        or stages != {"prepare", "train", "search", "score", "publish"}
        or totals != account["consumed"]
        or any(totals[k] + account["remaining"][k] != grant["resources"][k] for k in KEYS)
    ):
        raise ValueError("private and public cumulative debits differ")
    return account


def deliver(area, grant, accepted, balance, *, readonly):
    science = public.read_member(confined(area, "share/" + SCIENCE), MEMBER)
    account = check_account(
        accounting(area, grant, science["identity"], balance), science, accepted
    )
    records = {MEMBER: encode(science), ACCOUNT: encode(account)}
    path = confined(area, "share/" + ARCHIVE, exists=readonly)
    if not readonly and not path.exists():
        _atomic_write(path, archive_bytes(records))
    if read_canonical(path, (MEMBER, ACCOUNT)) != records:
        raise ValueError(
            "installed comparative release differs from the paid science and final debit"
        )
    return account


def verify(path, qualification):
    records = read_canonical(path, (MEMBER, ACCOUNT))
    science, account = (json.loads(records[k]) for k in (MEMBER, ACCOUNT))
    if records != {MEMBER: encode(science), ACCOUNT: encode(account)}:
        raise ValueError("noncanonical comparative records")
    checked_science(science, qualification)
    check_account(account, science, public.read_member(qualification, preparation.MEMBER))
    return science
