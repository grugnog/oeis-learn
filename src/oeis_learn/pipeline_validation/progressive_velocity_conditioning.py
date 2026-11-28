"""The paid, outcome-blind development sample for the two model input modes."""

from itertools import combinations

from oeis_learn.generalization.progressive.cohorts import STRATA, validate_prompt
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.generalization.progressive.trials import integer
from .identity import confined

PATH = "comparative/conditioning-plan.json"
MODES = ("true", "shuffled")
STAGE = "paired_conditioning"


def paid_inputs(root, folder, source, accepted, registration):
    from .progressive_round029 import preflight
    from .progressive_velocity_wave_host import panel

    estimate = preflight(root, source)["inventory"]["prices"]["launch_estimate"]
    if (
        estimate["sources"] != accepted["sources"]
        or estimate["methods"] != accepted["supported"]
        # A new timed checkpoint read shifts expected and tail by the same amount.
        or estimate["tail_us"] - estimate["expected_us"] != accepted["tail_us"] - accepted["expected_us"]
        or estimate["followup_reserve_us"] != accepted["followup_us"]
    ):
        raise ValueError("conditioning quote belongs to another wave")
    price = dict(
        estimate["items"]["paired_search_selected_and_banks"],
        **estimate["conditioning_components"],
        split_overhead_us=estimate["conditioning_extra_role_us"],
        methods=estimate["methods"],
        paired=estimate["sources"]["paired"],
        bank=estimate["sources"]["bank"],
    )
    witnesses = panel(root, folder, registration)
    return price, {
        r["task"]: validate_prompt(
            dict(
                id=r["task"],
                indices=witnesses[r["task"]]["indices"][:20],
                values=witnesses[r["task"]]["values"][:20],
            )
        )
        for r in registration["development"]
    }


def funding(accepted, price, total_us):
    headroom = integer(total_us) - accepted["proposed_total_us"]
    if headroom < 0:
        raise ValueError("grant is below the frozen full-wave tail and protected follow-up")
    overhead = price["split_overhead_us"]
    if set(overhead) != {"ordinary-common", "ordinary-full", "D-full"}:
        raise ValueError("paired search and score supervisor prices required for both modes")
    # The paid proxy supervised both modes together. Protect its full tail and bank
    # allowance, then price a second search/score startup for each split-mode pair.
    remaining = headroom
    funded, omitted = [], {}
    for method in ["00", "D", *(m for m in accepted["supported"] if m != "00")]:
        for profile in (("full",) if method == "D" else ("common", "full")):
            group = method + "-" + profile
            extra = integer(overhead[("D" if method == "D" else "ordinary") + "-" + profile])
            if remaining >= extra:
                funded.append(group)
                remaining -= extra
            else:
                omitted[group] = dict(required_us=extra, available_us=remaining, deficit_us=extra - remaining)
    return dict(total_us=total_us, headroom_us=headroom, extra_supervision_us=headroom - remaining,
                funded=funded, omitted=omitted)


def freeze(accepted, registration, price, prompts, total_us):
    components = ("search_us", "selected_us", "bank_us", "overhead_us")
    if (
        price["methods"] != accepted["supported"]
        or price["paired"] != accepted["sources"]["paired"]
        or price["bank"] != accepted["sources"]["bank"]
        or sum(integer(price[k]) for k in components) != integer(price["expected_us"], 1)
        or not price["expected_us"] <= integer(price["tail_us"]) <= accepted["tail_us"]
    ):
        raise ValueError("paired search, selected scoring and bank prices do not reconcile")
    members, missing, donors = [], {}, {}
    for s in STRATA:
        roster = [r for r in registration["development"] if r["stratum"] == s]
        pair = next(
            (
                (a, b)
                for a, b in combinations(roster, 2)
                if a["group"] != b["group"]
                and prompts[a["task"]]["values"] != prompts[b["task"]]["values"]
            ),
            None,
        )
        if pair is None:
            missing[s] = "fewer_than_two_families_with_differing_prefixes"
            continue
        a, b = pair
        members.extend(pair)
        donors.update({a["task"]: b["task"], b["task"]: a["task"]})
    return seal(
        dict(
            version="round029-conditioning-sample/v1",
            accepted=accepted["identity"],
            members=members,
            donors=donors,
            missing=missing,
            price=price,
            funding=funding(accepted, price, total_us),
            prefixes=digest([prompts[r["task"]] for r in members]),
        )
    )


def original(root, folder, source, accepted, registration, *, total_us=None):
    path = confined(folder, PATH, exists=False)
    if total_us is not None and not path.exists():
        price, prompts = paid_inputs(root, folder, source, accepted, registration)
        return freeze(accepted, registration, price, prompts, total_us)
    sample = load(path)
    if (sample != seal(sample) or sample["accepted"] != accepted["identity"]
            or (total_us is not None and sample["funding"]["total_us"] != total_us)):
        raise ValueError("original pre-score conditioning sample required")
    return sample


def opportunities(sample, registration, methods):
    if not sample["members"]:
        return
    for method in [*methods, "D"]:
        for profile in ("full",) if method == "D" else ("common", "full"):
            if method + "-" + profile not in sample["funding"]["funded"]:
                continue
            look = velocity_look(
                registration=registration["identity"],
                method=method,
                seed=registration["initialization"],
                look=4_000_000,
                profile=profile,
                development=sample["members"],
                retention=[],
            )
            for mode in MODES:
                ref = digest([sample["identity"], look["identity"], mode])
                yield ref, look, mode


def check_prompts(sample, look, prompts, donors):
    if (
        look["members"] != sample["members"]
        or digest(prompts) != sample["prefixes"]
        or donors != sample["donors"]
    ):
        raise ValueError(
            "conditioning recipients or within-stratum donors changed after authorization"
        )


def statistics(scope, measurements, key, costs):
    from oeis_learn.generalization.progressive import velocity_metrics as metrics
    from oeis_learn.generalization.progressive.velocity_report import public_family_cells

    panels, points = {}, {}
    for group, modes in scope["groups"].items():
        names = sorted(modes.values())
        items = {ref: measurements[ref] for ref in names}
        for mode, ref in modes.items():
            m = items[ref]
            probe = m["probe"]
            look = probe["look"]
            if (
                m != seal(m)
                or probe["mode"] != mode
                or look["method"] + "-" + look["profile"] != group
                or ref != digest([scope["sample"], look["identity"], mode])
            ):
                raise ValueError("conditioning measurement differs from its frozen coordinate")
            work = []
            for span in m["work"]:
                if costs.setdefault(span["receipt"], span) != span:
                    raise ValueError("conflicting conditioning role debit")
                work.append(span["receipt"])
            points[ref] = dict(
                model=digest(probe["model"]),
                score=m["scores"]["identity"],
                choices=m["scores"]["candidates"],
                work=work,
            )
        families, coords = metrics._frozen_rows(
            metrics._score_rows(items), {r: items[r]["probe"] for r in names}, names, ()
        )
        cells = []
        for s in STRATA:
            result = metrics.family_contrast(families, coords, dict.fromkeys(names, 1), s)
            cells.extend(public_family_cells(result, key=key)["families"])
        panels[group] = dict(rosters=names, families=cells)
    return dict(scope=scope, points=points, panels=panels)


def reduce_selected(data, costs, primary, schedule):
    from oeis_learn.generalization.progressive import velocity_donation_report as public
    from oeis_learn.generalization.progressive.velocity_metrics import family_contrast
    from .progressive_velocity_wave_report import expand

    public.closed(data, "scope points panels")
    scope, points, panels = data["scope"], data["points"], data["panels"]
    groups = scope["groups"]
    if panels.keys() != groups.keys() or set(points) != {
        r for modes in groups.values() for r in modes.values()
    }:
        raise ValueError("incomplete matched conditioning opportunities")
    spent, contrasts, used = 0, {}, set()
    barrier = {}
    for group, modes in groups.items():
        method, profile = group.split("-")
        if set(modes) != set(MODES) or modes["true"] == modes["shuffled"]:
            raise ValueError("both matched conditioning modes required")
        panel = panels[group]
        families = sorted(
            (
                {k: row[k] for k in ("stratum", "family", "frozen_size")}
                for row in panel["families"]
            ),
            key=lambda r: (r["stratum"], r["family"]),
        )
        if families != scope["families"]:
            raise ValueError("paired sample differs from the pre-score family commitment")
        rows = expand(panel, sorted(modes.values()), len(families))
        terms = {modes["true"]: 1, modes["shuffled"]: -1}
        contrasts[group] = {}
        for s in STRATA:
            result = family_contrast(rows, [], terms, s)
            contrasts[group][s] = {
                k: result[k]
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
            }
        models = set()
        for ref in modes.values():
            p = points[ref]
            public.closed(p, "model score choices work")
            public.hashes(p["model"], p["score"], p["choices"])
            models.add(p["model"])
            barrier[ref] = p["choices"]
            work = p["work"]
            if (
                used.intersection(work)
                or len(work) != len(set(work))
                or not set(work) <= costs.keys()
                or {costs[k]["role"] for k in work} != {"search", "score"}
                or any(costs[k]["beneficiaries"] != [method] for k in work)
            ):
                raise ValueError("unmatched conditioning dispatch charges")
            used.update(work)
            spent += sum(costs[k]["charged_us"] for k in work)
        parent = next(
            row["roster"]
            for row in schedule
            if row["method"] == method and row["profile"] == profile and row["look"] == 4_000_000
        )
        if len(models) != 1 or primary[parent]["model"] not in models:
            raise ValueError("conditioning modes require the same registered endpoint model")
    return dict(
        selected=contrasts,
        missing=scope["missing"],
        quote=scope["price"],
        funding=scope["funding"],
        charged_search_and_selected_us=spent,
        bank_status="deferred_to_T040",
        bank_dispatches=0,
        choices_barrier=(digest(dict(plan=scope["sample"], choices=barrier)) if groups else None),
    ), used
