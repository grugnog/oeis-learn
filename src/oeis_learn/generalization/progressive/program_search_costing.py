"""Price the requested generator looks using separate, original portfolio and scorer clocks."""

from math import ceil, isfinite

from oeis_learn.pipeline_validation.pilot_clocks import DRAIN, clock_record

from .cohorts import STRATA
from .contracts import seal

PROXIES = ("trial-05-1M", "retained_D")
CLOCK_FIELDS = (
    "policy",
    "prompt",
    "mode",
    "snapshot",
    "start",
    "scope_deadline",
    "finish",
    "expected",
    "started",
    "terminal",
)
MEASURES = (
    "cases",
    "timed",
    "closed",
    "scored",
    "portfolio_us",
    "open_portfolio_us",
    "open_overrun_us",
    "score_us",
    "observed_max_us",
)


def clock_cells(producers, members):
    """The incumbent's capability producer may contain additional cases and search modes."""
    cells = {s: dict.fromkeys(MEASURES, 0) for s in STRATA}
    for key, s in members.items():
        cells[s]["cases"] += 1
    seen = set()
    for gen, scores, gen_hash in producers:
        clock_summary = gen.get("clock")
        if scores["candidates"] != gen_hash or (
            clock_summary is not None and seal(clock_summary) != clock_summary
        ):
            raise ValueError("historical portfolio/scorer parent differs")
        clocks = {}
        original_clocks = (
            clock_summary["clocks"] if clock_summary and clock_summary["policy"] == DRAIN else []
        )
        for clock in original_clocks:
            key = clock["prompt"]
            if clock["mode"] != "correct" or key not in members:
                continue
            if (
                key in clocks
                or key not in gen["records"]
                or clock["snapshot"] != gen["model"]
                or clock["policy"] != DRAIN
                or clock_record(**{k: clock[k] for k in CLOCK_FIELDS}) != clock
            ):
                raise ValueError("historical model, prompt or clock changed")
            clocks[key] = clock
        for key in members.keys() & gen["records"].keys():
            if key in seen:
                raise ValueError("duplicate historical portfolio")
            seen.add(key)
            cell = cells[members[key]]
            clock = clocks.get(key)
            if clock:
                wall = ceil((clock["finish"] - clock["start"]) * 1e6)
                cell["timed"] += 1
                cell["closed"] += int(clock["cost_complete"])
                cell["portfolio_us"] += wall
                cell["observed_max_us"] = max(cell["observed_max_us"], wall)
                if not clock["cost_complete"]:
                    cell["open_portfolio_us"] += wall
                    cell["open_overrun_us"] += max(0, wall - 60_000_000)
            outcome = scores["outcomes"].get(key, {}).get("equal_budget", {}).get("scoring", {})
            seconds = outcome.get("elapsed_seconds")
            if seconds is not None:
                if type(seconds) not in (int, float) or not isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid original finite scorer clock")
                cell["scored"] += 1
                cell["score_us"] += ceil(seconds * 1e6)
    return cells


def forecast(cells, training, probes):
    """Full search rates for endpoints; conservatively use the full rate for the one-attempt probes."""
    rates = {}
    for proxy in PROXIES:
        rates[proxy] = {}
        for s in STRATA:
            c = cells[proxy][s]
            if (
                set(c) != set(MEASURES)
                or any(type(n) is not int or n < 0 for n in c.values())
                or not 0 <= c["closed"] <= c["timed"] <= c["cases"] == 32
                or c["scored"] > c["cases"]
                or c["open_portfolio_us"] > c["portfolio_us"]
                or c["open_overrun_us"] > c["open_portfolio_us"]
                or c["observed_max_us"] > c["portfolio_us"]
            ):
                raise ValueError("invalid historical clock denominator or totals")
            # Reserve at least the full registered 60s for each open or absent portfolio.
            open_cost = (c["cases"] - c["closed"]) * 60_000_000 + c["open_overrun_us"]
            rates[proxy][s] = (
                (c["portfolio_us"] - c["open_portfolio_us"] + open_cost + c["score_us"] + 31) // 32
                if c["scored"] == 32
                else None
            )
    if set(probes) != set(STRATA) or any(
        type(n) is not int or not 0 <= n <= 32 for n in probes.values()
    ):
        raise ValueError("descriptive look requires the reserved family count for each stratum")
    accounting = {
        "training_us": training["combined_training_us"],
        "shift_us": training["shift_us"],
        "reserve_us": training["evaluation_replication_reserve_us"],
        "rates_us": rates,
    }
    if any(n is None for v in rates.values() for n in v.values()):
        return dict(accounting, status="ORIGINAL_CLOCKS_INCOMPLETE")

    # Retention uses the same familiar parameter-transfer domain. The learned-arm
    # rate is the historical trial 05 checkpoint; D gets its own historical rate.
    def full(proxy):
        return 32 * (sum(rates[proxy].values()) + rates[proxy]["parameter"])

    learner = full(PROXIES[0])
    incumbent = full(PROXIES[1])
    single_look = sum(probes[s] * rates[PROXIES[0]][s] for s in STRATA)
    # Reserve both arms at both descriptive looks until an identical zero-look can actually be reused.
    pilot = training["combined_training_us"] + 4 * single_look + 2 * learner + incumbent
    repeat = training["combined_training_us"] + 4 * learner + incumbent
    return dict(
        accounting,
        status="HISTORICAL_ELAPSED_PROXY",
        endpoint_us=2 * learner + incumbent,
        descriptive_us=4 * single_look,
        pilot_us=pilot,
        conditional_replication_us=repeat,
        pilot_excess_us=max(0, pilot - training["shift_us"]),
        with_replication_excess_us=max(0, pilot + repeat - training["shift_us"]),
    )


def price_wave(plan, prices, *, remaining_us=None, credited=None):
    """Price only outstanding occurrences; missing receipts never imply free work."""
    from oeis_learn.generalization.budgets import integer

    result = {}
    for scope, items in plan.items():
        expected = tail = 0
        missing = []
        for kind, count in items.items():
            credit = integer((credited or {}).get(scope, {}).get(kind, 0))
            outstanding = integer(count) - credit
            if outstanding < 0:
                raise ValueError("reuse exceeds registered work")
            if not outstanding:
                continue
            if kind not in prices:
                missing.append(kind)
                continue
            price = prices[kind]
            typical, slow = integer(price["expected_us"]), integer(price["tail_us"])
            if slow < typical:
                raise ValueError("tail price below expected price")
            expected += outstanding * typical
            tail += outstanding * slow
        result[scope] = dict(
            expected_us=None if missing else expected,
            tail_us=None if missing else tail,
            known_expected_us=expected,
            known_tail_us=tail,
            missing=sorted(missing),
        )
    followup = result["followup"]["tail_us"]
    for scope, row in result.items():
        row["affordable_with_reserve"] = None
        if row["tail_us"] is not None and remaining_us is not None and followup is not None:
            row["affordable_with_reserve"] = row["tail_us"] + (
                0 if scope == "followup" else followup
            ) <= integer(remaining_us)
    return result


def original_probe_costs(cuts, historical_looks, costs):
    """Keep historical case and lease costs in the look/profile that actually incurred them."""
    from oeis_learn.generalization.budgets import integer

    rows = []
    for look in historical_looks:
        target, cases = integer(look["look"]), integer(look["denominator"], 1)
        for method, arm in (
            ("00", "ordinary"),
            ("first_divergence", "first_divergence"),
            ("D", "incumbent"),
        ):
            if method == "D" and target != 4_000_000:
                continue
            block = f"{arm}-{target}"
            portions = {}
            for role in ("search", "score"):
                stage = [
                    r
                    for r in costs
                    if r["role"] == role
                    and (
                        r["block"] == block
                        or r["block"].startswith((block + "-resume-", block + "-scoreretry-"))
                    )
                ]
                if not stage:
                    portions[role] = dict(
                        charged_us=None,
                        lease_us=None,
                        reason="original role clock unavailable or shared",
                    )
                    continue
                if len({r["identity"] for r in stage}) != len(stage) or any(
                    r != seal(r) for r in stage
                ):
                    raise ValueError("duplicated or inconsistent original stage cost")
                paid = sum(integer(r["charged_us"]) for r in stage)
                portions[role] = dict(
                    charged_us=paid,
                    lease_us=sum(integer(r["lease_us"]) for r in stage),
                    average_per_case_us=(paid + cases - 1) // cases,
                    reason=None,
                )
            cut = next(
                c for c in cuts if c["method"] == method and (method == "D" or c["look"] == target)
            )
            rows.append(
                dict(
                    method=method,
                    look=target,
                    attempts=look["attempts"],
                    cases=cases,
                    training_to_look_us=cut.get("training_to_look_us"),
                    roles=portions,
                )
            )
    return rows


def exposed_timing_rates(timing, panels):
    """Split the original exposed per-case clocks from each cold role's host overhead."""
    from .program_search_timing import BLOCKS, prices

    if timing != seal(timing) or panels != seal(panels) or timing["panel"] != panels["identity"]:
        raise ValueError("exposed timing and its historical panel differ")
    rates = {}
    for name in BLOCKS:
        block = timing["blocks"][name]
        proxy = PROXIES[1 if name == "incumbent-full" else 0]
        prior = panels["historical_clocks"][proxy]
        floors = {
            s: (c["score_us"] + c["cases"] - 1) // c["cases"]
            for s, c in prior.items()
            if c["cases"] and c["scored"] == c["cases"]
        }
        prices(
            block, full=name.endswith("-full"), score_floors={s: floors.get(s, 0) for s in STRATA}
        )
        row = {"cases": {}, "overhead_us": {}}
        for phase, device in (("search", "gpu"), ("score", "cpu")):
            row["overhead_us"][phase] = max(
                0,
                block[device + "_wall_us"] - sum(c[phase + "_us"] for c in block["cells"].values()),
            )
        for s, c in block["cells"].items():
            n = c["cases"]
            row["cases"][s] = cell = dict(
                observations=n, closed=c["closed"], empty=n - c["selected"], executed=c["executed"]
            )
            for phase, complete in (
                ("search", c["closed"] == n),
                (
                    "score",
                    c["scored"] == n
                    and c["executed"] == c["selected"]
                    and (c["executed"] > 0 or s in floors),
                ),
            ):
                # Empty selections cannot establish the price of checking a nonempty program.
                floor = floors.get(s, 0) if phase == "score" else 0
                cell[phase] = (
                    dict(
                        expected_us=max(floor, (c[phase + "_us"] + n - 1) // n),
                        tail_us=max(floor, c["max_" + phase + "_us"]),
                    )
                    if complete
                    else None
                )
        rates[name] = row
    return dict(
        receipt=timing["identity"],
        panel=panels["identity"],
        candidate=timing["candidate"],
        image_acquisition_us=timing["image_build_us"],
        blocks=rates,
    )


def velocity_forecast(info, policy):
    """Price the distinct registered opportunities from exposed clocks, before funding a screen."""
    from collections import Counter

    history = info["prices"].get("exposed_028_timing")
    blocks = history["blocks"] if history else {}
    prices = {}
    plan = {scope: Counter() for scope in ("common_probe", "wave", "followup")}

    def portfolios(scope, profile, method, counts, repeats):
        # One cold search and scorer per resident block. Treatments use ordinary for the
        # typical cost and include the slower learned checkpoint in the observed tail.
        if not any(counts.values()):
            return
        names = ["incumbent-full"] if method == "D" else ["ordinary-" + profile]
        if method == "learner":
            names.append("first_divergence-" + profile)
        for phase in ("search", "score"):
            for stratum, count in {"startup": 1, **counts}.items():
                if not count:
                    continue
                kind = ".".join((method, profile, phase, stratum))
                plan[scope][kind] += repeats * count
                if all(name in blocks for name in names):
                    rates = []
                    for name in names:
                        block = blocks[name]
                        if stratum == "startup":
                            clock = block["overhead_us"][phase]
                            rates.append(dict(expected_us=clock, tail_us=clock))
                        else:
                            domain = "parameter" if stratum == "retention" else stratum
                            rates.append(block["cases"][domain][phase])
                    if all(rate is not None for rate in rates):
                        prices[kind] = dict(
                            expected_us=rates[0]["expected_us"],
                            tail_us=max(r["tail_us"] for r in rates),
                        )

    supply = info["families"].get("by_role", {})
    development = supply.get("development", {})
    retained = supply.get("development_retention", {}).get("retention")
    common = {s: development[s]["families"] for s in STRATA if s in development}
    full = {s: min(32, development[s]["cases"]) for s in STRATA if s in development}
    if retained is not None:
        full["retention"] = min(32, retained["cases"])
    learners = len(policy["recipes"]) - 1
    looks = len(policy["profiles"]["common"]["looks"])
    for scope in ("common_probe", "wave"):
        if len(common) < len(STRATA) or not sum(common.values()):
            plan[scope]["development_family_census"] = 1
        for method, arms in (("ordinary", 1), ("learner", learners)):
            portfolios(scope, "one", method, common, looks * arms if scope == "wave" else 1)
    if retained is None or not sum(full.values()):
        plan["wave"]["development_and_retention_census"] = 1
    for method, arms in (("ordinary", 1), ("learner", learners), ("D", 1)):
        portfolios("wave", "full", method, full, arms)

    # A protected, incremental measurement with two existing full-profile models,
    # on one distinct OEIS family. No new optimizer or full-bank pass is involved.
    if "oeis" in development and not development["oeis"]["families"]:
        plan["followup"]["eligible_oeis_family"] = 1
    else:
        for method in ("ordinary", "learner"):
            portfolios("followup", "full", method, {"oeis": 1}, 1)

    work = original_probe_costs(info["cuts"], info["historical_scores"], info["original_clocks"])
    control = [c for c in info["cuts"] if c["method"] == "00"]
    missing_cuts = {c["look"] for c in control if not (c.get("weights") and c.get("lineage"))}
    broad = info["prices"].get("broader_updates")
    fresh = info["prices"].get("fresh_start")
    fresh_required = (broad or {}).get("disposition") == "requires_fresh_disjoint_start"
    if fresh_required:
        missing_cuts.update(c["look"] for c in control)
    missing_cuts.update(
        r["look"]
        for r in info["reuse"]["measurements"]
        if r["method"] == "00" and r["action"] == "rerun_control"
    )
    # The historical ordinary base trajectory is a proxy for ordinary positive updates
    # only. New optimizer batches, broader streams and auxiliary passes need their own prices.
    if missing_cuts:
        target = max(missing_cuts)
        key = f"ordinary_to_{target}_tokens"
        plan["wave"][key] = 1
        observed = next(c["training_to_look_us"] for c in control if c["look"] == target)
        if observed is not None and not fresh_required:
            prices[key] = dict(expected_us=observed, tail_us=observed)
    plan["wave"].update(
        {
            k: 1
            for k in (
                "six_treatment_trajectories",
                "matching_calibration_and_auxiliary_passes",
                "paired_conditioning_sample",
                "conventional_cpu_controls",
                "matched_full_bank_scoring",
                "verification_and_publication",
            )
        }
    )
    timed = info["prices"].get("timed_updates")
    if timed is not None:
        # Each update sample includes positive work and its actual auxiliary pass.
        # A broad stream has its own shapes and eligible donors. It needs its own price.
        del plan["wave"]["six_treatment_trajectories"]
        del plan["wave"]["matching_calibration_and_auxiliary_passes"]
        device = (fresh or broad or {}).get("device") or {}
        if fresh_required and "00" in device.get("methods", {}):
            prices["ordinary_to_4000000_tokens"] = device["methods"]["00"]
        for row in policy["recipes"]:
            if row["id"] == "00":
                continue
            key = "optimizer_to_4M_recipe_" + row["id"]
            plan["wave"][key] = 1
            if row["id"] in device.get("methods", {}):
                prices[key] = device["methods"][row["id"]]
            elif not fresh_required and row["stream"] == "ordinary" and row["id"] in timed["methods"]:
                prices[key] = timed["methods"][row["id"]]
        support = ((fresh or broad or {}).get("supply") or {}).get("matching", {}).get("ordinary")
        clock = device.get("calibration", {}).get("elapsed_us")
        if support is not None and not support["coverage"]["full"]:
            clock = None
        if clock is None and not fresh_required and timed.get("matching_scope", {}).get("full"):
            clock = timed["calibration_us"]
        plan["wave"]["ordinary_matching_calibration"] = 1
        if clock is not None:
            prices["ordinary_matching_calibration"] = dict(expected_us=clock, tail_us=clock)
        plan["wave"]["trajectory_startup_checkpoint_io_and_drain"] = 1
    result = price_wave(plan, prices)
    for scope, row in result.items():
        row["operations"] = dict(plan[scope])
        for estimate in ("expected_us", "tail_us"):
            row["uncached_" + estimate] = (
                row[estimate] + sum(history["image_acquisition_us"].values())
                if history is not None and row[estimate] is not None
                else None
            )
    reserve = result["followup"]["tail_us"]
    resident = info["prices"].get("resident_028")
    guards = []
    if resident and blocks:
        from oeis_learn.pipeline_validation.progressive_research_resources import velocity_guard

        for phase in ("search", "score"):
            observed = resident[phase + "_supervisor_us"]
            if phase == "score" and resident["nonempty_score_supervisor_us"] is not None:
                observed = max(observed or 0, resident["nonempty_score_supervisor_us"])
            # The production check is for a whole 160-case full-profile role. Use
            # the same maximum workload even when a smaller panel was censused.
            tails = []
            for name in ("ordinary-full", "first_divergence-full", "incumbent-full"):
                b = blocks[name]
                cells = [b["cases"][s][phase] for s in (*STRATA, "parameter")]
                if all(cell is not None for cell in cells):
                    tails.append(b["overhead_us"][phase] + 32 * sum(c["tail_us"] for c in cells))
            if resident["cases"] == 160 and observed is not None and len(tails) == 3:
                guards.append(
                    velocity_guard(
                        owner="resident_" + phase,
                        scope="one full-profile 160-case supervised role",
                        units="microseconds",
                        observed=observed,
                        projected=max(tails),
                        revision=resident["candidate"],
                        harm="terminating a valid portfolio or its scorer",
                        rehearsal="T058 cold/slow/empty portfolio",
                        exhaustion="drain and resume a smaller batch",
                    )
                )
    credits = info["reuse"]["distinct_score_credits"]
    if fresh_required:
        credits = len(
            {
                r["score"]
                for r in info["reuse"]["measurements"]
                if r["method"] == "D" and r["action"] == "reuse_score"
            }
        )
    return dict(
        **result,
        basis=(
            "round028_exposed_proxy_and_round029_timed_updates"
            if timed
            else "round028_exposed_proxy"
        ),
        **({"timed_updates": timed} if timed else {}),
        **({"broader_updates": broad} if broad else {}),
        **({"fresh_start": fresh} if fresh else {}),
        **({"broader_reason": info["prices"]["broader_reason"]} if "broader_reason" in info["prices"] else {}),
        **(
            {"timing_reason": info["prices"]["timing_reason"]}
            if info["prices"].get("timing_reason")
            else {}
        ),
        common_cases_per_arm=common,
        full_cases_per_arm=full,
        image_acquisition_us=history["image_acquisition_us"] if history else None,
        assumption="cached images; one resident search and one selected scorer per method and look",
        followup_scope="one matched full-profile OEIS family on existing ordinary and challenger weights",
        followup_reserve_us=reserve,
        followup_family_available=(
            bool(development["oeis"]["families"]) if "oeis" in development else None
        ),
        wave_with_reserve_us=(
            result["wave"]["tail_us"] + reserve
            if reserve is not None and result["wave"]["tail_us"] is not None
            else None
        ),
        missing_control_looks=sorted(missing_cuts),
        additional_control_tokens=max(missing_cuts, default=0),
        exposed_028_timing=history,
        prescore_028_balance=info["prices"].get("prescore_028_balance"),
        resident_028=resident,
        provisional_guards=guards,
        prior_score_credits=credits,
        inference_reuse_basis="aggregate screen assumes fresh requests until the ordered graph matches",
        original_028_workloads=work,
        unit_prices=prices,
    )


def supported_velocity_wave(policy, prices, fresh, cycles):
    """Separately price the measured ordinary-stream grid when input support settles the other arms."""
    if not fresh or not cycles or fresh.get("identity") != cycles.get("origin"):
        return None
    supply = fresh.get("supply") or {}
    matching = (supply.get("matching") or {}).get("ordinary") or {}
    if (fresh.get("disposition") != "fresh_ordinary_only"
            or not supply.get("slots") or supply.get("ordinary") != supply["slots"]
            or (matching.get("coverage") or {}).get("full") is not True or matching.get("eligible_slots") != 0):
        return None
    methods, omitted = [], {}
    for recipe in policy["recipes"]:
        reasons = []
        if recipe["stream"] == "broad":
            reasons.append("ordinary already has a distinct composition in every matched slot")
        if recipe["matching"]:
            reasons.append("complete census found zero eligible ordinary matching slots")
        if reasons:
            omitted[recipe["id"]] = reasons
        else:
            methods.append(recipe["id"])
    if (set(methods) != set(cycles["methods"])
            or not set(methods) <= set((fresh.get("device") or {}).get("methods", {}))):
        return None

    operations = dict(prices["wave"]["operations"])
    rates = dict(prices["unit_prices"])
    for method in omitted:
        del operations[f"optimizer_to_4M_recipe_{method}"]
    all_challengers = len(policy["recipes"]) - 1
    for name, count in list(operations.items()):
        if name.startswith("learner."):
            per_arm, remainder = divmod(count, all_challengers)
            if remainder:
                raise ValueError("challenger probe counts do not describe matched opportunities")
            operations[name] = per_arm * (len(methods) - 1)
    del operations["ordinary_matching_calibration"]
    del operations["trajectory_startup_checkpoint_io_and_drain"]
    operations["trajectory_startup_extra_checkpoints_and_drain"] = 1
    storage = {}
    for method in methods:
        io = cycles["methods"][method]["storage_us"]
        # Start, publish 1M, resume it, publish 4M. Final model scoring is priced separately.
        # Actual extra hourly checkpoints, restarts and final supervision are still unmeasured.
        rate = dict(expected_us=io["initial_save"] + io["first_save"] + io["resume_load"] + io["second_save"],
                    tail_us=io["initial_save"] + 2 * max(io["first_save"], io["second_save"])
                            + max(io["resume_load"], io["final_load"]))
        name = "scheduled_checkpoint_storage_" + method
        operations[name] = 1
        rates[name] = storage[method] = rate
    row = price_wave(dict(wave=operations, followup={}), rates)["wave"]
    return dict(row, methods=methods, omitted=omitted, operations=operations, storage=storage,
                fresh_receipt=fresh["identity"], checkpoint_receipt=cycles["receipt"],
                followup_reserve_us=prices["followup_reserve_us"])


def affordable_prefix(batches, prices, *, remaining_us, holdback_us, paid=(), opened=()):
    """Price whole registered batches against *unreserved* time. Open work still needs settlement."""
    from oeis_learn.generalization.budgets import integer

    remaining = integer(remaining_us)
    holdback = integer(holdback_us)
    paid, opened = set(paid), set(opened)
    known, ids = {}, set()
    for batch in batches:
        if batch["id"] in ids or not batch["operations"]:
            raise ValueError("unique registered nonempty batches required")
        ids.add(batch["id"])
        for key, kind in batch["operations"].items():
            if known.setdefault(key, kind) != kind:
                raise ValueError("shared operation work differs")
    if paid & opened or not (paid | opened) <= known.keys():
        raise ValueError("invalid paid or open work")
    spent = paid | opened
    result = dict(
        batches=[],
        expected_us=0,
        tail_us=0,
        holdback_us=holdback,
        next=None,
        deficit_us=0,
        missing=[],
        unsettled=sorted(opened),
    )
    for batch in batches:
        work = {k: v for k, v in batch["operations"].items() if k not in spent}
        missing = sorted({kind for kind in work.values() if kind not in prices})
        if missing:
            result.update(next=batch["id"], missing=missing, deficit_us=None)
            break
        expected = tail = 0
        for kind in work.values():
            typical = integer(prices[kind]["expected_us"])
            maximum = integer(prices[kind]["tail_us"], typical)
            expected += typical
            tail += maximum
        deficit = max(0, result["tail_us"] + tail + holdback - remaining)
        if deficit:
            result.update(next=batch["id"], deficit_us=deficit)
            break
        result["batches"].append(batch["id"])
        result["expected_us"] += expected
        result["tail_us"] += tail
        spent.update(work)
    return result


def occupied_time(intervals):
    """Union comparable intervals, dividing a shared span among its active beneficiaries."""
    from collections import Counter, defaultdict
    from fractions import Fraction
    from .trials import integer

    receipts = {}
    domains = defaultdict(lambda: defaultdict(Counter))
    unknown = []
    for r in intervals:
        receipt = r["receipt"]
        if receipt in receipts:
            if receipts[receipt] != r:
                raise ValueError("one shared receipt has conflicting clocks or beneficiaries")
            continue
        receipts[receipt] = r
        names = r["beneficiaries"]
        if not names or len(names) != len(set(names)):
            raise ValueError("explicit unique beneficiaries required")
        if any(r[k] is None for k in ("host", "boot", "start_us", "end_us")):
            unknown.append(receipt)
            continue
        start, end = integer(r["start_us"]), integer(r["end_us"])
        if end < start:
            raise ValueError("clock ran backwards")
        edges = domains[r["host"], r["boot"]]
        edges[start].update(names)
        edges[end].subtract(names)
    spans = []
    attributed = defaultdict(Fraction)
    for edges in domains.values():
        active = Counter()
        previous = min(edges)
        span = 0
        for at in sorted(edges):
            if active:
                elapsed = at - previous
                span += elapsed
                for name in active:
                    attributed[name] += Fraction(elapsed, len(active))
            active.update(edges[at])
            active = +active
            previous = at
        spans.append(span)
    comparable = len(domains) <= 1 and not unknown
    return dict(union_us=sum(spans) if comparable else None,
                attributed_us={k:float(v) for k,v in attributed.items()} if comparable else None,
                lower_us=max(spans, default=0), upper_us=None if unknown else sum(spans),
                unknown=unknown, receipt_count=len(receipts))


def workstation_accounting(records, *, launch=None, decision=None):
    """Report original debits separately from occupied wall and measured worker resources."""
    from collections import Counter
    from .trials import integer

    phases = {"historical": [], "incremental": []}
    seen = {}
    paid = set()
    for row in records:
        key = row["origin"], row["receipt"]
        if key in seen:
            if seen[key] != row:
                raise ValueError("one operation has conflicting accounting evidence")
            continue
        seen[key] = row
        phase, state = row["origin"], row["state"]
        if phase not in phases or state not in {
            "complete",
            "failed",
            "open",
            "reused",
            "unstarted",
        }:
            raise ValueError("registered work origin and disposition required")
        debit = integer(row["charged_us"])
        if state in {"reused", "unstarted"}:
            if debit or any(
                row.get(k) not in (None, 0)
                for k in ("worker_cpu_us", "lease_us", "covered_us", "busy_us")
            ):
                raise ValueError("a reused or unstarted operation has no new charge")
        else:
            if row["receipt"] in paid:
                raise ValueError("a paid operation cannot be charged in both cost scopes")
            paid.add(row["receipt"])
            start, end = row["start_us"], row["end_us"]
            if start is not None and end is not None and integer(end) - integer(start) != debit:
                raise ValueError("settled debit differs from its proven workstation interval")
            lease, covered, busy = (row.get(k) for k in ("lease_us", "covered_us", "busy_us"))
            if (covered is not None and lease is not None and covered > lease) or (
                busy is not None and (covered is None or busy > covered)
            ):
                raise ValueError("sampled activity must fit within coverage and the device lease")
        phases[phase].append(row)

    def summarize(rows):
        active = [r for r in rows if r["state"] not in {"unstarted", "reused"}]
        roles = Counter()
        resources = {}
        for r in active:
            roles[r["role"]] += r["charged_us"]
        for key in ("worker_cpu_us", "lease_us", "covered_us", "busy_us"):
            known = [integer(r[key]) for r in active if r.get(key) is not None]
            resources[key] = dict(observed=sum(known), missing_operations=len(active) - len(known))
        return dict(
            ledger_us=sum(roles.values()),
            roles=dict(roles),
            dispositions=dict(Counter(r["state"] for r in rows)),
            occupied=occupied_time(active),
            resources=resources,
        )

    elapsed = None
    if launch is not None and decision is not None:
        clocks = [(p.get("host"), p.get("boot")) for p in (launch, decision)]
        if (
            clocks[0] == clocks[1]
            and all(clocks[0])
            and all(p.get("at_us") is not None for p in (launch, decision))
        ):
            elapsed = integer(integer(decision["at_us"]) - integer(launch["at_us"]))
    historical = summarize(phases["historical"])
    incremental = summarize(phases["incremental"])
    work = [r for r in seen.values() if r["state"] not in {"reused", "unstarted"}]
    # A free reuse can have an unknown original production cost. Its zero debit
    # never supplies a workstation interval for that original.
    missing = {
        r["receipt"]: dict(r, host=None, boot=None, start_us=None, end_us=None)
        for r in seen.values() if r["state"] == "reused" and r["receipt"] not in paid
    }
    production = occupied_time([*work, *missing.values()])
    missing_names = {name for r in missing.values() for name in r["beneficiaries"]}
    names = sorted({name for r in seen.values() for name in r["beneficiaries"]})

    def share(occupied, name):
        if occupied["attributed_us"] is None:
            return None
        return occupied["attributed_us"].get(name, 0)

    return dict(
        historical=historical, incremental=incremental, production_occupied=production,
        by_beneficiary={
            name: dict(historical_us=None if name in missing_names else share(historical["occupied"], name),
                       incremental_us=share(incremental["occupied"], name),
                       production_us=share(production, name))
            for name in names
        },
        launch_to_decision_us=elapsed,
    )
