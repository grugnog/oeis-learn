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
        result[scope] = dict(expected_us=None if missing else expected,
                             tail_us=None if missing else tail, missing=sorted(missing))
    followup = result["followup"]["tail_us"]
    for scope, row in result.items():
        row["affordable_with_reserve"] = None
        if row["tail_us"] is not None and remaining_us is not None and followup is not None:
            row["affordable_with_reserve"] = (
                row["tail_us"] + (0 if scope == "followup" else followup) <= integer(remaining_us)
            )
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
