"""Prospective private trial inputs and public commitment for the ordinary initialization scope."""

import os

from oeis_learn.generalization.progressive.campaign import velocity_registration
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.graph import velocity_study_looks
from oeis_learn.generalization.progressive.velocity_donation_report import (
    QUALIFICATION,
    install,
    read_member,
)
from oeis_learn.generalization.progressive.velocity_records import SESSION
from oeis_learn.generalization.progressive.velocity_trials import select_start, recipe_bindings
from .identity import confined
from .preparation_recovery import retain
from .progressive_pilot_capacity import mount_demand
from .progressive_roles import ROUND029_ROLE_POLICY
from .progressive_velocity_initialization import VERSION
from . import progressive_velocity_checkpoint_host as checkpoints

MEMBER = "ordinary-initialization-qualification.json"


def parent(revision, marker):
    if marker != seal(marker):
        raise ValueError("original checkpoint qualification pointer differs")
    return digest([VERSION, revision, marker, ROUND029_ROLE_POLICY])


def authorize(root, revision, minutes):
    from .progressive_velocity_qualification import authorize_qualification

    folder = confined(root, "run-state/discovery/" + SESSION)
    marker = load(confined(folder, checkpoints.MARKER))
    return authorize_qualification(root, int(minutes), parent(revision, marker), dict(attempts=2))


def inputs(revision, info, original, binding, paid_plan):
    prices = info["prices"]
    wave = prices["supported_wave"]
    estimate = prices["launch_estimate"]
    fresh = prices["fresh_start"]
    start = fresh["start"]
    options = prices["launch_options"]
    if (
        estimate["missing"]
        or estimate["sources"]["fresh"] != fresh["identity"]
        or set(wave["methods"]) != set(paid_plan["methods"])
        or "00" not in wave["methods"]
        or estimate["sources"]["checkpoint"] != binding["checkpoint"]["receipt"]
        or paid_plan["origin"] != fresh["identity"]
        or not estimate["followup_reserve_us"]
        or options["scopes"]["ordinary_initialization"]["tail_us"] is None
        or original != seal(original)
    ):
        raise ValueError("complete paid recipe prices and fresh qualification required")

    def panel(role):
        return [
            dict(task=r["task"], group=r["family"], stratum=r["stratum"])
            for r in original["panels"][role]
        ]

    dev, retention = panel("development"), panel("development_retention")
    if not dev or not retention:
        raise ValueError("eligible development and retention families required")
    design = seal(
        dict(
            version=VERSION,
            role_policy=ROUND029_ROLE_POLICY,
            physical=binding["physical"],
            family_supply=info["families"]["measurement"],
            family_roster=original["identity"],
            sources=estimate["sources"],
            supported=sorted(wave["methods"]),
            omitted=wave["omitted"],
            start=start["identity"],
            checkpoint=binding["checkpoint"],
            seed=start["initialization"],
            stream=start["stream_seed"],
            followup_reserve_us=estimate["followup_reserve_us"],
        )
    )
    terms = dict(
        source=revision,
        design=design["identity"],
        development=dev,
        retention=retention,
        ordinary=start["ordinary"],
        broad=start["broad"],
        initialization=start["initialization"],
        stream_seed=start["stream_seed"],
        training_start=select_start(info, None, start),
    )
    registration = velocity_registration(**terms)
    graph = velocity_study_looks(registration)
    roster = {
        method: {
            profile: [
                dict(look=r["look"], members=len(r["members"]), receipt=r["identity"]) for r in rows
            ]
            for profile, rows in series.items()
        }
        for method, series in graph.items()
    }
    plan = seal(
        dict(
            version=VERSION,
            origin=fresh["identity"],
            start=start["identity"],
            registration=registration["identity"],
            design=design["identity"],
            graph=digest(roster),
            initial_weights=start["initial_weights"],
            projection=paid_plan["projection"],
            microbatch=paid_plan["microbatch"],
            learner=paid_plan["methods"]["00"]["package"],
            work=dict(attempts=1),
        )
    )
    return design, registration, roster, plan


def prepare(root, folder, area, revision, source, physical, paid_plan, floor, *, readonly=False):
    from .progressive_round029 import preflight
    from .progressive_study import open_velocity_study
    from .progressive_velocity_preparation import funded_inventory

    info = preflight(root, source, free_disk_gib=floor)["inventory"]
    marker = load(confined(folder, checkpoints.MARKER))
    pointer = load(confined(folder, "preparation/supply/families/funded.json"))
    public, original = funded_inventory(root, "families", pointer, with_original=True)
    if public != info["families"] or info["gpu_dispatches"] or info["assessment"] != "sealed":
        raise ValueError("preflight and original custodian receipt disagree")
    design, reg, roster, plan = inputs(
        revision, info, original, dict(checkpoint=marker, physical=physical), paid_plan
    )

    def record(path, value):
        if readonly:
            if load(confined(area, path)) != value:
                raise ValueError("private pre-score input differs")
        else:
            retain(area / path, value)

    record("study-design.json", design)
    record("look-roster.json", roster)
    if not readonly:
        open_velocity_study(folder, reg)
    trials = confined(folder, "registrations/" + reg["identity"][7:])
    if load(confined(trials, "velocity-registration.json")) != reg:
        raise ValueError("private trial registration differs")
    bindings = recipe_bindings(
        reg["identity"],
        **{
            k: reg[k]
            for k in ("initialization", "stream_seed", "ordinary", "broad", "training_start")
        },
    )
    trial_refs = {}
    for ordinal, (method, binding) in enumerate(bindings.items()):
        trial = load(confined(trials, "trial-" + method + ".json"))
        if trial != seal(trial) or trial["binding"] != binding or trial["ordinal"] != ordinal:
            raise ValueError("registered trial differs from its prospective input")
        trial_refs[method] = trial["identity"]
    record("trial-bindings.json", trial_refs)
    price = info["prices"]["launch_estimate"]
    # This is the original observed read latency on the first freeze. A later metadata read
    # has a different duration. Verify its other components and preserve the issued quote.
    quote = seal(
        dict(
            sources=price["sources"],
            expected_us=price["expected_us"],
            tail_us=price["tail_us"],
            followup_reserve_us=price["followup_reserve_us"],
            checkpoint_read_us=price["provisional_publication"]["checkpoint_read_us"],
            initial_role_us=info["prices"]["launch_preparation"]["initialization_price"]["tail_us"],
        )
    )
    price_file = confined(area, "quote.json", exists=False)
    if price_file.exists():
        original_quote = load(price_file)
        shift = quote["checkpoint_read_us"] - original_quote["checkpoint_read_us"]
        expected = seal(
            dict(
                quote,
                expected_us=quote["expected_us"] - shift,
                tail_us=quote["tail_us"] - shift,
                checkpoint_read_us=original_quote["checkpoint_read_us"],
            )
        )
        if original_quote != expected:
            raise ValueError("original launch quote no longer reconciles to the paid inputs")
        quote = original_quote
    else:
        record("quote.json", quote)
    return (
        plan,
        quote,
        info["prices"]["launch_preparation"]["wave_space"]["ordinary_initialization"],
    )


def dispatch_space(area, rows, floor, *, installed=None):
    """Charge every prefix owner on the actual attempt mount and re-observe its live free space."""
    from .progressive_velocity_broad_host import CapacityShortfall

    if not rows:
        raise ValueError("missing initialization mount requirements")
    owners = {o["owner"]: o for m in rows for o in m["owners"]}
    if set(owners) != {"learner_00", "first_account", "worker_staging"}:
        raise ValueError("incomplete ordinary initialization capacity owners")
    reserved = {u: max(m["reserved_" + u] for m in rows) for u in ("bytes", "inodes")}
    reserved["bytes"] = max(floor, reserved["bytes"])
    fs = os.statvfs(area)
    available = dict(bytes=fs.f_bavail * fs.f_frsize, inodes=fs.f_favail)
    pending = [o["incremental"] for o in owners.values()]
    if installed is not None:
        # Apply the same twofold staging guard to the current candidate wheel.
        old = owners["worker_staging"]["incremental"]
        pending.append({u: max(0, 2 * installed[u] - old[u]) for u in old})
    row = dict(
        device=area.stat().st_dev,
        **mount_demand(available=available, reserve=reserved, pending=pending),
    )
    if row["deficit_bytes"] or row["deficit_inodes"]:
        raise CapacityShortfall("ordinary_initialization", [row], None)
    return [row]


def publish(root, area, grant, binding, request, quote, plan, runtime, *, readonly=False):
    from .progressive_velocity_donation import retained_qualification

    roster = load(confined(area, "look-roster.json"))
    design = load(confined(area, "study-design.json"))
    if (
        digest(roster) != plan["graph"]
        or design != seal(design)
        or design["identity"] != plan["design"]
    ):
        raise ValueError("original look graph or design differs")
    value = seal(
        dict(
            version=VERSION,
            scope="ordinary_initialization",
            candidate=binding["candidate"],
            source=binding["source"],
            physical=binding["physical"],
            image=request["image"],
            wheel=runtime["wheel"]["sha256"],
            role_policy=ROUND029_ROLE_POLICY,
            grant=grant["identity"],
            total_us=grant["total_us"],
            request=request["identity"],
            registration=plan["registration"],
            plan=plan["identity"],
            design=design["identity"],
            graph=plan["graph"],
            looks=roster,
            trials=load(confined(area, "trial-bindings.json")),
            quote=quote,
            supported=design["supported"],
            omitted=design["omitted"],
            start=plan["start"],
            initial_weights=plan["initial_weights"],
            learner=plan["learner"],
            seed=design["seed"],
            stream_seed=design["stream"],
            target=0,
            score_authority=None,
        )
    )
    for dst in (area / "share" / QUALIFICATION, retained_qualification(root, grant)):
        if not readonly:
            install(dst, MEMBER, value)
        if read_member(dst, MEMBER) != value:
            raise ValueError("independently retained qualification differs")
    return value["identity"]


def host_operation(area, grant, name, request, build, *, snapshot=None, protected_us=0):
    """Bound host-owned registration and transport to the same original qualification grant."""
    import time
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from .progressive_velocity_operations import (
        funded_verification,
        replay_operation,
        run_operation,
    )
    from .progressive_velocity_image import preparation_boundary

    coords = dict(checkpoints.coordinates(grant, name), method="ordinary_initialization")

    def verify(out):
        saved = load(confined(out, "producer.json"))
        if saved != build(True):
            raise ValueError("published pre-score input differs from the funded preparation")
        return saved

    if snapshot is not None:
        return replay_operation(area, coords, request, grant, {}, snapshot, verify)["evidence"]

    def execute(out, allowance):
        deadline = time.monotonic_ns() // 1000 + allowance
        with preparation_boundary(out, deadline * 1000):
            value = funded_verification(lambda _: build(False), out, deadline)
            retain(out / "producer.json", value)

    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    return run_operation(
        area=area,
        grant=grant,
        journal=journal,
        coordinates=coords,
        request=request,
        cost={},
        execute=execute,
        verify=verify,
        max_attempts=2,
        protected_us=protected_us,
    )["evidence"]


def funded_inputs(
    root, folder, area, revision, source, physical, paid_plan, floor, grant, *, snapshot=None
):
    request = seal(
        dict(
            version=VERSION,
            candidate=revision,
            grant=grant["identity"],
            checkpoint=load(confined(folder, checkpoints.MARKER)),
            projection=paid_plan["identity"],
            physical=physical,
        )
    )

    def build(readonly):
        plan, quote, prefix = prepare(
            root, folder, area, revision, source, physical, paid_plan, floor, readonly=readonly
        )
        # Available space is re-observed immediately before the device reservation.
        guards = [
            {k: m[k] for k in ("device", "owners", "reserved_bytes", "reserved_inodes")}
            for m in prefix
        ]
        return seal(dict(plan=plan, quote=quote, guards=guards))

    saved = host_operation(area, grant, "freeze_trials", request, build, snapshot=snapshot)
    return saved["plan"], saved["quote"], saved["guards"]


def funded_publication(root, area, grant, binding, request, quote, plan, runtime, *, snapshot=None,
                       protected_us=0):
    def build(readonly):
        identity = publish(
            root, area, grant, binding, request, quote, plan, runtime, readonly=readonly
        )
        return seal(dict(prescore=identity, request=request["identity"]))

    return host_operation(area, grant, "publish_prescore", request, build, snapshot=snapshot,
                          protected_us=protected_us)[
        "prescore"
    ]


REPORT_MEMBER = "ordinary-initialization-report.json"


def public_result(record, balance):
    cut = record["initialization"]
    if balance["pending"] or balance["consumed"]["tokens"]:
        raise ValueError("unsettled or nonzero initialization token debit")
    return seal(
        dict(
            version="round029-ordinary-initialization-report/v1",
            receipt=record["identity"],
            prescore=record["prescore"],
            plan=cut["plan"],
            checkpoint=cut["checkpoint"]["sha256"],
            **{
                k: cut[k]
                for k in (
                    "weights",
                    "tokens",
                    "updates",
                    "state",
                    "restores",
                    "clocks",
                    "allocation",
                )
            },
            **{k: record[k] for k in ("lease_us", "peak_device_bytes", "capacity")},
            elapsed_us=balance["consumed"]["elapsed_us"],
            remaining_us=balance["remaining"]["elapsed_us"],
        )
    )


def verify_archives(report_path, qualification_path):
    from oeis_learn.generalization.budgets import integer
    from oeis_learn.generalization.progressive.campaign import velocity_policy
    from oeis_learn.generalization.progressive.velocity_donation_report import closed, hashes

    q = read_member(qualification_path, MEMBER)
    record = read_member(report_path, REPORT_MEMBER)
    closed(
        q,
        "version scope candidate source physical image wheel role_policy grant total_us request registration plan design graph looks trials quote supported omitted start initial_weights learner seed stream_seed target score_authority",
        sealed=True,
    )
    closed(
        record,
        "version receipt prescore plan checkpoint weights tokens updates state restores clocks allocation lease_us peak_device_bytes capacity elapsed_us remaining_us",
        sealed=True,
    )
    policy = velocity_policy()
    methods = {row["id"] for row in policy["recipes"]}
    hashes(
        *(
            q[k]
            for k in (
                "source",
                "physical",
                "image",
                "wheel",
                "grant",
                "request",
                "registration",
                "plan",
                "design",
                "graph",
                "start",
                "initial_weights",
                "learner",
            )
        ),
        *(record[k] for k in ("receipt", "checkpoint", "state")),
        *q["trials"].values(),
    )
    if (
        q["version"] != VERSION
        or q["scope"] != "ordinary_initialization"
        or q["role_policy"] != ROUND029_ROLE_POLICY
        or q["graph"] != digest(q["looks"])
        or set(q["looks"]) != methods | {policy["reference"], *policy["controls"]}
        or set(q["trials"]) != methods
        or q["supported"] != sorted(methods - set(q["omitted"])) or "00" not in q["supported"]
        or set(q["omitted"]) != methods - set(q["supported"])
        or q["quote"] != seal(q["quote"])
        or integer(q["quote"]["expected_us"], 1) > integer(q["quote"]["tail_us"], 1)
        or q["target"] != 0
        or q["score_authority"] is not None
        or record["version"] != "round029-ordinary-initialization-report/v1"
        or record["prescore"] != q["identity"]
        or record["plan"] != q["plan"]
        or record["weights"] != q["initial_weights"]
        or (integer(record["tokens"]), integer(record["updates"])) != (0, 0)
        or record["restores"] != [record["state"]] * 2
        or not record["capacity"]
        or integer(record["elapsed_us"]) < integer(record["lease_us"], 1)
        or integer(record["remaining_us"]) + record["elapsed_us"] != integer(q["total_us"], 1)
    ):
        raise ValueError(
            "ordinary zero-cut report differs from its independent pre-score commitment"
        )
    integer(q["quote"]["followup_reserve_us"], 1)
    for method, profiles in q["looks"].items():
        if set(profiles) != ({"common", "full"} if method in methods else {"full"}):
            raise ValueError("missing registered profile")
        for profile, looks in profiles.items():
            if [v["look"] for v in looks] != policy["profiles"][profile]["looks"]:
                raise ValueError("registered looks differ")
            for v in looks:
                closed(v, "members look receipt")
                hashes(v["receipt"])
                integer(v["members"], 1)
    clocks = record["clocks"]
    closed(clocks, "save load")
    if (
        len(clocks["save"]) != 1
        or len(clocks["load"]) != 2
        or sum(integer(t, 1) for t in clocks["save"] + clocks["load"]) > record["lease_us"]
    ):
        raise ValueError("ordinary checkpoint clock coverage differs")
    integer(record["peak_device_bytes"], 1)
    closed(record["allocation"], "bytes inodes")
    for u in ("bytes", "inodes"):
        integer(record["allocation"][u], 1)
        for m in record["capacity"]:
            if integer(m["deficit_" + u]) or integer(m["available_" + u]) < integer(
                m["reserved_" + u]
            ) + integer(m["required_" + u]):
                raise ValueError("ordinary checkpoint exceeded its physical guard")
    return record
