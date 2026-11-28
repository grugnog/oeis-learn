"""Freeze the supported exploration wave against its paid rehearsals before any study score."""

import re

from oeis_learn.generalization.progressive import velocity_donation_report as public
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    physical_root_id,
    validate_grant,
)
from . import progressive_velocity_checkpoint_host as checkpoints
from . import progressive_velocity_prescore as initial
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_velocity_broad_host import CapacityShortfall
from .progressive_velocity_donation import retained_qualification
from . import progressive_velocity_initialization_recovery as recovery
from .progressive_velocity_operations import InsufficientAuthority
from .progressive_velocity_qualification import authorize_qualification

VERSION = "round029-supported-trial-preparation/v1"
MEMBER = "supported-wave-qualification.json"
REPORT_MEMBER = "supported-wave-preparation-report.json"
INITIAL_MARKER = "preparation/ordinary-initialization.json"
SOURCES = {"fresh", "checkpoint", "controls", "paired", "bank", "accounting"}


def parent(reader, marker):
    if marker != seal(marker):
        raise ValueError("original initialization pointer differs")
    return digest([VERSION, reader, marker])


def accepted_preparation(folder):
    path = confined(folder, "accepted.json", exists=False)
    if path.exists():
        accepted = load(path)
        if accepted.get("version") == VERSION:
            if accepted != seal(accepted):
                raise ValueError("original trial preparation acceptance differs")
            public.hashes(accepted["preparation_grant"])
            return accepted
    return None


def authorize(root, reader, minutes):
    folder = confined(root, "run-state/discovery/" + SESSION)
    requested_parent = parent(reader, load(confined(folder, INITIAL_MARKER)))
    accepted = accepted_preparation(folder)
    if accepted is not None:
        original = load(
            confined(folder, "qualification/" + accepted["preparation_grant"][7:] + "/grant.json")
        )
        if (
            original["identity"] != accepted["preparation_grant"]
            or accepted["reader"] != reader
            or original["registration_or_parent_ref"] != requested_parent
            or original["total_us"] != minutes * 60_000_000
        ):
            raise ValueError(
                "reopen the original trial preparation with --recover-trial-preparation"
            )
    return authorize_qualification(root, minutes, requested_parent, {})


def schedule(looks, supported):
    """Bind both opportunities at 4M and the single full endpoint for each comparator."""
    policy = velocity_policy()
    stages = []
    for method in [*supported, "D", *policy["controls"]]:
        profiles = ("common", "full") if method in supported else ("full",)
        if set(looks[method]) != set(profiles):
            raise ValueError("unregistered trial profile")
        for profile in profiles:
            rows = looks[method][profile]
            if [r["look"] for r in rows] != policy["profiles"][profile]["looks"]:
                raise ValueError("registered trial looks differ")
            for r in rows:
                public.hashes(r["receipt"])
                stages.append(
                    dict(
                        method=method,
                        profile=profile,
                        look=r["look"],
                        members=integer(r["members"], 1),
                        roster=r["receipt"],
                    )
                )
    return stages


def training_requests(registration, methods):
    rows = []
    for method in sorted(methods):
        terms = methods[method]
        public.closed(terms, "package batch first second")
        public.hashes(terms["package"])
        if terms["batch"] != (16 if method in {"04", "06"} else 32) or not (
            integer(terms["second"], 1) > integer(terms["first"], 1)
        ):
            raise ValueError("registered update opportunity differs from the paid learner")
        previous = None
        for target in velocity_policy()["profiles"]["common"]["looks"]:
            row = seal(
                dict(
                    registration=registration,
                    method=method,
                    target=target,
                    learner=terms["package"],
                    previous=previous,
                )
            )
            rows.append(row)
            previous = row["identity"]
    return rows


def checked_space(rows):
    if not rows or len({r["device"] for r in rows}) != len(rows):
        raise ValueError("one current observation per wave mount required")
    selected = []
    for row in rows:
        m = dict(device=integer(row["device"]))
        for unit in ("bytes", "inodes"):
            for prefix in ("available", "reserved", "required", "deficit"):
                key = prefix + "_" + unit
                m[key] = integer(row[key])
            if m["deficit_" + unit] != max(
                0, m["required_" + unit] + m["reserved_" + unit] - m["available_" + unit]
            ):
                raise ValueError("wave capacity arithmetic differs")
        selected.append(m)
    if any(r["deficit_bytes"] or r["deficit_inodes"] for r in selected):
        raise CapacityShortfall("trial_preparation", selected, None)
    return selected


def verified_initialization(root, folder, reader_source, marker):
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant, binding = [load(confined(area, n + ".json")) for n in ("grant", "binding")]
    if (
        binding != seal(binding)
        or binding["version"] != initial.VERSION
        or grant["identity"] != marker["grant"]
        or binding["grant"] != grant["identity"]
    ):
        raise ValueError("original initialization grant or binding differs")
    validate_grant(
        grant,
        load(
            confined(
                root,
                "run-state/discovery-authorizations/"
                + grant["authorizing_record_ref"][7:]
                + ".json",
            )
        ),
        root=physical_root_id(root),
        parent=initial.parent(binding["candidate"], load(confined(folder, checkpoints.MARKER))),
        action="qualify",
    )
    source = recovery.worker_source(root, reader_source, binding["candidate"])
    state = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"], include_work=True
    )
    record = checkpoints.replay(root, folder, area, source, grant, binding, state)
    qualification = retained_qualification(root, grant)
    cut = initial.verify_archives(area / "share" / public.REPORT, qualification)
    if (
        state["pending"]
        or record != load(confined(area, "receipt.json"))
        or record["identity"] != marker["receipt"]
        or cut != initial.public_result(record, state)
    ):
        raise ValueError("original zero cut, grant or independent pre-score copy differs")
    q = public.read_member(qualification, initial.MEMBER)
    _, projection, _, matrix = checkpoints.inputs(root, folder, source)
    if (
        matrix["origin"] != q["quote"]["sources"]["fresh"]
        or matrix["methods"]["00"]["package"] != q["learner"]
        or set(matrix["methods"]) != set(q["supported"])
    ):
        raise ValueError("registered trial workers differ from the paid checkpoint rehearsals")
    return q, cut, matrix, projection


def commitment(root, folder, source, binding, grant, admission, area, readonly):
    q, cut, matrix, projection = verified_initialization(root, folder, source, binding["initial"])
    quote = q["quote"]
    if set(quote["sources"]) != SOURCES or quote["sources"] != admission["sources"]:
        raise ValueError("missing paid trial preparation prices")
    public.hashes(*quote["sources"].values())
    training = training_requests(q["registration"], matrix["methods"])
    reg = confined(folder, "registrations/" + q["registration"][7:] + "/velocity-registration.json")
    private = seal(
        dict(
            registration=reference(root, reg),
            projection=reference(root, projection),
            matrix=matrix,
            targets=training,
            initialization=binding["initial"],
        )
    )
    path = confined(area, "trial-inputs.json", exists=False)
    if not readonly:
        retain(path, private)
    if load(path) != private:
        raise ValueError("registered trial inputs or checkpoint targets differ")
    return seal(
        dict(
            version=VERSION,
            scope="exploration",
            candidate=q["candidate"],
            reader=binding["candidate"],
            owner=grant["state_root_id"],
            preparation_grant=grant["identity"],
            initialization=cut["receipt"],
            initialization_prescore=q["identity"],
            checkpoint=cut["checkpoint"],
            state=cut["state"],
            source=q["source"],
            image=q["image"],
            wheel=q["wheel"],
            physical=q["physical"],
            role_policy=q["role_policy"],
            registration=q["registration"],
            trials=q["trials"],
            graph=q["graph"],
            supported=q["supported"],
            omitted={m: digest(reason) for m, reason in q["omitted"].items()},
            seed=q["seed"],
            stream_seed=q["stream_seed"],
            learners=matrix["methods"],
            training=training,
            inputs=private["identity"],
            schedule=schedule(q["looks"], q["supported"]),
            policy=digest(velocity_policy()),
            sources=quote["sources"],
            expected_us=quote["expected_us"],
            tail_us=quote["tail_us"],
            followup_us=quote["followup_reserve_us"],
            proposed_total_us=quote["tail_us"] + quote["followup_reserve_us"],
            capacity=checked_space(admission["mounts"]),
            research_authority=None,
            assessment="sealed",
        )
    )


def stages(root, folder, area, source, grant, binding, *, snapshot=None):
    admission = load(confined(area, "admission.json"))
    if admission != seal(admission) or admission["binding"] != binding["identity"]:
        raise ValueError("original wave admission differs")
    accepted = initial.host_operation(
        area,
        grant,
        "freeze_wave",
        binding,
        lambda readonly: commitment(
            root, folder, source, binding, grant, admission, area, readonly
        ),
        snapshot=snapshot,
    )

    def publish(readonly):
        if not readonly and not retained_qualification(root, grant).exists():
            registration = confined(folder, "registrations/" + accepted["registration"][7:])
            if any(registration.glob("look-*.json")):
                raise ValueError("trial pre-score commitment must precede comparative observations")
        for path in (area / "share" / public.QUALIFICATION, retained_qualification(root, grant)):
            if not readonly:
                public.install(path, MEMBER, accepted)
            if public.read_member(path, MEMBER) != accepted:
                raise ValueError("independent supported-wave qualification differs")
        return seal(dict(accepted=accepted["identity"], grant=grant["identity"]))

    issued = initial.host_operation(
        area, grant, "publish_wave", accepted, publish, snapshot=snapshot
    )
    return accepted, seal(
        dict(
            version=VERSION,
            binding=binding["identity"],
            accepted=accepted["identity"],
            publication=issued["identity"],
            initialization=binding["initial"]["receipt"],
        )
    )


def report(record, grant, balance):
    return seal(
        dict(
            version=VERSION,
            accepted=record["accepted"],
            receipt=record["identity"],
            grant=grant["identity"],
            total_us=grant["total_us"],
            elapsed_us=balance["consumed"]["elapsed_us"],
            remaining_us=balance["remaining"]["elapsed_us"],
            charged_tokens=balance["consumed"]["tokens"],
            research_dispatches=0,
            scientific_verdict=None,
        )
    )


def published(root, folder):
    accepted = accepted_preparation(folder)
    if accepted is None:
        return None
    area = confined(folder, "qualification/" + accepted["preparation_grant"][7:])
    grant = load(confined(area, "grant.json"))
    marker = load(confined(folder, INITIAL_MARKER))
    validate_grant(
        grant,
        load(
            confined(
                root,
                "run-state/discovery-authorizations/"
                + grant["authorizing_record_ref"][7:]
                + ".json",
            )
        ),
        root=physical_root_id(root),
        parent=parent(accepted["reader"], marker),
        action="qualify",
    )
    record = load(confined(area, "receipt.json"))
    balance = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"]
    )
    external = retained_qualification(root, grant)
    if (
        accepted != public.read_member(external, MEMBER)
        or balance["pending"]
        or record != seal(record)
        or record["accepted"] != accepted["identity"]
        or record["initialization"] != marker["receipt"]
        or verify_archives(area / "share" / public.REPORT, external)
        != report(record, grant, balance)
    ):
        raise ValueError("accepted trial preparation differs from its original publication")
    return dict(
        accepted=accepted["identity"],
        registration=accepted["registration"],
        sources=accepted["sources"],
        supported=accepted["supported"],
        proposed_total_us=accepted["proposed_total_us"],
        research_authority=None,
        elapsed_us=balance["consumed"]["elapsed_us"], receipt=record["identity"],
    )


def verify_archives(report_path, qualification_path):
    q = public.read_member(qualification_path, MEMBER)
    r = public.read_member(report_path, REPORT_MEMBER)
    public.closed(
        q,
        "version scope candidate reader owner preparation_grant initialization initialization_prescore checkpoint state source image wheel physical role_policy registration trials graph supported omitted seed stream_seed learners training inputs schedule policy sources expected_us tail_us followup_us proposed_total_us capacity research_authority assessment",
        sealed=True,
    )
    public.closed(
        r,
        "version accepted receipt grant total_us elapsed_us remaining_us charged_tokens research_dispatches scientific_verdict",
        sealed=True,
    )
    public.hashes(
        r["receipt"],
        q["owner"],
        q["preparation_grant"],
        q["initialization"],
        q["initialization_prescore"],
        q["source"],
        q["image"],
        q["wheel"],
        q["physical"],
        q["role_policy"],
        q["registration"],
        q["graph"],
        q["checkpoint"],
        q["state"],
        q["inputs"],
        *q["sources"].values(),
        *q["trials"].values(),
        *q["omitted"].values(),
    )
    if (
        r["version"] != VERSION
        or q["version"] != VERSION
        or r["accepted"] != q["identity"]
        or r["grant"] != q["preparation_grant"]
        or q["research_authority"] is not None
        or q["assessment"] != "sealed"
        or q["scope"] != "exploration"
        or r["scientific_verdict"] is not None
        or integer(r["research_dispatches"])
        or integer(r["charged_tokens"])
        or set(q["sources"]) != SOURCES
        or integer(r["elapsed_us"]) + integer(r["remaining_us"]) != integer(r["total_us"], 1)
        or integer(q["expected_us"], 1) > integer(q["tail_us"], 1)
        or q["proposed_total_us"] != q["tail_us"] + integer(q["followup_us"], 1)
        or q["supported"] != sorted({r["id"] for r in velocity_policy()["recipes"]} - set(q["omitted"]))
        or "00" not in q["supported"]
        or set(q["learners"]) != set(q["supported"])
        or set(q["trials"]) != {r["id"] for r in velocity_policy()["recipes"]}
        or set(q["trials"]) != set(q["supported"]) | set(q["omitted"])
        or q["policy"] != digest(velocity_policy())
        or q["training"] != training_requests(q["registration"], q["learners"])
    ):
        raise ValueError("trial preparation differs from its priced pre-score scope")
    if not all(re.fullmatch("[0-9a-f]{40}", q[k]) for k in ("reader", "candidate")) or q[
        "capacity"
    ] != checked_space(q["capacity"]):
        raise ValueError("trial preparation revision or mount observation differs")
    looks = {}
    for row in q["schedule"]:
        public.closed(row, "method profile look members roster")
        looks.setdefault(row["method"], {}).setdefault(row["profile"], []).append(
            dict(look=row["look"], members=row["members"], receipt=row["roster"])
        )
    if q["schedule"] != schedule(looks, q["supported"]):
        raise ValueError("trial preparation schedule differs")
    return r


def qualify(args, root, folder, source, revision, grant):
    from .progressive_round029 import preflight, result

    marker = load(confined(folder, INITIAL_MARKER))
    if grant["registration_or_parent_ref"] != parent(revision, marker):
        raise ValueError("wave qualification requires the original initialization")
    accepted = accepted_preparation(folder)
    if accepted is not None and accepted["preparation_grant"] != grant["identity"]:
        raise ValueError("reopen the original trial preparation with --recover-trial-preparation")
    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    binding = seal(
        dict(version=VERSION, candidate=revision, grant=grant["identity"], initial=marker)
    )
    retain(area / "grant.json", grant)
    retain(area / "binding.json", binding)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    stage = "trial_capacity"
    extra = {}
    try:
        if not (area / "receipt.json").exists():
            info = preflight(root, source, free_disk_gib=getattr(args, "free_disk_gib", 5))[
                "inventory"
            ]
            prices = info["prices"]
            wave = prices["launch_options"]["scopes"]["wave"]
            mounts = checked_space(wave["mounts"])
            if (
                info["gpu_dispatches"]
                or info["assessment"] != "sealed"
                or prices["launch_estimate"]["missing"]
                or wave["fits"] is not True
                or prices["launch_options"]["research_authority"] is not None
            ):
                raise ValueError("complete paid wave prices and current physical capacity required")
            admission = seal(
                dict(
                    binding=binding["identity"],
                    mounts=mounts,
                    sources=prices["launch_estimate"]["sources"],
                )
            )
            saved = area / "admission.json"
            if not saved.exists():
                retain(saved, admission)
            else:
                earlier = load(confined(area, "admission.json"))
                demands = lambda entry: [
                    {k: v for k, v in row.items() if not k.startswith("available_")}
                    for row in entry["mounts"]
                ]
                if earlier["sources"] != admission["sources"] or demands(earlier) != demands(
                    admission
                ):
                    raise ValueError("registered wave requirements or paid prices changed")
        stage = "trial_publication"
        accepted, record = stages(root, folder, area, source, grant, binding)
        retain(area / "receipt.json", record)
        balance = journal.snapshot()
        if balance["pending"] or balance["consumed"]["tokens"]:
            raise ValueError("unsettled wave qualification")
        result_path = area / "share" / public.REPORT
        public.install(result_path, REPORT_MEMBER, report(record, grant, balance))
        verify_archives(result_path, retained_qualification(root, grant))
        retain(folder / "accepted.json", accepted)
        status, reason, nxt, code = (
            "COMPLETE",
            "Supported trial preparation and pre-score publication verified",
            "review_qualification",
            0,
        )
        extra = dict(
            last=record["identity"],
            receipt=record,
            receipt_path=str((area / "receipt.json").relative_to(root)),
        )
    except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as exc:
        reason = str(exc) or type(exc).__name__
        status, nxt, code = "FAILED", "inspect_attempt", 1
        if isinstance(exc, InsufficientAuthority):
            status, code = "BLOCKED", 2
        if isinstance(exc, CapacityShortfall):
            nxt = "inspect_capacity"
            extra.update(exc.deficits)
        error = seal(dict(stage=stage, error=type(exc).__name__, reason=reason))
        if not (area / "first-error.json").exists():
            retain(area / "first-error.json", error)
        extra["diagnostics"] = error
    if (area / "first-error.json").is_file():
        extra["first_error"] = load(confined(area, "first-error.json"))
    balance = journal.snapshot()
    value = result(
        "qualify",
        status,
        reason,
        nxt,
        scope="trial_preparation",
        stage=stage,
        remaining=balance["remaining"]["elapsed_us"],
        elapsed_us=balance["consumed"]["elapsed_us"],
        charged_tokens=balance["consumed"]["tokens"],
        research_dispatches=0,
        **extra,
    )
    if not (folder / "status.json").exists() or load(confined(folder, "status.json")) != seal(
        value
    ):
        revise(folder / "status.json", seal(value))
    return value, code
