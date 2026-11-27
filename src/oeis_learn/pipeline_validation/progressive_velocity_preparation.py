"""Charge and replay the two custodian census steps under one CPU-only grant."""

from contextlib import redirect_stdout
import sys
import time

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    physical_root_id,
    validate_grant,
)
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_research_inputs import checked_file
from .progressive_velocity_census import KINDS, SEED, quotas
from .progressive_velocity_donor import PILOT_REPORT
from .progressive_velocity_operations import InsufficientAuthority, replay_operation, run_operation
from .progressive_velocity_supply import FORMATS, _area, _counts

VERSION = "round029-donor-supply/v1"


def preparation_parent(revision):
    return digest([VERSION, velocity_policy(), revision, PILOT_REPORT, SEED, quotas(), list(KINDS)])


def binding_for(revision, grant):
    return seal(
        dict(
            version=VERSION,
            candidate=revision,
            grant=grant["identity"],
            parent=preparation_parent(revision),
        )
    )


def coordinates(grant, kind):
    return dict(
        study=grant["identity"],
        stage="preparation",
        method="custodian",
        seed=SEED,
        look=0,
        panel=kind,
        profile="original_donor",
        input_mode="exposed_sources",
        role="prepare",
        ordinal=0,
    )


def output_record(out, kind, *, root=None):
    measured = load(confined(out, "measurement.json"), maximum=256 << 20)
    inputs = load(confined(out, "inputs.json"))
    if (
        measured != seal(measured)
        or measured["version"] != FORMATS[kind][1]
        or inputs != seal(inputs)
        or inputs["kind"] != kind
        or not inputs["originals"]
    ):
        raise ValueError("invalid donor census output or missing original references")
    if root is not None:
        for ref in inputs["originals"]:
            checked_file(root, ref)
    return seal(
        dict(
            kind=kind,
            measurement=measured["identity"],
            source=inputs["identity"],
            counts=_counts(kind, measured),
        )
    )


def seal_output(out, kind):
    retain(out / "producer.json", output_record(out, kind))


def evidence(out, kind, *, root=None):
    proof = output_record(out, kind, root=root)
    if load(confined(out, "producer.json")) != proof:
        raise ValueError("donor supply differs from original worker evidence")
    return proof


def run_worker(root, source, out, kind, reserve):
    from .supervision import launch

    deadline = time.monotonic_ns() + reserve * 1000
    cleanup_seconds = 5
    seconds = (deadline - time.monotonic_ns()) / 1e9 - cleanup_seconds - 1
    if seconds <= 0:
        raise InsufficientAuthority("insufficient time for the CPU worker and monitored cleanup")
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module('oeis_learn.pipeline_validation.progressive_velocity_census',run_name='__main__')",
        str(source / "src"),
        str(root),
        str(out),
        kind,
    ]
    try:
        with redirect_stdout(sys.stderr):
            launch(argv, out, seconds=seconds, cleanup_seconds=cleanup_seconds)
    finally:
        if (out / "process.json").is_file():
            process = load(confined(out, "process.json"))
            retain(
                out / "cleanup.json",
                dict(
                    acknowledged=process["cleanup_failure"] is None,
                    within_bound=time.monotonic_ns() <= deadline,
                ),
            )
    if process["outcome"] != "COMPLETE":
        if (out / "failure.json").is_file():
            failed = load(confined(out, "failure.json"))
            if failed != seal(failed):
                raise ValueError("invalid original custodian diagnostic")
            raise ValueError(failed["error"] + ": " + failed["reason"])
        if process["returncode"] is None or "TimeoutError" in (process["failure"] or ""):
            raise InsufficientAuthority("custodian CPU deadline reached")
        raise ValueError("custodian worker exited " + str(process["returncode"]))
    seal_output(out, kind)


def reopen(root, area, binding, grant, snapshot, kind):
    if binding != binding_for(binding["candidate"], grant):
        raise ValueError("original preparation binding differs")
    return replay_operation(
        area,
        coordinates(grant, kind),
        binding,
        grant,
        {},
        snapshot,
        lambda out: evidence(out, kind, root=root),
    )


def combined(binding, receipts):
    return seal(
        dict(
            version=VERSION,
            binding=binding["identity"],
            measurements={k: receipts[k]["identity"] for k in KINDS},
        )
    )


def funded_inventory(root, kind, pointer):
    if pointer != seal(pointer) or pointer["kind"] != kind:
        raise ValueError("incorrect funded supply pointer")
    area = confined(root, f"run-state/discovery/{SESSION}/qualification/" + pointer["grant"][7:])
    grant = load(confined(area, "grant.json"))
    binding = load(confined(area, "binding.json"))
    if grant["identity"] != pointer["grant"]:
        raise ValueError("supply grant differs")
    approval = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant,
        approval,
        root=physical_root_id(root),
        parent=preparation_parent(binding["candidate"]),
        action="qualify",
    )
    snapshot = BudgetJournal.inspect_existing(
        confined(area, "accounting"), grant["identity"], grant["resources"]
    )
    paid = reopen(root, area, binding, grant, snapshot, kind)
    if pointer["receipt"] != paid["identity"]:
        raise ValueError("supply debit or evidence differs")
    return dict(
        **paid["evidence"]["counts"],
        registration=binding["identity"],
        measurement=paid["identity"],
        reason=None,
    )


def qualify(args, root, folder, source, revision, grant, host_factory):
    from .progressive_round029 import result

    area = confined(folder, "qualification/" + grant["identity"][7:], exists=False)
    area.mkdir(parents=True, exist_ok=True)
    binding = binding_for(revision, grant)
    retain(area / "binding.json", binding)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    receipts = {}
    errors = []
    for kind in KINDS:
        try:
            paid = run_operation(
                area=area,
                journal=journal,
                grant=grant,
                coordinates=coordinates(grant, kind),
                request=binding,
                cost={},
                minimum_us=6_100_000,
                execute=lambda out, reserve: run_worker(root, source, out, kind, reserve),
                verify=lambda out: evidence(out, kind),
            )
            receipts[kind] = paid
            path = _area(root, kind) / "funded.json"
            pointer = seal(dict(kind=kind, grant=grant["identity"], receipt=paid["identity"]))
            if not path.exists() or load(confined(root, path)) != pointer:
                revise(path, pointer)
        except (Exception, KeyboardInterrupt) as exc:
            if not (area / "first-error.json").exists():
                retain(
                    area / "first-error.json",
                    seal(
                        dict(
                            stage=kind,
                            error=type(exc).__name__,
                            reason=str(exc) or type(exc).__name__,
                        )
                    ),
                )
            errors.append(exc)
            if isinstance(exc, KeyboardInterrupt):
                break
    snapshot = journal.snapshot()
    extra = {}
    if (area / "first-error.json").exists():
        extra["first_error"] = load(confined(area, "first-error.json"))
    if errors:
        blocked = all(isinstance(e, InsufficientAuthority) for e in errors)
        state, code = ("BLOCKED", 2) if blocked else ("FAILED", 1)
        reason, next_action = str(errors[0]) or type(errors[0]).__name__, "inspect_attempt"
        if isinstance(errors[-1], KeyboardInterrupt):
            code = 130
    else:
        record = combined(binding, receipts)
        path = area / "receipt.json"
        retain(path, record)
        extra.update(
            last=record["identity"], receipt=record, receipt_path=str(path.relative_to(root))
        )
        state, code, reason, next_action = (
            "COMPLETE",
            0,
            "Completed both original donor supply measurements",
            "preflight",
        )
    value = result(
        "qualify",
        state,
        reason,
        next_action,
        scope="donor_supply",
        stage="preparation",
        remaining=snapshot["remaining"]["elapsed_us"],
        elapsed_us=snapshot["consumed"]["elapsed_us"],
        charged_tokens=0,
        remaining_tokens=snapshot["remaining"]["tokens"],
        research_dispatches=0,
        **extra,
    )
    path = folder / "status.json"
    if not path.exists() or load(confined(folder, path)) != seal(value):
        revise(path, seal(value))
    return value, code
