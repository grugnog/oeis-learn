"""Read-only replay of the last completed diagnostic on the physical donor or from its public pair."""

import fcntl

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.generalization.progressive.velocity_records import (
    SESSION,
    physical_root_id,
    validate_grant,
)
from oeis_learn.generalization.progressive import velocity_donation_report as public
from .identity import confined
from . import progressive_velocity_donation as donation
from .progressive_round029 import result
from .progressive_velocity_operations import replay_operation
from .progressive_velocity_qualification import _receipt, qualification_parent


def archive_result(report, qualification):
    evidence = public.verify_archives(report, qualification)
    return (
        result(
            "verify",
            "VERIFIED",
            "Recomputed donated next-update parity and its charges from the two public records",
            "review_qualification",
            last=evidence["receipt"],
            report_ref=evidence["identity"],
            verification_scope="paired_archives_donated_next_update",
        ),
        0,
    )


def _donation(root, area, binding, grant, snapshot):
    data, mounts = donation.donated_ordinary(root)
    request = donation.check_request(load(confined(area, "request.json")), binding, data)
    setup = replay_operation(
        area,
        donation.coordinates(grant, "prepare"),
        binding,
        grant,
        {},
        snapshot,
        lambda out: donation.prepare_receipt(area, out, binding, data),
    )
    seed = request["parent"]["milestone"]["seeds"]["initialization"]
    cost = dict(tokens=2 * request["plan"]["update_tokens"], attempts=2)
    paid = replay_operation(
        area,
        donation.coordinates(grant, "train", seed),
        request,
        grant,
        cost,
        snapshot,
        lambda out: donation.check_stage(area, out, request, mounts),
    )
    record = donation.combined_receipt(binding, setup, paid)
    q = public.validate_qualification(public.donation_qualification(binding, request, grant))
    report = public.validate_donation(public.donation_report(q, record, snapshot), q)
    if (
        public.read_member(
            donation.retained_qualification(root, grant), "donation-qualification.json"
        )
        != q
        or public.read_member(
            confined(area, "share/" + public.QUALIFICATION), "donation-qualification.json"
        )
        != q
        or public.verify_archives(
            confined(area, "share/" + public.REPORT), donation.retained_qualification(root, grant)
        )
        != report
    ):
        raise ValueError("public parity differs from the independently reopened original dispatch")
    return record, report["identity"], "private_donated_next_update"


def private_result(root, folder):
    def blocked(reason):
        return (
            result("verify", "BLOCKED", reason, "status", verification_scope="private_workspace"),
            2,
        )

    if not (folder / "status.json").is_file():
        return blocked("No completed round029 publication is available")
    with confined(folder, "run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return blocked("The session owner is updating the original workspace")
        saved = load(confined(folder, "status.json"))
        if saved != seal(saved) or saved["session_id"] != SESSION:
            raise ValueError("saved status differs")
        if saved["action"] != "qualify" or saved["mechanical_status"] != "COMPLETE":
            return blocked("The current session has no completed publication at this boundary")
        path = confined(root, saved["receipt_path"])
        area = path.parent
        grant = load(confined(area, "grant.json"))
        if path != confined(folder, "qualification/" + grant["identity"][7:] + "/receipt.json"):
            raise ValueError("receipt lies outside its authorized grant")
        binding = load(confined(area, "binding.json"))
        if binding != seal(binding) or binding["grant"] != grant["identity"]:
            raise ValueError("original worker binding differs")
        learner = binding["version"] == "round029-learner-rehearsal/v1"
        if not learner and binding["version"] != "round029-donation/v1":
            raise ValueError("unknown diagnostic publication")
        parent = qualification_parent if learner else donation.donation_parent
        approval = load(
            confined(
                root,
                "run-state/discovery-authorizations/"
                + grant["authorizing_record_ref"][7:]
                + ".json",
            )
        )
        validate_grant(
            grant,
            approval,
            root=physical_root_id(root),
            parent=parent(binding["candidate"]),
            action="qualify",
        )
        snapshot = BudgetJournal.inspect_existing(
            confined(area, "accounting"), grant["identity"], grant["resources"]
        )
        if (
            snapshot["pending"]
            or saved["elapsed_us"] != snapshot["consumed"]["elapsed_us"]
            or saved["charged_tokens"] != snapshot["consumed"]["tokens"]
            or saved["remaining_authority"] != snapshot["remaining"]["elapsed_us"]
        ):
            raise ValueError("published spending disagrees with the original journal")
        if learner:
            prior = load(path)
            record = _receipt(area, confined(area, prior["attempt"]), binding)
            if any(key not in snapshot["committed"] for key in record["debits"]):
                raise ValueError("original learner debit missing")
            report, scope = record["identity"], "private_learner_rehearsal"
        else:
            record, report, scope = _donation(root, area, binding, grant, snapshot)
        if (
            record != load(path)
            or saved["receipt"] != record
            or saved["last_committed_ref"] != record["identity"]
        ):
            raise ValueError("published receipt differs from original private evidence")
        return (
            result(
                "verify",
                "VERIFIED",
                "Reopened the original worker evidence and physical grant journal",
                "review_qualification",
                last=record["identity"],
                report_ref=report,
                verification_scope=scope,
            ),
            0,
        )
