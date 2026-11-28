"""Read-only replay of the last completed diagnostic on the physical donor or from its public pair."""

import fcntl

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest, load, seal
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
    import zipfile
    from . import progressive_velocity_prescore as prescore
    from . import progressive_velocity_trial_preparation as trials

    from . import progressive_velocity_wave_commitment as commitment
    with zipfile.ZipFile(qualification) as archive:
        exploration = archive.namelist() == [commitment.MEMBER]
        initialization = archive.namelist() == [prescore.MEMBER]
        trial_prep = archive.namelist() == [trials.MEMBER]
    if exploration or trial_prep:
        from . import progressive_velocity_wave_report as comparative

        with zipfile.ZipFile(report) as archive:
            research = archive.namelist() == [comparative.MEMBER, comparative.ACCOUNT]
        if research:
            value = comparative.verify(report, qualification)
            verified = result("verify", "VERIFIED", "Recomputed the comparative decision from anonymous paired observations",
                              "review_exploration", last=value["identity"], report_ref=value["identity"],
                              verification_scope="paired_archives_comparative_wave")
            verified["scientific_verdict"] = value["decision"]["verdict"]
            return verified, 0
        evidence = trials.verify_archives(report, qualification)
        return (result("verify", "VERIFIED", "Reopened the supported trial schedule and independent pre-score commitment",
                       "review_qualification", last=evidence["receipt"], report_ref=evidence["identity"],
                       verification_scope="paired_archives_trial_preparation"), 0)
    if initialization:
        evidence = prescore.verify_archives(report, qualification)
        return (
            result(
                "verify",
                "VERIFIED",
                "Reopened the exposed ordinary zero-cut report against its pre-score commitment",
                "review_qualification",
                last=evidence["receipt"],
                report_ref=evidence["identity"],
                verification_scope="paired_archives_ordinary_initialization",
            ),
            0,
        )
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


def donor_continuation(root, folder, source, cut):
    """Credit only the matching revision and cut after a full, read-only receipt replay."""
    from .progressive_round028 import candidate_revision

    if not cut.get("full_state") or not (folder / "qualification").is_dir():
        return None
    revision = candidate_revision(source, root)
    for path in sorted(folder.glob("qualification/*/receipt.json")):
        area = confined(root, path.parent)
        binding = load(confined(area, "binding.json"))
        if binding.get("version") != "round029-donation/v1" or binding.get("candidate") != revision:
            continue
        if binding.get("source") != digest(donation.collect_snapshot(source)):
            raise ValueError("registered donation source differs from the candidate checkout")
        request = load(confined(area, "request.json"))
        if (
            cut["full_state"] != request["plan"]["checkpoint"]
            or cut.get("lineage") != request["parent"]["milestone"]["stream"]
        ):
            continue
        grant = load(confined(area, "grant.json"))
        if (
            binding != seal(binding)
            or binding["grant"] != grant["identity"]
            or area.name != grant["identity"][7:]
            or request["binding"] != binding["identity"]
        ):
            raise ValueError("recorded donation grant or candidate binding differs")
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
            parent=donation.donation_parent(revision),
            action="qualify",
        )
        snapshot = BudgetJournal.inspect_existing(
            confined(area, "accounting"), grant["identity"], grant["resources"]
        )
        record, report, _ = _donation(root, area, binding, grant, snapshot)
        if record != load(confined(area, "receipt.json")):
            raise ValueError("original donation receipt differs")
        return dict(
            candidate=revision,
            full_state=cut["full_state"],
            receipt=record["identity"],
            report=report,
            image=request["image"],
            device=request["physical"],
            policy=request["plan"]["new_policy"],
        )
    return None


def private_result(root, folder, *, source=None, trial_grant=None):
    def blocked(reason):
        return (
            result("verify", "BLOCKED", reason, "status", verification_scope="private_workspace"),
            2,
        )

    if trial_grant is None and not (folder / "status.json").is_file():
        return blocked("No completed round029 publication is available")
    with confined(folder, "run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return blocked("The session owner is updating the original workspace")
        saved = None
        if trial_grant is None:
            saved = load(confined(folder, "status.json"))
            if saved != seal(saved) or saved["session_id"] != SESSION:
                raise ValueError("saved status differs")
            if saved["action"] == "run" and saved["mechanical_status"] == "COMPLETE":
                from . import progressive_velocity_wave as wave
                from .progressive_round028 import candidate_revision
                from .progressive_velocity_run import run

                marker = load(confined(folder, wave.MARKER))
                grant = load(confined(folder, "comparative/" + marker["grant"][7:] + "/grant.json"))
                approval = load(confined(root, "run-state/discovery-authorizations/" +
                                         grant["authorizing_record_ref"][7:] + ".json"))
                validate_grant(grant, approval, root=physical_root_id(root), parent=marker["accepted"], action="run")
                return run(None, root, folder, source, candidate_revision(source, root), grant, None, readonly=True)
            if saved["action"] != "qualify" or saved["mechanical_status"] != "COMPLETE":
                return blocked("The current session has no completed publication at this boundary")
            path = confined(root, saved["receipt_path"])
        else:
            public.hashes(trial_grant)
            path = confined(folder, "qualification/" + trial_grant[7:] + "/receipt.json")
        area = path.parent
        grant = load(confined(area, "grant.json"))
        if path != confined(folder, "qualification/" + grant["identity"][7:] + "/receipt.json"):
            raise ValueError("receipt lies outside its authorized grant")
        binding = load(confined(area, "binding.json"))
        if binding != seal(binding) or binding["grant"] != grant["identity"]:
            raise ValueError("original worker binding differs")
        learner = binding["version"] == "round029-learner-rehearsal/v1"
        from . import progressive_velocity_preparation as preparation
        from . import progressive_velocity_timing_host as timing

        from . import progressive_velocity_broad_host as broad
        from . import progressive_velocity_checkpoint_host as checkpoints

        from . import progressive_velocity_prescore as prescore

        from . import progressive_velocity_trial_preparation as trials
        trial_prep = binding["version"] == trials.VERSION
        if trial_grant is not None:
            from .progressive_velocity_initialization_recovery import worker_source

            authority = confined(
                root, "run-state/discovery-authorizations/grant-" + trial_grant[7:] + ".json"
            )
            if not trial_prep or grant["identity"] != trial_grant or load(authority) != grant:
                raise ValueError("original supported-wave grant required")
            # Authenticate the funded reader even when a newer reader performs the replay.
            source = worker_source(root, source, binding["candidate"])
        initial = binding["version"] == prescore.VERSION
        broader = binding["version"] == broad.VERSION
        checkpoint = binding["version"] == checkpoints.matrix.VERSION
        supply = binding["version"] == preparation.VERSION
        matrix = binding["version"] == timing.VERSION
        if learner:
            parent = qualification_parent(
                binding["candidate"], children=binding.get("child_restore", False)
            )
        elif trial_prep:
            marker = load(confined(folder, trials.INITIAL_MARKER))
            if binding["initial"] != marker:
                raise ValueError("supported-wave binding differs from the original zero cut")
            parent = trials.parent(binding["candidate"], marker)
        elif initial:
            parent = prescore.parent(
                binding["candidate"], load(confined(folder, checkpoints.MARKER))
            )
        elif checkpoint:
            parent = checkpoints.parent(binding["candidate"], binding["origin"])
        elif broader:
            parent = broad.parent(binding["candidate"], binding["origins"])
        elif supply:
            parent = preparation.preparation_parent(binding["candidate"])
        elif matrix:
            parent = timing.timing_parent(binding["candidate"], binding["pair_pointer"]["receipt"])
        elif binding["version"] == "round029-donation/v1":
            parent = donation.donation_parent(binding["candidate"])
        else:
            raise ValueError("unknown diagnostic publication")
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
            parent=parent,
            action="qualify",
        )
        snapshot = BudgetJournal.inspect_existing(
            confined(area, "accounting"),
            grant["identity"],
            grant["resources"],
            include_work=checkpoint or initial or trial_prep,
        )
        if snapshot["pending"] or (
            saved is not None
            and (
                saved["elapsed_us"] != snapshot["consumed"]["elapsed_us"]
                or saved["charged_tokens"] != snapshot["consumed"]["tokens"]
                or saved["remaining_authority"] != snapshot["remaining"]["elapsed_us"]
            )
        ):
            raise ValueError("published spending disagrees with the original journal")
        if learner:
            prior = load(path)
            record = _receipt(area, confined(area, prior["attempt"]), binding)
            if any(key not in snapshot["committed"] for key in record["debits"]):
                raise ValueError("original learner debit missing")
            report, scope = record["identity"], "private_learner_rehearsal"
        elif trial_prep:
            accepted, record = trials.stages(root, folder, area, source, grant, binding, snapshot=snapshot)
            copy = trials.verify_archives(area / "share" / public.REPORT, donation.retained_qualification(root, grant))
            if accepted != load(confined(folder, "accepted.json")) or copy != trials.report(record, grant, snapshot):
                raise ValueError("trial preparation differs from its accepted schedule or independent copy")
            report, scope = record["identity"], "private_trial_preparation"
        elif checkpoint or initial:
            record = checkpoints.replay(root, folder, area, source, grant, binding, snapshot)
            if initial:
                copied = prescore.verify_archives(
                    area / "share" / public.REPORT, donation.retained_qualification(root, grant)
                )
                if copied != prescore.public_result(record, snapshot):
                    raise ValueError("ordinary public report differs from the private checkpoint")
            report, scope = record["identity"], (
                "private_ordinary_initialization" if initial else "private_checkpoint_matrix"
            )
        elif broader:
            fresh = "overlap" in binding["origins"]
            if binding["origins"] != broad.original_pointers(root, folder, fresh=fresh):
                raise ValueError("broader inputs differ from the original paid records")
            if fresh:
                source = source or confined(
                    root, "run-state/discovery-tools/round028-source/" + binding["candidate"]
                )
                if (
                    broad.inventory(root, source)["identity"]
                    != binding["origins"]["overlap"]["receipt"]
                ):
                    raise ValueError("fresh qualification has a different paid exclusion")
            record = broad.replay(root, area, grant, binding, snapshot)
            report = record["identity"]
            scope = "private_fresh_start_timings" if fresh else "private_broader_timings"
        elif matrix:
            record = timing.replay(root, area, binding, grant, snapshot)
            report, scope = record["identity"], "private_update_timings"
        elif supply:
            paid = {
                k: preparation.reopen(root, area, binding, grant, snapshot, k)
                for k in preparation.KINDS
            }
            record = preparation.combined(binding, paid)
            report, scope = record["identity"], "private_donor_supply"
        else:
            record, report, scope = _donation(root, area, binding, grant, snapshot)
        if record != load(path) or (
            saved is not None
            and (saved["receipt"] != record or saved["last_committed_ref"] != record["identity"])
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
