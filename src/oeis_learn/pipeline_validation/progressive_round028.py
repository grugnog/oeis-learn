"""Foundation host actions for the registered CPU diagnosis."""
import fcntl
import os
from pathlib import Path
import re
import runpy
import subprocess
import time

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.campaign import diagnostic_registration, learned_search_policy
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import load, read_bound_record, reference, seal
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.program_search_report import public_audit, publish_archive, verify_archive
from .identity import confined
from .preparation_recovery import retain
from .progressive_outputs import round028_batch
from .progressive_round028_audit import Donor, FOLLOWUP, ORIGINAL, SESSION, audit, summarize
from .progressive_research_inputs import checked_file

ARCHIVE = "share/round028.zip"


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def candidate_revision(source, root):
    parent = root / "run-state/discovery-tools/round028-source"
    if source.parent == parent and re.fullmatch("[0-9a-f]{40}", source.name):
        helper = source / "specs/015-learned-program-search/validation/candidate_source.py"
        runpy.run_path(str(helper))["verify_tree"](root, source, source.name)
        return source.name
    if git(source, "rev-parse", "--show-toplevel") != str(source):
        raise ValueError("candidate source must be the checked cache or a clean Git checkout")
    if git(source, "status", "--porcelain", "--untracked-files=all") != "":
        raise ValueError("candidate checkout has uncommitted files")
    return git(source, "rev-parse", "HEAD")


def metadata(root, policy):
    refs = [ORIGINAL + "/prepare-research/frozen/campaign.json", FOLLOWUP + "/share/trial-00-2M.json",
            FOLLOWUP + "/share/1M-replay.json",
            f"run-state/discovery-control/{policy['campaign'][7:]}/transition-0096.json"]
    st = os.statvfs(root)
    head = git(root, "rev-parse", "HEAD")
    return dict(donor=head, expected_donor=policy["donor_revision"],
                qualified_checkout=head == policy["donor_revision"] and git(root, "symbolic-ref", "-q", "HEAD") is None
                and git(root, "rev-parse", "--show-toplevel") == str(root),
                files={p: confined(root, p, exists=False).is_file() for p in refs},
                filesystem=dict(device=root.stat().st_dev, free_bytes=st.f_bavail * st.f_frsize, free_inodes=st.f_favail),
                gpu_demand="unmeasured_pending_audit", gpu_dispatches=0)


def status(folder):
    path = folder / "run.lock"
    busy = False
    if path.is_file():
        with confined(folder, "run.lock").open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                busy = True
    if not (folder / "campaign.json").exists():
        return dict(mechanical_status="UNREGISTERED", scientific_status="UNMEASURED", owner_active=busy)
    binding = load(confined(folder, "campaign.json"))
    store = RunStore(folder, binding, diagnostic=True)
    scientific = "HISTORICAL_AUDIT" if store.state.get("publication") else "UNMEASURED"
    extra = {}
    for path in sorted(folder.glob("private/generator-rehearsal/*/progress.json"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)[:1]:
        progress = load(confined(folder, path.relative_to(folder)))
        if seal(progress) != progress:
            raise ValueError("generator rehearsal progress changed")
        extra["rehearsal"] = {k: progress[k] for k in ("candidate", "phase", "reason", "next_action") if k in progress}
    for panel_path in sorted(folder.glob("private/panel-supply/*/progress.json"),
                             key=lambda p:p.stat().st_mtime_ns, reverse=True)[:1]:
        panel_progress = load(confined(folder,panel_path.relative_to(folder)))
        if seal(panel_progress) != panel_progress:
            raise ValueError("panel supply progress changed")
        extra["panel_supply"] = dict(candidate=panel_path.parent.name,phase=panel_progress["phase"])
    timing_progress = folder / "private/evaluation-timing/progress.json"
    if timing_progress.is_file():
        measured = load(confined(folder,timing_progress))
        if (seal(measured) != measured or type(measured["completed_blocks"]) is not int
                or not 0 <= measured["completed_blocks"] <= 5
                or measured["portfolios"] != 8 * measured["completed_blocks"]):
            raise ValueError("candidate timing progress differs")
        extra["evaluation_timing"] = {k:measured[k] for k in ("completed_blocks","portfolios")}
    return dict(**extra, mechanical_status=store.state["status"], scientific_status=scientific,
                owner_active=busy, boundary=store.tip, decision=store.state.get("route"), active=store.state.get("active"))


def seal_audit(folder, registration, data, ordinal):
    batches = []
    for index, stratum in enumerate(STRATA):
        rows = sorted((r for r in data["cases"] if r["stratum"] == stratum), key=lambda r: r["task"])
        batch = seal(dict(version="round028-cases/v1", registration=registration["identity"], ordinal=index, cases=rows))
        path = confined(folder, "private/batches/" + batch["identity"][7:] + ".json", exists=False)
        retain(path, batch)
        batches.append(dict(authority="session", role="historical_audit", ordinal=index, registration=registration["identity"],
                            artifact=reference(folder, path), identity=batch["identity"]))
    item = seal(dict(version="round028-private-audit/v1", registration=registration["identity"],
                     batches=batches, **{k: v for k, v in data.items() if k != "cases"}))
    path = confined(folder, f"private/attempt-{ordinal}.json", exists=False)
    retain(path, item)
    return reference(folder, path)


def reopen(root, folder, registration, cut, *, recheck_inputs=True):
    value = read_bound_record(folder, cut, version="round028-private-audit/v1")
    if value["registration"] != registration["identity"] or len(value["batches"]) != 4:
        raise ValueError("private audit registration or batch roster differs")
    cases = []
    for i, binding in enumerate(value["batches"]):
        rows = round028_batch(folder, binding, registration=registration["identity"], ordinal=i)["cases"]
        if len(rows) != 32 or any(r["stratum"] != STRATA[i] for r in rows):
            raise ValueError("ordered audit stratum differs")
        cases.extend(rows)
    if len({r["task"] for r in cases}) != 128:
        raise ValueError("duplicate historical case in committed matrix")
    from oeis_learn.generalization.progressive.program_search_routing import project, route
    methods, pairs = summarize(cases, list(value["methods"]))
    votes = [dict(r["diagnosis"], task=r["task"], stratum=r["stratum"]) for r in cases
             if r["stratum"] in registration["policy"]["roster"]["primary"]]
    projected = project(votes, training_support=value["support"]["eligible_rows"] > 0, replay_budget_us=None)
    if (methods != value["methods"] or pairs != value["pairs"] or projected != value["projection"]
            or route(projected) != value["decision"]):
        raise ValueError("private case reduction differs from the sealed report")
    if recheck_inputs:
        for ref in value["inputs"]:
            checked_file(root, ref)
    return dict(value, cases=cases)


def elapsed_clock():
    return time.monotonic_ns() // 1000


def charge(journal, ordinal, start, boot, end):
    # CPU diagnostic reads use no scientific draws, model tokens or device budget.
    # Charge their observed elapsed interval once against the journal's integer counter.
    costs = dict.fromkeys(KEYS, 0)
    costs["elapsed_us"] = end - start
    debit = journal.reserve("historical-audit", "elapsed", ordinal, costs, boot=boot, monotonic_us=start)
    journal.commit(debit["key"], boot=boot, monotonic_us=end, termination_proven=True)


def audit_command(root, folder, store, registration):
    counter = dict.fromkeys(KEYS, 0)
    counter["elapsed_us"] = 2**63 - 1
    journal = BudgetJournal(folder / "accounting", registration["identity"], counter)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if store.state.get("registration") is None:
        store.append(dict(store.state, registration=registration["identity"], status="REGISTERED"))
    private = confined(folder, "private", exists=False)
    private.mkdir(mode=0o700, exist_ok=True)
    if private.stat().st_mode & 0o077:
        raise ValueError("private audit directory must be owner-only")

    active = store.state.get("active")
    if active:
        # An immutable end receipt is written before the debit. Replaying that gap
        # uses the original interval, including after a crash following the debit.
        finish = private / f"finish-{active['ordinal']}.json"
        if not finish.exists():
            retain(finish, seal(dict(ordinal=active["ordinal"], started=active["start"], boot=active["boot"],
                                     end=elapsed_clock() if active["boot"] == boot else None,
                                     basis="interruption_upper_bound" if active["boot"] == boot else "unavailable_cross_boot")))
        settle(store, journal, root, folder, registration, active, finish)

    cut = store.state.get("audit")
    reused = cut is not None
    if cut is None:
        ordinal = store.state.get("attempts", 0)
        active = dict(ordinal=ordinal, start=elapsed_clock(), boot=boot)
        store.append(dict(store.state, status="AUDITING", active=active, attempts=ordinal + 1))
        finish = private / f"finish-{ordinal}.json"
        try:
            data = audit(Donor(root, folder, registration))
            seal_audit(folder, registration, data, ordinal)
        finally:
            retain(finish, seal(dict(ordinal=ordinal, started=active["start"], boot=boot, end=elapsed_clock(),
                                     basis="observed_elapsed")))
            settle(store, journal, root, folder, registration, active, finish)
        cut = store.state["audit"]
    data = reopen(root, folder, registration, cut, recheck_inputs=reused)
    public = public_audit(registration, data, elapsed_us=journal.snapshot()["consumed"]["elapsed_us"],
                          private_identity=data["identity"])
    receipt = publish_archive(confined(folder, ARCHIVE, exists=False), public)
    if not store.state.get("publication"):
        store.append(dict(store.state, status="AUDIT_PUBLISHED", publication=receipt["report"],
                          route=data["decision"]["action"], pending=data["decision"].get("blocker")))
    if receipt["report"] != store.state["publication"]:
        raise ValueError("published report differs from committed audit")
    return dict(mechanical_status="BLOCKED" if data["decision"].get("blocker") else "COMPLETE",
                scientific_status="HISTORICAL_AUDIT", archive=ARCHIVE,
                decision=data["decision"], methods=data["methods"], pairs=data["pairs"]["trial-05-1M vs retained_D"],
                checkpoints=[c["disposition"] for c in data["checkpoints"]],
                missing_inputs=data["missing_inputs"], denominators=dict(historical=128, routing=64),
                unfinished_trial05_composition=public["trial05_unfinished_composition"],
                costs=journal.snapshot()["consumed"], receipt=receipt)


def settle(store, journal, root, folder, registration, active, finish):
    record = load(confined(folder, finish))
    if (seal(record) != record or (record["ordinal"], record["started"], record["boot"])
            != (active["ordinal"], active["start"], active["boot"])):
        raise ValueError("audit accounting boundary differs")
    if record["end"] is not None:
        charge(journal, active["ordinal"], active["start"], active["boot"], record["end"])
    path = confined(folder, f"private/attempt-{active['ordinal']}.json", exists=False)
    cut = None
    if path.exists():
        cut = reference(folder, path)
        reopen(root, folder, registration, cut, recheck_inputs=False)
    store.append(dict(store.state, active=None, audit=cut, status="AUDITED" if cut else "INTERRUPTED",
                      last_timing=reference(folder, finish)))


def audit_store(folder, revision):
    """Resume an unfinished audit on its owning revision or reuse its published result."""
    binding = confined(folder, "campaign.json", exists=False)
    registered = load(binding) if binding.exists() else diagnostic_registration(revision)
    store = RunStore(folder, registered, diagnostic=True)
    if registered["candidate"] != revision:
        if store.state.get("active") or not store.state.get("audit") or not store.state.get("publication"):
            raise ValueError("resume the unfinished audit with its registered implementation")
        published = verify_archive(confined(folder, ARCHIVE))
        if published["report"] != store.state["publication"]:
            raise ValueError("published diagnostic parent differs from its registered boundary")
    return store


def execute(args, source, *, host_factory=None):
    producer_revision = getattr(args, "producer_revision", None)
    if producer_revision is not None and args.action != "recover-rehearsal":
        raise ValueError("producer revision is restricted to original rehearsal recovery")
    if args.archive:
        if args.action != "verify":
            raise ValueError("archive authority is restricted to verification")
        from oeis_learn.generalization.progressive.program_search_rehearsal_report import verify_transport
        return verify_transport(Path(args.archive)), 0
    root = Path(args.workspace)
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise ValueError("absolute physical donor workspace required")
    folder = confined(root, SESSION, exists=False)
    if args.action == "status":
        return status(folder), 0
    policy = learned_search_policy()
    info = metadata(root, policy)
    if args.action == "recover-rehearsal":
        from .progressive_round028_receipts import recover_command
        if not info["qualified_checkout"]:
            return dict(mechanical_status="BLOCKED", reason="ORIGINAL_DONOR_REQUIRED", gpu_dispatches=0), 2
        candidate_revision(source, root)
        return recover_command(root, folder, source, producer_revision)
    if args.action == "preflight":
        return dict(info, mechanical_status="INSPECTED", scope="historical_audit",
                    admission="AVAILABLE" if info["qualified_checkout"] and all(info["files"].values()) else "BLOCKED"), 0
    if not info["qualified_checkout"] or not info["files"][ORIGINAL + "/prepare-research/frozen/campaign.json"]:
        return dict(info, mechanical_status="BLOCKED", scientific_status="UNMEASURED",
                    missing="qualified round027 donor and original frozen campaign required",
                    authenticated_cases=0, expected_cases=128), 2
    revision = candidate_revision(source, root)
    if args.action == "verify" and not (folder / ARCHIVE).is_file():
        return dict(mechanical_status="BLOCKED", missing="audit publication"), 2
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with RunStore.owner(folder):
            store = audit_store(folder, revision)
            registration = store.campaign
            if args.action == "verify":
                if not store.state.get("publication"):
                    raise ValueError("uncommitted public audit")
                private = reopen(root, folder, registration, store.state["audit"])
                repeated = audit(Donor(root, folder, registration))
                original = {r["task"]: r for r in private["cases"]}
                if (original != {r["task"]: r for r in repeated["cases"]}
                        or any(private[k] != v for k, v in repeated.items() if k != "cases")):
                    raise ValueError("private donor replay differs")
                receipt = verify_archive(confined(folder, ARCHIVE))
                journal = BudgetJournal(folder / "accounting", registration["identity"],
                                        dict(dict.fromkeys(KEYS, 0), elapsed_us=2**63 - 1))
                rebuilt = public_audit(registration, repeated, elapsed_us=journal.snapshot()["consumed"]["elapsed_us"],
                                       private_identity=private["identity"])
                if receipt["report"] != store.state["publication"] or rebuilt["identity"] != receipt["report"]:
                    raise ValueError("private/public audit mismatch")
                return dict(private_replay="VERIFIED", public_recomputation=receipt), 0
            from .progressive_round028_panels import PRODUCER, panels_command
            from .progressive_round028_timing import AREA, PANEL_ARCHIVE, timing_command
            if args.action == "timing" or (args.action == "run" and (folder / PANEL_ARCHIVE).is_file()):
                try:
                    observed, code = timing_command(args,source,root,folder,revision,store,host_factory)
                except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
                    import traceback
                    from .progressive_round028_rehearsal import failure_locations
                    area = confined(folder,AREA,exists=False)
                    area.mkdir(parents=True,mode=0o700,exist_ok=True)
                    (area/"exception.txt").write_text(traceback.format_exc())
                    return dict(mechanical_status="BLOCKED", scientific_status="PILOT_UNMEASURED",
                                reason="CANDIDATE_TIMING_COLLECTION_FAILED", error_type=type(exc).__name__,
                                failure_locations=failure_locations(source,area,role_folder="roles")), 2
                if args.action == "run":
                    return dict(observed,mechanical_status="BLOCKED",next_action=observed["reason"]), 2
                return observed,code
            prior_training = folder / "private/generator-rehearsal" / PRODUCER / "progress.json"
            if args.action == "panels" or (args.action == "run" and prior_training.is_file()
                                          and store.state.get("route") == "MATCHED_GENERATOR"):
                if not store.state.get("publication") or store.state.get("route") != "MATCHED_GENERATOR":
                    return dict(mechanical_status="BLOCKED", reason="MATCHED_GENERATOR_AUDIT_REQUIRED",
                                gpu_dispatches=0), 2
                try:
                    observed, code = panels_command(args, source, root, folder, revision, store, host_factory)
                except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
                    import traceback
                    from .progressive_round028_rehearsal import failure_locations
                    area = confined(folder,"private/panel-supply/"+revision,exists=False)
                    area.mkdir(parents=True,mode=0o700,exist_ok=True)
                    (area/"exception.txt").write_text(traceback.format_exc())
                    return dict(mechanical_status="BLOCKED", scientific_status="PILOT_UNMEASURED",
                                reason="PANEL_SUPPLY_COLLECTION_FAILED", error_type=type(exc).__name__,
                                failure_locations=failure_locations(source,area,role_folder="roles"), gpu_dispatches=0), 2
                if args.action == "run":
                    return dict(observed,mechanical_status="BLOCKED",next_action=observed["reason"]), 2
                return observed, code
            result = audit_command(root, folder, store, registration)
            if result["decision"].get("blocker"):
                return result, 2
            if args.action in {"run", "rehearse"} and result["decision"]["action"] == "MATCHED_GENERATOR":
                from .progressive_round028_rehearsal import rehearse
                observed, code = rehearse(args, root, source, folder, store, revision, result,
                                         host_factory=host_factory)
                if args.action == "rehearse" or code:
                    return observed, code
                return dict(observed, mechanical_status="BLOCKED", reason="PROSPECTIVE_ADMISSION_PENDING",
                            missing="fresh panel freeze, evaluation timing and campaign acceptance"), 2
            if args.action == "rehearse":
                return dict(result, mechanical_status="SKIPPED", reason="GENERATOR_NOT_SELECTED"), 0
            if args.action == "run" and result["decision"]["action"] != "RETAIN_REFERENCE":
                return dict(result, mechanical_status="BLOCKED", missing="registered next-stage execution"), 2
            return result, 0
    except ValueError as exc:
        if str(exc) != "another invocation owns this campaign":
            raise
        return dict(status(folder), mechanical_status="OWNED"), 2
