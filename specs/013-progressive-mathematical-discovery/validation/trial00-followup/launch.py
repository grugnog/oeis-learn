"""Run or verify the separately registered exploratory trial 00 follow-up."""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.pipeline_validation.identity import confined, file_hash
from oeis_learn.pipeline_validation.preparation_recovery import retain
from evidence import audit, need, prompts
from execution import accept, check_space, controller, execute, import_parent, receipt, task_for, task_request
from registration import (
    CID,
    HOME,
    LOOKS,
    PIN,
    RESTORE_HASH,
    SESSION,
    freeze,
    names,
    original_import,
    source,
)
from reports import history, publish_look


def run(root, restore_receipt, verify_only):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    need(head == PIN, "expected the qualified round027 revision " + PIN)
    binding = audit["saved"](root / SESSION / "session.json")["binding"]
    need(binding["source"] == digest(collect_snapshot(root)), "qualified package source changed")
    home = confined(root, HOME, exists=False)
    if verify_only:
        need((home / "share/1M-replay.json").is_file(), "no published follow-up to verify")
    home.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        for path, mode in ((home / "launcher.lock", "a"), (root / SESSION / "session.lock", "rb"),
                           (root / "run-state/discovery-control" / CID[7:] / "run.lock", "rb")):
            fp = locks.enter_context(path.open(mode))
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store, states = source(root)
        print("Checking the original full-state checkpoint and settled parent charges.", flush=True)
        saved = home / "original-screening-restore.json"
        if not saved.exists():
            need(restore_receipt.is_file() and file_hash(restore_receipt) == RESTORE_HASH,
                 "original HIP restore receipt differs")
            # Retain the exact original bytes. A rerun may use this local copy.
            from oeis_learn.generalization.progressive.contracts import _atomic_write
            _atomic_write(saved, restore_receipt.read_bytes())
        need(file_hash(saved) == RESTORE_HASH, "retained HIP restore receipt differs")
        init, spec, refs = original_import(root, home, store)
        c = freeze(root, home, store, states[0]["proposal"], init, spec)
        prompt = prompts(root, store.campaign)
        one_m, replayed = history(root, home, store, states, prompt, replay=verify_only)
        print("Verified six 1M looks. Frozen passes: none. Companion passes: trial 00.", flush=True)
        if not verify_only:
            # A completed follow-up can be checked offline, even if disk space
            # has since been spent or the engine is no longer running.
            for coord in reversed(LOOKS):
                final = home / "share" / f"trial-00-{coord//1000000}M.json"
                if final.is_file() and audit["saved"](
                        final)["next_action"] == f"STOP_AT_{coord//1000000}M":
                    verify_only = True
                    break
        if verify_only and not (home / "acceptance/acceptance.json").is_file():
            return
        host = None
        child, journal = controller(root, c)
        locks.enter_context(child.lease())
        if not child.state.get("choices"):
            need(not verify_only, "missing follow-up choices")
            child.choose_trial(states[0]["proposal"])
        need(not (child.root / "assessment-exposure.json").exists(), "locked assessment exposed")
        if verify_only:
            receipt(home / "acceptance", c)
        else:
            from foundation_local import Host
            req = task_request(root, c, task_for(child, names(LOOKS[0])[0]))
            host = Host(
                SimpleNamespace(**binding,
                                image=req["image"],
                                gpu_base_image=None,
                                free_disk_gib=5,
                                host_free_memory_gib=24))
            check_space(home, c,
                        [home / "tasks", root / "run-state/discovery-learning" / c["identity"][7:]])
            print("Checking campaign-bound CPU/native and HIP acceptance.", flush=True)
            accept(root, home, c, host)
            import_parent(root, home, child, journal, host, req["image"], spec, refs)
        drive_looks(root, home, store, child, journal, host, one_m, replayed, prompt, verify_only)


def drive_looks(root, home, store, child, journal, host, one_m, replayed, prompt, verify_only):
    cache = {}
    prior, report = one_m, replayed
    for coord in LOOKS:
        path = home / "share" / f"trial-00-{coord//1000000}M.json"
        if verify_only and not path.exists():
            need(
                not all(child.state["tasks"].get(n) == "COMPLETE" for n in names(coord)),
                "all workers are complete but this report is missing; use --run to publish " +
                str(path))
            print("Verified published boundaries. The next look has no published report yet.",
                  flush=True)
            break
        for name in names(coord):
            if verify_only:
                need(child.state["tasks"].get(name) == "COMPLETE",
                     "incomplete published task: " + name)
            else:
                print("Checking or executing " + name, flush=True)
                execute(root, home, child, journal, host, cache, name)
        prior, report = publish_look(root,
                                     home,
                                     store,
                                     child,
                                     journal,
                                     coord,
                                     one_m,
                                     prior,
                                     report,
                                     prompt,
                                     replay=verify_only)
        print(f"Verified {path}. Decision: {report['next_action']}", flush=True)
        if report["next_action"] != "CONTINUE_TO_4M":
            need(
                not any(k.startswith("trial-00-4000000-") for k in child.state["outputs"])
                or coord == LOOKS[1], "4M work exists beyond registered stop")
            break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify",
                      action="store_true",
                      help="reproduce all published measurements from original evidence")
    p.add_argument("--restore-receipt",
                   type=Path,
                   default=ROOT.parent /
                   "round027-trial00-restore-g813b4yq/original-screening-restore.json")
    args = p.parse_args()
    try:
        run(ROOT, args.restore_receipt, args.verify)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if args.run and (ROOT / HOME / "registration.json").is_file():
            try:
                retain(
                    ROOT / HOME / "private/failures" / f"{time.time_ns()}.json",
                    seal(
                        dict(error=type(exc).__name__,
                             detail=str(exc),
                             operation="trial00-followup")))
            except OSError:
                pass
        p.exit(2, f"Follow-up blocked: {exc}\nEvidence directory: {ROOT/HOME}\n")


if __name__ == "__main__":
    main()
