"""Supervisor-proven crash recovery; integrity failures never become resume authority."""

import json
from .identity import confined
from oeis_learn.generalization.progressive.contracts import load, verify_refs
from oeis_learn.generalization.progressive.recovery import Recovery


def recover(root, campaign, task, journal, output, error):
    if isinstance(error, ValueError):
        return None
    output = confined(root, str(output.relative_to(root)),exists=False)
    process = output / "supervision/process.json"
    cleanup = output / "supervision/cleanup.json"
    if not process.is_file() or not cleanup.is_file():
        return None
    p = json.loads(process.read_text())
    q = json.loads(cleanup.read_text())
    code = p.get("returncode")
    interrupted = (type(code) is int and (code < 0 or 128 <= code <= 192)) or (
        p.get("failure") or ""
    ).startswith(("TimeoutError:", "KeyboardInterrupt"))
    if (
        not interrupted
        or p.get("cleanup_failure") is not None
        or q.get("within_bound") is not True
        or q.get("acknowledged") is not True
    ):
        return None
    # A completed budget interval with a same-boot end is the authoritative proof.
    with journal.lock():
        events, pending, finished, _, _ = journal._read()
    prefix = task["payload"]["id"] + "/train/"
    proofs = [e for e in events if e["event"] == "COMMIT" and e["operation"].startswith(prefix)]
    if not proofs or proofs[-1]["end_us"] is None:
        return None
    state = (
        root
        / "run-state/discovery-learning"
        / campaign["identity"].split(":")[1]
        / task["identity"].split(":")[1]
    )
    if not (state / "binding.json").is_file():
        return None
    with Recovery(
        journal, task, task["payload"]["initializer"]["compatibility"], state_root=state
    ) as recovery:
        if (
            recovery.state["status"] in {"FAILED", "COMPLETE", "PARTIAL"}
            or recovery.state["checkpoint"] is None
        ):
            return None
        record = recovery.state["checkpoint"]
        from oeis_learn.generalization.progressive.state import restore

        cp = restore(
            record,
            root=recovery.store.root,
            compatibility=recovery.compatibility,
            campaign=campaign["identity"],
            task=task["identity"],
        )
        full = json.loads(
            verify_refs(recovery.store.root, [cp["full_state"]], max_bytes=64 << 20)[0]
        )
        from oeis_learn.generalization.state import restore as verify_full

        verify_full(
            full, contract=recovery.compatibility, cap=recovery.token_cap, root=recovery.store.root
        )
        for key, event in pending.items():
            if (
                key.startswith(task["payload"]["id"] + "/update/")
                and event["cost"]["elapsed_us"] == 0
            ):
                journal.commit(key)
        journal.reconcile(task)
        remaining = journal.task_remaining(task)
        from oeis_learn.generalization.progressive.iteration import margin

        status = (
            "PAUSED"
            if remaining["tokens"]
            and remaining["elapsed_us"]
            > task["payload"]["startup_us"]
            + task["payload"]["finalization_us"]
            + margin(task["payload"], research=task["purpose"] == "research")
            else "PARTIAL"
        )
        recovery.append(status=status)
        return status
