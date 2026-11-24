"""Publish exactly the replayed round027 decision, then resume its accepted campaign."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

ROOT = Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES, digest, load, record, reference, seal, verify_refs,
)
from oeis_learn.generalization.progressive.decisions import verify_gate_producers
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.milestones import gain_cost
from oeis_learn.generalization.progressive.projections import channel_value, describe, project
from oeis_learn.generalization.progressive.storage import owned_guard, register_invocation
from oeis_learn.pipeline_validation import progressive_outputs, progressive_roles
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.preparation_recovery import retain
from oeis_learn.pipeline_validation.progressive_decision_control import frozen_evidence, remaining_slots
from oeis_learn.pipeline_validation.progressive_device_acceptance import consume_receipt
from oeis_learn.pipeline_validation.progressive_oeis import _Derived, check_derived
from oeis_learn.pipeline_validation.progressive_quarantine_control import require_clean_inputs
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
from oeis_learn.pipeline_validation.progressive_workstation_state import read

QUALIFIED = "77667e67176ae32fa16457d52565bae1dbe41afc"
FIX = "f09d5471f88164f766ec97d14ae1b79a61b18be4"
MODULE = "src/oeis_learn/pipeline_validation/progressive_publication.py"
CAMPAIGN = "sha256:b35444dcbdc68693917a61d05ac3dc11efbdebee568a074b80f2596191035304"
REPLAY = "sha256:4d67a539c4f58ba4ec6b9d4002d15b1ba85575b9eee8bd5d0496d211916f20d9"
POLICY = "sha256:7916a40a39ee4b893fce105b27f7a1d6c15f8ae4cc587b68969e97054c9ef342"
SESSION = Path("run-state/discovery/013-research-round-027")
NAME = "diversity-comparison"
ORDINAL = 97
KEY = f"{NAME}/publish/{ORDINAL}"
INVOCATION = "round027-approved-publication"
SECONDS = 240


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replay_receipt(root):
    for path in sorted((root / "run-state/discovery-tools").glob("round027-comparison-*.zip")):
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("replay.json")
            if info.file_size > 16 << 20:
                continue
            value = json.loads(archive.read(info))
        if value.get("identity") == REPLAY and seal(value) == value:
            require(value["stage"] == "PASS" and value["campaign"] == CAMPAIGN, "replay scope differs")
            return value
    raise ValueError("restore the original round027-comparison-*.zip in run-state/discovery-tools")


def original_prefix(path, ref):
    h = hashlib.sha256()
    left = ref["bytes"]
    with path.open("rb") as stream:
        while left:
            block = stream.read(min(left, 1 << 20))
            require(block, "truncated original budget journal")
            h.update(block)
            left -= len(block)
    require("sha256:" + h.hexdigest() == ref["sha256"], "original budget prefix changed")


def charge_state(journal, replay):
    original_prefix(journal.path, replay["ledger"])
    old = replay["budget"]
    current = journal.snapshot()
    with journal.lock():
        events = journal._read()[0]
        tail = events[2 * len(old["committed"]):]
    require(not old["pending"] and events[2 * len(old["committed"]) - 1]["identity"] == old["watermark"],
            "original budget cut differs")
    require([e["event"] for e in tail] in ([], ["RESERVE"], ["RESERVE", "COMMIT"])
            and all(e["operation"] == KEY for e in tail), "another operation has already run")
    costs = dict.fromkeys(old["consumed"], 0)
    costs["elapsed_us"] = SECONDS * 1_000_000
    if tail:
        require(tail[0]["cost"] == costs, "publication reservation differs")
    for k in costs:
        delta = current["consumed"][k] - old["consumed"][k]
        require(delta == (tail[-1]["cost"][k] if tail else 0), "unexpected campaign charge")
    return tail, costs


def original_state(root, store, replay):
    old = read(checked_file(root, replay["transition"]))["state"]
    require(old["outputs"] == replay["outputs"] and old["tasks"].get(NAME) == "FAILED",
            "original comparison failure differs")
    current = store.state
    for key in ("tasks", "outputs"):
        require({k: v for k, v in current[key].items() if k != NAME}
                == {k: v for k, v in old[key].items() if k != NAME}, "campaign advanced outside publication")
    require(current["tasks"].get(NAME) in {"FAILED", "COMPLETE"} and current.get("choices") == old.get("choices"),
            "unexpected task or selection change")
    decisions = dict(current.get("decisions", {}))
    for key, cert in old.get("decisions", {}).items():
        require(decisions.pop(key, None) == cert, "an earlier decision changed")
    require(all(c["bundle"] == replay["decision_bundle"] for c in decisions.values()) and len(decisions) <= 1,
            "unrelated decision appeared")
    return old


def corrected_table(values):
    raw = subprocess.check_output(["git", "show", FIX + ":" + MODULE], cwd=ROOT)
    require(hashlib.sha256(raw).hexdigest() == "effe3ec15ce90935f99a90450fa04c03de5ee467e61b94c926d837903812d3ce",
            "reviewed publication adapter differs")
    namespace = {}
    exec(compile(raw, MODULE, "exec"), namespace)
    table, = namespace["decision_from_outputs"](values)
    return table


def check_table(table, replay):
    data = table["data"]
    bundle = seal(dict(version="progressive-decision-bundle/v1", decision=replay["decision"],
                       **{k: data[k] for k in ("source_slot", "target_slot", "rows", "roster", "seeds", "design", "kwargs")}))
    require(bundle["identity"] == replay["decision_bundle"] and
            replay["decision"]["payload"]["next_action"] == "RETAIN_BASELINE" and
            replay["decision"]["payload"]["evidence_verdict"] == "INCOMPLETE", "replayed rows or decision differ")
    return bundle


def build_input(root, store, task, request, journal, staging, replay):
    with tempfile.TemporaryDirectory(dir=staging, prefix="resolve-") as tmp:
        tmp = Path(tmp)
        views, old_authority = progressive_outputs.resolve(root, store, task, request, tmp)
        require_clean_inputs(root, journal, views)
        permitted = [r["artifact"] for r in store.campaign["payload"]["references"] if r["role"] == "verifier"]
        permitted.extend(check_derived(old_authority, campaign=store.campaign, task=task, operation="publish"))
        project(root, tmp / "original", "publish", views, registry=progressive_roles.REGISTRY, authorized=permitted)
        table = corrected_table([channel_value(root, v) for v in views["public"]])
        bundle = check_table(table, replay)
        require(verify_gate_producers(root, store.campaign, bundle, store=store) == replay["gate_registration"],
                "original producer registration differs")
        path = staging / "decision-input.json"
        retain(path, table)
        artifact = describe(root, str(path.relative_to(root)))
    return artifact, bundle


def terminal(target, expected):
    process = load(target / "supervision/process.json")
    cleanup = load(target / "supervision/cleanup.json")
    result = read(target / "role/result.json")
    require(process["returncode"] == 0 and process.get("cleanup_failure") is None
            and cleanup["within_bound"] is True and cleanup["acknowledged"] is True and cleanup.get("failure") is None,
            "qualified publisher has no clean supervised completion")
    require(result["operation"] == "publish" and result["mechanical_status"] == "COMPLETE"
            and result["next_action"] == "RETAIN_BASELINE" and result["scientific_status"] == "INCOMPLETE",
            "qualified publisher returned a different disposition")
    actual = json.loads(verify_refs(target / "role", [result["metrics"]["decision_bundle"]])[0])
    require(actual == expected, "installed decision bundle differs from replay")
    return result


def finish(root, store, task, journal, target, bundle, replay, old):
    base = target / "role"
    campaign = store.campaign
    require(verify_gate_producers(root, campaign, bundle, store=store) == replay["gate_registration"],
            "original producer check failed")
    policy, panels, designs = frozen_evidence(root, campaign)
    require(bundle["source_slot"] == task["payload"]["slot"], "foreign decision slot")
    remaining = remaining_slots(journal, campaign, policy)
    certificate = store.authorize(bundle, policy=policy, panels=panels, designs=designs,
                                  remaining=remaining, evidence_root=root)
    require(certificate["decision"] == replay["decision"]["identity"] and certificate["bundle"] == bundle["identity"]
            and certificate["source_slot"] == bundle["source_slot"] and certificate["target_slot"] == bundle["target_slot"]
            and certificate["verdict"] == "INCOMPLETE" and certificate["action"] == "RETAIN_BASELINE",
            "decision certificate differs")
    # This disposition authorizes no exploratory spending. Preserve its exact budget cut
    # by byte length, file digest and chain tip; the append-only ledger retains every event.
    retain(base / "decision-audit.json", seal(dict(version="round027-publication-audit/v1",
           bundle=bundle["identity"], kwargs=bundle["kwargs"], prior_decisions=list(old["decisions"].values()),
           replay=REPLAY, original_budget_prefix=replay["ledger"], budget_prefix=reference(root, journal.path),
           watermark=journal.snapshot()["watermark"], authorization=reference(root, target.parent/"authorization.json"))))
    nodes = {n["id"]: n["dependencies"] for n in campaign["payload"]["graph"]["nodes"]}
    source = bundle["source_slot"]
    ancestors = set()
    pending = list(nodes[source])
    while pending:
        parent = pending.pop()
        if parent not in ancestors:
            ancestors.add(parent)
            pending.extend(nodes[parent])
    costs, prior = {}, {}
    for plan in campaign["payload"]["tasks"]:
        if plan["slot"] == source or plan["slot"] in ancestors:
            dest = costs if plan["slot"] == source else prior
            dest[plan["id"]] = plan["caps"]["elapsed_us"] - journal.plan_remaining(plan)["elapsed_us"]
    retain(base / "gain-cost.json", gain_cost(bundle["decision"], bundle["rows"], bundle["roster"], bundle["seeds"],
                                               bundle["design"], costs=costs, ancestor_costs=prior))
    if NAME not in store.state["outputs"]:
        progressive_outputs.commit(root, store, task, base, "publish")
    require(store.state["outputs"][NAME] == dict(task=task["identity"], operation="publish",
                                                  result=reference(root, base / "result.json")), "published cut differs")
    if store.state["tasks"][NAME] != "COMPLETE":
        store.boundary(task, "COMPLETE", INVOCATION)
    next_task = store.next_task()
    require(next_task and next_task["payload"]["id"] == "trial-00-propose", "first screening task differs")


def recover(root, binding, folder, replay):
    from foundation_local import Host
    frozen = folder / "prepare-research/frozen"
    campaign = load(frozen / "campaign.json", maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(campaign["identity"] == CAMPAIGN, "expected frozen round027")
    control = confined(root, "run-state/discovery-control/" + CAMPAIGN[7:])
    account = confined(root, "run-state/discovery-accounting/" + CAMPAIGN[7:])
    require(not (control / "assessment-exposure.json").exists() and not (folder / "assess").exists(), "assessment exposed")
    first = folder / "run-research/execution-0000"
    require(read(first/"result.json")["identity"] == replay["failure"], "original failure differs")
    run = read(first / "run.json")
    require(run["campaign"] == CAMPAIGN and confined(root, run["campaign_path"]) == frozen / "campaign.json",
            "original execution binding differs")
    accepted = confined(root, run["acceptance_path"])
    receipt = read(accepted / "acceptance.json")
    require(receipt["campaign"] == CAMPAIGN and receipt["runtime"] == campaign["payload"]["runtime"]
            and receipt["status"] == "COMPLETE" and receipt["qualified"] is True
            and receipt["scope"] == "progressive-lifecycle", "campaign acceptance differs")
    consume_receipt(accepted, receipt)
    for path in (control / "campaign.json", control / "run.lock", account / "budget-binding.json",
                 account / "budget.lock", account / "journal-storage.json", account / "budget.jsonl"):
        require(path.is_file() and not path.is_symlink(), "missing original campaign file: " + str(path))
    store = RunStore(control, campaign)
    journal = BudgetJournal(account, CAMPAIGN, campaign["payload"]["caps"],
                            journal_bytes=campaign["payload"]["storage"]["journal_bytes"])
    with store.lease():
        old = original_state(root, store, replay)
        tail, costs = charge_state(journal, replay)
        for cut in replay["outputs"].values():
            base = checked_file(root, cut["result"])
            for ref in read(base)["metrics"].values():
                if isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}:
                    checked_file(base.parent, ref)
        for ref in replay["task_bindings"] + replay["storage_inventories"]:
            checked_file(root, ref)
        for cp in replay["training_checkpoints"].values():
            for ref in cp["files"]:
                checked_file(root, ref)
        plan, = [p for p in campaign["payload"]["tasks"] if p["id"] == NAME]
        task = record("task", dict(plan, campaign=CAMPAIGN), created_at=campaign["created_at"],
                      purpose=campaign["purpose"], track=campaign["track"])
        require(plan["policy"] == POLICY, "original publication policy differs")
        req_ref, = [r["artifact"] for r in campaign["payload"]["references"] if r["name"] == "requests/" + NAME]
        require(req_ref["sha256"] == POLICY, "original request differs")
        request = load(checked_file(root, req_ref))
        require(request["operation"] == "publish" and request["reservation"]["native_work"] == 1_000_000
                and journal.task_remaining(task)["native_work"] == 0, "original native work charge differs")
        if not tail:
            left = journal.task_remaining(task)["elapsed_us"]
            require(SECONDS <= request["seconds"] and
                    costs["elapsed_us"] + plan["startup_us"] + plan["finalization_us"] <= left,
                    "insufficient publication time after protecting finalization")
        home = confined(root, str(SESSION / "run-research/publication-recovery"), exists=False)
        home.mkdir(exist_ok=True)
        staging = home / "inputs"
        target = home / "publisher"
        stage_proof = register_invocation(root, campaign, task, account, staging)
        stage_guard = owned_guard(dict(campaign=campaign, task=task, storage=stage_proof), output=staging)
        staging.mkdir(exist_ok=True)
        with stage_guard.activate():
            artifact, bundle = build_input(root, store, task, request, journal, staging, replay)
        stage_guard.check()
        # The new authorization is an explicit amendment to this particular frozen
        # request. It grants one typed input and one elapsed-only installed invocation.
        retain(home / "authorization.json", seal(dict(version="round027-publication-amendment/v1",
               campaign=CAMPAIGN, task=task["identity"], original_request=req_ref, replay=REPLAY,
               original_failure=replay["failure"], original_transition=replay["transition"],
               original_ledger=replay["ledger"], adapter=FIX, input=artifact.ref, operation=KEY,
               reservation=costs, expected_bundle=bundle["identity"], action="RETAIN_BASELINE")))
        if not tail:
            require(not target.exists(), "unowned publisher output already exists")
            runtime = load(frozen / "runtime.json")
            host = Host(SimpleNamespace(**binding, image=request["image"], gpu_base_image=None,
                                        free_disk_gib=5, host_free_memory_gib=24))
            probe = Path(tempfile.mkdtemp(prefix="runtime-", dir=home))
            checked = progressive_roles.current_runtime(host, root, probe, runtime,
                      producer_root=confined(root, runtime["payload"]["qualification"][0]["path"]).parent)
            authority = _Derived(CAMPAIGN, task["identity"], "publish", POLICY, (artifact.ref,))
            print("Running the qualified publisher on the verified comparison table.", flush=True)
            progressive_roles.invoke(host, request["image"], root, "publish", {"public": [artifact]},
                    output=str(target.relative_to(root)), campaign=campaign, task=task, journal=journal,
                    ordinal=ORDINAL, reservation=costs, seconds=SECONDS, runtime=checked, derived=authority)
        require((target / "supervision/cleanup.json").is_file() and (target / "role/result.json").is_file(),
                "publication was reserved but has no complete receipts; preserve " + str(target))
        terminal(target, bundle)
        tail, _ = charge_state(journal, replay)
        if len(tail) == 1:
            # Both terminal receipts survived but the supervisor's final journal write
            # was interrupted. Charge the entire reserved interval, with no time refund.
            journal.commit(KEY)
            tail, _ = charge_state(journal, replay)
        require(len(tail) == 2, "missing completed publication charge")
        proof = register_invocation(root, campaign, task, account, target)
        with owned_guard(dict(campaign=campaign, task=task, storage=proof), output=target).activate():
            finish(root, store, task, journal, target, bundle, replay, old)
        print("Diversity comparison recorded: INCOMPLETE / RETAIN_BASELINE. Next task: trial-00-propose.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue", action="store_true", dest="approved", required=True)
    parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == QUALIFIED, "run from the qualified round027 checkout " + QUALIFIED)
    folder = confined(ROOT, str(SESSION))
    # Serialize two copies of this launcher across the handoff to the workstation,
    # which acquires its own session lock after the publisher releases it.
    with (folder / "round027-publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return continue_session(folder)


def continue_session(folder):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    with (folder/"session.lock").open("rb") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = read(folder/"session.json")["binding"]
        require(binding["source"] == digest(collect_snapshot(ROOT)), "qualified source files changed")
        steps = read(folder/"journal.json")["steps"]
        require(steps["prepare-research/acceptance"]["status"] == "COMPLETE", "retained acceptance required")
        require(not (folder/"run-research/execution-0001").exists()
                and "run-research/invocation-0001" not in steps, "a subsequent research invocation already exists")
        print("Checking round027's saved results, four checkpoints and original ledger.", flush=True)
        recover(ROOT, binding, folder, replay_receipt(ROOT))
    from oeis_learn.pipeline_validation.progressive_workstation import main as workstation
    args = ["--engine", binding["engine"], "--memory-gib", str(binding["memory_gib"]),
            "--cpus", str(binding["cpus"]), "--session", str(SESSION)]
    if binding["render_node"]:
        args.extend(["--render-node", binding["render_node"]])
    code = workstation(ROOT, args + ["run-research"])
    # The workstation's phase-wide exit also counts invocation-0000's retained failure.
    steps = read(folder/"journal.json")["steps"]
    if all(steps.get("run-research/invocation-0001" + suffix, {}).get("status") == "COMPLETE"
           for suffix in ("", "-report", "-verify")):
        return 0
    return code or 2


if __name__ == "__main__":
    raise SystemExit(main())
