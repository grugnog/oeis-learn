"""Freeze a separately accounted continuation from the accepted round027 templates."""
import json
import os
from pathlib import Path

from oeis_learn.generalization.progressive.campaign import (
    frozen_task_records,
    initializer_origin,
    validate_campaign,
    validate_task,
)
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    record,
    reference,
    seal,
    verify_refs,
)
from oeis_learn.generalization.progressive.import_state import inspect
from oeis_learn.generalization.progressive.iteration import validate_producer
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.trial_scopes import materialize, resolve
from oeis_learn.pipeline_validation.identity import confined, file_hash
from oeis_learn.pipeline_validation.preparation_recovery import retain
from oeis_learn.pipeline_validation.progressive_device_acceptance import consume_receipt
from oeis_learn.pipeline_validation.progressive_outputs import validate_requests
from oeis_learn.pipeline_validation.progressive_study import original
from evidence import READER_HASH, audit, need

PIN = "77667e67176ae32fa16457d52565bae1dbe41afc"
CID = audit["CAMPAIGN"]
SESSION = Path("run-state/discovery/013-research-round-027")
HOME = Path("run-state/discovery/013-round027-trial00-followup")
STUDY = "sha256:416978cd5bd6e48434224d42a59cc03a63ac81f83d9f3b0cd400b9e802620795"
STUDY_TIP = "sha256:9e948130aefa271aa22328c40ad4551b4f42398576fee954e6dcd48c9b07e53b"
CONTROL_TIP = "sha256:f44d46fad1ec950a91f23a6ecb0f1c5aa14a611729bb3bdafaf5c2ba83a06b5b"
PARENT = "trial-00-1000000-train"
CHECKPOINT = "sha256:7c2b666225e179685ebf88245c1e536b8460c33d6c452ddd99e1d76bec4fc0b4"
ACCOUNT = "sha256:a69de2e8a1fd9af84f001d13282685310d83f61ec1ef50cbcd14512c2c0f5f2d"
LEDGER = "sha256:a16d0db8bed3b78ccd8115583cef8ef2a1c31293a3ff9cce908646fe07cc777e"
RESTORE_HASH = "sha256:56382afce7a0ca2aae7588dcc8a129caeb81287c4657b54dafe45b59058cc572"
TASK_IDS = {
    1000000: "baf4ae65d7c4625be96ed9d3de285310a2ad6d956f32ae5b3da4564f9a322ddd",
    2000000: "083a1295315b1cdf2514c54285ac1dd10250624c48aad785f5dafd21befee3ba",
    4000000: "90d4649c533e32731f58bfae32c32a38e0b292d8728cebf8b813462a9698f8b1"
}
PARAMS = dict(dropout=0,
              effective_batch=16,
              lr_multiplier=1,
              schedule="constant",
              weight_decay=0.01)
LOOKS = (2000000, 4000000)
SUFFIXES = ("train", "retention-generate", "retention-score", "development-generate",
            "development-score")
RULE = dict(
    selection="exploratory, retrospectively selected from six original infeasible trials",
    coordinates=list(LOOKS),
    advance="verified 2M companion passes; always stop at 4M",
    development="at least 95 percent complete in each original 32-case screening stratum",
    retention="at most two percentage points conservative loss in 128 paired cases",
    companion=
    "fully evidenced 1024-token no-EOS or independently witnessed control_range; all other gaps stay unresolved"
)


def names(coordinate):
    return [f"trial-00-{coordinate}-{suffix}" for suffix in SUFFIXES]


def task_record(c, plan):
    return record("task",
                  dict(plan, campaign=c["identity"]),
                  created_at=c["created_at"],
                  purpose=c["purpose"],
                  track=c["track"])


def one_ref(c, name):
    (ref, ) = [r["artifact"] for r in c["payload"]["references"] if r["name"] == name]
    return ref


def content(root, ref):
    return json.loads(verify_refs(root, [ref], max_bytes=CAMPAIGN_DOCUMENT_BYTES)[0])


def source(root):
    folder = root / SESSION
    frozen = folder / "prepare-research/frozen"
    c = audit["saved"](frozen / "campaign.json")
    need(
        c["identity"] == CID and audit["saved"](frozen / "result.json")["identity"]
        == "sha256:3e616f8997234a8b052df2d8a1f4158fd320b8fd68a9f7a713e1abdfd310d75d",
        "wrong source freeze")
    run = audit["saved"](folder / "run-research/execution-0000/run.json")
    accepted = confined(root, run["acceptance_path"])
    receipt = audit["saved"](accepted / "acceptance.json")
    need(
        receipt["identity"]
        == "sha256:380bb65b32fd92c158bbf3b1b7a93be89a57b36c35e9602aab90a45d134167f2"
        and run["campaign"] == receipt["campaign"] == CID
        and confined(root, run["campaign_path"]) == frozen / "campaign.json",
        "original acceptance binding differs")
    consume_receipt(accepted, receipt)
    control = confined(root, "run-state/discovery-control/" + CID[7:])
    need((control / "campaign.json").is_file()
         and not (control / "assessment-exposure.json").exists()
         and not (folder / "assess").exists(), "original ownership or locked exposure differs")
    store = RunStore(control, c)
    states, tip = audit["study_results"](root, c)
    need(tip == STUDY_TIP and store.tip == CONTROL_TIP and set(range(6)) <= states.keys(),
         "paused six-trial cut differs")
    need(
        file_hash(root / "run-state/discovery-accounting" / CID[7:] / "budget.jsonl") == LEDGER,
        "paused journal differs")
    proposal = audit["checked"](states[0]["proposal"])
    need(
        proposal["study"] == STUDY and proposal["ordinal"] == 0 and proposal["parameters"] == PARAMS
        and proposal["seed"] == 3123489707, "saved baseline proposal differs")
    return store, states


def original_import(root, home, store):
    c = store.campaign
    p = next(t for t in c["payload"]["tasks"] if t["id"] == PARENT)
    task = task_record(c, store.resolved_plan(p))
    need(task["identity"] == "sha256:" + TASK_IDS[1000000], "wrong original 1M parent")
    cp = original(root, store, PARENT, "train", "checkpoint_record")
    final = confined(root, cp["path"]).parent / "accounting-final.json"
    state_root = str(
        Path("run-state/discovery-learning") / CID[7:] / TASK_IDS[1000000] / "checkpoints")
    spec = dict(artifact=cp, accounting=reference(root, final), state_root=state_root)
    state, costs, _, full, refs = inspect(root, spec)
    need((state["identity"], costs["identity"], full["ledger"]["consumed"],
          full["ledger"]["updates"],
          costs["milestone"]["total_cost_us"]) == (CHECKPOINT, ACCOUNT, 1000564, 1278, 578661207),
         "original checkpoint or settled charges differ")
    retain(home / "source-task.json", task)
    origin = seal(
        dict(version="progressive-initializer-origin/v1",
             checkpoint=cp,
             state=cp,
             state_root=state_root,
             campaign=reference(root, root / SESSION / "prepare-research/frozen/campaign.json"),
             task=reference(root, home / "source-task.json"),
             exposure=one_ref(c, "exposure"),
             parent=None))
    retain(home / "origin.json", origin)
    init = dict(mode="full_state",
                artifact=cp,
                origin=reference(root, home / "origin.json"),
                compatibility=task["payload"]["initializer"]["compatibility"])
    initializer_origin(root, init, track=c["track"], purpose=c["purpose"])
    return init, spec, refs


def plans_for(source_store, init):
    c = source_store.campaign
    selected = []
    for name in [PARENT, *names(LOOKS[0]), *names(LOOKS[1])]:
        template = next(t for t in c["payload"]["tasks"] if t["id"] == name)
        plan = source_store.resolved_plan(template)
        need(plan and template["trial_choices"][plan["trial_choice"]]["parameters"] == PARAMS,
             "original chosen parameters differ: " + name)
        if plan["kind"] == "training":
            need(
                task_record(c, plan)["identity"] == "sha256:" + TASK_IDS[plan["token_target"]],
                "training task differs")
        # Removing the unused alternatives shrinks every worker authority. The installed
        # selector still materializes the exact original training package and update law.
        plan = dict(plan,
                    output=str(HOME / ("import" if name == PARENT else "tasks") / name),
                    trial_choices=[template["trial_choices"][plan["trial_choice"]]])
        plan.pop("trial_choice")
        if name == PARENT:
            plan.update(initializer=init, caps=dict(plan["caps"], tokens=0))
        selected.append(plan)
    return selected


def compile_campaign(c, plans, registration, refs, availability):
    p = c["payload"]
    storage = dict(p["storage"])
    storage["task_checkpoints"] = [
        r for r in storage["task_checkpoints"] if r["task"] in {t["id"]
                                                                for t in plans}
    ]
    storage["milestone_pins"] = 0
    storage["task_store_bytes"] = sum(t["caps"]["disk_bytes"] for t in plans)
    required = sum(storage[k] for k in ("task_store_bytes", "projection_bytes", "shard_bytes",
                                        "journal_bytes", "package_bytes"))
    original_inodes_per_task = (p["storage"]["required_inodes"] - 16384) // len(p["tasks"])
    inodes = 16384 + len(plans) * max(original_inodes_per_task,
                                      2 * storage["checkpoint_inodes"] + 256)
    need(
        availability["bytes"] >= required and availability["inodes"] >= inodes,
        f"follow-up needs {required:,} free bytes and {inodes:,} inodes for registration; "
        f"available {availability['bytes']:,} bytes and {availability['inodes']:,} inodes")
    storage.update(required_bytes=required,
                   required_inodes=inodes,
                   available_bytes=availability["bytes"],
                   available_inodes=availability["inodes"])
    costs = {k: sum(t["caps"][k] for t in plans) for k in p["caps"]}
    costs["elapsed_us"] += plans[0]["trial"]["shared_finalization_us"]
    zero = dict.fromkeys(costs, 0)
    nodes = [
        dict(n, caps=costs if n["id"] == "tuning-screening" else zero) for n in p["graph"]["nodes"]
        if n["id"] in {"capability", "diversity", "tuning-screening"}
    ]
    graph = dict(nodes=nodes,
                 edges=[],
                 diagnostic_pool=zero,
                 selected_stages=["capability", "diversity", "tuning"])
    payload = dict(p,
                   tasks=plans,
                   graph=graph,
                   storage=storage,
                   caps=dict(costs, disk_bytes=required),
                   references=refs,
                   blueprint=registration["identity"])
    campaign = record("campaign",
                      payload,
                      parents=[c["identity"], registration["identity"]],
                      created_at="2026-11-25T00:00:00Z",
                      purpose=c["purpose"],
                      track=c["track"])
    validate_campaign(campaign)
    frozen_task_records(campaign)
    return campaign


def freeze(root, home, store, proposal, init, spec):
    c = store.campaign
    plans = plans_for(store, init)
    by_policy = {
        r["artifact"]["sha256"]: r
        for r in c["payload"]["references"] if r["kind"] == "policy"
    }
    paths, runtime_refs = set(), []
    calibrations = []
    for plan in plans:
        runtime_refs.append(plan["trial_choices"][0]["runtime"])
        if plan["kind"] == "training":
            producer = plan["iteration_budget"]["producer"]
            if producer not in calibrations:
                validate_producer(root, resolve(plan, 0), content(root, runtime_refs[-1]),
                                  c["payload"]["references"])
                calibrations.append(producer)
        if plan["id"] == PARENT:
            continue
        req = content(root, by_policy[plan["policy"]]["artifact"])
        op = plan["id"].rsplit("-", 1)[-1]
        need(req["operation"] == op and op in {"train", "generate", "score"},
             "unregistered worker request")
        if op == "train":
            # Reconstruct both future packages before reserving any new GPU work.
            (channel, ) = req["artifacts"]["training"]
            (entry, ) = [
                r["artifact"] for r in c["payload"]["references"]
                if r["role"] == "learner" and r["artifact"]["path"] == channel
            ]
            envelope = audit["checked"](content(root, entry))
            from oeis_learn.pipeline_validation.progressive_learner import compatibility, validate_input
            base = json.loads(
                verify_refs(root, [envelope["data"]["artifact"]], max_bytes=256 << 20)[0])
            need(
                envelope["kind"] == "training_request"
                and compatibility(base) == envelope["data"]["compatibility"],
                "frozen training envelope differs")
            variant = resolve(plan, 0)
            package = materialize(base, variant)
            validate_input(package,
                           dict(purpose=c["purpose"], track=c["track"], payload=variant),
                           allow_pending=True)
            need(package["update_policy"]["schedule_horizon_tokens"] == 4000000,
                 "training horizon changed")
        for items in req["artifacts"].values():
            paths.update(i for i in items if isinstance(i, str))
    policy_ids = {p["policy"] for p in plans}
    refs = [
        r for r in c["payload"]["references"] if r["name"] in {
            "research-reservation-v2", "research_resource_policy", "acceptance_plan",
            "study-policy", "calibration_reuse", "exposure"
        } or r["artifact"]["path"] in paths or r["artifact"] in runtime_refs
        or r["artifact"] in calibrations or r["artifact"]["sha256"] in policy_ids
    ]
    registration = seal(
        dict(version="trial00-followup/v1",
             campaign=CID,
             study=STUDY,
             study_tip=STUDY_TIP,
             controller=CONTROL_TIP,
             proposal=proposal,
             source=spec,
             checkpoint=CHECKPOINT,
             ledger=LEDGER,
             restore=RESTORE_HASH,
             evidence_reader=READER_HASH,
             policy=RULE,
             donor=PARENT,
             operations=[t["id"] for t in plans[1:]],
             implementation={
                 f.name: file_hash(f)
                 for f in sorted(Path(__file__).parent.glob("*.py"))
                 if not f.name.startswith("test_")
             },
             audit=file_hash(Path(__file__).resolve().parents[1] / "audit-round027-screening.py")))
    snapshot = home / "storage-at-freeze.json"
    if snapshot.exists():
        available = audit["saved"](snapshot)
    else:
        fs = os.statvfs(home)
        available = seal(dict(bytes=fs.f_bavail * fs.f_frsize, inodes=fs.f_favail))
    campaign = compile_campaign(c, plans, registration, refs, available)
    retain(snapshot, available)
    validate_requests(root,
                      plans[1:],
                      refs,
                      campaign["payload"]["graph"],
                      purpose=c["purpose"],
                      track=c["track"])
    for plan in plans:
        validate_task(task_record(campaign, resolve(plan, 0)), campaign=campaign)
    retain(home / "registration.json", registration)
    retain(home / "frozen/campaign.json", campaign)
    retain(home / "frozen/runtime.json",
           audit["saved"](root / SESSION / "prepare-research/frozen/runtime.json"))
    return campaign
