"""Run the registered original workers with fresh campaign ownership and bounded charges."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid

from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.storage_plan import checkpoint_bound
from oeis_learn.pipeline_validation import progressive_outputs, progressive_roles, progressive_study
from oeis_learn.pipeline_validation.generalization_roles import supervise_owned
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.preparation_recovery import retain
from oeis_learn.pipeline_validation.progressive_device_acceptance import consume_receipt
from oeis_learn.pipeline_validation.progressive_evidence import finalize_learning
from oeis_learn.pipeline_validation.progressive_handoff import acceptance_command
from oeis_learn.pipeline_validation.progressive_quarantine_control import require_clean_inputs
from registration import CHECKPOINT, PARENT, audit, content, need, one_ref, task_record


def receipt(root, campaign):
    value = consume_receipt(root, audit["saved"](root / "acceptance.json"))
    need(
        value["campaign"] == campaign["identity"]
        and value["runtime"] == campaign["payload"]["runtime"],
        "follow-up acceptance belongs to a different campaign or runtime")
    return value


def accept(root, home, c, host):
    target = home / "acceptance"
    if not (target / "acceptance.json").exists():
        # A power loss between package export and publication can leave the complete
        # original producer cut behind. Republish it only after ordinary verification.
        if (target / "acceptance.pending.json").is_file():
            from oeis_learn.pipeline_validation.progressive_device_acceptance import export_package
            pending = consume_receipt(target, audit["saved"](target / "acceptance.pending.json"))
            need(
                pending["campaign"] == c["identity"]
                and pending["runtime"] == c["payload"]["runtime"], "foreign pending acceptance")
            plan = content(root, one_ref(c, "acceptance_plan"))
            q = content(root, plan["qualification"])
            if not (target / "diagnostics.zip").exists():
                export_package(target,
                               confined(root, plan["qualification"]["path"]).parent,
                               q,
                               receipt_path=target / "acceptance.pending.json")
            import zipfile
            with zipfile.ZipFile(target / "diagnostics.zip") as archive:
                manifest = audit["checked"](json.loads(archive.read("manifest.json")))
                import hashlib
                need(
                    archive.testzip() is None
                    and json.loads(archive.read("acceptance.json")) == pending
                    and all("sha256:" +
                            hashlib.sha256(archive.read(r["path"])).hexdigest() == r["sha256"]
                            for r in manifest["files"]), "acceptance package incomplete")
            retain(target / "acceptance.json", pending)
        else:
            need(not target.exists(),
                 "incomplete native acceptance; originals retained in " + str(target))
            args = SimpleNamespace(campaign=str((home / "frozen/campaign.json").relative_to(root)),
                                   output=str(target.relative_to(root)))
            _, code = acceptance_command(args, root, lambda _: host)
            need(code == 0, "new physical acceptance did not complete")
    return receipt(target, c)


def controller(root, campaign):
    tag = campaign["identity"][7:]
    return (RunStore(root / "run-state/discovery-control" / tag, campaign),
            BudgetJournal(root / "run-state/discovery-accounting" / tag,
                          campaign["identity"],
                          campaign["payload"]["caps"],
                          journal_bytes=campaign["payload"]["storage"]["journal_bytes"]))


def task_for(store, name):
    template = next(t for t in store.campaign["payload"]["tasks"] if t["id"] == name)
    return task_record(store.campaign, store.resolved_plan(template))


def state_dir(root, task):
    return root / "run-state/discovery-learning" / task["payload"]["campaign"][7:] / task[
        "identity"][7:]


def process_complete(target):
    process = load(target / "supervision/process.json")
    cleanup = load(target / "supervision/cleanup.json")
    need(
        process.get("outcome") == "COMPLETE" and process.get("returncode") == 0
        and process.get("cleanup_failure") is None and cleanup.get("within_bound") is True
        and cleanup.get("acknowledged") is True,
        "worker termination and cleanup must be proven: " + str(target))


def import_parent(root, home, store, journal, host, image, spec, refs):
    from oeis_learn.container_runtime import private_runtime_environment
    task = task_for(store, PARENT)
    state = state_dir(root, task)
    state.mkdir(parents=True, exist_ok=True)
    target = home / "import"
    target.mkdir(exist_ok=True)
    storage = store.campaign["payload"]["storage"]
    request = seal(
        dict(task=task,
             parent=CHECKPOINT,
             source=spec,
             caps=store.campaign["payload"]["caps"],
             journal_bytes=storage["journal_bytes"],
             checkpoint_bytes=checkpoint_bound(storage, task["payload"])["checkpoint_bytes"]))
    retain(target / "request.json", request)
    owner = target / "worker"
    if not owner.exists():
        owner.mkdir()
        mounts = [(str(confined(root, r["path"])), "/source/" + r["path"], False) for r in refs]
        mounts.extend([(str(target / "request.json"), "/inputs/import.json", False),
                       (str(Path(__file__).with_name("import_parent.py").resolve()),
                        "/inputs/import_parent.py", False), (str(state), "/learning-state", True),
                       (str(journal.root), "/accounting", True),
                       (str(owner), "/work/output", True)])
        name = "trial00-import-" + uuid.uuid4().hex
        command = host.engine.command(image=image,
                                      name=name,
                                      mounts=mounts,
                                      argv=["python", "/inputs/import_parent.py"],
                                      network=False,
                                      gpu=False,
                                      memory_bytes=host.policy["memory_bytes"],
                                      cpus=host.policy["cpus"],
                                      tmp_bytes=2 << 30,
                                      environment=private_runtime_environment(owner))
        supervise_owned(host,
                        name,
                        command,
                        owner,
                        deadline=time.monotonic() + task["payload"]["caps"]["elapsed_us"] / 1000000,
                        cleanup_seconds=60)
    process_complete(owner)
    value = audit["saved"](state / "import.json")
    need(
        value["task"] == task["identity"] and value["parent"] == CHECKPOINT
        and value["new_tokens"] == value["new_updates"] == 0, "donor import differs")
    if store.state["tasks"].get(PARENT) != "COMPLETE":
        store.boundary(task, "COMPLETE", None)


def task_request(root, c, task):
    matching = [
        r["artifact"] for r in c["payload"]["references"]
        if r["kind"] == "policy" and r["artifact"]["sha256"] == task["payload"]["policy"]
    ]
    need(bool(matching), "original task request missing")
    return content(root, matching[0])


def runtime_for(root, home, host, task, cache):
    ref = task["payload"]["trial_runtime"]
    if ref["sha256"] not in cache:
        value = content(root, ref)
        probe = Path(tempfile.mkdtemp(prefix="runtime-", dir=home))
        cache[ref["sha256"]] = progressive_roles.current_runtime(
            host,
            root,
            probe,
            value,
            producer_root=confined(root, value["payload"]["qualification"][0]["path"]).parent)
    return cache[ref["sha256"]]


def finish(root, store, journal, task, target, key, operation):
    process_complete(target)
    owner = seal(
        dict(campaign=store.campaign["identity"],
             task=task["identity"],
             output=str(target.relative_to(root))))
    binding = journal.root / "storage-inventory" / (digest(owner)[7:] + ".json")
    need(audit["saved"](binding) == owner, "foreign worker output directory")
    result = audit["saved"](target / "role/result.json")
    need(
        result.get("operation") == operation and result.get("version") == "discovery-result/v1"
        and result.get("mechanical_status") == result.get("metrics", {}).get(
            "task_status", "COMPLETE") == "COMPLETE"
        and result.get("task") in (None, task["identity"])
        and result.get("campaign") in (None, store.campaign["identity"]),
        "worker result differs from successful exit")
    # If the host died after the worker, keep its one reservation and charge the
    # full reserved interval in the absence of the original monotonic end proof.
    snap = journal.snapshot()
    need(key in snap["committed"] or key in snap["pending"],
         "role result lacks its original reservation")
    if key in snap["pending"]:
        journal.commit(key)
    need(not journal.snapshot()["pending"], "unsettled update charges at role boundary")
    if operation == "train":
        final = target / "role/accounting-final.json"
        if not final.exists():
            finalize_learning(journal, task, state_dir(root, task), target / "role")
        cp = audit["saved"](target / "role/checkpoint.json")
        costs = audit["saved"](final)
        need(
            costs["task"] == cp["payload"]["task"] == task["identity"]
            and costs["checkpoint"] == cp["identity"] and not costs["pending"]
            and costs["milestone"]["tokens"] >= task["payload"]["token_target"],
            "checkpoint or training milestone incomplete")
    name = task["payload"]["id"]
    if name not in store.state.get("outputs", {}):
        progressive_outputs.commit(root, store, task, target / "role", operation)
    need(
        store.state["outputs"][name] == dict(task=task["identity"],
                                             operation=operation,
                                             result=reference(root, target / "role/result.json")),
        "committed producer changed")
    if store.state["tasks"].get(name) != "COMPLETE":
        store.boundary(task, "COMPLETE", None)


def execute_attempt(root, home, store, journal, host, cache, name, attempt):
    task = task_for(store, name)
    c, plan = store.campaign, task["payload"]
    req = task_request(root, c, task)
    operation = req["operation"]
    need(operation in {"train", "generate", "score"} and name.endswith("-" + operation),
         "unregistered operation")
    folder = home / "tasks" / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"worker-{attempt:04d}"
    key = f"{name}/{operation}/{attempt}"
    snap = journal.snapshot()
    if key not in snap["pending"] and key not in snap["committed"]:
        need(not target.exists() and name not in store.state.get("outputs", {}),
             "ambiguous prior worker ownership")
        runtime = runtime_for(root, home, host, task, cache)
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        start = time.monotonic_ns() // 1000
        debit = dict.fromkeys(plan["caps"], 0)
        debit["elapsed_us"] = journal.task_remaining(task)["elapsed_us"]
        # Each owned preparation interval is recorded, including on retries before dispatch.
        prefix = name + "/host-input/"
        need(not any(k.startswith(prefix) for k in snap["pending"]),
             "previous input-preparation interval is unsettled")
        ordinal = sum(k.startswith(prefix) for k in snap["committed"])
        charge = journal.reserve(name, "host-input", ordinal, debit, boot=boot, monotonic_us=start)
        try:
            inputs, derived = progressive_outputs.resolve(
                root, store, task, req, folder / f"inputs-{ordinal:04d}" / "slots")
            inputs, derived = progressive_study.project_variant(
                root, store, task, operation, inputs, derived,
                folder / f"inputs-{ordinal:04d}" / "variant")
            require_clean_inputs(root, journal, inputs)
        finally:
            journal.commit(charge["key"],
                           boot=boot,
                           monotonic_us=time.monotonic_ns() // 1000,
                           termination_proven=True)
        remaining = journal.task_remaining(task)
        seconds = min(req["seconds"], remaining["elapsed_us"] / 1000000)
        reservation = dict(req["reservation"], elapsed_us=int(seconds * 1000000))
        progressive_roles.invoke(host,
                                 req["image"],
                                 root,
                                 operation,
                                 inputs,
                                 output=str(target.relative_to(root)),
                                 campaign=c,
                                 task=task,
                                 journal=journal,
                                 ordinal=attempt,
                                 reservation=reservation,
                                 seconds=seconds,
                                 runtime=runtime,
                                 derived=derived)
    finish(root, store, journal, task, target, key, operation)


def execute(root, home, store, journal, host, cache, name):
    from oeis_learn.pipeline_validation.progressive_interruption import recover
    attempt = 0
    folder = home / "tasks" / name
    # A new attempt is allowed only from a real paused learner with a settled
    # same-boot supervisor interval. Every attempt shares one frozen task budget.
    while True:
        target = folder / f"worker-{attempt:04d}"
        resumed = target / "resume.json"
        if resumed.is_file():
            marker = audit["saved"](resumed)
            need(
                name.endswith("-train") and marker["task"] == task_for(store, name)["identity"]
                and marker["attempt"] == attempt and marker["status"] == "PAUSED",
                "foreign resume marker")
            attempt += 1
            continue
        try:
            return execute_attempt(root, home, store, journal, host, cache, name, attempt)
        except (RuntimeError, OSError, ValueError, KeyboardInterrupt) as exc:
            if not name.endswith("-train") or not (target / "supervision/process.json").is_file():
                raise
            task = task_for(store, name)
            # Use the retained process proof even when this invocation merely
            # discovered an interrupted worker left by an earlier launcher.
            status = recover(root, store.campaign, task, journal, target,
                             RuntimeError("inspect prior interruption"))
            if status != "PAUSED":
                raise
            retain(resumed, seal(dict(task=task["identity"], attempt=attempt, status=status)))
            if isinstance(exc, KeyboardInterrupt):
                raise
            attempt += 1


def check_space(home, campaign, owned):
    fs = os.statvfs(home)
    storage = campaign["payload"]["storage"]
    # Only credit allocations already in the registered task partitions. A partially
    # filled disk needs space for the unpaid portion, not a second complete campaign.
    device = home.stat().st_dev
    seen = set()
    allocated = count = 0
    for base in owned:
        if not base.exists():
            continue
        for directory, dirs, files in os.walk(base):
            for path in [Path(directory), *(Path(directory) / f for f in files)]:
                st = path.lstat()
                key = st.st_dev, st.st_ino
                if key not in seen and st.st_dev == device:
                    seen.add(key)
                    allocated += st.st_blocks * 512
                    count += 1
    need(
        fs.f_bavail * fs.f_frsize >= max(
            0, storage["required_bytes"] - min(allocated, storage["task_store_bytes"]))
        and fs.f_favail >= max(0, storage["required_inodes"] - count),
        f"follow-up requires {storage['required_bytes']:,} bytes across its partitions; "
        f"{allocated:,} allocated, {fs.f_bavail*fs.f_frsize:,} free")
