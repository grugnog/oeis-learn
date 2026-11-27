"""Reproduce the round024 native deadline on HIP with four training-only prompts."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

BASE = Path("run-state/discovery/013-research-round-024")
OUT = Path("run-state/discovery-tools/round024-random-rehearsal")
CAMPAIGN = "sha256:ea3266af2f6950610ce472af244f735575c38db21243edc9b3a9d77d0267cf91"
TIP = "sha256:825eb5e890a71fb0e990d0a59c3d43d88ced798a56b0967f92fbc21e65fb090f"
DEVICE = "sha256:cff5ab0142f953321e85749692b5ccd8d09aa3af5bfbb6672b56b6008d4205b7"
TASK = "capability-random-00-generate"
PATCH = "88922141bb678730853b949054d9c222cbb2f88d"
MODULE = "src/oeis_learn/generalization/progressive/budgets.py"
OLD = "aa2ace4fdc3ac9c831569095162219c5417ef711efdbf7cf7df39c4af4ff8855"
NEW = "64819084108419e7ec6885199210cc06deb2252a95263f4421596c69363f7b0b"
MEMBERS = 4
WORK_SECONDS = 1800
SPEC = Path("specs/013-progressive-mathematical-discovery/validation")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def patched_meter(path):
    import oeis_learn.generalization.progressive.budgets as original

    need(hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == OLD and
         hashlib.sha256(path.read_bytes()).hexdigest() == NEW, "generator budget source differs")
    spec = importlib.util.spec_from_file_location(original.__name__, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[original.__name__] = module
    sys.modules[original.__package__].budgets = module
    return module


def configure(root, folder):
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file, _json

    base = root / BASE / "prepare-research"
    campaign_path = base / "frozen/campaign.json"
    campaign = _json(campaign_path)
    need(seal(campaign) == campaign and campaign["identity"] == CAMPAIGN, "round024 campaign differs")
    refs = campaign["payload"]["references"]
    refs_by_path = {r["artifact"]["path"]: r["artifact"] for r in refs}
    request_ref, = [r["artifact"] for r in refs if r["name"] == "requests/" + TASK]
    req = _json(checked_file(root, request_ref))
    plan, = [p for p in campaign["payload"]["tasks"] if p["id"] == TASK]
    need(plan["policy"] == request_ref["sha256"] and req["operation"] == "generate" and
         set(req["artifacts"]) == {"visible", "model"} and len(req["artifacts"]["visible"]) == 64,
         "expected the original 64-member generator")
    models = [_json(checked_file(root, refs_by_path[p])) for p in req["artifacts"]["model"]]
    recipe, = [v["data"]["artifact"] for v in models if v["kind"] == "recipe"]
    checkpoint, = [v for v in models if v["kind"] == "checkpoint"]
    need(len(models) == 2 and checkpoint["data"]["control"] == "random", "random model required")
    device = base / "acceptance/attempts/0000/device/integrated/request.json"
    d = _json(device)
    need(seal(d) == d and d["identity"] == DEVICE and d["campaign"] == CAMPAIGN and
         d["image"] == req["image"], "accepted device or image differs")
    support = _json(base / "materials/materials.json")["support"]
    previous = runpy.run_path(str(root / SPEC / "rehearse-round023-generator.py"))
    need(support["sha256"] == previous["SUPPORT"], "training support differs")
    prompts = previous["training_prompts"](_json(checked_file(root, support)), MEMBERS)
    scope = seal(dict(original=CAMPAIGN, task=TASK, diagnostic_only=True, patch=PATCH,
                      prompts=prompts, models=models, recipe=recipe, support=support,
                      reservation=req["reservation"], image=d["image"]))
    retain(folder / "scope.json", scope)
    patch = subprocess.check_output(["git", "show", PATCH + ":" + MODULE], cwd=root)
    need(hashlib.sha256(patch).hexdigest() == NEW, "fetched fix differs")
    module_path = folder / "budgets.py"
    if module_path.exists():
        need(module_path.read_bytes() == patch, "staged budget source differs")
    else:
        with module_path.open("xb") as stream:
            stream.write(patch)
    return d["image"], [(Path(__file__).resolve(), "rehearse.py"), (module_path, "budgets.py"),
                         (folder / "scope.json", "scope.json"), (device, "device.json"),
                         (checked_file(root, recipe), "recipe.json"), (campaign_path, "campaign.json")]


def meter(module, original, scope, out):
    from oeis_learn.generalization.progressive.contracts import digest, record, seal

    need(seal(original) == original and original["identity"] == CAMPAIGN, "original campaign required")
    # Distinct campaign identity, local journal and task output. The source campaign
    # is mounted read-only. Its account and research outputs are never mounted.
    plan, = [p for p in original["payload"]["tasks"] if p["id"] == TASK]
    plan["output"] = str(OUT / scope["identity"][7:] / "generator")
    diagnostic = seal(original)
    task = record("task", dict(plan, campaign=diagnostic["identity"]),
                  created_at=diagnostic["created_at"], purpose=diagnostic["purpose"], track=diagnostic["track"])
    values = scope["models"] + [seal(dict(version="discovery-channel/v1", kind="visible_prompt",
                                           role="generator", data=p)) for p in scope["prompts"]]
    cost = {k: v * MEMBERS // 64 for k, v in scope["reservation"].items()}
    cost["elapsed_us"] = WORK_SECONDS * 1_000_000 + plan["finalization_us"]
    journal = module.BudgetJournal(out / "diagnostic-account", diagnostic["identity"], diagnostic["payload"]["caps"])
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    remaining = journal.task_remaining(task)
    need(all(0 <= cost[k] <= remaining[k] for k in cost), "diagnostic reservation exceeds task")
    charge = journal.reserve(TASK, "generate", 0, cost, boot=boot, monotonic_us=time.monotonic_ns() // 1000)
    need(not charge.get("reserved") and not charge["committed"], "prior diagnostic dispatch")
    event, = map(json.loads, journal.path.read_bytes().splitlines())
    authority = seal(dict(version="progressive-role-budget/v1", campaign=diagnostic, task=task,
                          operation="generate", inputs=digest(sorted(values, key=lambda v: v["identity"])),
                          reservation=event))
    return module.RoleWorkBudget(authority, "generate", values, out), journal, charge["key"]


def records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def attempt(row):
    return row["prompt"], row["mode"], row["attempt"]


def exercise(inputs, out, module, scope):
    from oeis_learn.generalization.progressive.contracts import digest, reference, seal
    from oeis_learn.pipeline_validation.generalization_execution import verify_execution
    from oeis_learn.pipeline_validation.progressive_qualification import hardware_facts, require_hip
    from oeis_learn.pipeline_validation.progressive_evaluation import generate
    from oeis_learn.pipeline_validation.progressive_research_inputs import _json
    from oeis_learn.synthesis.client import SynthesisClient

    device = _json(inputs / "device.json")
    require_hip()
    need(seal(device) == device and device["identity"] == DEVICE and device["hardware"] == hardware_facts()
         and device["execution_policy"] == verify_execution() and device["image"] == os.environ["OEIS_QUALIFICATION_IMAGE"],
         "original HIP hardware and numerical policy required")
    recipe_path = inputs / "recipe.json"
    need(reference(inputs, recipe_path)["sha256"] == scope["recipe"]["sha256"], "frozen recipe differs")
    recipe = _json(recipe_path)
    budget, journal, key = meter(module, _json(inputs / "campaign.json"), scope, out)
    log = out / "generation/attempts/attempts.jsonl"
    delayed = None
    try:
        with SynthesisClient(digest(recipe)) as client:
            worker_call = client.worker.call

            def observe(request, *, deadline):
                nonlocal delayed
                response = worker_call(request, deadline=deadline)
                if (delayed is None and request["operation"] == "fit" and response["complete"]
                        and response["results"] and time.monotonic() < deadline):
                    last = records(log)[-1]
                    if last["event"] == "GENERATED":
                        delayed = (attempt(last), budget.used["native_work"], len(request["candidates"]))
                        time.sleep(max(0, deadline - time.monotonic()))
                return response

            client.worker.call = observe
            budget.client(client)
            ref = generate(inputs, "generation", scope["prompts"], recipe, client=client, output_root=out,
                           model_control="random", seed=recipe["model"]["seed"], device="cuda",
                           budget=budget, seconds=WORK_SECONDS)
    finally:
        # On an interrupted diagnostic retain the full reservation, including time.
        journal.commit(key)
    data = _json(out / ref["path"])
    need(delayed is not None and data["model_control"] == "random" and
         set(data["records"]) == {p["id"] for p in scope["prompts"]} and
         data["clock"]["cost_complete"] and data["clock"]["expected_portfolios"] == MEMBERS * 4,
         "missing delayed fit or incomplete diagnostic roster")
    target, debit, candidates = delayed
    events = records(log)
    terminal, = [i for i, row in enumerate(events) if attempt(row) == target and row["event"] == "INCOMPLETE"]
    fitting = events[terminal]["payload"]["fitting"]
    prompt, mode, position = target
    details = data["records"][prompt]["costs"][mode]["details"]
    detail, = [row for row in details if row["position"] == position]
    need(detail["fitting"]["admission"] == [] and detail["fitting"]["reason"] in {
        "native service deadline", "native/reference service deadline; late candidate excluded"
    }, "delayed fit must have no admitted candidate")
    charges = records(budget.path)
    need(fitting["complete"] is False and fitting["outcome"] == "resource_exhausted" and
         any(r["event"] == "STARTED" for r in events[terminal+1:]) and
         any(r["used"]["native_work"] == debit and debit - (charges[i-1]["used"]["native_work"] if i else 0)
             == candidates for i, r in enumerate(charges)) and budget.used["native_work"] > debit,
         "delayed fit was not recorded, charged and followed by more native work")
    return seal(dict(scope="DIAGNOSTIC_ONLY", inputs=scope["identity"], patch=PATCH,
                     campaign=CAMPAIGN, device="actual_HIP", members=MEMBERS, portfolios=MEMBERS*4,
                     delayed_fit="INCOMPLETE", native_work=budget.used["native_work"],
                     result=ref, events=reference(out, log), debits=reference(out, budget.path),
                     account=reference(out, journal.path)))


def finish(folder):
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_research_inputs import _json, checked_file

    stage = folder / "stage"
    process, cleanup, proof = [_json(stage / name) for name in ("process.json", "cleanup.json", "proof.json")]
    need(process["outcome"] == "COMPLETE" and process["returncode"] == 0 and not process.get("cleanup_failure")
         and cleanup["acknowledged"] and cleanup["within_bound"] and not cleanup["failure"],
         "successful supervised exit and cleanup required")
    scope = _json(folder / "scope.json")
    need(seal(scope) == scope and scope["original"] == CAMPAIGN and scope["patch"] == PATCH
         and seal(proof) == proof and proof["inputs"] == scope["identity"]
         and proof["scope"] == "DIAGNOSTIC_ONLY" and proof["patch"] == PATCH and proof["campaign"] == CAMPAIGN
         and proof["device"] == "actual_HIP" and proof["members"] == MEMBERS
         and proof["portfolios"] == MEMBERS * 4 and proof["delayed_fit"] == "INCOMPLETE",
         "rehearsal identity or result differs")
    for k in ("result", "events", "debits", "account"):
        checked_file(stage, proof[k])
    retain(folder / "result.json", seal(dict(status="PASS", **{k:v for k,v in proof.items() if k != "identity"})))


def main():
    if sys.argv[1:] == ["--inside"]:
        from oeis_learn.generalization.progressive.contracts import atomic_write, seal
        inputs, out = Path("/work/inputs"), Path("/work/output")
        scope = json.loads((inputs / "scope.json").read_bytes())
        need(seal(scope) == scope and scope["original"] == CAMPAIGN and scope["patch"] == PATCH, "unexpected scope")
        module = patched_meter(inputs / "budgets.py")
        atomic_write(out / "proof.json", exercise(inputs, out, module, scope))
        return
    need(len(sys.argv) == 1, "run the host rehearsal without arguments")
    root = Path.cwd()
    sys.path.insert(0, str(root / "src"))
    from oeis_learn.pipeline_validation.progressive_handoff import stage
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import read

    control = root / "run-state/discovery-control" / CAMPAIGN[7:]
    folder = root / OUT
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "rehearsal.lock").open("a") as lock, (control / "run.lock").open("rb") as original:
        for handle in (lock, original):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failure = None
        try:
            need(read(sorted(control.glob("transition-*.json"))[-1])["identity"] == TIP and
                 not (control / "assessment-exposure.json").exists(), "round024 state advanced")
            if not (folder / "stage").exists():
                image, paths = configure(root, folder)
                binding = read(root / BASE / "session.json")["binding"]
                host = Workflow(SimpleNamespace(root=root, folder=root/BASE),
                                SimpleNamespace(**binding, phase="prepare-research")).host()
                print("Starting four training-only random-model prompts on HIP.", flush=True)
                stage(host, root, image, folder / "stage", ["python", "/work/inputs/rehearse.py", "--inside"],
                      gpu=True, seconds=WORK_SECONDS+300,
                      readonly_inputs=[(path, "/work/inputs/"+name) for path, name in paths])
            finish(folder)
            print("PASS: delayed fit charged and excluded; subsequent work completed.", flush=True)
        except Exception as exc:
            failure = dict(type=type(exc).__name__, frames=[
                [Path(f.filename).name, f.lineno, f.name] for f in traceback.extract_tb(exc.__traceback__)
            ])
            raise
        finally:
            tmp = folder / "diagnostics.tmp"
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for name in ("result.json", "stage/process.json", "stage/cleanup.json", "stage/proof.json"):
                    if (folder/name).is_file():
                        z.write(folder/name, name)
                if failure:
                    z.writestr("host-failure.json", json.dumps(failure))
                # Only frame locations from the container traceback are shareable.
                stderr = folder/"stage/stderr.log"
                if stderr.is_file():
                    frames = [s.strip() for s in stderr.read_text(errors="replace").splitlines()
                              if s.lstrip().startswith('File "')]
                    z.writestr("frames.txt", "\n".join(frames))
            tmp.replace(folder/"diagnostics.zip")
            print("Share diagnostics:", folder/"diagnostics.zip", flush=True)


if __name__ == "__main__":
    main()
