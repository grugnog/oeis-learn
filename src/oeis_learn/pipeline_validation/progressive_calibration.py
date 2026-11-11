"""Bounded prepared-input calibration; the host remains standard-library-only."""

import copy
import json
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    atomic_write,
    digest,
    load,
    reference,
    seal,
    validate,
    verify_refs,
)
from oeis_learn.pipeline_validation.identity import confined

# Bound the total campaign inventory by the existing calibration inode ceiling,
# including distinct source/library inputs beyond the two tuning trajectories.
# Generic Q retains its separate 48-case hardware matrix limit.
MAX_CASES = (1000000 - 256) // 64
MAX_SECONDS = 86400
METADATA_BYTES = 16 << 20


def selected_inputs(root, prepared):
    """Resolve only frozen learner channels, including every finite trial choice."""
    from oeis_learn.generalization.progressive.projections import _validate_channel
    from oeis_learn.generalization.progressive.trial_scopes import materialize, variants

    from .progressive_learner import compatibility, validate_input

    refs = prepared["payload"]["references"]
    policies = {r["artifact"]["sha256"]: r["artifact"] for r in refs if r["kind"] == "policy"}
    channels = {
        r["artifact"]["path"]: r for r in refs if r["kind"] == "prepared" and r["role"] == "learner"
    }
    selected, bindings = {}, []
    originals = {}
    def original_json(ref, maximum=4<<20):
        # A verified immutable base is shared by finite policy alternatives;
        # materialize() changes only profile/update policy, not training rows.
        key=(ref['path'],ref['bytes'],ref['sha256'])
        if key not in originals:
            originals[key]=json.loads(verify_refs(root,[ref],max_bytes=maximum)[0])
        return originals[key]
    for index, original in enumerate(prepared["payload"]["task_plans"]):
        for plan in variants(original):
            if plan["kind"] != "training":
                continue
            if plan["policy"] not in policies:
                raise ValueError("original prepared training policy required for calibration")
            request = original_json(policies[plan["policy"]])
            paths = request.get("artifacts", {}).get("training", [])
            if request.get('operation')=='train':
                paths=[path for path in paths if isinstance(path,str) and path in channels and json.loads(verify_refs(root,[channels[path]['artifact']])[0]).get('kind')=='training_request']
            if request.get('operation') in {'source-cycle','oeis-cycle'}:
                frozen=json.loads(verify_refs(root,[reference(root,request['artifacts']['cycle'][0])],max_bytes=32<<20)[0])
                paths=[e['channel'] for e in frozen['training'] if e['task']==plan['id']]
            if (
                request.get('operation') not in {'train','source-cycle','oeis-cycle'}
                or len(paths) != 1
                or not isinstance(paths[0], str)
                or paths[0] not in channels
            ):
                raise ValueError("one original prepared learner channel required for calibration")
            channel = original_json(channels[paths[0]]["artifact"])
            if (
                seal(channel) != channel
                or channel.get("role") != "learner"
                or channel.get("kind") != "training_request"
            ):
                raise ValueError("original prepared learner channel identity differs")
            _validate_channel("training_request", channel["data"])
            package = original_json(channel["data"]["artifact"],256<<20)
            if compatibility(package) != channel["data"]["compatibility"]:
                raise ValueError("original prepared compatibility differs")
            package = materialize(package, plan)
            validate_input(
                package,
                {"purpose": prepared["purpose"], "track": prepared["track"], "payload": plan},
                allow_pending=True,
            )
            key = compatibility(package)
            if key != plan["initializer"]["compatibility"]:
                raise ValueError("calibration task compatibility differs")
            selected.setdefault(key, {"compatibility": key, "package": package, "plan": plan})
            bindings.append({"task": index, "choice": plan.get("trial_choice"), "case": key})
    if not 1 <= len(selected) <= MAX_CASES:
        raise ValueError("bounded nonempty prepared calibration inventory required")
    return [selected[k] for k in sorted(selected)], bindings


def prepare(root, blueprint, prepared, output, args, host_factory):
    """Measure once inside the blueprint's prospective setup reservation."""
    began = time.monotonic()
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, vector
    from oeis_learn.generalization.progressive.campaign import validate_runtime
    from oeis_learn.generalization.progressive.freeze import trial_runtime
    from oeis_learn.generalization.progressive.storage import StorageGuard

    from .generalization_qualification_io import current_facts
    from .progressive_handoff import named_plan, original_authority, stage, validate_device_proofs

    plan = named_plan(
        root,
        blueprint,
        "iteration_calibration_plan",
        "progressive-calibration-plan/v1",
        {"seconds", "caps", "checkpoint_bytes", "maximum_inodes"},
        optional={'checkpoint_retention'},
    )
    if plan is None:
        if prepared["purpose"] == "research" and any(
            p["kind"] == "training" for p in prepared["payload"]["task_plans"]
        ):
            return seal(
                dict(
                    prepared,
                    payload=dict(
                        prepared["payload"],
                        deficits=sorted(
                            set(prepared["payload"]["deficits"] + ["iteration_calibration_plan"])
                        ),
                    ),
                )
            )
        return prepared
    from .progressive_calibration_storage import POLICY as RETIREMENT_POLICY
    if 'checkpoint_retention' in plan and plan['checkpoint_retention']!=RETIREMENT_POLICY:
        raise ValueError('registered calibration snapshot retention policy required')
    caps = vector(plan["caps"])
    if (
        any(
            caps[k] + prepared["payload"]["caps"][k] > blueprint["payload"]["proposed_caps"][k]
            for k in caps
        )
        or caps["elapsed_us"] != plan["seconds"] * 1000000
        or type(plan["checkpoint_bytes"]) is not int
        or not 0 < plan["checkpoint_bytes"] <= 64 << 30
        or type(plan["maximum_inodes"]) is not int
        or not 0 < plan["maximum_inodes"] <= 1000000
    ):
        raise ValueError("finite calibration reservation must fit prospective allowance")
    cases, bindings = selected_inputs(root, prepared)
    tokens = sum(
        2
        * c["package"]["update_policy"]["effective_batch"]
        * max(len(r["tokens"]) for shard in c["package"]["shards"] for r in shard)
        for c in cases
    )
    input_bytes = sum((len(json.dumps(c['package'],sort_keys=True,indent=2,allow_nan=False).encode())+1)*2 for c in cases)
    from .progressive_calibration_storage import required_bytes as storage_required
    required_bytes=storage_required(len(cases),plan['checkpoint_bytes'],input_bytes,
        retention=plan.get('checkpoint_retention'),metadata_bytes=METADATA_BYTES)
    if (
        caps["tokens"] < tokens
        or caps["disk_bytes"] < required_bytes
        or plan["maximum_inodes"] < 64 * len(cases) + 256
    ):
        raise ValueError("calibration token/storage reservation deficit")
    guard = StorageGuard([output], max_bytes=caps["disk_bytes"], max_inodes=plan["maximum_inodes"])
    guard.check(required_bytes, 64 * len(cases) + 256)
    runtime = validate("runtime", load(confined(root, prepared["payload"]["runtime"]["path"])))
    qref = reference(root, confined(root, args.qualification) / "qualification.json")
    if qref not in runtime["payload"]["qualification"]:
        raise ValueError("calibration requires original preparation qualification")
    q, qroot = original_authority(root, qref)
    source = digest(collect_snapshot(root))
    # A stable blueprint reservation survives changed output directories and
    # failures. Reserve returns an existing ordinal, which must never rerun work.
    identity = digest(["iteration-calibration", blueprint["identity"]])
    journal = BudgetJournal(
        Path(root) / "run-state/discovery-calibration" / identity[7:], identity, caps
    )
    reserved = journal.reserve(
        "preparation", "calibrate", 0, caps, boot=identity, monotonic_us=time.monotonic_ns() // 1000
    )
    if "reserved" in reserved:
        raise ValueError(
            "calibration reservation already consumed; retain original output or prospectively revise blueprint"
        )
    # Conservative commit before dispatch: failed/unproven setup work never
    # refunds this finite attempt or grants time by selecting a sibling output.
    journal.commit(reserved["key"])
    host = host_factory(args)
    matrix, _ = validate_device_proofs(
        qroot, q, source=source, image=q["images"][1], engine=host.engine.name
    )
    facts = current_facts(
        host,
        root,
        output / "inspection",
        images=q["images"],
        source_ref=q["source_closure"],
        build_refs=q["build_dependencies"],
        producer_root=qroot,
        pac_seconds=max(1, min(120, plan["seconds"] // 6)),
    )
    for case in cases:
        selected = trial_runtime(root, case["plan"], runtime)
        validate_runtime(selected, root=root, current_facts=facts, inspected=True)
        model = digest(
            {k: v for k, v in case["package"]["profile"]["model"].items() if k != "seed"}
        )
        if selected["payload"]["scope"]["model"] != model:
            raise ValueError("prepared calibration model/runtime differs")
    device = output / "calibration"
    device.mkdir()
    projected = []
    with guard.activate():
        atomic_write(output / "calibration-input.json", prepared)
        for case in cases:
            path = device / "inputs" / (case["compatibility"][7:] + ".json")
            atomic_write(path, case["package"])
            projected.append(
                {"compatibility": case["compatibility"], "artifact": reference(device, path)}
            )
        remaining = int(plan["seconds"] - (time.monotonic() - began)) - 10
        if remaining <= 0:
            raise TimeoutError("calibration allowance exhausted during inspection")
        request = seal(
            {
                "version": "progressive-calibration-request/v1",
                "source": source,
                "image": q["images"][1],
                "engine": host.engine.name,
                "hardware": matrix["hardware"],
                "execution_policy": q["runtime"]["execution_policy"],
                "blueprint": blueprint["identity"],
                "prepared": prepared["identity"],
                "cases": projected,
                "seconds": remaining,
                "checkpoint_bytes": plan["checkpoint_bytes"],
                "maximum_inodes": plan["maximum_inodes"],
                "caps": caps,
                **({'checkpoint_retention':plan['checkpoint_retention']} if 'checkpoint_retention' in plan else {}),
            }
        )
        atomic_write(device / "request.json", request)
    code = (
        "from oeis_learn.pipeline_validation.progressive_calibration import run_inside;"
        "run_inside('/work/output')"
    )
    stage(host, root, q["images"][1], device, ["python", "-c", code], gpu=True, seconds=remaining)
    if time.monotonic() - began > plan["seconds"]:
        raise TimeoutError("calibration aggregate allowance exhausted")
    guard.check()
    result = load(device / "producer.json")
    producers = verify_result(device, request, result)
    payload = copy.deepcopy(prepared["payload"])
    from oeis_learn.generalization.progressive.iteration import validate_producer

    refs = {}
    for case, producer in zip(projected, producers):
        key = case["compatibility"]
        ref = reference(root, confined(device, result["producers"][key]["path"]))
        refs[key] = dict(producer=ref, **{k: 2 * v for k, v in producer["measurements"].items()})
        payload["references"].append(
            {
                "name": "iteration:" + key,
                "role": "custodian",
                "kind": "qualification",
                "artifact": ref,
            }
        )
    for binding in bindings:
        task = payload["task_plans"][binding["task"]]
        if binding["choice"] is None:
            task["iteration_budget"] = refs[binding["case"]]
        else:
            task["trial_choices"][binding["choice"]]["iteration_budget"] = refs[binding["case"]]
    payload["references"].extend(
        [
            {
                "name": "iteration_calibration",
                "role": "custodian",
                "kind": "qualification",
                "artifact": reference(root, device / "producer.json"),
            },
            {
                "name": "iteration_calibration_request",
                "role": "custodian",
                "kind": "policy",
                "artifact": reference(root, device / "request.json"),
            },
            {
                "name": "iteration_calibration_cost",
                "role": "custodian",
                "kind": "source",
                "artifact": reference(root, journal.path),
            },
            {
                "name": "iteration_calibration_input",
                "role": "custodian",
                "kind": "policy",
                "artifact": reference(root, output / "calibration-input.json"),
            },
        ]
    )
    from oeis_learn.generalization.progressive.trial_scopes import variants

    for task in payload["task_plans"]:
        for variant in variants(task):
            if variant["kind"] == "training":
                validate_producer(
                    root, variant, trial_runtime(root, variant, runtime), payload["references"]
                )
    return seal(dict(prepared, payload=payload))


def verify_result(root, request, result):
    """No label can substitute for the complete original per-input measurements."""
    validate_request(request)
    if load(Path(root) / "request.json", maximum=32 << 20) != request:
        raise ValueError("original calibration request changed")
    keys = {c["compatibility"] for c in request["cases"]}
    if (
        set(result)
        != {"version", "identity", "request", "device", "status", "producers", "actual_elapsed_us"}
            | ({'retirements'} if 'checkpoint_retention' in request else set())
        or seal(result) != result
        or result["version"] != "progressive-calibration-run/v1"
        or result["request"] != request["identity"]
        or result["device"] != "actual_HIP"
        or result["status"] != "PASS"
        or set(result["producers"]) != keys
        or type(result["actual_elapsed_us"]) is not int
        or not 0 < result["actual_elapsed_us"] <= request["seconds"] * 1000000
    ):
        raise ValueError("complete original bounded calibration producer required")
    producers = []
    if 'checkpoint_retention' in request and set(result['retirements'])!=keys:
        raise ValueError('complete original calibration retirement inventory required')
    for case in request["cases"]:
        producer_ref=result['producers'][case['compatibility']]
        producer=verify_case(root,request,case,producer_ref)
        if 'checkpoint_retention' in request:
            from .progressive_calibration_storage import verify as verify_retirement
            base=confined(root,producer_ref['path']).parent
            rref=result['retirements'][case['compatibility']]
            rpath=confined(root,rref['path'])
            if rpath.parent!=base:
                raise ValueError('retirement belongs to another calibration case')
            verify_refs(root,[rref])
            verify_retirement(base,reference(base,rpath),producer)
        producers.append(producer)
    return producers


def verify_case(root,request,case,producer_ref):
    """Verify original measurements before optional temporary snapshot cleanup."""
    from oeis_learn.generalization.progressive.iteration import measured_producer
    producer=json.loads(verify_refs(root,[producer_ref])[0])
    measured_producer(producer,source=request['source'],images=[request['image']],
                      compatibility=case['compatibility'])
    package=json.loads(verify_refs(root,[case['artifact']],max_bytes=256<<20)[0])
    if (producer['effective_batch']!=package['update_policy']['effective_batch']
            or producer['maximum_target_tokens']!=max(len(r['tokens']) for s in package['shards'] for r in s)
            or producer['checkpoint_bytes']>request['checkpoint_bytes']):
        raise ValueError('calibration prepared work/storage scope differs')
    return producer


def validate_request(request):
    from oeis_learn.generalization.progressive.budgets import vector

    fields = {
        "version",
        "identity",
        "source",
        "image",
        "engine",
        "hardware",
        "execution_policy",
        "blueprint",
        "prepared",
        "cases",
        "seconds",
        "checkpoint_bytes",
        "maximum_inodes",
        "caps",
    }
    from .progressive_calibration_storage import POLICY as RETIREMENT_POLICY
    if 'checkpoint_retention' in request and request['checkpoint_retention']!=RETIREMENT_POLICY:
        raise ValueError('registered calibration snapshot retention policy required')
    if (
        set(request)-{'checkpoint_retention'} != fields
        or seal(request) != request
        or request["version"] != "progressive-calibration-request/v1"
        or type(request["seconds"]) is not int
        or not 0 < request["seconds"] <= MAX_SECONDS
        or not isinstance(request["cases"], list)
        or not 1 <= len(request["cases"]) <= MAX_CASES
        or any(set(c) != {"compatibility", "artifact"} for c in request["cases"])
        or len({c["compatibility"] for c in request["cases"]}) != len(request["cases"])
        or type(request["checkpoint_bytes"]) is not int
        or not 0 < request["checkpoint_bytes"] <= 64 << 30
        or type(request["maximum_inodes"]) is not int
        or not 0 < request["maximum_inodes"] <= 1000000
    ):
        raise ValueError("closed finite installed calibration request required")
    caps = vector(request["caps"])
    from .progressive_calibration_storage import required_bytes as storage_required
    if (
        request["seconds"] * 1000000 > caps["elapsed_us"]
        or storage_required(len(request['cases']),request['checkpoint_bytes'],0,
                            retention=request.get('checkpoint_retention'),metadata_bytes=METADATA_BYTES)
        > caps["disk_bytes"]
    ):
        raise ValueError("installed calibration exceeds prospective reservation")


def validate_prepared(root, blueprint, prepared):
    """Freeze replays the retained original setup run, not just its timing labels."""
    names = {
        "iteration_calibration",
        "iteration_calibration_request",
        "iteration_calibration_input",
        "iteration_calibration_cost",
    }
    refs = {
        r["name"]: r["artifact"] for r in prepared["payload"]["references"] if r["name"] in names
    }
    if not refs:
        return
    if set(refs) != names:
        raise ValueError("complete original prepared calibration closure required")
    request = json.loads(verify_refs(root, [refs["iteration_calibration_request"]])[0])
    original = json.loads(
        verify_refs(root, [refs["iteration_calibration_input"]], max_bytes=CAMPAIGN_DOCUMENT_BYTES)[0]
    )
    validate("prepared", original)
    if request["blueprint"] != blueprint["identity"] or request["prepared"] != original["identity"]:
        raise ValueError("original calibrated preparation/blueprint differs")
    result = json.loads(verify_refs(root, [refs["iteration_calibration"]])[0])
    base = confined(root, refs["iteration_calibration"]["path"]).parent
    producers = verify_result(base, request, result)
    expected = {p["compatibility"] for p in producers}
    cases, _ = selected_inputs(root, prepared)
    if expected != {c["compatibility"] for c in cases}:
        raise ValueError("frozen calibration input inventory differs")
    raw = verify_refs(root, [refs["iteration_calibration_cost"]], max_bytes=64 << 20)[0]
    events = [json.loads(line) for line in raw.splitlines()]
    if (
        len(events) != 2
        or [e["event"] for e in events] != ["RESERVE", "COMMIT"]
        or any(
            seal(e) != e
            or e["cost"] != request["caps"]
            or e["operation"] != "preparation/calibrate/0"
            for e in events
        )
        or events[1]["previous"] != events[0]["identity"]
    ):
        raise ValueError("original conservative calibration debit required")


def run_inside(root):
    """Installed actual-HIP entry point, supervised by the existing GPU launcher."""
    import os

    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.generalization.progressive.storage import StorageGuard

    from .generalization_execution import verify_execution
    from .progressive_learner import compatibility
    from .progressive_qualification import hardware_facts, require_hip

    root = Path(root)
    request = load(root / "request.json", maximum=32 << 20)
    validate_request(request)
    began = time.monotonic()
    require_hip()
    if (
        request["source"] != digest(source_snapshot())
        or request["image"] != os.environ.get("OEIS_QUALIFICATION_IMAGE")
        or request["engine"] != os.environ.get("OEIS_QUALIFICATION_ENGINE")
        or request["hardware"] != hardware_facts()
        or request["execution_policy"] != verify_execution()
    ):
        raise ValueError("current exact calibration device/source/image/policy required")
    guard = StorageGuard(
        [root],
        max_bytes=request["caps"]["disk_bytes"],
        max_inodes=request["maximum_inodes"],
        reserve_bytes=METADATA_BYTES,
    )
    producers = {}
    retirements = {}
    failures = []
    outcomes=[dict(index=i,compatibility=c['compatibility'],status='PENDING')
              for i,c in enumerate(request['cases'])]
    def publish_outcome(index):
        row=outcomes[index]
        atomic_write(root/'case-events'/str(index)/row['status']/'calibration-case.json',seal(dict(
            version='progressive-calibration-case/v1',request=request['identity'],case=row)))
    blocked=None
    with guard.activate():
        for index, case in enumerate(request["cases"]):
            if blocked:
                outcomes[index].update(status='BLOCKED',reason=blocked)
                publish_outcome(index)
                continue
            folder = root / "cases" / case["compatibility"][7:]
            # Fresh processes contain per-configuration OOMs and allocator
            # residue. Every remaining independent case has a bounded share.
            import sys
            import traceback
            from .progressive_acceptance import _bounded_process, _record_failure
            code=('import sys;from oeis_learn.pipeline_validation.progressive_calibration import run_one;'
                  'run_one(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]))')
            launched=False
            try:
                package=json.loads(verify_refs(root,[case['artifact']],max_bytes=256<<20)[0])
                if compatibility(package)!=case['compatibility']:
                    raise ValueError('prepared calibration compatibility differs')
                remaining=request['seconds']-(time.monotonic()-began)
                if remaining<=1:
                    blocked='CALIBRATION_DEADLINE'
                    raise TimeoutError('calibration deadline before next input')
                case_seconds=min(3600,remaining/(len(request['cases'])-index))
                outcomes[index].update(status='RUNNING',seconds=case_seconds)
                publish_outcome(index)
                launched=True
                status=_bounded_process([sys.executable,'-c',code,str(root),str(index),str(case_seconds)],
                    cwd=root,env=dict(os.environ),log=root/f'case-{index}.log',
                    seconds=case_seconds,maximum=65536)
                if status: raise ValueError('exact-input calibration process failed')
                producer_ref=reference(root,folder/'producer.json')
                producer=verify_case(root,request,case,producer_ref)
                if 'checkpoint_retention' in request:
                    from .progressive_calibration_storage import retire
                    retired=retire(folder,producer,maximum_bytes=2*request['checkpoint_bytes'])
                    retirements[case['compatibility']]=reference(root,confined(folder,retired['path']))
                producers[case['compatibility']]=producer_ref
                outcomes[index].update(status='PASS',producer=producer_ref)
            except (OSError,ValueError,TimeoutError) as error:
                outcomes[index].update(status='FAIL',error=type(error).__name__,message=str(error))
                _record_failure(root, failures, dict(compatibility=case['compatibility'],error=type(error).__name__,traceback=traceback.format_exc()))
                remaining=request['seconds']-(time.monotonic()-began)
                try:
                    if launched and (remaining<=1 or _bounded_process([sys.executable,'-c',
                        'from oeis_learn.pipeline_validation.progressive_qualification import require_hip; t=require_hip(); t.cuda.synchronize()'],
                        cwd=root,env=dict(os.environ),log=root/f'health-{index}.log',seconds=min(30,remaining),maximum=65536)):
                        raise RuntimeError('device health probe failed or calibration allowance exhausted')
                except (OSError,ValueError,TimeoutError,RuntimeError) as health_error:
                    _record_failure(root, failures, dict(compatibility=case['compatibility'],error='DEVICE_BLOCKED',message=str(health_error)))
                    blocked='DEVICE_BLOCKED'
            publish_outcome(index)
        atomic_write(root/'calibration-cases.json',seal(dict(
            version='progressive-calibration-cases/v1',request=request['identity'],cases=outcomes)))
        if failures:
            atomic_write(root/'failures.json',failures)
            atomic_write(root/'partial.json',seal(dict(version='progressive-calibration-failure/v1',
                status='FAIL',request=request['identity'],producers=producers,failures=failures,
                **({'retirements':retirements} if 'checkpoint_retention' in request else {}),
                missing=[c['compatibility'] for c in request['cases'] if c['compatibility'] not in producers])))
            raise ValueError('exact calibration incomplete; original per-case diagnostics retained')
        result = seal(
            {
                "version": "progressive-calibration-run/v1",
                "request": request["identity"],
                "device": "actual_HIP",
                "status": "PASS",
                "producers": producers,
                "actual_elapsed_us": max(1, int((time.monotonic() - began) * 1000000)),
                **({'retirements':retirements} if 'checkpoint_retention' in request else {}),
            }
        )
        verify_result(root, request, result)
        atomic_write(root / "producer.json", result)
    return result


def run_one(root,index,seconds):
    """Installed child; the parent owns the finite request and external deadline."""
    from .progressive_qualification import calibrate_iteration
    from oeis_learn.generalization.progressive.storage import StorageGuard
    root=Path(root)
    request=load(root/'request.json',maximum=32<<20)
    validate_request(request)
    case=request['cases'][index]
    package=json.loads(verify_refs(root,[case['artifact']],max_bytes=256<<20)[0])
    with StorageGuard([root],max_bytes=request['caps']['disk_bytes'],
                      max_inodes=request['maximum_inodes'],reserve_bytes=METADATA_BYTES).activate():
        calibrate_iteration(package,root/'cases'/case['compatibility'][7:],
            source=request['source'],image=request['image'],device='cuda',
            checkpoint_bytes=request['checkpoint_bytes'],seconds=seconds)
