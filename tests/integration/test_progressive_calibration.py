"""Real CPU measurements and explicitly simulated host/device boundaries."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)


def inputs(root):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture

    campaign, _, _, _ = fixture(root, interrupt=False)
    return {
        "payload": {
            "task_plans": campaign["payload"]["tasks"],
            "references": campaign["payload"]["references"],
        },
        "purpose": "acceptance",
        "track": "synthetic_strict",
    }


def test_calibration_resolves_original_prepared_inputs_and_deduplicates_continuation(tmp_path):
    from oeis_learn.pipeline_validation.progressive_calibration import selected_inputs

    prepared = inputs(tmp_path)
    selected, bindings = selected_inputs(tmp_path, prepared)
    assert len(selected) == 1
    assert len(bindings) == 2
    assert selected[0]["package"] == load(tmp_path / "training-input.json")
    assert {b["case"] for b in bindings} == {selected[0]["compatibility"]}
    changed = copy.deepcopy(prepared)
    changed["payload"]["references"] = [
        r for r in changed["payload"]["references"] if r["kind"] != "prepared"
    ]
    with pytest.raises(ValueError, match="original prepared"):
        selected_inputs(tmp_path, changed)
    changed = copy.deepcopy(prepared)
    changed["payload"]["task_plans"][0]["initializer"]["compatibility"] = digest("different")
    with pytest.raises(ValueError, match="compatibility"):
        selected_inputs(tmp_path, changed)


def test_installed_calibration_measures_cpu_but_cannot_claim_hip(tmp_path):
    from oeis_learn.pipeline_validation.progressive_qualification import calibrate_iteration

    inputs(tmp_path / "data")
    package = load(tmp_path / "data/training-input.json")
    code = """from pathlib import Path
import sys
from oeis_learn.generalization.progressive.contracts import load,digest
from oeis_learn.pipeline_validation.progressive_qualification import calibrate_iteration
root=Path(sys.argv[1])
calibrate_iteration(load(root/'data/training-input.json'),root/'measured',source=digest('cpu-diagnostic'),image=digest('image'),device='cpu',checkpoint_bytes=64<<20,seconds=60)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], check=True, timeout=90, env=os.environ.copy()
    )
    result = load(tmp_path / "measured/producer.json")
    assert result["device"] == "cpu"
    assert result["checkpoint_bytes"] > 0 and result["checkpoint_inodes"] > 0
    assert all(v > 0 for v in result["measurements"].values())
    assert result["effective_batch"] == package["update_policy"]["effective_batch"]
    with pytest.raises(TimeoutError, match="checkpoint"):
        calibrate_iteration(
            package,
            tmp_path / "too-small",
            source=digest("cpu-diagnostic"),
            image=digest("image"),
            device="cpu",
            checkpoint_bytes=1,
            seconds=60,
        )
    assert not list((tmp_path / "too-small").rglob("*.pt"))
    assert not (tmp_path / "too-small/producer.json").exists()


def test_calibration_input_resolution_is_stdlib_only(tmp_path):
    prepared = inputs(tmp_path)
    atomic_write(tmp_path / "prepared.json", prepared)
    code = """import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from oeis_learn.pipeline_validation.progressive_calibration import selected_inputs
root=Path(sys.argv[2])
assert len(selected_inputs(root,json.loads((root/'prepared.json').read_text()))[0])==1
assert not {'torch','numpy','optuna','jsonschema'} & set(sys.modules)
"""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            code,
            str(Path(__file__).resolve().parents[2] / "src"),
            str(tmp_path),
        ],
        check=True,
        timeout=30,
    )


def test_expired_calibration_cannot_construct_model(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_qualification as q
    from oeis_learn.pipeline_validation import progressive_training as training

    inputs(tmp_path / "data")
    package = load(tmp_path / "data/training-input.json")
    monkeypatch.setattr(
        training, "PolicyUpdate", lambda *a, **kw: pytest.fail("late model construction")
    )
    with pytest.raises(ValueError, match="finite calibration"):
        q.calibrate_iteration(
            package,
            tmp_path / "out",
            source=digest("s"),
            image=digest("i"),
            device="cpu",
            seconds=0,
            checkpoint_bytes=64 << 20,
        )


def host_inputs(root, monkeypatch):
    """Fixture receipts are explicitly diagnostic; never hardware qualification."""
    from oeis_learn.pipeline_validation import generalization_qualification_io as io
    from oeis_learn.pipeline_validation import progressive_handoff as h
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from tests.unit.test_progressive_freeze import (
        test_complete_diagnostic_freeze_binds_original_producer_and_task_roster,
    )

    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(root, monkeypatch)

    def put(path, value):
        path.write_text(json.dumps(value))

    b, p = load(root / "test-blueprint.json"), load(root / "test-prepared.json")
    campaign, _, _, _ = fixture(root / "learning", interrupt=False)
    campaign["payload"]["caps"]["disk_bytes"] = 3 << 30
    channel = load(root / "learning/training-channel.json")
    channel["data"]["artifact"] = reference(root, "learning/training-input.json")
    put(root / "learning/training-channel.json", seal(channel))
    refs = []
    for item in campaign["payload"]["references"]:
        if item["kind"] not in {"policy", "prepared"}:
            continue
        path = root / "learning" / item["artifact"]["path"]
        if item["kind"] == "policy":
            request = load(path)
            request["artifacts"]["training"] = ["learning/training-channel.json"]
            put(path, request)
            for task in campaign["payload"]["tasks"]:
                if task["policy"] == item["artifact"]["sha256"]:
                    task["policy"] = reference(root, path)["sha256"]
        refs.append(dict(item, name="learning:" + item["name"], artifact=reference(root, path)))
    package = load(root / "learning/training-input.json")
    model = digest({k: v for k, v in package["profile"]["model"].items() if k != "seed"})
    q = load(root / "qualification/qualification.json")
    producer = load(root / "qualification/producer.json")
    producer["capabilities"] = q["covered_capabilities"] = [
        "model:" + model,
        "language:bigint-body/v1",
    ]
    put(root / "qualification/producer.json", seal(producer))
    q["producer"]["receipt"] = reference(root / "qualification", "producer.json")
    put(root / "qualification/qualification.json", seal(q))
    runtime = load(root / "runtime.json")
    runtime["payload"]["scope"]["model"] = model
    runtime["payload"]["qualification"] = [reference(root, "qualification/qualification.json")]
    runtime["payload"]["operations"] = ["train"]
    put(root / "runtime.json", seal(runtime))
    p["payload"].update(
        runtime=reference(root, "runtime.json"),
        graph=campaign["payload"]["graph"],
        task_plans=campaign["payload"]["tasks"],
        caps=campaign["payload"]["caps"],
    )
    for item in p["payload"]["references"]:
        if item["kind"] == "qualification":
            item["artifact"] = reference(root, "qualification/qualification.json")
    p["payload"]["references"] += refs
    caps = dict.fromkeys(campaign["payload"]["caps"], 0)
    caps.update(tokens=1000, elapsed_us=60000000, disk_bytes=1 << 30)
    plan = seal(
        {
            "version": "progressive-calibration-plan/v1",
            "seconds": 60,
            "caps": caps,
            "checkpoint_bytes": 64 << 20,
            "maximum_inodes": 1024,
        }
    )
    put(root / "calibration-plan.json", plan)
    b["payload"]["proposed_caps"] = {k: v + caps[k] for k, v in campaign["payload"]["caps"].items()}
    b["payload"]["references"].append(
        {
            "name": "iteration_calibration_plan",
            "role": "custodian",
            "kind": "policy",
            "artifact": reference(root, "calibration-plan.json"),
        }
    )
    payload = dict(p["payload"])
    payload.pop("blueprint")
    put(
        root / "preparation-plan.json",
        {"version": "progressive-preparation-plan/v1", "payload": payload},
    )
    b["payload"]["references"].append(
        {
            "name": "preparation_plan",
            "role": "custodian",
            "kind": "policy",
            "artifact": reference(root, "preparation-plan.json"),
        }
    )
    b = seal(b)
    put(root / "blueprint.json", b)
    monkeypatch.setattr(
        h,
        "original_authority",
        lambda *a: (load(root / "qualification/qualification.json"), root / "qualification"),
    )
    monkeypatch.setattr(
        h, "validate_device_proofs", lambda *a, **kw: ({"hardware": {"boundary_only": True}}, [])
    )
    from oeis_learn.generalization.qualification import FACTS

    monkeypatch.setattr(io, "current_facts", lambda *a, **kw: {k: q[k] for k in FACTS})
    return b


@pytest.mark.parametrize("change", ["none", "cpu", "stale", "missing", "timeout", "overspend"])
@pytest.mark.parametrize('retirement',[False,True])
def test_canonical_prepare_calibrates_then_freezes_or_preserves_failed_debit(
    tmp_path, monkeypatch, capsys, change, retirement
):
    from oeis_learn.pipeline_validation import progressive_handoff as h
    from oeis_learn.pipeline_validation.progressive_commands import main

    host_inputs(tmp_path, monkeypatch)
    if retirement:
        from oeis_learn.pipeline_validation.progressive_calibration_storage import POLICY
        plan=seal(dict(load(tmp_path/'calibration-plan.json'),checkpoint_retention=POLICY))
        (tmp_path/'calibration-plan.json').write_text(json.dumps(plan))
        b=load(tmp_path/'blueprint.json')
        next(r for r in b['payload']['references'] if r['name']=='iteration_calibration_plan')['artifact']=reference(tmp_path,'calibration-plan.json')
        (tmp_path/'blueprint.json').write_text(json.dumps(seal(b)))
    calls = []

    def stage(host, root, image, output, argv, *, gpu, seconds):
        calls.append((gpu, seconds))
        request = load(output / "request.json")
        assert all(c["artifact"]["path"].startswith("inputs/") for c in request["cases"])
        assert gpu and 0 < seconds < 60
        if change == "timeout":
            raise TimeoutError("deliberate supervised timeout")
        producers = {}; retired={}
        for case in request["cases"]:
            package = json.loads((output / case["artifact"]["path"]).read_text())
            value = seal(
                {
                    "version": "progressive-iteration-calibration/v1",
                    "source": digest("stale") if change == "stale" else request["source"],
                    "image": image,
                    "device": "cpu" if change == "cpu" else "actual_HIP",
                    "status": "PASS",
                    "compatibility": case["compatibility"],
                    "maximum_target_tokens": max(
                        len(r["tokens"]) for s in package["shards"] for r in s
                    ),
                    "effective_batch": package["update_policy"]["effective_batch"],
                    "measurements": {"update_us": 10, "probe_us": 20, "checkpoint_us": 30},
                    "measurement_scope": "TEST BOUNDARY ONLY; NOT ACTUAL HIP",
                    "checkpoint_bytes": 32 << 20,
                    "checkpoint_inodes": 32,
                }
            )
            path = output / "cases" / case["compatibility"][7:] / "producer.json"
            atomic_write(path, value)
            if change != "missing":
                producers[case["compatibility"]] = reference(output, path)
                if retirement:
                    from oeis_learn.pipeline_validation.progressive_calibration_storage import retire
                    # This is an explicitly simulated installed boundary. The
                    # separate CPU test actually saves/restores both snapshots.
                    atomic_write(path.parent/'checkpoints/generation/manifest.json',{'test_boundary_only':True})
                    rref=retire(path.parent,value,maximum_bytes=2*request['checkpoint_bytes'])
                    retired[case['compatibility']]=reference(output,path.parent/rref['path'])
        atomic_write(
            output / "producer.json",
            seal(
                {
                    "version": "progressive-calibration-run/v1",
                    "request": request["identity"],
                    "device": "actual_HIP",
                    "status": "PASS",
                    "producers": producers,
                    **({'retirements':retired} if retirement else {}),
                    "actual_elapsed_us": request["seconds"] * 1000000 + 1
                    if change == "overspend"
                    else 100,
                }
            ),
        )

    monkeypatch.setattr(h, "stage", stage)
    host = SimpleNamespace(engine=SimpleNamespace(name="docker"))
    args = SimpleNamespace(
        operation="prepare", blueprint="blueprint.json", qualification="qualification", output="out"
    )
    code = main(args, tmp_path, lambda _: host)
    capsys.readouterr()
    assert len(calls) == 1
    if change == "none":
        assert code == 0
        p = load(tmp_path / "out/prepared.json")
        assert (
            len([r for r in p["payload"]["references"] if r["name"].startswith("iteration:")]) == 1
        )
        assert all(t["iteration_budget"]["update_us"] == 20 for t in p["payload"]["task_plans"])
        args = SimpleNamespace(
            operation="freeze",
            blueprint="blueprint.json",
            prepared="out",
            qualification="qualification",
            output="frozen",
        )
        assert main(args, tmp_path, None) == 0
        c = load(tmp_path / "frozen/campaign.json")
        assert c["payload"]["storage"]["checkpoint_bytes"] >= 32 << 20
        if retirement:
            from oeis_learn.pipeline_validation.progressive_calibration import verify_result
            base=tmp_path/'out/calibration'
            result=load(base/'producer.json');request=load(base/'request.json')
            changed=seal(dict(result,retirements={}))
            with pytest.raises(ValueError,match='retirement inventory'):
                verify_result(base,request,changed)
            rref=next(iter(result['retirements'].values()))
            record=load(base/rref['path']);record['bytes']+=1
            (base/rref['path']).write_text(json.dumps(seal(record)))
            changed=copy.deepcopy(result)
            key=next(iter(changed['retirements']))
            changed['retirements'][key]=reference(base,rref['path'])
            with pytest.raises(ValueError,match='inventory differs'):
                verify_result(base,request,seal(changed))
    else:
        assert code != 0
        assert not (tmp_path / "out/prepared.json").exists()
    journal = next((tmp_path / "run-state/discovery-calibration").glob("*/budget.jsonl"))
    original = journal.read_bytes()
    args = SimpleNamespace(
        operation="prepare",
        blueprint="blueprint.json",
        qualification="qualification",
        output="sibling",
    )
    assert main(args, tmp_path, lambda _: host) != 0
    assert len(calls) == 1 and journal.read_bytes() == original


@pytest.mark.parametrize("axis", ["tokens", "disk_bytes", "elapsed_us", "maximum_inodes"])
def test_calibration_deficit_stops_before_engine_or_debit(tmp_path, monkeypatch, axis):
    from oeis_learn.pipeline_validation.progressive_commands import main

    host_inputs(tmp_path, monkeypatch)
    plan = load(tmp_path / "calibration-plan.json")
    if axis == "maximum_inodes":
        plan[axis] = 1
    else:
        plan["caps"][axis] = 1
    (tmp_path / "calibration-plan.json").write_text(json.dumps(seal(plan)))
    b = load(tmp_path / "blueprint.json")
    next(r for r in b["payload"]["references"] if r["name"] == "iteration_calibration_plan")[
        "artifact"
    ] = reference(tmp_path, "calibration-plan.json")
    (tmp_path / "blueprint.json").write_text(json.dumps(seal(b)))
    args = SimpleNamespace(
        operation="prepare", blueprint="blueprint.json", qualification="qualification", output="out"
    )
    assert (
        main(
            args, tmp_path, lambda _: pytest.fail("engine contacted with invalid setup reservation")
        )
        != 0
    )
    assert not (tmp_path / "out/prepared.json").exists()
    assert not (tmp_path / "run-state/discovery-calibration").exists()


@pytest.mark.parametrize("source", [False, True])
def test_calibration_resolves_closed_host_cycle_children(tmp_path, source):
    from oeis_learn.pipeline_validation.progressive_calibration import selected_inputs

    if source:
        from tests.integration.test_progressive_short_cycle import fixture

        campaign, _, _ = fixture(tmp_path)
    else:
        from tests.helpers.progressive_oeis_fixture import fixture

        campaign, _, _, _ = fixture(tmp_path)
    prepared = {
        "purpose": campaign["purpose"],
        "track": campaign["track"],
        "payload": {
            "task_plans": campaign["payload"]["tasks"],
            "references": campaign["payload"]["references"],
        },
    }
    selected, bindings = selected_inputs(tmp_path, prepared)
    assert len(selected) == 1 and len(bindings) == 4


def test_calibration_resolves_component_template_before_owned_targets_exist(
    tmp_path, fitting_client
):
    from oeis_learn.pipeline_validation.progressive_calibration import selected_inputs
    from tests.integration.test_progressive_component_flow import fixture

    campaign, _, _ = fixture(tmp_path, fitting_client)
    prepared = {
        "purpose": campaign["purpose"],
        "track": campaign["track"],
        "payload": {
            "task_plans": campaign["payload"]["tasks"],
            "references": campaign["payload"]["references"],
        },
    }
    selected, bindings = selected_inputs(tmp_path, prepared)
    assert len(selected) == len(bindings) == 1
    assert selected[0]["package"]["component_rules"]
    assert not (tmp_path / "mine").exists()


def test_full_registered_learning_domain_has_exact_calibration_cases(tmp_path):
    from itertools import product
    from oeis_learn.generalization.progressive.tuning import SPACE
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
    from oeis_learn.pipeline_validation.progressive_learner import compatibility
    from oeis_learn.pipeline_validation.progressive_calibration import selected_inputs
    prepared = inputs(tmp_path)
    plan = copy.deepcopy(prepared['payload']['task_plans'][0])
    package = load(tmp_path/'training-input.json')
    choices = []
    for values in product(*SPACE.values()):
        params = dict(zip(SPACE, values))
        profile = resolve_profile(package['profile']['base_recipe'], seed=plan['seed'], dropout=params['dropout'])
        policy = resolve_update_policy(profile,
            schedule_horizon_tokens=package['update_policy']['schedule_horizon_tokens'],
            microbatch_size=min(package['update_policy']['microbatch_size'],params['effective_batch']),
            **{k:v for k,v in params.items() if k!='dropout'})
        material = seal(dict(package, profile=profile, update_policy=policy))
        choices.append(dict(parameters=params, compatibility=compatibility(material), policy=plan['policy']))
    plan.update(trial_choices=choices, trial={'study':digest('full-domain'), 'ordinal':0, 'phase':'training', 'coordinate':0})
    prepared['payload']['task_plans'] = [plan]
    cases, bindings = selected_inputs(tmp_path, prepared)
    assert len(cases) == len(bindings) == 162
    assert {c['compatibility'] for c in cases} == {v['compatibility'] for v in choices}
