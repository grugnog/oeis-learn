"""Freeze rejects incomplete authority before touching engine or learning state."""

import json
import pytest
from tests.helpers.progressive_fixture import blueprint, task, caps, graph, envelope
from oeis_learn.generalization.progressive.freeze import freeze, IncompleteFreeze
from oeis_learn.generalization.progressive.campaign import validate_campaign, validate_task
from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.pipeline_validation.identity import digest, seal


def prepared(root, b):
    for name in [
        "runtime",
        "checkpoint",
        "targets",
        "recipe",
        "source",
        "image",
        "groups",
        "policy",
        "qualification",
        "prepared",
    ]:
        (root / (name + ".json")).write_text("{}")
    plan = task()["payload"]
    plan.pop("campaign")
    return envelope(
        "prepared",
        dict(
            blueprint=b["identity"],
            runtime=reference(root, "runtime.json"),
            graph=graph(),
            task_plans=[plan],
            references=[
                dict(
                    name=kind, role="custodian", kind=kind, artifact=reference(root, kind + ".json")
                )
                for kind in [
                    "checkpoint",
                    "recipe",
                    "source",
                    "image",
                    "groups",
                    "policy",
                    "qualification",
                    "prepared",
                ]
            ],
            caps=caps(),
            checkpoint=reference(root, "checkpoint.json"),
            milestone_pins=0,
            shard_bytes=0,
            package_bytes=100,
            required_inodes=10,
            target_policy=reference(root, "targets.json"),
            deficits=[],
        ),
    )


def test_unresolved_blueprint_and_named_deficits_block_freeze(tmp_path):
    b = blueprint()
    p = prepared(tmp_path, b)
    with pytest.raises(IncompleteFreeze) as exc:
        freeze(tmp_path, b, p, qualification_root="unused")
    assert set(exc.value.deficits) == set(b["payload"]["unresolved"]) - {"qualification"}
    p["payload"]["deficits"] = ["insufficient_fresh_groups"]
    p = seal(p)
    with pytest.raises(IncompleteFreeze, match="insufficient_fresh_groups"):
        freeze(tmp_path, b, p, qualification_root="unused")


def campaign():
    plan = task()["payload"]
    plan.pop("campaign")
    plan["slot"] = "diversity"
    g = graph()
    cap = caps(**{k: 200 * v for k, v in caps().items()})
    return envelope(
        "campaign",
        dict(
            blueprint=digest("b"),
            runtime=digest("r"),
            graph=g,
            tasks=[plan],
            references=[
                dict(
                    name="policy",
                    role="custodian",
                    kind="policy",
                    artifact=dict(path="policy.json", bytes=2, sha256=digest("p")),
                )
            ],
            caps=cap,
            storage=dict(
                checkpoint_bytes=100,
                atomic_generations=2,
                milestone_pins=2,
                shard_bytes=300,
                package_bytes=100,
                required_bytes=800,
                available_bytes=900,
                required_inodes=10,
                available_inodes=20,
            ),
            target_policy=digest("targets"),
        ),
    )


def test_storage_graph_and_frozen_child_task_entitlement():
    c = campaign()
    validate_campaign(c)
    t = envelope("task", dict(c["payload"]["tasks"][0], campaign=c["identity"]))
    validate_task(t, campaign=c)
    for changed in [
        dict(c["payload"]["storage"], required_bytes=100),
        dict(c["payload"]["storage"], available_bytes=799),
        dict(c["payload"]["storage"], available_inodes=9),
    ]:
        with pytest.raises(ValueError):
            validate_campaign(seal(dict(c, payload=dict(c["payload"], storage=changed))))
    bad = seal(dict(t, payload=dict(t["payload"], id="sibling", output="new/path")))
    with pytest.raises(ValueError):
        validate_task(bad, campaign=c)


def test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(tmp_path, monkeypatch):
    """Synthetic original bytes test assembly only; no device qualification claim."""
    from oeis_learn.pipeline_validation.progressive_roles import ROLE_POLICY
    from oeis_learn.generalization.progressive.targets import defaults
    from oeis_learn.experiments import source_snapshot

    def put(folder, name, value):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(json.dumps(value))
        return reference(folder, name)

    source = {"diagnostic_fixture": digest("source")}
    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda _: source)
    qroot = tmp_path / "qualification"
    sourceref = put(qroot, "source.json", source)
    buildref = put(qroot, "build.json", {"build": "fixture"})
    from oeis_learn.fresh.bigint_config import resolve

    recipe = resolve(dict(version="bigint-bootstrap/v1"))
    policy = digest("numeric")
    model = digest({k: v for k, v in recipe["model"].items() if k != "seed"})
    device = seal(
        dict(
            version="generalization-device-evidence/v1",
            device="hip",
            loaded_weights=digest("weights"),
            finite_forward=True,
            finite_backward=True,
            parameters_changed=True,
            resume_equal=True,
            execution_policy=policy,
        )
    )
    dref = put(qroot, "device.json", device)
    facts = dict(
        source_closure=sourceref,
        build_dependencies=[buildref],
        images=[digest("cpu"), digest("gpu")],
        runtime=dict(
            hip="fixture", driver="fixture", python="3.12", torch="fixture", execution_policy=policy
        ),
        hardware=dict(
            hip_available=True, device="DIAGNOSTIC_ONLY_NOT_ACTUAL_HARDWARE", memory_bytes=1024
        ),
        precision="fp32",
        resources=dict(
            memory_bytes=1024,
            cpus=1,
            torch_intraop=2,
            torch_interop=1,
            native_threads=1,
            learners=1,
        ),
        engine_security=dict(
            engine="docker",
            runtime_socket=False,
            engine_version="fixture",
            runtime="fixture",
            rootless=True,
            selinux=False,
            network=False,
            read_only_root=True,
            no_new_privileges=True,
            capabilities="none",
        ),
    )
    suites = ["bigint-cpu", "bigint-gpu"]
    capabilities = ["model:" + model, "language:bigint-body/v1"]
    producer = seal(
        dict(
            version="generalization-producer/v1",
            status="PASS",
            facts=facts,
            suites=suites,
            capabilities=capabilities,
            artifacts=[dref],
            dependencies=[],
            feature_acceptance=False,
        )
    )
    pref = put(qroot, "producer.json", producer)
    q = seal(
        dict(
            version="generalization-qualification/v2",
            **facts,
            suite_selection=suites,
            covered_capabilities=capabilities,
            producer=dict(receipt=pref, status="PASS", artifacts=[dref]),
            artifact_inventory=[],
            status="PASS",
        )
    )
    put(qroot, "qualification.json", q)
    b = blueprint()
    b["payload"]["unresolved"] = []
    b["payload"]["proposed_caps"] = caps(**{k: 200 * v for k, v in caps().items()})
    b = seal(b)
    p = prepared(tmp_path, b)
    runtime = envelope(
        "runtime",
        dict(
            scope=dict(
                source=sourceref["sha256"],
                images=facts["images"],
                model=model,
                language="bigint-body/v1",
                precision="fp32",
                device="cpu",
                execution_policy=policy,
            ),
            qualification=[reference(tmp_path, qroot / "qualification.json")],
            operations=["census", "audit-language", "publish"],
            role_policy=ROLE_POLICY,
            fixture=True,
        ),
    )
    put(tmp_path, "runtime.json", runtime)
    target = defaults()
    target["payload"]["selected"] = True
    target = seal(target)
    put(tmp_path, "targets.json", target)
    put(tmp_path, "recipe.json", recipe)
    put(tmp_path, "image.json", dict(version="discovery-images/v1", images=facts["images"]))
    put(
        tmp_path,
        "groups.json",
        envelope(
            "exposure", dict(previous=None, detector_limits=["diagnostic fixture"], events=[])
        ),
    )
    for item in p["payload"]["references"]:
        if item["kind"] == "qualification":
            item["artifact"] = reference(tmp_path, qroot / "qualification.json")
        elif item["kind"] == "policy":
            item["artifact"] = reference(tmp_path, "targets.json")
        else:
            item["artifact"] = reference(tmp_path, item["kind"] + ".json")
    p["payload"]["runtime"] = reference(tmp_path, "runtime.json")
    p["payload"]["target_policy"] = reference(tmp_path, "targets.json")
    p["payload"]["caps"] = b["payload"]["proposed_caps"]
    p = seal(p)
    from oeis_learn.generalization.progressive.freeze import learning_compatibility

    p["payload"]["task_plans"][0]["policy"] = target["identity"]
    p["payload"]["task_plans"][0]["initializer"]["compatibility"] = learning_compatibility(
        recipe, p["payload"]["references"], runtime["payload"]["scope"], b["track"]
    )
    p = seal(p)
    campaign, runtime, tasks = freeze(tmp_path, b, p, qualification_root="qualification")
    assert len(tasks) == 1 and tasks[0]["payload"]["campaign"] == campaign["identity"]
    assert campaign["payload"]["storage"]["checkpoint_bytes"] == p["payload"]["checkpoint"]["bytes"]
    assert campaign["payload"]["runtime"] == runtime["identity"]
    # Full successful freeze, including recipe/original-qualification checks, is
    # host-stdlib-only. The source mapping is an explicit diagnostic test seam.
    import subprocess
    import sys
    from pathlib import Path

    put(tmp_path, "test-blueprint.json", b)
    put(tmp_path, "test-prepared.json", p)
    code = """import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from oeis_learn.experiments import source_snapshot
from oeis_learn.generalization.progressive.freeze import freeze
root=Path(sys.argv[2])
source_snapshot.collect_snapshot=lambda _:json.loads((root/'qualification/source.json').read_text())
freeze(root,json.loads((root/'test-blueprint.json').read_text()),json.loads((root/'test-prepared.json').read_text()),qualification_root='qualification')
assert not {'torch','optuna','jsonschema'} & set(sys.modules)
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            code,
            str(Path(__file__).resolve().parents[2] / "src"),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
