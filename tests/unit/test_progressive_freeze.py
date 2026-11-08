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
    assert campaign["payload"]["storage"]["checkpoint_bytes"] >= p["payload"]["checkpoint"]["bytes"]
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


def panel_inputs(root, monkeypatch, *, role="confirmation", second_role=None):
    from tests.unit.test_progressive_cohorts import fixture

    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(root, monkeypatch)
    b = json.loads((root / "test-blueprint.json").read_text())
    p = json.loads((root / "test-prepared.json").read_text())
    c, e = fixture(root, role=role)
    # Bind the original detector/exposure producer, not a fresh replacement cut.
    e = json.loads((root / "groups.json").read_text())
    c["payload"]["exposure"] = e["identity"]
    panels = [seal(c)]
    if second_role:
        other = json.loads(json.dumps(c))
        other["payload"].update(role=second_role, panel="other", decision="other")
        panels.append(seal(other))
    for i, panel in enumerate(panels):
        path = root / f"panel-{i}.json"
        path.write_text(json.dumps(panel))
        p["payload"]["references"].append(
            dict(name=f"panel-{i}", kind="groups", role="custodian", artifact=reference(root, path))
        )
    return b, seal(p), panels, e


@pytest.mark.parametrize("other_role", ["confirmation", "development", "assessment"])
def test_freeze_rejects_cross_panel_overlap(tmp_path, monkeypatch, other_role):
    b, p, _, _ = panel_inputs(tmp_path, monkeypatch, second_role=other_role)
    with pytest.raises(ValueError, match="cross-panel"):
        freeze(tmp_path, b, p, qualification_root="qualification")


def test_freeze_rejects_missing_exposure_producer(tmp_path, monkeypatch):
    b, p, panels, _ = panel_inputs(tmp_path, monkeypatch)
    panels[0]["payload"]["exposure"] = digest("missing original cut")
    (tmp_path / "panel-0.json").write_text(json.dumps(seal(panels[0])))
    p["payload"]["references"][-1]["artifact"] = reference(tmp_path, "panel-0.json")
    with pytest.raises(ValueError, match="exposure cut"):
        freeze(tmp_path, b, seal(p), qualification_root="qualification")


def test_freeze_rejects_historical_exposure_hidden_by_new_group_name(tmp_path, monkeypatch):
    b, p, panels, e = panel_inputs(tmp_path, monkeypatch)
    from oeis_learn.generalization.progressive.exposure import append

    row = panels[0]["payload"]["rows"][0]
    truth = json.loads((tmp_path / row["truth"]["path"]).read_text())
    newer = append(
        e,
        [
            dict(
                group="old-name",
                relations=["window:" + digest(truth["values"][:20])],
                role="development",
                endpoint="parameter",
                source="fresh-002",
                access="truth",
                campaign="historical",
            )
        ],
        created_at=e["created_at"],
    )
    (tmp_path / "later-exposure.json").write_text(json.dumps(newer))
    p["payload"]["references"].append(
        dict(
            name="later-exposure",
            kind="groups",
            role="custodian",
            artifact=reference(tmp_path, "later-exposure.json"),
        )
    )
    with pytest.raises(ValueError, match="exposed group"):
        freeze(tmp_path, b, seal(p), qualification_root="qualification")


def test_freeze_detects_value_overlap_despite_distinct_declared_fingerprints(tmp_path, monkeypatch):
    from tests.unit.test_progressive_cohorts import fixture

    b, p, _, e = panel_inputs(tmp_path, monkeypatch)
    other, _ = fixture(tmp_path, role="development", panel="unrelated-label")
    other["payload"]["exposure"] = e["identity"]
    (tmp_path / "other.json").write_text(json.dumps(seal(other)))
    p["payload"]["references"].append(
        dict(
            name="other",
            kind="groups",
            role="custodian",
            artifact=reference(tmp_path, "other.json"),
        )
    )
    with pytest.raises(ValueError, match="cross-panel"):
        freeze(tmp_path, b, seal(p), qualification_root="qualification")


def test_freeze_accepts_bound_single_panel_and_rejects_erased_history(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.exposure import append

    b, p, _, e = panel_inputs(tmp_path, monkeypatch)
    freeze(tmp_path, b, p, qualification_root="qualification")
    changed = append(e, [], created_at=e["created_at"])
    changed["payload"]["detector_limits"] = ["erased detector scope"]
    (tmp_path / "history.json").write_text(json.dumps(seal(changed)))
    p["payload"]["references"].append(
        dict(
            name="history",
            kind="groups",
            role="custodian",
            artifact=reference(tmp_path, "history.json"),
        )
    )
    with pytest.raises(ValueError, match="history cannot be erased"):
        freeze(tmp_path, b, seal(p), qualification_root="qualification")


def test_freeze_rejects_strict_import_without_original_lineage(tmp_path, monkeypatch):
    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(tmp_path, monkeypatch)
    b = json.loads((tmp_path / "test-blueprint.json").read_text())
    p = json.loads((tmp_path / "test-prepared.json").read_text())
    p["payload"]["task_plans"][0]["initializer"].update(
        mode="weights_only", artifact=p["payload"]["checkpoint"]
    )
    with pytest.raises(ValueError, match="strict.*origin"):
        freeze(tmp_path, b, seal(p), qualification_root="qualification")


@pytest.mark.parametrize("origin_track", ["synthetic_strict", "external_assisted"])
def test_whole_freeze_retained_strict_origin(tmp_path, monkeypatch, origin_track):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture as learning_fixture

    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(tmp_path, monkeypatch)
    _, tasks, _, _ = learning_fixture(tmp_path, interrupt=False, weights_only=True)
    init = tasks[0]["payload"]["initializer"]
    if origin_track == "external_assisted":
        origin = load(tmp_path / init["origin"]["path"])
        exposure = load(tmp_path / origin["exposure"]["path"])
        (tmp_path / "assisted-exposure.json").write_text(
            json.dumps(seal(dict(exposure, track=origin_track)))
        )
        origin["exposure"] = reference(tmp_path, "assisted-exposure.json")
        (tmp_path / "assisted-origin.json").write_text(json.dumps(seal(origin)))
        init = dict(init, origin=reference(tmp_path, "assisted-origin.json"))
    b, p = (load(tmp_path / ("test-" + name + ".json")) for name in ("blueprint", "prepared"))
    p["payload"]["references"].append(
        dict(name="strict-retained", role="learner", kind="checkpoint", artifact=init["artifact"])
    )
    original = p["payload"]["task_plans"][0]["initializer"]
    original.update(mode="weights_only", artifact=init["artifact"], origin=init["origin"])
    if origin_track == "synthetic_strict":
        frozen, _, _ = freeze(tmp_path, b, seal(p), qualification_root="qualification")
        assert frozen["payload"]["tasks"][0]["initializer"]["origin"] == init["origin"]
    else:
        with pytest.raises(ValueError, match="assisted"):
            freeze(tmp_path, b, seal(p), qualification_root="qualification")


def test_freeze_measures_tensor_closure_not_manifest_bytes(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.storage import checkpoint_size

    # The manifest is tiny; the original tensor producer is not.
    (tmp_path / "weights.pt").write_bytes(b"w" * 100_000)
    (tmp_path / "optimizer.pt").write_bytes(b"o" * 200_000)
    full = seal(
        dict(
            version="generalization-learner-state/v1",
            states={
                "weights": reference(tmp_path, "weights.pt"),
                "optimizer": reference(tmp_path, "optimizer.pt"),
            },
        )
    )
    (tmp_path / "state.json").write_text(json.dumps(full))
    measured, inodes = checkpoint_size(
        tmp_path, reference(tmp_path, "state.json"), checkpoint_root="."
    )
    assert measured >= 300_000 and inodes >= 3


def test_freeze_reserves_all_retained_task_stores(tmp_path, monkeypatch):
    test_complete_diagnostic_freeze_binds_original_producer_and_task_roster(tmp_path, monkeypatch)
    b, p = (
        json.loads((tmp_path / ("test-" + name + ".json")).read_text())
        for name in ("blueprint", "prepared")
    )
    first = p["payload"]["task_plans"][0]
    second = json.loads(json.dumps(first))
    second.update(id="second-store", output="second-store", slot="capability", stage="capability")
    p["payload"]["task_plans"].append(second)
    frozen, _, _ = freeze(tmp_path, b, seal(p), qualification_root="qualification")
    assert (
        frozen["payload"]["storage"]["task_store_bytes"]
        == first["caps"]["disk_bytes"] + second["caps"]["disk_bytes"]
    )
    assert (
        frozen["payload"]["storage"]["required_bytes"]
        >= frozen["payload"]["storage"]["task_store_bytes"]
    )


def test_storage_measurement_includes_nested_projection_payload(tmp_path):
    from oeis_learn.generalization.progressive.storage import projection_size

    (tmp_path / "model.pt").write_bytes(b"w" * 100_000)
    channel = seal(
        dict(
            version="discovery-channel/v1",
            kind="checkpoint",
            role="generator",
            data=dict(artifact=reference(tmp_path, "model.pt")),
        )
    )
    (tmp_path / "channel.json").write_text(json.dumps(channel))
    descriptor = reference(tmp_path, "channel.json")
    measured, inodes = projection_size(tmp_path, [descriptor])
    assert measured == 100_000 + descriptor["bytes"]
    assert inodes >= 2


def test_research_storage_rejects_weights_only_or_incomplete_manifest(tmp_path):
    from oeis_learn.generalization.progressive.storage import checkpoint_size

    (tmp_path / "weights.pt").write_bytes(b"w" * 100)
    with pytest.raises(ValueError, match="full-state"):
        checkpoint_size(tmp_path, reference(tmp_path, "weights.pt"), research=True)
    partial = seal(
        dict(
            version="generalization-learner-state/v1",
            contract=digest("contract"),
            ledger=dict(cap=100),
            states=dict(weights=reference(tmp_path, "weights.pt")),
        )
    )
    (tmp_path / "partial.json").write_text(json.dumps(partial))
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_size(
            tmp_path, reference(tmp_path, "partial.json"), checkpoint_root=".", research=True
        )
