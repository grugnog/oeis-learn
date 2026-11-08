import pytest

from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.pipeline_validation.identity import seal
from tests.unit.test_progressive_freeze import campaign


def test_finite_slots_stop_when_predecessor_is_missing(tmp_path):
    c = campaign()
    store = RunStore(tmp_path, c)
    assert store.next_task() is None  # no task allocated to the prerequisite slot
    before = list(tmp_path.iterdir())
    assert store.read()["status"] == "FROZEN" and list(tmp_path.iterdir()) == before


def test_pause_is_owned_pending_request_and_resume_invocation_is_distinct(tmp_path):
    c = campaign()
    store = RunStore(tmp_path, c)
    store.append({"tasks": {}, "status": "RUNNING", "invocation": "first"})
    assert store.pause() == "PENDING_EXTERNAL_DEADLINE"
    assert store.pause_path().is_file()
    store.append({"tasks": {}, "status": "PAUSED", "invocation": "first"})
    store.append({"tasks": {}, "status": "RUNNING", "invocation": "second"})
    assert not store.pause_path().exists()
    changed = seal(dict(c, purpose="research"))
    with pytest.raises(ValueError):
        RunStore(tmp_path, changed)


@pytest.fixture(scope="module")
def installed_lifecycle(tmp_path_factory):
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    target = tmp_path_factory.mktemp("lifecycle-wheel")
    metadata = target / "metadata"
    metadata.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "egg_info",
            "--egg-base",
            str(metadata),
            "build",
            "--build-base",
            str(target / "build"),
            "bdist_wheel",
            "--bdist-dir",
            str(target / "bdist"),
            "--dist-dir",
            str(target / "dist"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert built.returncode == 0, built.stderr[-4096:]
    [wheel] = (target / "dist").glob("*.whl")
    installed = target / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    return installed


def test_actual_installed_role_interrupt_resume_and_successor(tmp_path, installed_lifecycle):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    c, tasks, channel, charge = fixture(tmp_path)
    first, _one = invoke_cpu(
        tmp_path, c, tasks[0], channel, ordinal=0, pythonpath=installed_lifecycle
    )
    assert first["mechanical_status"] == "PARTIAL", first
    assert first["metrics"]["task_status"] == "PAUSED"
    resumed, two = invoke_cpu(
        tmp_path, c, tasks[0], channel, ordinal=1, pythonpath=installed_lifecycle
    )
    assert resumed["mechanical_status"] == "COMPLETE", resumed
    successor, three = invoke_cpu(
        tmp_path, c, tasks[1], channel, ordinal=2, pythonpath=installed_lifecycle
    )
    assert successor["mechanical_status"] == "COMPLETE", successor
    assert successor["metrics"]["charged_tokens"] == 5 * charge
    assert successor["metrics"]["discarded_tokens"] == charge
    assert successor["metrics"]["useful_tokens"] == 4 * charge
    from oeis_learn.generalization.progressive.contracts import load

    assert load(two / "role/checkpoint.json")["payload"]["model_update"] == 2
    assert load(three / "role/checkpoint.json")["payload"]["model_update"] == 4
    assert list((three / "role/progress").glob("*.json"))
    final = load(three / "role/accounting-final.json")
    parent = load(two / "role/accounting-final.json")
    assert final["milestone"]["total_cost_us"] > parent["milestone"]["total_cost_us"] > 0
    assert final["milestone"]["ancestor_tokens"] == 3 * charge
    from oeis_learn.pipeline_validation.progressive_acceptance import export_lifecycle

    archive = export_lifecycle(tmp_path, [_one, two, three], tmp_path / "continuation.zip")
    assert archive["bytes"] > 0


def test_installed_role_partial_and_failure_publish_original_diagnostics(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    partial = tmp_path / "partial"
    c, tasks, channel, charge = fixture(partial, interrupt=False, first_cap_updates=1)
    result, out = invoke_cpu(partial, c, tasks[0], channel, ordinal=0)
    assert result["metrics"]["task_status"] == "PARTIAL", result
    assert result["metrics"]["charged_tokens"] == charge
    failed = tmp_path / "failed"
    c, tasks, channel, _ = fixture(failed, interrupt=False, fail=True)
    result, out = invoke_cpu(failed, c, tasks[0], channel, ordinal=0)
    assert result["mechanical_status"] == "FAILED"
    assert result["producer_error"]["type"] == "ValueError"
    assert result["producer_error"]["local_traceback"]
    assert (out / "role/private/producer-traceback.txt").is_file()


def test_weights_only_child_is_new_initialization_with_fresh_clocks(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    c, tasks, channel, _ = fixture(tmp_path, interrupt=False, weights_only=True)
    control = tmp_path / "control"
    atomic_write(
        control / ("pause-" + digest("invocation-0").split(":")[1] + ".json"),
        seal({"campaign": c["identity"], "requested": True}),
    )
    result, out = invoke_cpu(tmp_path, c, tasks[0], channel, ordinal=0)
    assert result["metrics"]["initialization"] == "new_initialization", result
    assert result["metrics"]["charged_tokens"] == 0
    assert result["metrics"]["ancestor_cost_available"] is False
    checkpoint = load(out / "role/checkpoint.json")
    assert checkpoint["payload"]["model_update"] == 0
    completed, _ = invoke_cpu(tmp_path, c, tasks[0], channel, ordinal=1)
    assert completed["mechanical_status"] == "COMPLETE"
    successor, out = invoke_cpu(tmp_path, c, tasks[1], channel, ordinal=2)
    assert successor["mechanical_status"] == "COMPLETE"
    assert successor["metrics"]["ancestor_cost_available"] is False
    assert load(out / "role/accounting-final.json")["ancestor_cost_available"] is False


def test_installed_eligible_diagnostic_and_complete_partial_failed_exports(
    tmp_path, installed_lifecycle
):
    import copy
    import zipfile

    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import digest, load
    from oeis_learn.pipeline_validation.progressive_acceptance import (
        export_lifecycle,
        fixture,
        invoke_cpu,
        invoke_decision_cpu,
        receipt,
    )
    from tests.unit.test_progressive_cohorts import fixture as panel_fixture
    from tests.unit.test_progressive_decisions import fixture as outcomes

    c, _tasks, channel, _charge = fixture(tmp_path, interrupt=False)
    data, roster, design, policy = outcomes()
    # Explicit mechanical outcome fixture: no native evaluation/research claim.
    for row in data:
        if row["seed"] == 2 and row["arm"] == "candidate" and row["stratum"] != "retention":
            row["outcome"] = "failure"
    gain = next(t for t in policy["payload"]["targets"] if t["id"] == "gain")
    gain.update(allowed_actions=["RETAIN_BASELINE", "EXPLORE_RESERVED"], budget_ref="diag")
    diag_caps = copy.deepcopy(c["payload"]["graph"]["nodes"][1]["caps"])
    policy["payload"]["diagnostics"] = [
        {
            "id": "diag",
            "failure_class": "heldout-deficit",
            "trigger": "paired-heldout-deficit",
            "signal": "heldout_yield",
            "design": digest(design),
            "panel": "heldout",
            "task_slots": ["diagnostic"],
            "reservation": diag_caps,
            "maximum_visits": 1,
            "terminal_action": "DEFER_BRANCH",
            "assessment": False,
        }
    ]
    policy = seal(policy)
    source = copy.deepcopy(c["payload"]["tasks"][0])
    source.update(id="decision", kind="packaging", token_target=0)
    child = copy.deepcopy(c["payload"]["tasks"][0])
    child.update(id="diagnostic", slot="diagnostic", stage="diversity", output="diagnostic")
    graph = c["payload"]["graph"]
    graph["edges"][0]["condition"] = "CONFIRMED_GAIN"
    graph["nodes"].append(
        {
            "id": "diagnostic",
            "stage": "diversity",
            "kind": "diagnostic",
            "dependencies": ["first"],
            "maximum_visits": 1,
            "caps": diag_caps,
        }
    )
    graph["edges"].append(
        {
            "source": "first",
            "target": "diagnostic",
            "condition": "INCONCLUSIVE",
            "action": "EXPLORE_RESERVED",
        }
    )
    graph["diagnostic_pool"] = diag_caps
    c["payload"]["tasks"] = [source, child]
    c["payload"]["caps"] = {k: sum(n["caps"][k] for n in graph["nodes"]) for k in diag_caps}
    c["payload"]["target_policy"] = policy["identity"]
    c = seal(c)
    store = RunStore(tmp_path / "dag", c)
    assert store.next_task()["payload"]["id"] == "decision"
    request = {
        "version": "progressive-decision-input/v1",
        "source_slot": "first",
        "target_slot": "diagnostic",
        "rows": data,
        "roster": roster,
        "seeds": [1, 2],
        "design": design,
        "policy": policy,
        "remaining": {"diag": diag_caps},
        "kwargs": {
            "target_id": "gain",
            "action": "EXPLORE_RESERVED",
            "trigger": "paired-heldout-deficit",
        },
    }
    result, decision_output = invoke_decision_cpu(
        tmp_path, request, ordinal=0, pythonpath=installed_lifecycle
    )
    assert result["scientific_status"] == "INCONCLUSIVE"
    bundle = load(decision_output / "role/decision-bundle.json")
    panel, _ = panel_fixture(tmp_path / "panels")
    # Bind the exact public outcome roster to this mechanical confirmation panel.
    template = panel["payload"]["rows"][0]
    panel["payload"]["rows"] = [
        dict(template, id=r["task"], stratum=r["stratum"], groups=[r["group"]]) for r in roster
    ]
    panel = seal(panel)
    store.authorize(bundle, policy=policy, remaining={"diag": diag_caps}, panels={"heldout": panel})
    original = store.state["decisions"].copy()
    assert next(iter(original.values()))["panel"] == panel["identity"]
    source_task = store.next_task()
    store.boundary(source_task, "COMPLETE", "decision")
    selected = store.next_task()
    assert selected["payload"]["slot"] == "diagnostic"
    trained, train_output = invoke_cpu(
        tmp_path, c, selected, channel, ordinal=0, pythonpath=installed_lifecycle
    )
    assert trained["mechanical_status"] == "COMPLETE", trained
    store.boundary(selected, "COMPLETE", "train")
    assert store.next_task() is None
    assert store.state["decisions"] == original
    with pytest.raises(ValueError):
        store.authorize(
            seal(dict(bundle, target_slot="second")),
            policy=policy,
            remaining={"diag": diag_caps},
            panels={"heldout": panel},
        )
    outputs = [decision_output, train_output]
    for name, kwargs in [("partial", {"first_cap_updates": 1}), ("failed", {"fail": True})]:
        folder = tmp_path / name
        campaign, plans, ch, _ = fixture(folder, interrupt=False, **kwargs)
        outcome, out = invoke_cpu(
            folder, campaign, plans[0], ch, ordinal=0, pythonpath=installed_lifecycle
        )
        assert outcome["mechanical_status"] == ("FAILED" if name == "failed" else "PARTIAL")
        outputs.append(out)
    archive = export_lifecycle(tmp_path, outputs, tmp_path / "terminal-paths.zip")
    with zipfile.ZipFile(tmp_path / "terminal-paths.zip") as exported:
        assert all(
            "private" not in n and not n.endswith(".pt") and "inputs" not in n
            for n in exported.namelist()
        )
        failed = __import__("json").loads(exported.read("invocation-3/result.json"))
        assert failed["producer_error"]["local_traceback"]
    replay = tmp_path / "replay"
    campaign, plans, ch, _ = fixture(replay)
    for ordinal, plan in enumerate((plans[0], plans[0], plans[1])):
        result, out = invoke_cpu(
            replay, campaign, plan, ch, ordinal=ordinal, pythonpath=installed_lifecycle
        )
        outputs.append(out)
    checks = dict.fromkeys(
        [
            "interruption",
            "resume",
            "continuation",
            "partial",
            "failure",
            "eligible_diagnostic",
            "export",
        ],
        True,
    )
    archive = export_lifecycle(tmp_path, outputs, tmp_path / "all-paths.zip")
    value = receipt(tmp_path, outputs, checks=checks, archive=archive)
    assert not value["qualified"] and value["actual_elapsed_hours"] < 1
    assert value["sustained_duration_status"] == value["hardware_status"] == "PENDING"
    assert (
        BudgetJournal(tmp_path / "accounting", c["identity"], c["payload"]["caps"]).task_remaining(
            selected
        )["tokens"]
        < selected["payload"]["caps"]["tokens"]
    )


@pytest.mark.parametrize("bad_origin", ["unknown", "assisted"])
def test_installed_learner_rechecks_strict_import_origin(tmp_path, bad_origin):
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        load,
        reference,
        record,
    )
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_learner import run

    c, tasks, _, _ = fixture(tmp_path, interrupt=False, weights_only=True)
    package = load(tmp_path / "training-input.json")
    init = c["payload"]["tasks"][0]["initializer"]
    if bad_origin == "unknown":
        del init["origin"]
        del package["origin"]
    else:
        origin = load(tmp_path / init["origin"]["path"])
        exposure = load(tmp_path / origin["exposure"]["path"])
        atomic_write(tmp_path / "assisted.json", seal(dict(exposure, track="external_assisted")))
        origin["exposure"] = reference(tmp_path, "assisted.json")
        atomic_write(tmp_path / "altered-origin.json", seal(origin))
        init["origin"] = package["origin"] = reference(tmp_path, "altered-origin.json")
    c = seal(c)
    task = record(
        "task",
        dict(c["payload"]["tasks"][0], campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    authority = seal(
        dict(
            version="progressive-supervisor/v1",
            campaign=c,
            task=task,
            clock={},
            invocation="invalid-origin",
        )
    )
    with pytest.raises(ValueError, match="strict|assisted"):
        run(
            seal(package),
            authority,
            accounting=tmp_path / "unused-accounting",
            control=tmp_path / "unused-control",
            output=tmp_path / "unused-output",
            state_root=tmp_path / "unused-state",
            initializer_root=tmp_path,
        )
    assert not (tmp_path / "unused-state").exists()


def test_installed_initial_checkpoint_preflights_storage(tmp_path):
    from oeis_learn.generalization.progressive.contracts import load, record
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture
    from oeis_learn.pipeline_validation.progressive_learner import run

    c, _, _, _ = fixture(tmp_path, interrupt=False)
    c["payload"]["tasks"][0]["caps"]["disk_bytes"] = c["payload"]["storage"]["package_bytes"] - 1
    c = seal(c)
    t = record(
        "task",
        dict(c["payload"]["tasks"][0], campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    authority = seal(
        dict(
            version="progressive-supervisor/v1", campaign=c, task=t, clock={}, invocation="no-room"
        )
    )
    with pytest.raises(TimeoutError, match="storage"):
        run(
            load(tmp_path / "training-input.json"),
            authority,
            accounting=tmp_path / "unused-accounting",
            control=tmp_path / "unused-control",
            output=tmp_path / "unused-output",
            state_root=tmp_path / "unused-state",
        )
    assert not (tmp_path / "unused-state").exists()


def test_resume_storage_counts_retained_invocation_outputs(tmp_path):
    from oeis_learn.generalization.progressive.storage import register_invocation, owned_guard
    from oeis_learn.generalization.progressive.contracts import atomic_write, record

    c = campaign()
    plan = c["payload"]["tasks"][0]
    t = record(
        "task",
        dict(plan, campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    register_invocation(tmp_path, c, t, tmp_path / "journal", tmp_path / "old")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "retained.bin").write_bytes(b"x" * (plan["caps"]["disk_bytes"] - 1000))
    proof = register_invocation(tmp_path, c, t, tmp_path / "journal", tmp_path / "resumed")
    assert proof["prior_bytes"] == plan["caps"]["disk_bytes"] - 1000
    guard = owned_guard(dict(campaign=c, task=t, storage=proof), output=tmp_path / "resumed")
    with guard.activate(), pytest.raises(TimeoutError, match="storage"):
        atomic_write(tmp_path / "resumed" / "new.json", {"value": "x" * 2000})
    assert not (tmp_path / "resumed").exists()
