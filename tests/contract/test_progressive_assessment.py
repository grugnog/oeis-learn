import pytest
from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    reference,
    seal,
    record,
    digest,
)
from oeis_learn.generalization.progressive.assessment import validate_selection, expose
from tests.contract.test_progressive_roles import artifact
from tests.unit.test_progressive_freeze import campaign


def fixture(root):
    (root / "weights.pt").write_bytes(b"identified diagnostic weights")
    checkpoint = reference(root, "weights.pt")
    candidate = artifact(
        root, "candidate.json", "candidate", "assessment", dict(checkpoint=checkpoint["sha256"])
    )
    truth = artifact(
        root,
        "truth.json",
        "assessment_truth",
        "assessment",
        dict(id="t", indices=list(map(str, range(100))), values=["0"] * 100),
    )
    c = campaign()
    c["payload"]["tasks"][0].update(kind="packaging", token_target=0)
    c["payload"]["references"].append(
        dict(name="final-panel", role="assessment", kind="source", artifact=truth.ref)
    )
    c = seal(c)
    t = record(
        "task",
        dict(c["payload"]["tasks"][0], campaign=c["identity"]),
        created_at=c["created_at"],
        purpose=c["purpose"],
        track=c["track"],
    )
    selection = seal(
        dict(
            version="progressive-locked-selection/v1",
            campaign=c["identity"],
            task=t,
            checkpoint=checkpoint,
            library=None,
            panel=[truth.ref],
            candidates=candidate.ref,
        )
    )
    return c, selection, truth


def test_selection_panel_and_checkpoint_binding(tmp_path):
    c, s, _ = fixture(tmp_path)
    assert validate_selection(tmp_path, s, c) == s
    for changes in [
        dict(campaign=digest("wrong")),
        dict(checkpoint=dict(s["checkpoint"], sha256=digest("wrong"))),
        dict(panel=[]),
        dict(library=dict(s["checkpoint"], path="missing")),
    ]:
        with pytest.raises((ValueError, FileNotFoundError)):
            validate_selection(tmp_path, seal(dict(s, **changes)), c)


def test_assessment_exposure_is_one_way_and_blocks_adaptive_resume(tmp_path):
    from oeis_learn.generalization.progressive.lifecycle import RunStore

    c, s, _ = fixture(tmp_path)
    validate_selection(tmp_path, s, c)
    e = expose(tmp_path, c, s)
    assert e["untouched"] is False
    with pytest.raises(ValueError):
        expose(tmp_path, c, s)
    control = tmp_path / "run-state/discovery-control" / c["identity"].split(":")[1]
    with pytest.raises(ValueError, match="assessment exposed"):
        RunStore(control, c).next_task()


@pytest.mark.parametrize(
    "operation,channel",
    [
        ("train", "training"),
        ("propose-trial", "search"),
        ("review-claim", "dossier"),
        ("score", "score_truth"),
    ],
)
def test_other_roles_cannot_project_assessment_truth(tmp_path, operation, channel):
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    _, _, truth = fixture(tmp_path)
    with pytest.raises(ValueError):
        project(tmp_path, "forbidden", operation, {channel: [truth]}, registry=REGISTRY)


def test_native_assessment_scores_only_locked_final_checkpoint(tmp_path, fitting_client):
    import time
    import torch
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.baselines import load_random
    from oeis_learn.pipeline_validation.progressive_evaluation import generate
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, role_work_requirements
    from oeis_learn.pipeline_validation.progressive_role import execute
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    c, s, truth_descriptor = fixture(tmp_path)
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    e, d, _ = load_random(recipe, seed=13)
    torch.save(dict(encoder=e.state_dict(), decoder=d.state_dict()), tmp_path / "actual-weights.pt")
    truth = load(tmp_path / truth_descriptor.ref["path"])["data"]
    checkpoint = reference(tmp_path, "actual-weights.pt")
    candidate_ref = generate(
        tmp_path,
        "generated",
        [dict(id=truth["id"], indices=truth["indices"][:20], values=truth["values"][:20])],
        recipe,
        client=fitting_client,
        checkpoint=checkpoint,
        methods=("polynomial",),
        seconds=10,
    )
    generated = load(tmp_path / candidate_ref["path"])
    candidate = artifact(
        tmp_path, "actual-candidate-channel.json", "candidate", "assessment", generated
    )
    s = seal(dict(s, checkpoint=checkpoint, candidates=candidate.ref))
    validate_selection(tmp_path, s, c)
    expose(tmp_path, c, s)
    locked = artifact(tmp_path, "locked.json", "locked_selection", "assessment", s)
    descriptors = [candidate, truth_descriptor, locked]
    project(
        tmp_path,
        "projected",
        "assess",
        dict(candidates=[candidate], assessment_truth=[truth_descriptor], selection=[locked]),
        registry=REGISTRY,
    )
    values = [load(tmp_path / a.ref["path"]) for a in descriptors]
    costs = role_work_requirements("assess", values)
    costs["elapsed_us"] = 30_000_000
    t = s["task"]
    j = BudgetJournal(tmp_path / "budget", c["identity"], c["payload"]["caps"])
    j.task_remaining(t)
    boot = __import__("pathlib").Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    j.reserve(
        t["payload"]["id"], "assess", 0, costs, boot=boot, monotonic_us=time.monotonic_ns() // 1000
    )
    authority = seal(
        dict(
            version="progressive-role-budget/v1",
            campaign=c,
            task=t,
            operation="assess",
            inputs=digest(sorted(values, key=lambda v: v["identity"])),
            reservation=j._read()[0][-1],
        )
    )
    atomic_write(tmp_path / "authority.json", authority)
    result = execute(
        "assess",
        inputs=tmp_path / "projected",
        output=tmp_path / "result",
        authority=tmp_path / "authority.json",
    )
    assert result["mechanical_status"] == "COMPLETE"
    scored = load(tmp_path / "result/scoring/scores.json")
    assert scored["outcomes"]["t"]["polynomial"]["success"]
    assert load(tmp_path / "actual-candidate-channel.json")["data"] == generated


def test_component_selection_cannot_drop_library_dependency(tmp_path):
    c, s, _ = fixture(tmp_path)
    candidate = artifact(
        tmp_path,
        "component-candidate.json",
        "candidate",
        "assessment",
        dict(
            checkpoint=s["checkpoint"]["sha256"],
            records={"t": dict(costs=dict(components=dict(library=digest("library"))))},
        ),
    )
    s = seal(dict(s, candidates=candidate.ref))
    with pytest.raises(ValueError, match="selected library"):
        validate_selection(tmp_path, s, c)
