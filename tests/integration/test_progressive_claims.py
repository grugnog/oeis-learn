"""Replayable polynomial promotion through immutable dossier review."""

import copy
import json
import os
import subprocess
import sys

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
from tests.unit.test_progressive_dossiers import SOURCE, dossier, selected
from tests.integration.test_progressive_lifecycle import installed_lifecycle  # noqa: F401


def proof(root, **changes):
    value = (
        dict(
            version="progressive-polynomial-certificate/v1",
            source=digest(SOURCE),
            coefficients=["1", "1"],
            arithmetic="mathematical_integers",
            assumptions=[],
            machine_domain=None,
        )
        | changes
    )
    atomic_write(root / "proof.json", value)
    return reference(root, "proof.json")


def test_scoped_proof_replay_is_immutable_and_never_proves_novelty_or_sequence(tmp_path):
    from oeis_learn.generalization.progressive.claims import review_dossier, validate_dossier

    value = dossier(tmp_path)
    before = (tmp_path / "dossier.json").read_bytes()
    claim = dict(value["payload"]["claim"], statement="proved", proof=proof(tmp_path))
    reviewed = review_dossier(tmp_path, value, claim, reviewers={}, output="proved.json")
    p = reviewed["payload"]
    assert p["claim"]["novelty"] == "not_reviewed" and not p["training_eligible"]
    assert p["proof_check"]["scope"] == "program_equals_polynomial_only"
    assert p["proof_check"]["sequence_law"] == "unproved"
    assert p["proof_check"]["machine_safety"] == "unverified"
    assert p["machine_safety"] == "finite_execution_only_no_unbounded_safety"
    assert reviewed["parents"] == [value["identity"]]
    assert (tmp_path / "dossier.json").read_bytes() == before
    assert validate_dossier(tmp_path, reviewed) == reviewed
    again = review_dossier(tmp_path, reviewed, claim, reviewers={}, output="replayed.json")
    assert again["payload"]["proof_check"] == p["proof_check"]
    assert again["parents"] == [reviewed["identity"]]


@pytest.mark.parametrize(
    "change", ["scope", "checker", "coefficient", "machine", "assumptions", "missing"]
)
def test_resealed_proof_scope_cannot_override_replay(tmp_path, change):
    from oeis_learn.generalization.progressive.claims import review_dossier, validate_dossier

    value = dossier(tmp_path)
    claim = dict(value["payload"]["claim"], statement="proved", proof=proof(tmp_path))
    reviewed = review_dossier(tmp_path, value, claim, reviewers={}, output="proved.json")
    bad = copy.deepcopy(reviewed)
    if change == "missing":
        del bad["payload"]["proof_check"]
    elif change == "assumptions":
        bad["payload"]["assumptions"] = ["unproved recurrence"]
    else:
        key, val = {
            "scope": ("sequence_law", "proved"),
            "checker": ("checker", digest("other checker")),
            "coefficient": ("coefficients", ["0", "1"]),
            "machine": ("machine_safety", "finite_numeric_range_verified"),
        }[change]
        bad["payload"]["proof_check"][key] = val
        bad["payload"]["proof_check"] = seal(bad["payload"]["proof_check"])
    with pytest.raises(ValueError):
        validate_dossier(tmp_path, seal(bad))


def test_range_argument_does_not_expand_other_dossier_evidence(tmp_path):
    from oeis_learn.generalization.progressive.claims import review_dossier

    value = dossier(tmp_path)
    claim = dict(
        value["payload"]["claim"],
        statement="proved",
        proof=proof(tmp_path, machine_domain={"lower": "-100", "upper": "1000"}),
    )
    reviewed = review_dossier(tmp_path, value, claim, reviewers={}, output="proved.json")
    assert reviewed["payload"]["proof_check"]["machine_safety"] == "finite_numeric_range_verified"
    assert reviewed["payload"]["proof_check"]["resource_safety"] == "unverified"
    for field in ("tested", "untested_scope", "counterexamples", "generation_budget"):
        assert reviewed["payload"][field] == value["payload"][field]


def test_true_program_identity_cannot_promote_a_refuted_sequence_claim(tmp_path):
    from oeis_learn.generalization.progressive.claims import create_dossier, review_dossier

    candidate = selected(tmp_path)
    value = create_dossier(
        tmp_path,
        candidate,
        indices=list(map(str, range(100))),
        expected=["0"] * 100,
        output="refuted.json",
    )
    claim = dict(value["payload"]["claim"], statement="proved", proof=proof(tmp_path))
    with pytest.raises(ValueError, match="counterexample"):
        review_dossier(tmp_path, value, claim, reviewers={}, output="forbidden.json")
    assert not (tmp_path / "forbidden.json").exists()


@pytest.mark.parametrize("change", ["coefficients", "source", "legacy", "assumptions", "bytes"])
def test_failed_promotion_preserves_original_dossier(tmp_path, change):
    from oeis_learn.generalization.progressive.claims import review_dossier

    value = dossier(tmp_path)
    before = (tmp_path / "dossier.json").read_bytes()
    changes = {
        "coefficients": dict(coefficients=["0", "1"]),
        "source": dict(source=digest(SOURCE + " ")),
        "legacy": dict(version="legacy-ir-proof/v1"),
        "assumptions": dict(assumptions=["assuming the theorem"]),
        "bytes": {},
    }[change]
    ref = proof(tmp_path, **changes)
    if change == "bytes":
        ref["bytes"] = (256 << 10) + 1
    with pytest.raises(ValueError):
        review_dossier(
            tmp_path,
            value,
            dict(value["payload"]["claim"], statement="proved", proof=ref),
            reviewers={},
            output="forbidden.json",
        )
    assert (tmp_path / "dossier.json").read_bytes() == before
    assert not (tmp_path / "forbidden.json").exists()


def native_review_inputs(root, client, *, already_proved=False, invalid=False):
    from oeis_learn.generalization.progressive.claims import review_dossier
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
    from tests.contract.test_progressive_roles import artifact

    value = dossier(root, client)
    claim = dict(
        value["payload"]["claim"],
        statement="proved",
        proof=proof(root, coefficients=["2", "1"] if invalid else ["1", "1"]),
    )
    if already_proved:
        value = review_dossier(root, value, claim, reviewers={}, output="proved.json")
    descriptors = [
        artifact(root, "dossier-channel.json", "sealed_dossier", "review", value),
        artifact(root, "review-channel.json", "review_evidence", "review", dict(claim=claim)),
    ]
    project(root, "projection", "review-claim", {"dossier": descriptors}, registry=REGISTRY)
    return value, [load(root / d.ref["path"]) for d in descriptors]


@pytest.mark.parametrize("already_proved", [False, True])
def test_installed_wheel_replays_proof_with_native_execution_and_prepaid_work(
    tmp_path, fitting_client, installed_lifecycle, already_proved  # noqa: F811
):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.claims import validate_dossier
    from tests.unit.test_progressive_budgets import role_authority

    value, values = native_review_inputs(tmp_path, fitting_client, already_proved=already_proved)
    before = (tmp_path / "dossier.json").read_bytes()
    authority, journal, task = role_authority(
        tmp_path, "review-claim", values, useful_us=30_000_000
    )
    assert authority["reservation"]["cost"]["attempts"] == 5
    atomic_write(tmp_path / "authority.json", authority)
    script = """
import sys
from pathlib import Path
from oeis_learn.generalization.progressive import polynomial_certificate
from oeis_learn.pipeline_validation.progressive_role import execute
assert Path(polynomial_certificate.__file__).is_relative_to(Path(sys.argv[1]))
root = Path.cwd()
result = execute('review-claim', inputs=root/'projection', output=root/'result',
                 authority=root/'authority.json')
assert result['mechanical_status'] == 'COMPLETE', result
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(installed_lifecycle)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(installed_lifecycle)),
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reviewed = load(tmp_path / "result/dossier.json")
    assert validate_dossier(tmp_path / "projection/dossier", reviewed) == reviewed
    assert reviewed["parents"] == [value["identity"]]
    assert reviewed["payload"]["claim"]["statement"] == "proved"
    assert reviewed["payload"]["claim"]["novelty"] == "not_reviewed"
    assert reviewed["payload"]["proof_check"]["sequence_law"] == "unproved"
    assert not reviewed["payload"]["training_eligible"]
    assert (tmp_path / "dossier.json").read_bytes() == before
    used = json.loads((tmp_path / "result/work-debits.jsonl").read_text().splitlines()[-1])["used"]
    assert used["attempts"] == used["rejections"] == (5 if already_proved else 3)
    assert used["native_work"] == 1
    assert journal.task_remaining(task)["attempts"] == task["payload"]["caps"]["attempts"] - 5


def test_native_invalid_proof_retains_debits_without_publishing(tmp_path, fitting_client):
    from oeis_learn.pipeline_validation.progressive_role import execute
    from tests.unit.test_progressive_budgets import role_authority

    _, values = native_review_inputs(tmp_path, fitting_client, invalid=True)
    before = (tmp_path / "dossier.json").read_bytes()
    authority, journal, task = role_authority(
        tmp_path, "review-claim", values, useful_us=30_000_000
    )
    atomic_write(tmp_path / "authority.json", authority)
    with pytest.raises(ValueError, match="coefficients"):
        execute(
            "review-claim",
            inputs=tmp_path / "projection",
            output=tmp_path / "result",
            authority=tmp_path / "authority.json",
        )
    assert not (tmp_path / "result/dossier.json").exists()
    assert (tmp_path / "dossier.json").read_bytes() == before
    used = json.loads((tmp_path / "result/work-debits.jsonl").read_text().splitlines()[-1])["used"]
    assert used["attempts"] == used["rejections"] == 2
    assert used["native_work"] == 1
    assert journal.task_remaining(task)["attempts"] == task["payload"]["caps"]["attempts"] - 5


@pytest.mark.parametrize("axis", ["attempts", "rejections", "native_work"])
def test_proof_review_refuses_insufficient_reservation_before_dispatch(tmp_path, axis):
    from oeis_learn.generalization.progressive.budgets import RoleWorkBudget, role_work_requirements
    from tests.unit.test_progressive_budgets import role_authority

    _, values = native_review_inputs(tmp_path, None)
    costs = role_work_requirements("review-claim", values)
    costs[axis] -= 1
    authority, journal, _ = role_authority(tmp_path, "review-claim", values, costs=costs)
    before = journal.path.read_bytes()
    with pytest.raises(ValueError, match="insufficient enclosing"):
        RoleWorkBudget(authority, "review-claim", values, tmp_path / "never")
    assert journal.path.read_bytes() == before
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize('promote', [False, True])
def test_installed_counterexample_quarantines_consumed_checkpoint_and_dossier(tmp_path, fitting_client, monkeypatch, promote):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture,invoke_cpu
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.quarantine import Journal
    from oeis_learn.generalization.progressive.claims import seal_candidate,create_dossier
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
    from tests.contract.test_progressive_roles import artifact
    from tests.unit.test_progressive_budgets import role_authority
    from oeis_learn.pipeline_validation.progressive_role import execute
    from oeis_learn.pipeline_validation.progressive_quarantine_control import reconcile
    from oeis_learn.generalization.progressive.recovery import Recovery
    c,tasks,channel,_=fixture(tmp_path/'learn',interrupt=False)
    result,out=invoke_cpu(tmp_path/'learn',c,tasks[0],channel,ordinal=0)
    assert result['mechanical_status']=='COMPLETE'
    journal_root=tmp_path/'learn/accounting'
    q=Journal(journal_root)
    binding=next((q.root/'sources').glob('*.json'))
    source=load(binding)['source']
    checkpoint=load(out/'role/checkpoint.json')
    state=tmp_path/'learn/learning'/tasks[0]['identity'][7:]/'checkpoints'
    full=load(state/checkpoint['payload']['full_state']['path'])
    weights=full['states']['weights']['sha256']
    review=tmp_path/'review';review.mkdir()
    atomic_write(review/'snapshot.json',{'source':'explicit diagnostic counterexample panel'})
    selection=seal(dict(version='discovery-candidates/v1',model=digest('model'),checkpoint=weights,
        model_control='retained_D',recipe=digest('recipe'),methods=['correct'],prompts=digest('visible'),clock={},counts={},elapsed_us=1,
        records={'candidate':dict(selections={'equal_budget':dict(selected=dict(source=source,verified=True))},costs={})}))
    atomic_write(review/'selection.json',selection)
    candidate=seal_candidate(review,reference(review,'selection.json'),'candidate',source_snapshot=reference(review,'snapshot.json'),offset='0',
        budget={k:1 for k in c['payload']['caps']},output='candidate.json',purpose='acceptance',track='synthetic_strict')
    dossier=create_dossier(review,candidate,indices=list(map(str,range(100))),expected=['0']*100,output='original.json',client=fitting_client)
    requested=dict(dossier['payload']['claim'])
    if promote:requested['statement']='proved'
    values=[artifact(review,'dossier-channel.json','sealed_dossier','review',dossier),
            artifact(review,'claim-channel.json','review_evidence','review',dict(claim=requested))]
    project(review,'inputs','review-claim',{'dossier':values},registry=REGISTRY)
    authority,_,_=role_authority(review,'review-claim',[load(review/v.ref['path']) for v in values],useful_us=30000000)
    atomic_write(review/'authority.json',authority)
    reviewed=execute('review-claim',inputs=review/'inputs',output=review/'output',authority=review/'authority.json')
    assert reviewed['mechanical_status']==('FAILED' if promote else 'COMPLETE')
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    journal=BudgetJournal(journal_root,c['identity'],c['payload']['caps'])
    before=journal.path.read_bytes()
    overlay=reconcile(tmp_path,review/'output',reviewed,journal)
    assert checkpoint['identity'] in overlay['affected']
    graph,affected,_,_=q.read()
    assert reviewed['metrics']['dossier'] in affected
    assert all(n['id'] in affected for n in graph if n['kind']=='checkpoint' and checkpoint['identity'] in n['parents'])
    with Recovery(journal,tasks[0],checkpoint['payload']['compatibility'],state_root=state.parent) as owner:
        with pytest.raises(ValueError,match='quarantined'):owner.restore()
    assert journal.path.read_bytes()==before
    q.observe([dict(id='clean-ancestor',kind='checkpoint',parents=[])])
    assert q.recovery([checkpoint['identity'],'clean-ancestor'],spent={'tokens':17})['checkpoint']=='clean-ancestor'
    assert q.recovery([checkpoint['identity']],spent={'tokens':17})['status']=='CONTAMINATED'
