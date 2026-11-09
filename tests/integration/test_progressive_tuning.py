"""Actual CPU proposer/learner/native scorer diagnostic, never research authority."""

import time

import torch

from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.contracts import digest, reference
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.generalization.progressive.trials import TrialJournal
from oeis_learn.generalization.progressive.tuning import propose
from oeis_learn.pipeline_validation.progressive_evaluation import generate, score
from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate, resolve_update_policy


def test_real_cpu_proposal_learning_frozen_generation_and_native_score(tmp_path, fitting_client):
    began = time.monotonic_ns()
    j = TrialJournal(tmp_path / "journal", study=digest("diagnostic-tuning"))
    proposal = propose(j, 0)
    assert propose(TrialJournal(tmp_path / "journal", study=j.binding["study"]), 0) == proposal
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    p = resolve_profile(recipe, seed=proposal["seed"])
    runner = PolicyUpdate(
        p, resolve_update_policy(p, schedule_horizon_tokens=4000000), device="cpu"
    )
    from oeis_learn.decoder.bigint_codec import BOS_ID, encode_body

    ids = [BOS_ID, *encode_body("(program (return (add $index (int 1))))")]
    tokens = torch.tensor([ids] * 16)
    j.dispatch(0)
    result, ledger = runner.update(
        [list(range(1, 21))] * 16,
        tokens,
        ledger=Ledger(4000000).reserve("diagnostic-update", 16 * (len(ids) - 1)),
    )
    assert ledger.updates == 1 and result["loss"] > 0
    torch.save(
        {"encoder": runner.encoder.state_dict(), "decoder": runner.decoder.state_dict()},
        tmp_path / "weights.pt",
    )
    truth = [
        {
            "id": str(k),
            "indices": list(map(str, range(100))),
            "values": [str(i + k) for i in range(100)],
        }
        for k in range(2)
    ]
    prompts = [
        {"id": r["id"], "indices": r["indices"][:20], "values": r["values"][:20]} for r in truth
    ]
    candidates = generate(
        tmp_path,
        "generated",
        prompts,
        recipe,
        client=fitting_client,
        checkpoint=reference(tmp_path, "weights.pt"),
        seconds=10,
        methods=("correct",),
    )
    scored = score(tmp_path, candidates, truth, client=fitting_client, output="scored", seconds=30)
    # "correct" is the conditioning mode; the primary scored portfolio is
    # equal_budget (raw and rescue are separately reported diagnostics).
    assert scored["denominator"] == len(truth)
    value = sum(r["equal_budget"]["success"] for r in scored["outcomes"].values()) / len(truth)
    j.tell(
        0,
        {
            "status": "PARTIAL",
            "tokens": ledger.consumed,
            "elapsed_us": (time.monotonic_ns() - began) // 1000,
            "value": value,
            "role": "development",
        },
    )
    assert not j.equal_token_eligible(target=4000000)


def test_canonical_installed_study_asks_learns_scores_and_tells_originals(tmp_path,monkeypatch,fitting_client):
    from types import SimpleNamespace
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation import progressive_commands as commands,progressive_study as study
    from oeis_learn.generalization.progressive.selection import rank
    c,p,host,charge=build(tmp_path,monkeypatch)
    token=commands._INVOCATION.set('native-study')
    try:result,code=commands.lifecycle_command(SimpleNamespace(operation='run',campaign='study-campaign.json',acceptance='accepted',output='study-run'),tmp_path,lambda _:host)
    finally:commands._INVOCATION.reset(token)
    assert code==0,result
    journal=study.journal(tmp_path,c,p)
    assert journal.trials['0']['status']=='COMPLETE'
    result=journal.trials['0']['result']
    assert result['value']==1 and result['tokens']==charge
    assert rank([dict(result,id='0')],target=charge,root=tmp_path)['ordered']==['0']
    assert study.journal(tmp_path,c,p).trials==journal.trials


import pytest


@pytest.mark.parametrize('mode',['failed','partial','interrupted'])
def test_actual_installed_study_preserves_negative_and_interrupted_trials(tmp_path,monkeypatch,fitting_client,mode):
    from types import SimpleNamespace
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation import progressive_commands as commands,progressive_study as study
    c,p,host,charge=build(tmp_path,monkeypatch,fail=mode=='failed',partial=mode=='partial',interrupt=mode=='interrupted')
    def run(operation,**kw):
        token=commands._INVOCATION.set(operation)
        try:return commands.lifecycle_command(SimpleNamespace(operation=operation,**kw),tmp_path,lambda _:host)
        finally:commands._INVOCATION.reset(token)
    result,code=run('run',campaign='study-campaign.json',acceptance='accepted',output='one')
    j=study.journal(tmp_path,c,p)
    if mode=='interrupted':
        assert j.trials['0']['recovery_pending'],result
        before=j.trials['0']['tokens']
        result,code=run('resume',run='one',output='two')
        assert code==0,result
        j=study.journal(tmp_path,c,p)
        assert j.trials['0']['status']=='COMPLETE'
        assert j.trials['0']['tokens']>before>0
    else:
        assert j.trials['0']['status']==mode.upper(),result
        assert not j.equal_token_eligible(target=p['target_tokens'])


def test_installed_selection_runs_fresh_paired_confirmation_without_sampler_feedback(tmp_path,monkeypatch,fitting_client):
    from types import SimpleNamespace
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation import progressive_commands as commands,progressive_study as study
    from oeis_learn.generalization.progressive.contracts import load
    c,p,host,_=build(tmp_path,monkeypatch,confirmation=True)
    token=commands._INVOCATION.set('fresh-confirmation')
    try:result,code=commands.lifecycle_command(SimpleNamespace(operation='run',campaign='study-campaign.json',acceptance='accepted',output='confirmed'),tmp_path,lambda _:host)
    finally:commands._INVOCATION.reset(token)
    assert code==0,result
    journal=study.journal(tmp_path,c,p)
    assert set(journal.trials)=={'0'} and journal.trials['0']['result']['value']==1
    locked=load(journal.root/'selection.json')
    assert locked['selected']==['0'] and locked['seeds']==[17,19] and locked['sampler_feedback'] is False
    trained=[];scored=[]
    for path in (tmp_path/'confirmed').glob('task-*/role/result.json'):
        value=load(path)
        if value['operation']=='train' and value['metrics']['initialization']=='fresh_random':trained.append(value)
        if value['operation']=='score':scored.append(load(path.parent/'scoring/scores.json'))
    assert len(trained)==3 # screening plus two independently initialized seeds
    assert len(scored)==5 # three screening/retention plus paired confirmation
    assert all(not any(r['equal_budget']['success'] for r in s['outcomes'].values()) for s in scored if next(iter(s['outcomes'])).startswith('fresh-'))
    assert study.journal(tmp_path,c,p).trials==journal.trials


def test_real_six_trial_native_study_prunes_only_after_five_comparable_completions(tmp_path,monkeypatch,fitting_client):
    """Short diagnostic coordinate; production retains its 1M/2M lower bounds."""
    from types import SimpleNamespace
    from tests.helpers.progressive_study import build
    from oeis_learn.pipeline_validation import progressive_commands as commands,progressive_study as study
    from oeis_learn.generalization.progressive.selection import rank
    from oeis_learn.generalization.progressive.contracts import load
    c,p,host,_=build(tmp_path,monkeypatch,trials=6)
    token=commands._INVOCATION.set('six-native-trials')
    try:result,code=commands.lifecycle_command(SimpleNamespace(operation='run',campaign='study-campaign.json',acceptance='accepted',output='six'),tmp_path,lambda _:host)
    finally:commands._INVOCATION.reset(token)
    assert code==0,result
    j=study.journal(tmp_path,c,p)
    assert [j.trials[str(i)]['status'] for i in range(6)]==['COMPLETE']*5+['PRUNED']
    assert all(j.trials[str(i)]['result']['value']==1 for i in range(5))
    assert j.trials['5']['result']['value']==0
    assert result['metrics']['tasks']['t5-late-train']=='SKIPPED'
    assert not any(load(p)['task']['payload']['id']=='t5-late-train' for p in (tmp_path/'six').glob('task-*/authority/supervisor.json'))
    rows=[dict(s['result'],id=k) for k,s in j.trials.items()]
    expected=[r['id'] for r in sorted(rows[:5],key=lambda r:(-r['value'],r['elapsed_us'],r['id']))]
    assert rank(rows,target=p['target_tokens'],root=tmp_path)['ordered']==expected
    assert study.journal(tmp_path,c,p).trials==j.trials
