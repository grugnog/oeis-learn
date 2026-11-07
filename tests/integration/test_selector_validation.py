"""Follow-up adapters use canonical generation/scoring and preserve role isolation."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.selector_validation import CONTROLS
from tests.unit.test_selector_validation import GOOD,BAD,PROMPT,envelope


def test_one_generation_pool_produces_two_selections_then_independent_scoring(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_evaluation as evaluation,pilot_clocks
    from oeis_learn.pipeline_validation import selector_validation_roles as roles
    from oeis_learn.fresh import bigint_diagnostics
    from oeis_learn.synthesis import client
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.decoder.bigint_codec import encode_body
    (tmp_path/'weights').write_bytes(b'fixture')
    prompts=[dict(id='p',indices=list(PROMPT.indices),values=list(PROMPT.values))]
    monkeypatch.setattr(evaluation,'load_retained',lambda *a,**k:('encoder','decoder',{}))
    class Evidence:
        def __init__(self,*a,**kw):self.clock_summary=dict(cost_complete=True,identity='clock')
        def finish(self):pass
    monkeypatch.setattr(pilot_clocks,'FullClockEvidence',Evidence)
    calls=[]
    def generate(*args,**kwargs):
        calls.append(args)
        assert kwargs['retain_selector_candidates'] is True
        assert kwargs['diagnostic_attempts']==dict(correct=8,shuffled=8,absent=8,template=8)
        return dict(records={'p':dict(selections={c:envelope(BAD) for c in CONTROLS},
            selector_candidates={c:[envelope(BAD),envelope(GOOD)] for c in CONTROLS},
            arm_evidence={c:dict(complete=True,reason='fixture') for c in CONTROLS})})
    monkeypatch.setattr(bigint_diagnostics,'generate',generate)
    root=tmp_path/'generation'
    evaluation.evaluate_retained(tmp_path,reference(tmp_path/'weights',tmp_path),prompts,{},root,
        client=None,device='cpu',study='study',source='source',compare_selectors=True,pac=True)
    assert len(calls)==1
    original=json.loads((root/'selections.json').read_text())
    alternative=json.loads((root/'alternative-selections.json').read_text())
    assert 'selector_candidates' not in original['p']
    pools=json.loads((root/'selector-candidates.json').read_text())
    assert set(pools['p'])==set(CONTROLS) and len(pools['p']['bank'])==2
    generation=json.loads((root/'evaluation.json').read_text())
    assert generation['selector_comparison']['artifacts']['selector-candidates.json']==reference(root/'selector-candidates.json',root)
    assert all(original['p']['selections'][c]['selected']['source']==BAD for c in CONTROLS)
    assert all(alternative['p']['selections'][c]['selected']['source']==GOOD for c in CONTROLS)
    supervision=tmp_path/'supervision'
    write(supervision/'targets.json',[dict(id='p',role='training',indices=list(map(str,range(100))),
        values=list(map(str,range(100))),source=GOOD,tokens=encode_body(GOOD))])
    write(supervision/'roster.json',{'p':dict(family='f',category='training')})
    native=SimpleNamespace(limits=ResourceProfile(),execute=lambda source,indices,**kw:
        dict(outcome='ok',values=list(evaluate(source,indices).values)))
    monkeypatch.setattr(client,'SynthesisClient',lambda *a,**kw:nullcontext(native))
    settings=dict(phase='probes',study='study',targets=reference(supervision/'targets.json',supervision),
        roster=reference(supervision/'roster.json',supervision),
        **{k:reference(root/v,root) for k,v in [('original','selections.json'),('alternative','alternative-selections.json'),('observations','selector-observations.json')]})
    result=roles.score(dict(selection_supervision=supervision,selection_candidates=root),tmp_path/'score',settings,seconds=30)
    for control in CONTROLS:
        measured=result['metrics']['training'][control]
        assert measured['original']['success']==0 and measured['alternative']['success']==1
        assert measured['joint']['recoveries']==1
    # An unavailable selector trace cannot masquerade as a complete successful decision.
    observations=json.loads((root/'selector-observations.json').read_text())
    observations['p']['equal_budget']['complete']=False
    write(root/'selector-observations.json',observations,exclusive=False)
    settings['observations']=reference(root/'selector-observations.json',root)
    incomplete=roles.score(dict(selection_supervision=supervision,selection_candidates=root),tmp_path/'incomplete',settings,seconds=30)
    assert incomplete['trace_incomplete']==1
    assert incomplete['outcomes']['alternative']['p']['equal_budget']['fallback_success']
    assert not incomplete['outcomes']['alternative']['p']['equal_budget']['complete']


def test_training_score_rejects_assessment_before_client_creation(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_validation_roles as roles
    from oeis_learn.synthesis import client
    monkeypatch.setattr(client,'SynthesisClient',lambda *a,**kw:pytest.fail('assessment reached executor'))
    write(tmp_path/'targets.json',[dict(id='p',role='assessment',source=GOOD,indices=list(map(str,range(100))),
        values=list(map(str,range(100))),family='f',stratum='structural')])
    write(tmp_path/'roster.json',{'p':dict(family='f',category='training')})
    settings=dict(phase='probes',study='study',targets=reference(tmp_path/'targets.json',tmp_path),
        roster=reference(tmp_path/'roster.json',tmp_path),original=None,alternative=None,observations=None)
    with pytest.raises(ValueError):roles.score(dict(selection_supervision=tmp_path),tmp_path/'out',settings,seconds=30)


def fixture_campaign(root):
    """Minimal producer-bound local chain; tensor payload stays opaque to the host."""
    campaign=root/'campaign';base=campaign/'admission/execution/execution/role/private/assembly/resolved'
    write(base/'private/structural.json',[])
    from oeis_learn.generalization.structural import original_cells
    originals=[]
    for cell in original_cells():
        path=root/'cells'/(cell['name']+'.json');write(path,[])
        originals.append(dict(cell,ref=reference(path,root)))
    admission=seal(dict(status='PASS',purpose='research',source='old-source',originals=originals,
        structural=reference(base/'private/structural.json',base),experiment='old-study'))
    write(base/'admission.json',admission)
    write(base/'recipe.json',{})
    write(base/'training/private/inventory.json',{})
    write(base/'training/private/probes.json',[])
    entries=[dict(stage='admission',artifacts=[reference(p,campaign/'admission') for p in sorted(base.rglob('*.json'))])]
    for arm in ('B','D'):
        folder=campaign/('train-'+arm)/'execution';checkpoints=folder/('training-'+arm)/'role/private/learner/checkpoints'
        checkpoints.mkdir(parents=True);(checkpoints/'weights.pt').write_bytes(b'opaque weights')
        descriptor=seal(dict(kind='full_state',pins=['latest'],tokens=25000000,
                             weights=reference(checkpoints/'weights.pt',checkpoints)))
        roster=seal(dict(study='trained-study',arm=arm,seed=20261016,latest=descriptor['identity'],checkpoints=[descriptor]))
        write(checkpoints/'retained.json',roster)
        result=seal(dict(status='PASS',study='trained-study',inventory=[reference(checkpoints/'retained.json',folder)],
                         result=dict(runs=[dict(arm=arm,result=dict(retained=roster['identity']))])))
        write(folder/'result.json',result)
        entries.append(dict(stage='train-'+arm,artifacts=[reference(folder/'result.json',folder.parent)]))
    plan=seal(dict(research=True,arms=['B','C','D'],source='old-source',inputs=[r['ref'] for r in originals]))
    write(campaign/'plan.json',plan);write(campaign/'workflow.json',seal(dict(plan=plan['identity'],receipts=entries)))
    return campaign


def test_preflight_uses_exact_inventory_and_never_calls_engine(tmp_path):
    from oeis_learn.pipeline_validation.selector_validation_commands import probes,discover
    campaign=fixture_campaign(tmp_path)
    args=SimpleNamespace(campaign='campaign',output='preflight',preflight_only=True)
    assert probes(args,tmp_path,lambda *_:pytest.fail('preflight used engine'))==0
    verified=json.loads((tmp_path/'preflight/inputs.json').read_text())
    assert set(verified['checkpoints'])=={'B','D'}
    assert not (tmp_path/'preflight/plan.json').exists()
    weights=campaign/'train-D/execution/training-D/role/private/learner/checkpoints/weights.pt'
    weights.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='Artifact hash/size mismatch'):discover(tmp_path,'campaign')


def test_freeze_requires_completed_probe_and_rejects_changed_source(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_validation_commands as host
    from oeis_learn.generalization.selector_coverage import POLICY
    probe=tmp_path/'probes';probe.mkdir()
    plan=seal(dict(source='current',selector=POLICY,phase='probes',inputs={},qualification='q'))
    write(probe/'plan.json',plan)
    scoring={}
    for arm in ('B','D'):
        path=probe/(arm+'.json')
        write(path,seal(dict(study=plan['identity'],trace_incomplete=0,
            metrics=dict(training=dict(equal_budget={v:dict(complete=244) for v in ('original','alternative')})))))
        scoring[arm]=reference(path,tmp_path)
    result=seal(dict(version='pac-selector-validation-result/v1',phase='probes',plan=plan['identity'],
                     freeze_eligible=True,status='PASS',scoring=scoring))
    write(probe/'result.json',result)
    monkeypatch.setattr(host,'source',lambda _: 'changed')
    args=SimpleNamespace(probes='probes',output='fresh')
    with pytest.raises(ValueError,match='Completed compatible'):host.freeze(args,tmp_path)
    monkeypatch.setattr(host,'source',lambda _: 'current')
    assert host.freeze(args,tmp_path)==0
    frozen=host.read(tmp_path/'fresh/plan.json',sealed=True)
    assert frozen['phase']=='structural' and frozen['protocol']['seed']==20261108
    assert frozen['probe_result']==reference(probe/'result.json',tmp_path)
    host.verify_structural_plan(tmp_path,tmp_path/'fresh/plan.json',frozen)
    changed=seal(dict(frozen,inputs=dict(checkpoints='different frozen weights')))
    with pytest.raises(ValueError,match='differs from completed probe'):
        host.verify_structural_plan(tmp_path,tmp_path/'fresh/plan.json',changed)
    with pytest.raises(ValueError,match='differs from completed probe'):
        host.verify_structural_plan(tmp_path,tmp_path/'other/plan.json',frozen)
    with pytest.raises(ValueError,match='Fresh freeze'):host.freeze(args,tmp_path)


def test_fresh_custodian_excludes_observed_cohort_and_stops_before_native(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import selector_validation_roles as roles
    from oeis_learn.generalization import structural,preparation
    from oeis_learn.synthesis import client
    cells=[dict(c,count=1) for c in structural.original_cells()]
    monkeypatch.setattr(structural,'original_cells',lambda:cells)
    protocol=seal(dict(cells=[dict(family='fresh',count=1)]))
    monkeypatch.setattr(roles,'fresh_design',lambda:protocol)
    row=dict(id='new',family='fresh',source=GOOD,group='g',variant='v',values=list(map(str,range(100))),indices=list(map(str,range(100))))
    unrelated=dict(row,id='other',family='other',source='other',group='other',variant='other',values=['-1']*100)
    monkeypatch.setattr(roles,'fresh_draw',lambda _: [row])
    monkeypatch.setattr(preparation,'reference_candidate',lambda r:((r,None),None))
    reads=[]
    def read(root,ref,cap=None):
        reads.append(ref['path'])
        return [row]*128 if ref['path']=='observed' else [] if ref['path']=='probes' else [unrelated]
    monkeypatch.setattr(roles,'_read',read)
    monkeypatch.setattr(client,'SynthesisClient',lambda *a,**kw:pytest.fail('quota deficit reached native'))
    result=roles.publish_structural(dict(isolation_truth=tmp_path),tmp_path/'out',
        dict(protocol=protocol,originals=[dict(c,ref=dict(path=c['name'])) for c in cells],
             observed=dict(path='observed'),probes=dict(path='probes'),study='study'),seconds=30)
    assert result['status']=='BLOCKED' and result['deficits']=={'fresh':1}
    assert len(reads)==18
    availability=json.loads((tmp_path/'out/availability.json').read_text())
    assert availability['exclusion_reasons']['finite_window']==1


def test_failure_bundle_keeps_actual_exception_without_engine(tmp_path,monkeypatch,capsys):
    from oeis_learn.pipeline_validation import selector_validation_commands as host
    def failed(args,workspace,host_factory):
        args._selector_output=workspace/'new';args._selector_output.mkdir()
        raise FileNotFoundError('missing exact checkpoint: /workstation/weights.pt')
    monkeypatch.setattr(host,'probes',failed)
    args=SimpleNamespace(selector_phase='probes')
    assert host.main(args,tmp_path,lambda _:None)==1
    error=host.read(tmp_path/'new/failure.json',sealed=True)
    assert '/workstation/weights.pt' in error['message']
    import zipfile
    with zipfile.ZipFile(tmp_path/'new/failure.zip') as z:
        assert 'FileNotFoundError' in z.read('traceback.txt').decode()
    assert 'missing exact checkpoint' in capsys.readouterr().out
