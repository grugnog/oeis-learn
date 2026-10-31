"""Synthetic originals exercise integrity only; these are not device evidence."""
import json
import pytest
from oeis_learn.pipeline_validation.identity import seal,file_hash
from oeis_learn.pipeline_validation.receipts import Ledger
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.qualification import qualify_reuse
from tests.unit.test_generalization_qualification import producer

# Independently enumerated acceptance obligations, not imported from production.
OBLIGATIONS={
 'retained-eval':(('weights_loaded','visible_only','matched_budgets'),('checkpoint_identity','evaluation')),
 'refresh':(('fixed_support','fresh_shards','finite_update'),('preparation','training')),
 'parameter-targets':(('literal_update','parameterized_update','grounding_verified'),('literal_training','parameterized_training','target_audit')),
 'replay-resume':(('training_only_admission','paired_control','full_state_resume','finite_update'),('admission','continued_sft','verified_replay','resume')),
 'failure-packaging':(('failure_preserved','archive_verified','private_payloads_excluded'),('failure_ledger','archive_verification')),
}


def fixture(root):
    receipt,facts=producer(root);cases={}
    for name,(checks,labels) in OBLIGATIONS.items():
        folder=root/name;folder.mkdir();artifacts={}
        for label in labels:
            path=folder/(label+'.json');path.write_text('{"synthetic_fixture":true}')
            artifacts[label]=reference(path,root)
        result=seal(dict(version='generalization-acceptance-result/v1',case=name,qualification=receipt['identity'],
            status='PASS',checks=dict.fromkeys(checks,True),artifacts=artifacts))
        (folder/'case-result.json').write_text(json.dumps(result))
        ledger=Ledger(folder,'acceptance.'+name,dict(case=name,qualification=receipt['identity']))
        ledger.transition('READY');ledger.transition('RUNNING')
        ledger.transition('PASS',files={p.name:file_hash(p) for p in folder.glob('*.json')})
        cases[name]=reference(folder/'receipt.json',root)
    feature=seal(dict(version='generalization-feature-acceptance/v2',qualification=receipt['identity'],status='PASS',cases=cases))
    return receipt,facts,feature


def check(root,receipt,facts,feature):
    path=root/'feature.json';path.write_text(json.dumps(seal(feature)))
    return qualify_reuse(receipt,facts,root=root,required_suites={'hip'},required_capabilities={'load'},
        research=True,feature_receipt=reference(path,root))


def test_all_original_cases_authorize_research(tmp_path):
    receipt,facts,feature=fixture(tmp_path)
    assert check(tmp_path,receipt,facts,feature)==receipt['identity']


@pytest.mark.parametrize('change',['missing','extra','version','qualification','status','unknown','names'])
def test_feature_authority_rejects_incomplete_or_changed_manifest(tmp_path,change):
    receipt,facts,feature=fixture(tmp_path)
    if change=='missing':del feature['cases']['refresh']
    elif change=='extra':feature['cases']['other']=feature['cases']['refresh']
    elif change=='names':feature['cases']=list(feature['cases'])
    elif change=='unknown':feature['skip_checks']=True
    elif change=='version':feature['version']='generalization-feature-acceptance/v1'
    elif change=='status':feature['status']='PARTIAL'
    else:feature['qualification']='sha256:'+'0'*64
    with pytest.raises(ValueError):check(tmp_path,receipt,facts,feature)


@pytest.mark.parametrize('change',['FAIL','PARTIAL','REUSED','unknown','stage','binding','payload','missing-file',
 'false-check','missing-check','unknown-check','missing-artifact','swapped-artifact','result-version','result-status','result-case','result-qualification','result-extra'])
def test_original_case_and_all_mandatory_artifacts_are_verified(tmp_path,change):
    receipt,facts,feature=fixture(tmp_path)
    folder=tmp_path/'refresh';path=folder/'receipt.json';stage=json.loads(path.read_text())
    if change in {'FAIL','PARTIAL','REUSED'}:stage['state']=change
    elif change=='unknown':stage['unregistered']=True
    elif change=='stage':stage['stage']='acceptance.retained-eval'
    elif change=='binding':stage['binding']['qualification']='sha256:'+'0'*64
    elif change=='payload':(folder/'training.json').write_text('tampered')
    elif change=='missing-file':(folder/'training.json').unlink()
    else:
        result_path=folder/'case-result.json';result=json.loads(result_path.read_text())
        if change=='false-check':result['checks']['finite_update']=False
        elif change=='missing-check':del result['checks']['finite_update']
        elif change=='unknown-check':result['checks']['extra']=True
        elif change=='missing-artifact':del result['artifacts']['training']
        elif change=='swapped-artifact':result['artifacts']['training']=result['artifacts']['preparation']
        elif change=='result-version':result['version']='unsupported/v1'
        elif change=='result-status':result['status']='FAIL'
        elif change=='result-case':result['case']='retained-eval'
        elif change=='result-qualification':result['qualification']='sha256:'+'0'*64
        elif change=='result-extra':result['skip']=True
        result_path.write_text(json.dumps(seal(result)))
        stage['files']['case-result.json']=file_hash(result_path)
    path.write_text(json.dumps(seal(stage)));feature['cases']['refresh']=reference(path,tmp_path)
    with pytest.raises(ValueError):check(tmp_path,receipt,facts,feature)


def test_builder_and_complete_projection_closure(tmp_path):
    from oeis_learn.generalization.feature_acceptance import assemble_feature,feature_artifacts,verify_feature
    receipt,facts,feature=fixture(tmp_path)
    built=assemble_feature(feature['cases'],qualification=receipt['identity'],root=tmp_path)
    assert built==feature
    refs=feature_artifacts(feature,qualification=receipt['identity'],root=tmp_path)
    projected=tmp_path/'projection';projected.mkdir()
    for ref in refs:
        path=projected/ref['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((tmp_path/ref['path']).read_bytes())
    assert verify_feature(feature,qualification=receipt['identity'],root=projected)==feature['identity']
    assert len(refs)==23  # five receipts, five case results, thirteen mandatory payloads
    (projected/'refresh/training.json').unlink()
    with pytest.raises(ValueError):verify_feature(feature,qualification=receipt['identity'],root=projected)


@pytest.mark.parametrize('private',[False,True])
def test_transitive_case_artifacts_are_verified_and_private_files_rejected(tmp_path,private):
    receipt,facts,feature=fixture(tmp_path)
    child=tmp_path/('private/child.json' if private else 'child.json');child.parent.mkdir(exist_ok=True)
    child.write_text('{}')
    artifact=tmp_path/'refresh/training.json';artifact.write_text(json.dumps({'child':reference(child,tmp_path)}))
    result_path=tmp_path/'refresh/case-result.json';result=json.loads(result_path.read_text())
    result['artifacts']['training']=reference(artifact,tmp_path);result_path.write_text(json.dumps(seal(result)))
    stage_path=tmp_path/'refresh/receipt.json';stage=json.loads(stage_path.read_text())
    stage['files'].update({'training.json':file_hash(artifact),'case-result.json':file_hash(result_path)})
    stage_path.write_text(json.dumps(seal(stage)));feature['cases']['refresh']=reference(stage_path,tmp_path)
    if private:
        with pytest.raises(ValueError,match='public'):check(tmp_path,receipt,facts,feature)
    else:
        assert check(tmp_path,receipt,facts,feature)
        child.write_text('changed')
        with pytest.raises(ValueError):check(tmp_path,receipt,facts,feature)
