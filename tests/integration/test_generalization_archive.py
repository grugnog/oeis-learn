"""Offline archive boundary checks require no engine or HIP execution."""
import json
import zipfile
import pytest
from oeis_learn.pipeline_validation.bundle import pack
from oeis_learn.pipeline_validation.identity import write


def test_archive_reconciles_evidence_after_safe_extraction(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.generalization import bundles
    root=tmp_path/'input';root.mkdir()
    write(root/'summary.json',{'selected_scope':'PARTIAL'})
    write(root/'evidence.json',{'study':{'path':'study.json'}})
    write(root/'study.json',{'identity':'study'})
    archive=pack(root,mandatory=['summary.json','evidence.json','study.json'])
    called=[]
    def verify(record,*,expected_study,root):
        called.append(expected_study)
        assert json.loads((root/'study.json').read_text())['identity']=='study'
        return dict(integrity='PASS',software_status='PARTIAL',scientific_status='UNMEASURED')
    monkeypatch.setattr(bundles,'verify_bundle',verify)
    result=verify_archive(archive)
    assert result['software_status']=='PARTIAL' and called==['study']


def test_generic_recovery_archive_cannot_claim_generalization_verification(tmp_path):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    root=tmp_path/'input';root.mkdir();write(root/'summary.json',{'selected_scope':'FAIL'})
    archive=pack(root,mandatory=['summary.json'])
    with pytest.raises(ValueError,match='evidence'):verify_archive(archive)


def test_archive_rejects_total_expansion_before_reading_payloads(tmp_path):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('large.json','0'*1025)
    with pytest.raises(ValueError,match='cap'):verify_archive(archive,cap=1024)


def test_final_controller_ledger_is_verified_without_rewriting_payloads(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.pipeline_validation.generalization_finalization import publish
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.generalization import bundles
    root=tmp_path/'run';report=root/'report';report.mkdir(parents=True)
    write(report/'summary.json',dict(selected_scope='PASS'))
    write(report/'evidence.json',{'study':{'path':'study.json'}});write(report/'study.json',{'identity':'study'})
    archive=pack(report,mandatory=['summary.json','evidence.json','study.json'])
    summary=seal(dict(selected_scope='PASS',integrity='PASS',archive=reference(archive,root),scientific_status='UNMEASURED'))
    workflow=seal(dict(status='PASS',plan=[{'name':n} for n in ('report','verify-bundle')],receipts=[dict(stage=n,exit_code=0) for n in ('report','verify-bundle')]))
    verification=root/'verify-bundle';verification.mkdir()
    write(verification/'summary.json',summary)
    write(verification/'result.json',seal(dict(version='generalization-verification-result/v1',status='COMPLETE',archive=summary['archive'],summary=reference(verification/'summary.json',verification),exit_code=0)))
    final=publish(root,workflow,summary)
    monkeypatch.setattr(bundles,'verify_bundle',lambda *a,**kw:dict(integrity='PASS',software_status='PASS'))
    assert verify_archive(final)['software_status']=='PASS'
    target=root/'final-export'
    write(target/'workflow-final.json',seal(dict(workflow,status='FAIL')),exclusive=False)
    (target/'manifest.json').unlink()
    (root/'final-export.zip').unlink(missing_ok=True)
    tampered=pack(target,mandatory=['summary.json','evidence.json','study.json','workflow-final.json','verification-final.json','report-manifest.json','verification-terminal.json'])
    with pytest.raises(ValueError,match='workflow status'):verify_archive(tampered)


def pac_record(status='FAIL'):
    from oeis_learn.pipeline_validation.identity import seal,digest
    return seal(dict(version='pac-public-evidence/v1',plan=digest('plan'),source=digest('source'),images=[digest('cpu'),digest('gpu')],
        backend=digest('backend'),helper=dict(requested=digest('helper'),observed=digest('helper')),scope=dict(research=False,arms=['B','C'],seed=7,target_tokens=25000000,parameter='DEFERRED',assessment='NOT_EXECUTED'),
        required=['admission','qualification','calibration','acceptance'],stages=[],software_status=status,scientific_status='UNMEASURED',
        errors=[dict(stage='admission',kind='Injected',status=status)] if status!='PASS' else [],omissions=['native admission unavailable'],
        inventory=[],qualification=None,acceptance=None,native_admission=None,calibration=None))


@pytest.mark.parametrize('status',['FAIL','PARTIAL','INTERRUPTED','BLOCKED','INCONCLUSIVE'])
def test_pac_negative_archive_has_verified_terminal_and_preserved_scope(tmp_path,status):
    from oeis_learn.pipeline_validation.generalization_finalization import publish_pac
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    archive=publish_pac(pac_record(status),tmp_path/'report')
    result=verify_archive(archive)
    assert result['software_status']==status and result['integrity']=='PASS' and not result['feature_qualified']
    with zipfile.ZipFile(archive) as z:assert {'terminal.json','verification.json','evidence.json'}<=set(z.namelist())


def test_pac_stale_pass_missing_quota_or_failed_process_cannot_qualify(tmp_path):
    from oeis_learn.generalization.acceptance_archive import verify_pac_evidence
    from oeis_learn.pipeline_validation.identity import seal,digest
    record=pac_record('PASS')
    with pytest.raises(ValueError):verify_pac_evidence(record,root=tmp_path,finalized=False)
    stages=[dict(stage=n,status='PASS',producer=digest(n),binding=digest(n+'binding'),process=dict(outcome='FAIL',returncode=1),cleanup=dict(within_bound=True,acknowledged=True)) for n in record['required']]
    with pytest.raises(ValueError):verify_pac_evidence(seal(dict(record,stages=stages)),root=tmp_path,finalized=False)


def test_pac_complete_synthetic_proof_closure_roundtrips_real_collector_and_verifier(tmp_path):
    """Synthetic device labels test archive wiring; no hardware evidence is produced."""
    from copy import deepcopy
    from pathlib import Path
    from oeis_learn.pipeline_validation.identity import seal,digest
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_pac_export import export
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    from oeis_learn.pipeline_validation.generalization_execution import POLICY_ID
    from oeis_learn.generalization.qualification import FACTS
    from oeis_learn.generalization.structural import original_cells
    from tests.unit.test_generalization_metrics import pac_roster
    root=tmp_path/'campaign';root.mkdir()
    source=dict(synthetic_fixture=digest('not actual hardware'));source_id=digest(source)
    images=[digest('synthetic CPU'),digest('synthetic GPU')]
    environment=digest(dict(images=images,engine='docker',backend=POLICY_ID))
    qroot=root/'qualification/execution/qualification';qroot.mkdir(parents=True)
    write(qroot/'source.json',source)
    write(qroot/'build/Dockerfile',dict(synthetic_fixture=True))
    write(qroot/'device.json',seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights=digest('synthetic'),
        finite_forward=True,finite_backward=True,parameters_changed=True,resume_equal=True,execution_policy=POLICY_ID)))
    q=deepcopy(json.loads((Path(__file__).resolve().parents[1]/'fixtures/generalization/authority.json').read_text())['qualification'])
    device=reference(qroot/'device.json',qroot)
    q.update(images=images,source_closure=reference(qroot/'source.json',qroot),build_dependencies=[reference(qroot/'build/Dockerfile',qroot)],
        suite_selection=['bigint-cpu','bigint-gpu'],covered_capabilities=['weight-load','finite-update','native-exact'])
    q['runtime']['execution_policy']=POLICY_ID
    facts={k:q[k] for k in FACTS}
    producer=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,suites=q['suite_selection'],
        capabilities=q['covered_capabilities'],artifacts=[device],dependencies=[],feature_acceptance=False))
    write(qroot/'producer.json',producer)
    q.update(producer=dict(receipt=reference(qroot/'producer.json',qroot),status='PASS',artifacts=[device]),
        artifact_inventory=[reference(qroot/n,qroot) for n in ('source.json','device.json','producer.json','build/Dockerfile')])
    write(qroot/'qualification.json',seal(q))
    aroot=root/'acceptance/execution/execution/role/acceptance';aroot.mkdir(parents=True)
    for boundary in (1,2):
        write(aroot/f'proof-{boundary}/training-proof.json',seal(dict(status='PASS',resume_equal=True,pac_equal=True,
            runs=[dict(checkpoint=digest('synthetic final'),weights=digest('synthetic weights'),updates=16,finite=True,changed=True)]*2)))
    write(aroot/'checkpoint-evaluation.json',seal(dict(version='pac-checkpoint-forward/v1',status='PASS',device='hip',scope='training probes only',
        checkpoint=digest('synthetic final'),weights=digest('synthetic weights'),rows=2,tokens=10,loss_sum=3.,recall=[False,False])))
    failed=aroot/'failure';failed.mkdir();write(failed/'summary.json',dict(selected_scope='FAIL'))
    archive=pack(failed,mandatory=['summary.json'])
    acceptance=seal(dict(version='pac-feature-acceptance/v2',status='PASS',source=source_id,environment=environment,device='hip',research_eligible=True,
        cadence=dict(block_updates=2,cycle_updates=8,review_interval=1,common_updates=8),updates=16,valid_rewards=6,graded_anchors=1,resume_boundaries=[1,2],encoded_token_minimum=100,
        proofs={str(i):reference(aroot/f'proof-{i}/training-proof.json',aroot) for i in (1,2)},
        checkpoint_evaluation=reference(aroot/'checkpoint-evaluation.json',aroot),failure_archive=reference(archive,aroot)))
    write(aroot/'pac-acceptance.json',acceptance)
    croot=root/'calibration/execution/execution/role';croot.mkdir(parents=True)
    from oeis_learn.pipeline_validation.generalization_pac_calibration import forecast,REQUIRED
    samples={k:[1.] for k in REQUIRED}
    write(croot/'calibration.json',seal(dict(version='pac-calibration/v1',status='PASS',source=source_id,environment=environment,
        production_recipe=digest('synthetic model'),updates=64,tokens=100,valid_rewards=6,forecast_updates=10000,weights_are_research_initializers=False,
        samples=samples,forecasts={a:forecast(samples,updates=10000,arm=a,existing={}) for a in 'BC'})))
    nroot=root/'admission/execution/execution/role/private/assembly/resolved';nroot.mkdir(parents=True)
    write(nroot/'admission.json',seal(dict(status='PASS',source=source_id,images=images,originals=original_cells(),roster=pac_roster())))
    names=['admission','qualification','calibration','acceptance']
    plan=seal(dict(source=source_id,images=images,helper=digest('synthetic helper'),research=False,arms=['B','C'],stages=[dict(name=n,extra=['synthetic context','7'] if n=='admission' else []) for n in names]))
    write(root/'plan.json',plan)
    write(root/'workflow.json',seal(dict(plan=plan['identity'],status='PASS',errors=[],receipts=[dict(stage=n,status='PASS',producer=digest(n),binding=digest(['binding',n]),
        process=dict(outcome='COMPLETE',returncode=0),cleanup=dict(acknowledged=True,within_bound=True),elapsed_seconds=1,reused=False,directory=n,
        artifacts=[reference(p,root/n) for p in sorted((root/n).rglob('*')) if p.is_file()]) for n in names])))
    result=verify_archive(export(root,tmp_path/'public'))
    assert result['software_status']=='PASS' and result['feature_qualified']
    (aroot/'proof-1/training-proof.json').write_text('{}')
    minimal=verify_archive(export(root,tmp_path/'tampered'))
    assert minimal['software_status']=='FAIL' and not minimal['feature_qualified']
