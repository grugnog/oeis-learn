import copy
import hashlib
import json
import pytest
from oeis_learn.generalization.qualification import qualify_reuse
from oeis_learn.pipeline_validation.identity import seal

def producer(tmp_path):
    evidence=seal(dict(version='generalization-device-evidence/v1',device='hip',loaded_weights='a',finite_forward=True,finite_backward=True,parameters_changed=True))
    raw=json.dumps(evidence).encode(); (tmp_path/'update.json').write_bytes(raw)
    ref=dict(path='update.json',sha256='sha256:'+hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    facts=dict(source_closure='code',build_dependencies='build',images=['sha256:image'],runtime={'hip_available':True,'version':'1'},hardware={'device':'gpu'},precision='fp32',resources={'memory':100},engine_security={'socket':False})
    receipt=seal(dict(version='generalization-producer/v1',status='PASS',facts=facts,suites=['cpu','hip'],capabilities=['load','update'],artifacts=[ref],dependencies=[],feature_acceptance=False))
    return receipt,facts

def test_original_pass_and_exact_current_facts(tmp_path):
    receipt,facts=producer(tmp_path)
    assert qualify_reuse(receipt,facts,required_suites={'cpu','hip'},required_capabilities={'load'},root=tmp_path,research=False)
    for key in facts:
        changed=copy.deepcopy(facts); changed[key]='changed'
        with pytest.raises(ValueError): qualify_reuse(receipt,changed,required_suites={'hip'},required_capabilities={'load'},root=tmp_path)
    with pytest.raises(ValueError): qualify_reuse(seal(dict(receipt,status='REUSED')),facts,required_suites={'hip'},required_capabilities={'load'},root=tmp_path)
    with pytest.raises(ValueError): qualify_reuse(receipt,facts,required_suites={'hip'},required_capabilities={'new'},root=tmp_path)
    with pytest.raises(ValueError): qualify_reuse(receipt,facts,required_suites={'hip'},required_capabilities={'load'},root=tmp_path,research=True)
    (tmp_path/'update.json').write_text('tampered')
    with pytest.raises(ValueError): qualify_reuse(receipt,facts,required_suites={'hip'},required_capabilities={'load'},root=tmp_path)


def test_qualification_can_bind_effective_resource_policy():
    """T035 boundary: computed effective limits must survive external authority.

    This is a Phase-B exit counterexample, not a relaxed integration validator.
    Dropping these fields would let different CPU/thread/learner allocations share
    the same external qualification identity despite the exact-match contract.
    """
    from pathlib import Path
    from oeis_learn.generalization.contracts import freeze
    from oeis_learn.generalization.resources import compute_resources
    authority=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['qualification']
    effective=compute_resources(physical_bytes=128<<30,cgroup_bytes=112<<30,
        effective_cpus=16,peak_bytes=64<<30,headroom_bytes=8<<30,native_bytes=4<<30)
    authority['resources'].update(effective)
    frozen=freeze('qualification',authority).record()
    assert all(frozen['resources'][key]==value for key,value in effective.items())


def test_qualification_v1_and_omitted_execution_facts_rejected():
    from pathlib import Path
    from oeis_learn.generalization.contracts import freeze
    authority=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['qualification']
    with pytest.raises(ValueError):freeze('qualification',dict(authority,version='generalization-qualification/v1'))
    for section in ('resources','engine_security'):
        for key in authority[section]:
            omitted=copy.deepcopy(authority);del omitted[section][key]
            with pytest.raises(ValueError):freeze('qualification',omitted)


def test_each_effective_resource_and_security_change_invalidates_reuse(tmp_path):
    from pathlib import Path
    fixture=json.loads(Path('tests/fixtures/generalization/authority.json').read_text())['qualification']
    receipt,facts=producer(tmp_path)
    facts.update(resources=fixture['resources'],engine_security=fixture['engine_security'])
    receipt=seal(dict(receipt,facts=facts))
    for section in ('resources','engine_security'):
        for key,value in facts[section].items():
            changed=copy.deepcopy(facts)
            changed[section][key]=not value if type(value) is bool else value+1 if type(value) is int else value+'-changed'
            with pytest.raises(ValueError):qualify_reuse(receipt,changed,root=tmp_path,required_suites={'hip'},required_capabilities={'load'})
