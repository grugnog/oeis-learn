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
