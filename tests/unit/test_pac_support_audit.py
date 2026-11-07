"""Outcome-blind support witnesses never establish a fresh cohort or model result."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oeis_learn.pipeline_validation.identity import seal


def module():
    path=Path(__file__).resolve().parents[2]/'scripts/pac_fresh_support_audit.py'
    spec=importlib.util.spec_from_file_location('support_audit',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_witnesses_cover_law_and_match_exact_reference():
    from oeis_learn.generalization.selector_validation import fresh_design
    from oeis_learn.generalization.targets import ground
    from oeis_learn.synthesis.reference import evaluate
    audit=module();protocol=fresh_design();windows=audit.witnesses(protocol)
    assert sum(len(v) for v in windows.values())==10244
    assert sorted(len(v) for v in windows.values())==[2049,2049,2049,4097]
    for family,items in windows.items():
        source=protocol['families'][family]['parameter_source']
        for parameter in [min(items),0,max(items)]:
            if family.startswith('sha256:ee89'):
                p0=max(-1024,min(1024,parameter));assignment={'p0':str(p0),'p1':str(parameter-p0)}
            elif family.startswith('sha256:5c47'):assignment={'p0':'-13','p1':str(parameter)}
            elif family.startswith('sha256:83a8'):assignment={'p0':str(parameter),'p1':'17'}
            else:assignment={'p0':str(parameter)}
            result=evaluate(ground(source,assignment),list(map(str,range(100))))
            assert result.outcome=='ok'
            start=13 if family.startswith('sha256:5c47') else 0
            assert tuple(result.values[start:start+20])==items[parameter]


def fixture(tmp_path,*,covered):
    audit=module()
    from oeis_learn.generalization.selector_validation import fresh_design
    protocol=fresh_design();rows=[]
    if covered:
        for windows in audit.witnesses(protocol).values():
            rows.extend(dict(values=list(w)+['0']*80) for w in windows.values())
    def save(name,rows):
        data=json.dumps(rows).encode();(tmp_path/name).write_bytes(data)
        return dict(path=name,bytes=len(data),sha256='sha256:'+hashlib.sha256(data).hexdigest())
    candidates=save('originals.json',rows);empty=save('empty.json',[])
    originals=[dict(name=name,count=len(rows) if i==0 else 0,ref=candidates if i==0 else empty)
        for i,name in enumerate(audit.ORIGINALS)]
    plan=seal(dict(phase='structural',protocol=protocol,inputs=dict(originals=originals,
        observed=save('observed.json',[dict(values=['0']*100)]*128),
        probes=save('probes.json',[dict(values=['0']*100,role='training')]*256))))
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    return audit,'sha256:'+hashlib.sha256((tmp_path/'plan.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('covered',[False,True])
def test_census_is_diagnostic_and_preserves_inputs(tmp_path,covered):
    audit,expected=fixture(tmp_path,covered=covered)
    before=(tmp_path/'plan.json').read_bytes()
    result=audit.run(tmp_path,'plan.json','out',expected)
    assert result['scientific_status']=='DIAGNOSTIC_ONLY' and not result['freeze_eligible']
    assert all(r['all_assignments_blocked'] is covered for r in result['families'].values())
    assert (tmp_path/'plan.json').read_bytes()==before
    with pytest.raises(ValueError,match='fresh'):audit.run(tmp_path,'plan.json','out',expected)
    assert not any('values' in r for r in result['families'].values())


def test_tampered_or_unsafe_input_cannot_publish(tmp_path):
    audit,expected=fixture(tmp_path,covered=False)
    (tmp_path/'observed.json').write_text('[]')
    with pytest.raises(ValueError,match='reference mismatch'):
        audit.run(tmp_path,'plan.json','out',expected)
    assert not (tmp_path/'out/report.zip').exists()
    with pytest.raises(ValueError,match='plan hash'):
        audit.run(tmp_path,'plan.json','other','sha256:'+'0'*64)
    with pytest.raises(ValueError):audit.confined(tmp_path,'../outside')
    (tmp_path/'link').symlink_to(tmp_path/'plan.json')
    with pytest.raises(ValueError,match='symlink'):audit.confined(tmp_path,'link')


def test_changed_protocol_cannot_use_old_witness_formula():
    from oeis_learn.generalization.selector_validation import fresh_design
    audit=module();protocol=fresh_design()
    with pytest.raises(ValueError,match='reviewed protocol'):
        audit.witnesses(seal(dict(protocol,numeric_law='different')))


def test_copied_command_needs_only_stdlib_and_reports_its_own_bytes(tmp_path):
    audit,expected=fixture(tmp_path,covered=False)
    copied=tmp_path/'run-state/audit.py';copied.parent.mkdir()
    copied.write_bytes(Path(audit.__file__).read_bytes())
    result=subprocess.run([sys.executable,'-S',str(copied),'--workspace',str(tmp_path),
        '--plan','plan.json','--expected-plan-sha256',expected,'--output','cli-output'],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    record=json.loads((tmp_path/'cli-output/result.json').read_text())
    assert record['audit_source']=='sha256:'+hashlib.sha256(copied.read_bytes()).hexdigest()
    assert all(not r['all_assignments_blocked'] for r in record['families'].values())


def test_expired_census_cannot_publish_a_conclusion(tmp_path):
    audit,expected=fixture(tmp_path,covered=False);audit.SECONDS=0
    with pytest.raises(TimeoutError,match='deadline'):
        audit.run(tmp_path,'plan.json','expired',expected)
    assert not (tmp_path/'expired/report.zip').exists()
