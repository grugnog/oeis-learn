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


def reference_values(protocol,family,p0,p1=0):
    from oeis_learn.generalization.targets import ground
    from oeis_learn.synthesis.reference import evaluate
    source=protocol['families'][family]['parameter_source']
    assignment={'p0':str(p0)}
    if '(param.get p1)' in source:assignment['p1']=str(p1)
    value=evaluate(ground(source,assignment),list(map(str,range(100))))
    assert value.outcome=='ok'
    return tuple(map(int,value.values))


def test_all_positions_recognize_every_true_window_including_boundaries():
    from oeis_learn.generalization.selector_validation import fresh_design
    audit=module();protocol=fresh_design();families=tuple(audit.witnesses(protocol))
    for family in families:
        # Equal-coefficient, zero, one-term crossing and extreme-law cases.
        cases=[(-1024,-512),(1024,512),(0,0),(2,1),(5,-2),(-41,0),(41,1),(-42,0),(42,1)]
        for p0,p1 in cases:
            values=reference_values(protocol,family,p0,p1)
            for start in range(81):
                census=audit.AllWindows(families)
                census.project(values[start:start+20])
                assert census.blocked(family,p0,p1),(family,p0,p1,start)


def test_projection_has_no_false_blocks_against_independent_small_grid():
    from oeis_learn.generalization.selector_validation import fresh_design
    audit=module();protocol=fresh_design();families=tuple(audit.witnesses(protocol))
    pool={(f,p0,p1):reference_values(protocol,f,p0,p1) for f in families
          for p0 in (-1024,-42,-41,-2,0,2,41,42,1024) for p1 in (-1024,-2,0,1,2,1024)}
    windows=[]
    for f in families:
        v=reference_values(protocol,f,2,-1)
        windows.extend(v[k:k+20] for k in (0,1,12,13,22,23,24,40,41,42,80))
    windows.extend([tuple(range(20)),tuple(-n for n in range(20)),(7,)*20,tuple(n**3 for n in range(20))])
    windows.extend(tuple(a*j*j+b*j+c for j in range(20)) for a,b,c in [(1,47,552),(2,164,3362),(1,-43,0),(3,3,1),(-2,4,10)])
    for window in windows:
        census=audit.AllWindows(families);census.project(window)
        for (f,p0,p1),values in pool.items():
            expected=any(values[k:k+20]==window for k in range(81))
            assert census.blocked(f,p0,p1)==expected,(f,p0,p1,window)


def test_shifted_windows_can_block_assignments_with_uncovered_initial_window():
    from oeis_learn.generalization.selector_validation import fresh_design
    audit=module();families=tuple(audit.witnesses(fresh_design()))
    family=next(f for f in families if f.startswith('sha256:3f7f'))
    census=audit.AllWindows(families)
    window=tuple(10-n for n in range(20));census.project(window)
    assert census.blocked(family,90,0)  # Matches at candidate offset 80.
    assert not census.blocked(family,91,0)
    assert census.summary()[family]['window_free_assignments']==2049-81


@pytest.mark.parametrize('covered',[False,True])
def test_all_window_reader_and_copied_stdlib_cli(tmp_path,covered):
    audit,expected=fixture(tmp_path,covered=covered)
    result=audit.run(tmp_path,'plan.json','full',expected,all_windows=True)
    assert result['version']=='pac-fresh-support-audit/v2' and not result['freeze_eligible']
    assert all((v['window_free_assignments']==0) is covered for v in result['families'].values())
    copied=tmp_path/'run-state/audit.py';copied.parent.mkdir();copied.write_bytes(Path(audit.__file__).read_bytes())
    done=subprocess.run([sys.executable,'-S',str(copied),'--workspace',str(tmp_path),
        '--plan','plan.json','--expected-plan-sha256',expected,'--output','full-cli','--all-windows'],
        capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stderr
    other=json.loads((tmp_path/'full-cli/result.json').read_text())
    assert other['families']==result['families']


def test_full_row_accelerator_agrees_with_individual_window_projection():
    from oeis_learn.generalization.selector_validation import fresh_design
    audit=module();protocol=fresh_design();families=tuple(audit.witnesses(protocol))
    rows=[reference_values(protocol,f,41,1) for f in families]
    rows.append(tuple(10**30 if j%23==0 else j*j-42*j for j in range(100)))
    for values in rows:
        fast=audit.AllWindows(families);slow=audit.AllWindows(families)
        fast.observe(list(map(str,values)))
        for j in range(81):slow.project(values[j:j+20])
        assert fast.summary()==slow.summary()


def test_assignment_counts_handle_overlaps_and_nonuniform_sum_multiplicity():
    from oeis_learn.generalization.selector_validation import fresh_design
    audit=module();protocol=fresh_design();families=tuple(audit.witnesses(protocol))
    branch=next(f for f in families if f.startswith('sha256:5c47'))
    sums=next(f for f in families if f.startswith('sha256:ee89'))
    census=audit.AllWindows(families)
    values=reference_values(protocol,branch,2,1)
    census.project(values[:20])  # Equal-coefficient mixed window blocks one point.
    assert census.summary()[branch]['window_blocked_assignments']==1
    census.project(values[13:33])  # Entire p1 column subsumes that point.
    assert census.summary()[branch]['window_blocked_assignments']==2049
    census.project(tuple(range(20)))  # Sums 0..80, with triangular multiplicity.
    assert census.summary()[sums]['window_blocked_assignments']==sum(2049-s for s in range(81))
    assert census.summary()[sums]['window_free_output_classes']==4097-81
