import importlib.util
from pathlib import Path
import time
import pytest
from oeis_learn.synthesis.admission import source_identity
from oeis_learn.synthesis.contracts import identity

spec = importlib.util.spec_from_file_location('post_pilot_analysis', Path(__file__).with_name('post_pilot_analysis.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def envelope():
    source = '(program (return (int 3)))'
    selected = dict(source=source, source_sha256=source_identity(source), verified=True, visible_count=20, cost=0)
    selected.update(seal_sha256=identity(selected), selection='visible-only/v1')
    return dict(original_source='(program (return (int 7)))',
        original_source_sha256=source_identity('(program (return (int 7)))'),
        transformed_source='(program (return (param.get p0)))', assignment={'p0':'3'},
        selected=selected, verification=dict(indices=list(map(str,range(20))), values=['3']*20,
        scope='visible_only', admitted=True, source_sha256=source_identity(source)))


def test_grounding_and_independent_execution():
    e=envelope(); m.check_envelope(e)
    p=(e['verification']['indices'],e['verification']['values'])
    assert m.execute(e['original_source'],p,time.monotonic()+5)['visible_match'] is False
    assert m.execute(e['selected']['source'],p,time.monotonic()+5)['visible_match'] is True


@pytest.mark.parametrize('change', ['source','assignment','original','seal','scope'])
def test_invalid_selected_provenance_rejected(change):
    e=envelope()
    if change=='source': e['selected']['source']='(program (return (int 4)))'
    if change=='assignment': e['assignment']['p0']='4'
    if change=='original': e['original_source']+=' '
    if change=='seal': e['selected']['seal_sha256']='sha256:bad'
    if change=='scope': e['verification']['scope']='assessment'
    with pytest.raises(ValueError): m.check_envelope(e)


def test_decode_source_and_hash_must_agree():
    from oeis_learn.decoder.bigint_codec import encode_body,decode_body
    tokens=encode_body('(program (return (int 7)))'); source=decode_body(tokens)
    a=dict(tokens=tokens,source=source,source_sha256=source_identity(source),generation=dict(complete=True))
    assert m.check_attempt(a)['parse_status']=='ok'
    a['tokens']=encode_body('(program (return (int 8)))')
    with pytest.raises(ValueError): m.check_attempt(a)
    a.update(source=None, generation=dict(complete=False))
    assert m.check_attempt(a)['parse_status']=='unavailable'


def test_equal_totals_do_not_mean_same_successes_and_incomplete_stays_unresolved():
    def row(success,complete=True):return {'equal_budget':dict(success=success,complete=complete,reason=None)}
    a={'a':row(True),'b':row(False),'c':row(False,False)}
    b={'a':row(False),'b':row(True),'c':row(False,False)}
    assert m.compare(a,b)['equal_budget']==dict(both_success=0,static_only=1,refreshed_only=1,neither_success=1)
    assert m.count_outcomes(a)['equal_budget']['unresolved']==1
    b['d']=row(False)
    with pytest.raises(ValueError):m.compare(a,b)
    a['c']['equal_budget']['success']=True
    with pytest.raises(ValueError):m.count_outcomes(a)


def test_missing_prompt_is_unavailable_and_prefix_match_is_not_continuation_success():
    s='(program (local $b0 bigint) (if (lt $index (int 20)) (then (set $b0 (int 3))) (else (set $b0 (int 9)))) (return (get $b0)))'
    missing=m.execute(s,None,time.monotonic()+5)
    assert missing['status']=='ok' and missing['visible_match'] is None
    assert missing['visible_values_available'] is False
    assert m.execute(None,None,time.monotonic()+5)['status']=='unavailable'
    result=m.execute(s,(list(map(str,range(20))),['3']*20),time.monotonic()+5)
    assert result['visible_match'] and 'success' not in result
    assert result['first_values']==['3']*5


def test_conflicting_visible_prompts_rejected():
    e=envelope(); f=envelope(); f['verification']['values'][3]='99'
    with pytest.raises(ValueError):m.prompt_for({'selections':{'raw':e,'rescue':f}})


def test_archive_tamper_and_overwrite(tmp_path):
    import json,zipfile
    from oeis_learn.pipeline_validation.bundle import pack
    root=tmp_path/'input';root.mkdir();(root/'summary.json').write_text(json.dumps({'selected_scope':'PASS'}))
    archive=pack(root,mandatory=['summary.json'])
    assert m.verify_input(archive)['version']=='pipeline-bundle/v1'
    with zipfile.ZipFile(archive,'a') as z:z.writestr('extra.json','{}')
    with pytest.raises(ValueError):m.verify_input(archive)
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(FileExistsError):m.run(archive,out,seconds=5)


def test_real_prompt_hash_and_provenance_are_checked():
    from oeis_learn.synthesis.admission import VisiblePrompt
    e=envelope();v=e['verification'];e['prompt_sha256']=v['prompt_sha256']=VisiblePrompt(tuple(v['indices']),tuple(v['values'])).sha256
    m.check_envelope(e)
    v['values'][0]='4'
    with pytest.raises(ValueError):m.check_envelope(e)


def test_legacy_bank_recovery_is_explicit_hash_bound_and_nonmutating():
    import copy
    e=envelope();original=e['original_source'];e['original_source']=e['transformed_source'];before=copy.deepcopy(e)
    with pytest.raises(ValueError):m.check_envelope(e)
    fixed,finding=m.bank_envelope(e,dict(proposals=[dict(source=original)]))
    assert fixed['original_source']==original and finding['kind']=='bank_original_source_mismatch'
    assert e==before
    m.check_envelope(fixed)
    with pytest.raises(ValueError):m.bank_envelope(e,dict(proposals=[dict(source='wrong')]))


@pytest.mark.parametrize('operator,threshold,expected', [('lt',20,['p1']),('lt',19,[]),('eq',46,['p0']),('eq',19,[])])
def test_unobserved_branch_parameters(operator,threshold,expected):
    source=f'(program (local $b0 bigint) (if ({operator} $index (int {threshold})) (then (set $b0 (mul $index (param.get p0)))) (else (set $b0 (add $index (param.get p1))))) (return (get $b0)))'
    assert m.branch_visibility(source)['unseen_parameters']==expected


def test_unresolved_sample_chooses_failed_fit_not_merely_completed_generation():
    a=dict(complete=True,generation={'complete':True},fitting={'complete':False})
    assert m.incomplete_attempt(a)
    a['fitting']['complete']=True
    assert not m.incomplete_attempt(a)


def test_branch_witness_checks_execution_and_does_not_count_cancelled_parameter():
    source='(program (local $b0 bigint) (if (lt $index (int 20)) (then (set $b0 (param.get p0))) (else (set $b0 (param.get p1)))) (return (get $b0)))'
    witness=m.branch_witness(source,['p1'],time.monotonic()+5)
    assert witness['visible_equal'] and witness['first_difference_index']==20
    cancelled=source.replace('(set $b0 (param.get p1))','(set $b0 (sub (param.get p1) (param.get p1)))')
    assert m.branch_witness(cancelled,['p1'],time.monotonic()+5) is None


def test_failure_preserves_verified_partial_archive(tmp_path):
    import zipfile,json
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('manifest.json','{}')
    out=tmp_path/'audit'
    with pytest.raises(RuntimeError):m.run(bad,out,seconds=5)
    assert json.loads((out/'summary.json').read_text())['selected_scope']=='FAIL'
    assert m.verify_input(out.with_suffix('.zip'))['summary_state']=='FAIL'


def test_distinct_originals_can_converge_to_the_same_fitted_program():
    def record(original,final):return dict(details={'correct':{'details':[{'source':original}]}},selections={'equal_budget':{'selected':{'source':final}}})
    a={'t':record('(program (return (int 2)))','(program (return (int 5)))')}
    b={'t':record('(program (return (int 3)))','(program (return (int 5)))')}
    c=m.paired_candidate_counts(a,b)
    assert c['first_original_equal']==0 and c['selected_source_equal']==1 and c['first_structure_equal']==1


def test_missing_input_still_packages_failure(tmp_path):
    import json
    out=tmp_path/'audit'
    with pytest.raises(RuntimeError):m.run(tmp_path/'missing.zip',out,seconds=5)
    assert json.loads((out/'summary.json').read_text())['selected_scope']=='FAIL'
    m.verify_input(out.with_suffix('.zip'))


def test_expired_witness_is_unavailable_not_evidence_of_no_ambiguity():
    source='(program (return (param.get p0)))'
    with pytest.raises(ValueError):m.branch_witness(source,['p0'],time.monotonic()-1)
