import copy
import json
from pathlib import Path
import pytest
from oeis_learn.generalization.contracts import freeze, validate, verify_artifact, training_ancestry, resolve_study

FIXTURE = Path('tests/fixtures/generalization/authority.json')

def example(name):
    return copy.deepcopy(json.loads(FIXTURE.read_text())[name])

def test_identity_and_immutable_bytes():
    raw = example('preparation')
    frozen = freeze('preparation', raw)
    raw['numeric_law'] = 'changed'
    assert frozen.record()['numeric_law'] != 'changed'
    changed = frozen.record(); changed['numeric_law'] = 'changed'
    with pytest.raises(ValueError): validate('preparation', changed)
    changed = frozen.record(); changed['hidden_terms'] = []
    with pytest.raises(ValueError): freeze('preparation', changed)

def test_confined_verified_file(tmp_path):
    import hashlib
    (tmp_path / 'ok').write_bytes(b'abc')
    ref = dict(path='ok', bytes=3, sha256='sha256:'+hashlib.sha256(b'abc').hexdigest())
    assert verify_artifact(tmp_path, ref) == b'abc'
    for update in ({'bytes': 2}, {'path': '../ok'}, {'sha256': 'sha256:'+'0'*64}):
        with pytest.raises(ValueError): verify_artifact(tmp_path, dict(ref, **update))
    (tmp_path / 'link').symlink_to(tmp_path / 'ok')
    with pytest.raises(ValueError): verify_artifact(tmp_path, dict(ref, path='link'))

def test_ancestry_and_failed_preparation():
    training_ancestry(('mechanical', 'training'))
    for origin in ('oeis', 'selection', 'assessment', 'unknown', 'named_teacher'):
        with pytest.raises(ValueError): training_ancestry(('mechanical', origin))
    prep = freeze('preparation', example('preparation'))
    with pytest.raises(ValueError, match='shortfall'):
        resolve_study(prep, example('study'), (), shortfalls=('loop_active:2',))


def test_p4_cannot_freeze_random_initialization_without_common_retained_authority():
    # P4 must name the common selection-designated literal initializer before
    # search/distillation. A free runtime initializer would contradict this
    # frozen study and could silently import arbitrary weights.
    study=example('study')
    study.update(recipe='P4-replay/v1',arms=['continued_sft','verified_replay'],changing_factors=['verified_replay'])
    study['progression']['stage']='P4'
    study['budgets'].update(target_tokens=4_000_000,seconds=7200)
    study['checkpoints']['milestones']=[1_000_000,2_000_000,4_000_000]
    assert study['model']['initialization']=='random'
    with pytest.raises(ValueError,match='initializer|initialization'):
        freeze('study',study)


def pac_shell():
    from oeis_learn.pipeline_validation.identity import digest
    h=digest('fixture')
    return dict(version='PAC-screen/v1',arms=['B','C'],comparison_factors=['allocation_and_review'],seed=20261016,
                initializer=h,initializer_kind='random',pool=h,probes=h,targets=h,inventory=h,
                policy='pac-exp3s-leitner/v1',budget_graph=[],metrics=['certified_family_macro'],
                control=h,control_mode='reference-only',frozen=True,outcomes_seen=False,
                target_tokens=25_000_000,model=h,optimizer=h,backend=h,order=h,source=h,exposure=h,diagnostics=h,literal_targets=h)


def test_pac_closed_shell_and_unknown_version():
    from oeis_learn.generalization.contracts import pac_record,pac_public
    raw=pac_shell();frozen=pac_record('experiment',raw,sealed=False)
    assert pac_public('experiment',frozen.record())==frozen.record()
    for extra in ({'hidden_terms':[]},{'version':'PAC-screen/v2'},{'policy':'visit-scheduler/v1'}):
        with pytest.raises(ValueError):pac_record('experiment',dict(raw,**extra),sealed=False)
    with pytest.raises(ValueError):pac_record('adaptive',dict(version='visit-scheduler/v1'),sealed=False)
    with pytest.raises(ValueError):pac_record('unregistered',raw,sealed=False)
    changed=frozen.record();changed['seed']+=1
    with pytest.raises(ValueError):pac_record('experiment',changed)


def test_all_pac_object_shapes_are_closed():
    schema=json.loads(Path('src/oeis_learn/generalization/schemas/pac.schema.json').read_text())
    def walk(node):
        if isinstance(node,dict):
            if node.get('type')=='object':
                assert node['additionalProperties'] is False
                assert set(node['required'])==set(node['properties'])
            for value in node.values():walk(value)
        elif isinstance(node,list):
            for value in node:walk(value)
    walk(schema)


def test_pac_schema_rng_and_public_receipt_cannot_carry_arbitrary_private_payloads():
    from oeis_learn.generalization.contracts import pac_record,pac_public
    from oeis_learn.pipeline_validation.identity import digest
    from jsonschema import Draft202012Validator
    schema=json.loads(Path('src/oeis_learn/generalization/schemas/pac.schema.json').read_text())
    rng=schema['$defs']['adaptive']['properties']['bandit_rng']
    assert not Draft202012Validator(rng).is_valid([{'hidden_terms':['1','2']}])
    h=digest('fixture')
    receipt=dict(version='pac-stage-receipt/v1',input=h,source=h,image=h,policy=h,helper=h,parents=[],command=h,
                 application='PASS',process='PASS',cleanup='PASS',artifacts=[dict(path='reports/metrics.json',sha256=h,bytes=1)],omissions=[],errors=[])
    assert pac_public('receipt',pac_record('receipt',receipt,sealed=False).record())['application']=='PASS'
    for path in ('/home/private/weights.pt','reports/private/truth.json','reports/../weights.pt','reports/checkpoint.json'):
        changed=copy.deepcopy(receipt);changed['artifacts'][0]['path']=path
        with pytest.raises(ValueError):pac_public('receipt',pac_record('receipt',changed,sealed=False).record())
    changed=dict(receipt,errors=[dict(stage='export',reason='failed /home/user/private')])
    with pytest.raises(ValueError):pac_public('receipt',pac_record('receipt',changed,sealed=False).record())
