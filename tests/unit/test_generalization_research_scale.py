"""Independent counterexamples for preparation at the registered research scale."""
import copy
import json
import random
from pathlib import Path

import pytest

from oeis_learn.generalization.cohorts import choose
from oeis_learn.generalization.exclusions import ExclusionIndex
from oeis_learn.pipeline_validation.identity import digest


def slow_choice(rows, count, quotas, structural):
    remaining=sorted(rows,key=lambda r:r['id']); selected=[]
    counts={k:0 for k in quotas}; families={k:set() for k in ('loop_active','branch_active')}
    while remaining and len(selected)<count:
        def score(r):
            return sum(k in r['tags'] and counts[k]<v for k,v in quotas.items())+sum(
                structural and k in r['tags'] and len(fs)<2 and r['family'] not in fs
                for k,fs in families.items())
        best=max(range(len(remaining)),key=lambda i:score(remaining[i]))
        row=remaining.pop(best);selected.append(row)
        for k in counts:counts[k]+=k in row['tags']
        for k,fs in families.items():
            if k in row['tags']:fs.add(row['family'])
    return selected


def test_quota_choice_matches_independent_original_order():
    rng=random.Random(915)
    for _ in range(80):
        rows=[dict(id=f'{i:04}',family=str(rng.randrange(5)),tags=[t for t in
            ('A','B','loop_active','branch_active','family:0') if rng.randrange(2)]) for i in range(40)]
        rng.shuffle(rows); count=rng.randrange(1,41)
        quotas={t:rng.randrange(count+1) for t in ('A','B','family:0')}
        structural=bool(rng.randrange(2))
        assert list(choose(rows,count=count,quotas=quotas,structural=structural).rows)==slow_choice(rows,count,quotas,structural)


def test_32768_row_choice_does_not_rescan_every_remaining_candidate():
    class Tags(list):
        probes=0
        def __contains__(self,value):
            type(self).probes+=1
            assert type(self).probes<2_000_000, 'quadratic full-shard quota search'
            return super().__contains__(value)
    rows=[dict(id=f'{i:05}',family=str(i%4),tags=Tags(['A'])) for i in range(32768)]
    result=choose(rows,count=32768,quotas={'A':16384})
    assert result.complete and len(result.rows)==32768


def exrow(i,role='training'):
    return dict(id=str(i),role=role,family=str(i%7),holdout='parameter',source=str(i),
                values=[str(i*1000+j*j) for j in range(100)],group=str(i),variant=str(i))


def oracle_reasons(row,old,window=20):
    found=set(); windows={tuple(row['values'][i:i+window]) for i in range(101-window)}
    for p in old:
        if p['role']==row['role']:continue
        if (p['holdout']=='structural' or row['holdout']=='structural') and p['family']==row['family']:found.add('structural_family')
        for key in ('source','group'):
            if p[key]==row[key]:found.add(key)
        if p['family']==row['family'] and p['variant']==row['variant']:found.add('numeric_variant')
        if any(tuple(p['values'][i:i+window]) in windows for i in range(101-window)):found.add('finite_window')
    return tuple(sorted(found))


def test_bulk_exclusions_preserve_independent_reasons_and_prior_index():
    rows=[exrow(i) for i in range(60)]
    original=ExclusionIndex(); expanded=original.add_many(rows)
    assert original.records==() and len(expanded.records)==60
    for i in range(80):
        r=exrow(i,'selection')
        if i%2:r['holdout']='structural'
        if i%3:r['values']=r['values'][1:]+['999999999']
        assert expanded.reasons(r)==oracle_reasons(r,rows)
    with pytest.raises(ValueError,match='cross-role'):
        expanded.add_many([exrow(500,'selection'),exrow(0,'assessment')])
    assert len(expanded.records)==60
    assert expanded.add_many([exrow(0)]).records[-1]==expanded.records[0]


def test_bulk_checks_collisions_within_the_new_batch():
    with pytest.raises(ValueError,match='cross-role'):
        ExclusionIndex().add_many([exrow(0,'selection'),exrow(0,'assessment')])


def test_full_static_shard_variant_manifest_is_supported():
    from oeis_learn.generalization.contracts import freeze
    ref=dict(path='x.json',sha256=digest('x'),bytes=1)
    payload=dict(version='generalization-cohort/v1',role='training',family_rule='fixed-descriptor/v1',
        sources=[digest('source')]*32768,variants=[str(i) for i in range(32768)],
        groups=[digest('group')]*32768,quotas={},coverage={},exclusions=ref,
        indices=list(map(str,range(100))),public_payloads=[],private_payloads=[ref],preparation_ledger=ref)
    assert freeze('cohort',payload).record()['variants']==payload['variants']
    payload['variants'].append('overflow')
    with pytest.raises(ValueError):freeze('cohort',payload)


def test_research_preparation_envelope_is_not_acceptance_smoke_limit():
    from oeis_learn.pipeline_validation.generalization_commands import preparation_seconds
    assert preparation_seconds('research',86400)==86400
    assert preparation_seconds('acceptance',86400)==600
    assert preparation_seconds('research',21600)==21600
    assert preparation_seconds('research',1800)==1800
    assert preparation_seconds('acceptance',900)==600
    assert preparation_seconds('acceptance',180)==180
    for purpose,seconds in [('research',86401),('unknown',60),('research',0),('research',float('nan'))]:
        with pytest.raises(ValueError):preparation_seconds(purpose,seconds)


def test_large_cohort_manifest_survives_real_custodian_projection(tmp_path):
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.generalization_preparation import reference
    from oeis_learn.pipeline_validation.generalization_runtime import execution_refs
    write(tmp_path/'empty.json',[]); small=reference(tmp_path/'empty.json',tmp_path)
    cohort=dict(sources=[digest(str(i)) for i in range(32768)],public_payloads=[],private_payloads=[])
    write(tmp_path/'cohort.json',cohort); large=reference(tmp_path/'cohort.json',tmp_path)
    assert 1<<20<large['bytes']<16<<20
    study=dict(model=dict(initialization='random'),source_closure=small,
        data=dict(generator=small,descriptors=small,exclusions=small,manifests=[large]))
    assert large in execution_refs(tmp_path,study,dict(recipe=small,replay=None))


def test_family_quota_is_derived_from_verified_descriptor_not_candidate_tags(tmp_path):
    from oeis_learn.data.bigint_programs import BigintSampler,materialize
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell
    sampler=BigintSampler(20261016)
    rows=[sampler.sample(i) for i in (0,1,16,17)]
    assert len({r['family'] for r in rows})==2
    def execute(source,indices):
        from oeis_learn.synthesis.reference import evaluate
        result=evaluate(source,indices)
        return dict(outcome=result.outcome,values=result.values)
    quotas={'family:'+family:1 for family in {r['family'] for r in rows}}
    result=prepare_cell(rows,role='training',holdout='parameter',count=2,quotas=quotas,
        structural=False,output=tmp_path/'cell',execute=execute)
    assert result['status']=='PASS'
    selected=json.loads((tmp_path/'cell/private/selected.json').read_text())
    assert {r['family'] for r in selected}=={r['family'] for r in rows}


def test_length_bins_are_frozen_and_bounded(tmp_path):
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_cell
    for value in (True,0,4097,16.5):
        with pytest.raises(ValueError,match='length bin'):
            prepare_cell([],role='training',holdout='parameter',count=1,quotas={},structural=False,
                output=tmp_path/'bad',execute=lambda *_:None,length_boundary=value)
    assert not (tmp_path/'bad').exists()


def test_schema_failure_never_echoes_protected_payload():
    from oeis_learn.generalization.contracts import shape
    secret='protected-target-'+'9'*10000
    with pytest.raises(ValueError) as caught:
        shape('visible-prompt',dict(id='row',indices=['0']*20,values=[secret]*20))
    assert secret[:32] not in str(caught.value)
    assert len(str(caught.value))<512
