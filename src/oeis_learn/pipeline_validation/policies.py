"""Visible-only serial policy comparison; hidden scores are never inputs."""
from copy import deepcopy
from .identity import seal


def compare_records(candidates, alternatives):
    from oeis_learn.fresh.synthesis_policy import select_visible
    from oeis_learn.fresh.candidate_archive import retain_alternatives, training_representative
    old_fields={'id','structure','source','cost','verified','verification','track','profile','assignment'}
    # The old serial collector retained the cheapest duplicate. Preserve that
    # interpretation in this comparison, rather than relabeling it as v2.
    legacy={}
    for row in alternatives:
        old={k:deepcopy(row[k]) for k in old_fields}
        if old['id'] not in legacy or old['cost']<legacy[old['id']]['cost']:
            legacy[old['id']]=old
    a=retain_alternatives(list(legacy.values()))
    b=retain_alternatives(alternatives,policy='structure-pareto/v2')
    selections={version:select_visible(candidates,policy='visible-only/'+version) for version in ('v1','v2')}
    representatives={'v1':training_representative(a),'v2':training_representative(b)}
    return seal(dict(version='pipeline-policy-comparison/v1',selections=selections,
        archives={'v1':a,'v2':b},training_representatives=representatives,
        differences=dict(selection=selections['v1']!=selections['v2'],archive=a!=b,
            training_representative=representatives['v1']!=representatives['v2']),
        authority='policy effects on supplied independently admitted visible records only; no hidden-score selection or performance adoption'))


def fixture_report():
    """Small declared fixture set; records are not actual model/native evidence."""
    from oeis_learn.synthesis.admission import source_identity
    candidates=[];alternatives=[]
    for ordinal,cost in ((0,9),(1,1)):
        source='(program (return $index))'
        candidates.append(dict(source=source,source_sha256=source_identity(source),verified=True,
            visible_count=20,cost=cost,attempt_ordinal=ordinal,job_id=source_identity(str(ordinal)),
            provenance=dict(prompt='fixture',snapshot='fixture',policies={'selection':'v2'},
                original_source_sha256=source_identity(source),transformed_source_sha256=source_identity(source),
                assignment={},verification=source_identity('fixture-verification'))))
        alternatives.append(dict(id=source_identity(source),source=source,structure='fixture',cost=cost,
            verified=True,verification=source_identity('fixture-verification'),track='generic_only',
            profile='bigint-body/v1',assignment={},attempt_ordinal=ordinal,job_id=source_identity(str(ordinal))))
    return seal(dict(version='pipeline-policy-fixtures/v1',fixtures_only=True,
        records=[compare_records(candidates,alternatives)]))
