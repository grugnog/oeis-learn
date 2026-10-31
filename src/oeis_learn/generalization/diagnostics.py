"""Candidate-only structure diagnostics; no hidden reference needed."""
from oeis_learn.synthesis.reference_parser import parse, SourceError
from oeis_learn.pipeline_validation.identity import digest


def signature(source):
    try:
        p=parse(source,allow_parameters=True)
    except (SourceError,ValueError,TypeError):
        return dict(parse_status='invalid',signature=None,parameters=None,source_hash=digest(source))
    params=set()
    def walk(n):
        if isinstance(n,str): return n
        if n[0]=='param.get': params.add(n[1]); return ('param.get',n[1])
        if n[0]=='int': return ('int','<literal>')
        return tuple(walk(x) for x in n)
    tree=walk(('program',*p.statements,('return',p.result)))
    return dict(parse_status='ok',signature=digest([p.declarations,tree]),parameters=len(params),source_hash=digest(source))


def join_attempts(attempts, fits):
    ids=[a['id'] for a in attempts]
    if len(set(ids))!=len(ids) or set(fits)-set(ids): raise ValueError('attempt identity mismatch')
    result=[]
    for attempt in attempts:
        attrs=signature(attempt.get('source'))
        fit=fits.get(attempt['id'],dict(status='unavailable',reason='historical_field_missing'))
        result.append(dict(attempt,**attrs,duplicate_group=attrs['source_hash'],fit=fit,
                           final_source_hash=digest(fit['final_source']) if fit.get('final_source') is not None else None,
                           first_error=fit.get('reason'),queue_seconds=fit.get('queue_seconds'),
                           service_seconds=fit.get('service_seconds'),cleanup_seconds=fit.get('cleanup_seconds')))
    return tuple(result)


def target_diagnostics(attempts, truth, *, frozen_selection, authorized_selection):
    if not frozen_selection or frozen_selection!=authorized_selection:
        raise ValueError('target diagnostics require frozen assessment authorization')
    return tuple(dict(id=a['id'],finite_match=tuple(a.get('values',()))==tuple(truth)) for a in attempts)
