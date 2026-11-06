"""Exact paired targets; independent control-taint check mirrors the native boundary.

Native parameters/dependence/fit remain the production fitting path. This policy
never searches coefficients or replaces that fitter. Nonlinear terms retain the
bounded SMT path; whole-program control-dependent parameters are unsupported.
"""
from oeis_learn.synthesis.reference_parser import parse
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.contracts import canonical_integer, ResourceProfile
from oeis_learn.data.bigint_programs import render
from oeis_learn.decoder.bigint_codec import encode_body, decode_body


def learning_target(proof):
    """Retain unsupported rows with their original literal target and disposition.

    The original pair proof is kept separately for auditing. A control-sensitive
    parameter is never taught as an admissible fitted target.
    """
    if proof['disposition']=='supported':
        return proof['parameter_source'],proof['parameter_tokens']
    if proof['disposition']=='unsupported_control_dependency':
        return proof['literal_source'],proof['literal_tokens']
    raise ValueError('unknown paired-target disposition')


def ground(source,assignment):
    p=parse(source,allow_parameters=True)
    names=set()
    def visit(n):
        if isinstance(n,str): return n
        if n[0]=='param.get':
            names.add(n[1]); canonical_integer(assignment[n[1]])
            return ('int',assignment[n[1]])
        return tuple(visit(x) for x in n)
    try:
        tree=('program',*(('local',n,k) for n,k in p.declarations),*(visit(s) for s in p.statements),('return',visit(p.result)))
    except KeyError as exc: raise ValueError('missing grounding') from exc
    if names!=set(assignment) or len(names)>8: raise ValueError('assignment/parameter mismatch')
    result=render(tree); parse(result)
    return result


def _control_taint(p):
    locals_={n:set() for n,_ in p.declarations}
    def dependencies(n):
        if isinstance(n,str): return set()
        if n[0]=='param.get': return {n[1]}
        if n[0]=='get': return locals_[n[1]]
        return set().union(*(dependencies(x) for x in n[1:]))
    def assignments(nodes):
        for n in nodes:
            if n[0]=='set': locals_[n[1]].update(dependencies(n[2]))
            elif n[0]=='if': assignments(n[2][1:]); assignments(n[3][1:])
            elif n[0]=='loop': assignments(n[3:])
    while True:
        old={k:set(v) for k,v in locals_.items()}; assignments(p.statements)
        if old==locals_: break
    protected=set()
    def walk(n):
        if isinstance(n,str): return
        if n[0] in ('eq','lt','le','to_control','control.lt'): protected.update(dependencies(n))
        if n[0] in ('if','select'): protected.update(dependencies(n[1]))
        if n[0]=='loop': protected.update(dependencies(n[2]))
        for x in n[1:]: walk(x)
    for n in (*p.statements,p.result): walk(n)
    return protected


def pair(concrete,parameterized,assignment,*,indices,sharing,limits=ResourceProfile()):
    if sharing not in {'explicit','distinct-occurrences'}: raise ValueError('registered sharing policy required')
    final=ground(parameterized,assignment)
    p=parse(parameterized,limits,allow_parameters=True)
    if parse(final,limits)!=parse(concrete,limits): raise ValueError('grounding is not the paired concrete program')
    counts={}; nonlinear=False
    def degree(n):
        nonlocal nonlinear
        if isinstance(n,str): return 0
        if n[0]=='param.get': counts[n[1]]=counts.get(n[1],0)+1; return 1
        ds=[degree(x) for x in n[1:]]
        d=sum(ds) if n[0]=='mul' else max(ds,default=0)
        nonlinear|=d>1
        return d
    for n in (*p.statements,p.result): degree(n)
    if sharing=='distinct-occurrences' and any(c>1 for c in counts.values()): raise ValueError('implicit parameter sharing')
    literal_tokens=encode_body(concrete); parameter_tokens=encode_body(parameterized)
    if parse(decode_body(literal_tokens))!=parse(concrete) or parse(decode_body(parameter_tokens),allow_parameters=True)!=p:
        raise ValueError('codec roundtrip mismatch')
    a=evaluate(concrete,indices,limits); b=evaluate(final,indices,limits)
    if a.outcome!='ok' or b.outcome!='ok' or a.values!=b.values: raise ValueError('independent grounding execution failed')
    return dict(disposition='unsupported_control_dependency' if _control_taint(p) else 'supported',
                literal_source=concrete,parameter_source=parameterized,final_source=final,assignment=dict(assignment),
                literal_tokens=literal_tokens,parameter_tokens=parameter_tokens,parameter_count=len(assignment),sharing=sharing,
                protected_parameters=sorted(_control_taint(p)),fit_class='nonlinear_bounded_smt' if nonlinear or p.statements else 'affine_or_bounded_smt',
                verification=dict(indices=list(indices),values=list(a.values),scope='finite exact agreement'))


def paired_learning_view(concrete,parameterized,assignment,*,indices,sharing,limits=ResourceProfile()):
    """PAC retains unsupported *whole rows*; malformed pair evidence still fails.

    An over-cap proposal is grounded only to audit its original literal. It is
    never encoded as a parameter target or submitted to the at-most-eight fitter.
    Existing pair()/ground() and all historical callers keep their strict limits.
    """
    if sharing not in {'explicit','distinct-occurrences'}:raise ValueError('registered sharing policy required')
    if len(assignment)<=8:
        proof=pair(concrete,parameterized,assignment,indices=indices,sharing=sharing,limits=limits)
        source,tokens=learning_target(proof)
        return dict(proof,target_source=source,target_tokens=tokens)
    import re
    if not isinstance(parameterized,str) or len(parameterized.encode())>limits.source_bytes:
        raise ValueError('source limit')
    counts={}
    def replace(match):
        name=match.group(1);counts[name]=counts.get(name,0)+1
        if name not in assignment:raise ValueError('missing grounding')
        canonical_integer(assignment[name]);return '(int '+assignment[name]+')'
    grounded=re.sub(r'\(\s*param\.get\s+(p(?:0|[1-9][0-9]*))\s*\)',replace,parameterized,flags=re.ASCII)
    if set(counts)!=set(assignment) or sharing=='distinct-occurrences' and any(n>1 for n in counts.values()):
        raise ValueError('assignment/parameter sharing mismatch')
    if parse(grounded,limits)!=parse(concrete,limits):raise ValueError('grounding is not paired concrete program')
    result=evaluate(concrete,indices,limits)
    if result.outcome!='ok':raise ValueError('literal fallback execution failed')
    tokens=encode_body(concrete)
    if parse(decode_body(tokens),limits)!=parse(concrete,limits):raise ValueError('literal codec roundtrip mismatch')
    return dict(disposition='unsupported_parameter_count',literal_source=concrete,parameter_source=parameterized,
                final_source=concrete,assignment=dict(assignment),sharing=sharing,parameter_count=len(assignment),
                literal_tokens=tokens,parameter_tokens=None,target_source=concrete,target_tokens=tokens,
                protected_parameters=[],fit_class='literal_fallback',
                verification=dict(indices=list(indices),values=list(result.values),scope='finite exact agreement'))
