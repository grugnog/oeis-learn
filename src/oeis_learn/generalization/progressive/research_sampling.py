"""Outcome-blind, support-conditioned *evaluation* draws for preparation v3.

This is a new finite sampling commitment, never a learner, a replacement search,
or an inference about model performance. Native/reference validation and the
original exposure detectors still decide eligibility for every attempted draw.
"""
import random
import re
from .contracts import digest
from .generator import structure_id


def source_eligibility(row):
    """Unsupported exact integers are ineligible; malformed originals are errors."""
    from oeis_learn.synthesis.contracts import canonical_integer, ContractError
    if (not isinstance(row,dict) or not isinstance(row.get('id'),str)
            or len(row.get('indices',[]))!=100 or len(row.get('values',[]))!=100):
        raise ValueError('complete original indexed source required')
    for value in row['indices']+row['values']:
        if not isinstance(value,str) or re.fullmatch(r'0|-?[1-9][0-9]*',value,flags=re.ASCII) is None:
            raise ValueError('noncanonical original integer')
    # Offset/index errors are integrity errors, not value-profile exclusions.
    indices=[canonical_integer(v) for v in row['indices']]
    if indices!=list(range(indices[0],indices[0]+100)):
        raise ValueError('original source indices are not consecutive')
    try:
        for value in row['values']:canonical_integer(value)
    except ContractError as exc:
        if str(exc) not in {'integer length limit','integer bit limit'}:raise
        return 'outside_bigint_4096_profile'
    return None


class Sampler:
    """Alternate parameter variants and new combinations of seen subexpressions.

    No evaluation row, OEIS value, metadata or model outcome is an input. Literal
    variants use a broader predeclared range than the exhausted [-8,8] training
    law. Control counts remain <=4 and branch thresholds stay within [1,99].
    Every counter is attempted once, including duplicate/unsafe draws.
    """
    def __init__(self,support,*,seed):
        from oeis_learn.synthesis.reference_parser import parse
        from oeis_learn.data.bigint_programs import render
        probes={p['id'] for p in support['layout']['probes'].values()}
        rows=[r for r in support['shards'][0] if r['draw_identity'] not in probes]
        self.sources=sorted({r['source'] for r in rows})
        expressions=set()
        def pure(n):
            return n=='$index' if isinstance(n,str) else (n[0] in {'int','add','sub','mul','select','lt','le','eq'}
                and (n[0]=='int' or all(pure(c) for c in n[1:])))
        def visit(n):
            if not isinstance(n,tuple):return
            if n[0] in {'int','add','sub','mul','select'} and pure(n):expressions.add(render(n))
            for c in n[1:]:visit(c)
        operators=set()
        for s in self.sources:
            p=parse(s)
            for node in (*p.statements,p.result):visit(node)
            operators.update(re.findall(r'\((add|sub|mul)\b',s))
        self.expressions=sorted(expressions);self.operators=sorted(operators)
        if not self.sources or not self.expressions or not self.operators:
            raise ValueError('common initial support lacks composable arithmetic expressions')
        self.identity=digest(dict(version='support-conditioned-evaluation/v1',support=support['identity'],
            seed=seed,sources=self.sources,expressions=self.expressions,operators=self.operators,
            literal_range=[-65536,65536],control_range=[1,4],guard_range=[1,99],
            draw_limit=32768,law='alternate parameter mutation and binary composition; no replacement'))

    def draw(self,ordinal):
        from oeis_learn.synthesis.reference_parser import parse
        from oeis_learn.data.bigint_programs import render
        if type(ordinal) is not int or not 0<=ordinal<32768:raise ValueError('finite evaluation counter required')
        rng=random.Random(int(digest([self.identity,ordinal])[7:],16))
        if ordinal%2==0:source=rng.choice(self.sources)
        else:
            source='(program (return ('+rng.choice(self.operators)+' '+rng.choice(self.expressions)+' '+rng.choice(self.expressions)+')))'
        p=parse(source)
        def mutate(n,mode=None):
            if not isinstance(n,tuple):return n
            if n[0]=='int':
                lo,hi=(1,4) if mode=='control' else (1,99) if mode=='guard' else (-65536,65536)
                return ('int',str(rng.randint(lo,hi)))
            child_mode='control' if n[0]=='to_control' else 'guard' if n[0] in {'lt','le','eq'} else mode
            return (n[0],*(mutate(c,child_mode) for c in n[1:]))
        tree=('program',*(('local',n,k) for n,k in p.declarations),
              *(mutate(s) for s in p.statements),('return',mutate(p.result)))
        source=render(tree)
        return dict(ordinal=ordinal,source=source,structure=structure_id(source),
            source_policy=self.identity,draw_identity=digest([self.identity,ordinal]),
            detector_limit='support-conditioned finite parameter/composition census; exact finite exclusions; no semantic independence claim')
