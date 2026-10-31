"""Observed bounded reference execution, not syntax-as-coverage."""
from dataclasses import dataclass
from collections import Counter
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.contracts import ResourceProfile, canonical_integer
from oeis_learn.synthesis.reference_parser import parse

@dataclass(frozen=True)
class Coverage:
    stage: str
    source_bytes: int
    nodes: int
    loop_iterations: int
    true_branches: int
    false_branches: int
    above_64_bits: bool
    composition: tuple[tuple[str,int], ...]

    @property
    def loop_active(self): return self.loop_iterations>0

    @property
    def branch_active(self): return self.true_branches>0 and self.false_branches>0


def observe(source, indices, *, stage, limits=ResourceProfile()):
    if stage not in ('A','B','C','D'): raise ValueError('unknown mechanical stage')
    counts=Counter()
    def observer(kind,value):
        if kind=='loop': counts['loops']+=value
        elif kind=='branch': counts['true' if value else 'false']+=1
    result=evaluate(source,indices,limits,observer=observer)
    if result.outcome!='ok': raise ValueError('coverage execution failed: '+str(result.reason))
    program=parse(source,limits)
    ops=Counter()
    def visit(node):
        if isinstance(node,tuple):
            ops[node[0]]+=1
            for child in node[1:]: visit(child)
    for node in (*program.statements, program.result): visit(node)
    return Coverage(stage,len(source.encode()),sum(ops.values()),counts['loops'],counts['true'],counts['false'],
                    any(abs(canonical_integer(v,bits=limits.intermediate_bits)).bit_length()>64 for v in result.values),tuple(sorted(ops.items())))


def distribution(observations, *, length_boundary):
    if type(length_boundary) is not int or length_boundary<1: raise ValueError('frozen length bin required')
    counts=Counter()
    for c in observations:
        for tag in (c.stage,'long' if c.nodes>=length_boundary else 'short','above64' if c.above_64_bits else 'within64'):
            counts[tag]+=1
        counts['loop_active']+=c.loop_active; counts['branch_active']+=c.branch_active
    return dict(counts)
