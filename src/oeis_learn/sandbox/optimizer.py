"""Small typed rewrite catalog preserving writes, traps and control ordering."""
from dataclasses import replace
import math
from oeis_learn.data.models import CanonicalProgramArtifact
from oeis_learn.decoder.wat_grammar import tokenize_wat
from oeis_learn.sandbox.wat_ast import parse, ExecutionFailure


def _rewrite(nodes):
    result=[]; index=0
    while index<len(nodes):
        node=nodes[index]
        if node.op=='nop': index+=1; continue
        # These pushes are total and effect-free in the validated typed scope.
        if node.op in ('i32.const','i64.const','local.get') and index+1<len(nodes) and nodes[index+1].op=='drop': index+=2; continue
        result.append(replace(node,body=_rewrite(node.body),otherwise=_rewrite(node.otherwise)))
        index+=1
    return tuple(result)


def _render(nodes):
    words=[]
    for node in nodes:
        if node.op in ('block','loop','if'):
            result=' ( result '+' '.join(node.results)+' )' if node.results else ''
            if node.op=='if':
                body=' ( then '+_render(node.body)+' )'
                if node.has_else: body+=' ( else '+_render(node.otherwise)+' )'
            else: body=' '+_render(node.body)
            words.append('( '+node.op+result+body+' )')
        else: words.append(node.op+(' '+str(node.arg) if node.arg is not None else ''))
    return ' '.join(words)


def optimize_wat_program(wat_code,hard_waste_threshold=0.30):
    if isinstance(hard_waste_threshold,bool) or not isinstance(hard_waste_threshold,(float,int)) or not math.isfinite(hard_waste_threshold) or not 0<=hard_waste_threshold<=1: raise ValueError('waste threshold must be finite in0..1')
    optimized=wat_code; passes=[]
    try:
        program=parse(wat_code); rewritten=_rewrite(program.instructions)
        if rewritten!=program.instructions:
            optimized=parse(_render(rewritten)).canonical_source
            passes=['typed-nop-and-pure-drop-v1']
    except ExecutionFailure:
        # Unsupported legacy/full-module rewrites return exactly the original bytes.
        pass
    raw_count=len(tokenize_wat(wat_code)); opt_count=len(tokenize_wat(optimized))
    waste=max(0,raw_count-opt_count)/max(1,raw_count)
    return CanonicalProgramArtifact(wat_code,optimized,raw_count,opt_count,waste,passes,waste>hard_waste_threshold)
