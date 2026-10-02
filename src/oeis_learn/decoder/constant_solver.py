"""Compatibility APIs routed through the shared contained grounding service."""
from oeis_learn.data.models import ASTSkeleton, ConstantSolverResult, GroundedCandidate
from oeis_learn.decoder.wat_grammar import tokenize_wat
from oeis_learn.decoder.grounding import ground, affine, splice, source_hash


def parse_ast_placeholders(wat_code):
    indices = [i for i,w in enumerate(tokenize_wat(wat_code)) if w in ('i64.const_?','i256.const_?')]
    return ASTSkeleton(raw_wat=wat_code, placeholder_count=len(indices), is_linear=affine(wat_code), placeholder_indices=indices, basis_signatures=[])


def abstract_wat_constants(wat_code, max_placeholders=4):
    """Only abstract whole logical values; scalar helper and counter literals are structural."""
    if type(max_placeholders) is not int or not 0 <= max_placeholders <= 8: raise ValueError('parameter cap')
    words=tokenize_wat(wat_code); result=[]; i=count=0
    while i<len(words):
        if words[i]=='i256.const' and count<max_placeholders and i+1<len(words):
            result.append('i256.const_?'); i+=2; count+=1
        else: result.append(words[i]); i+=1
    return ' '.join(result)


def splice_constants_into_wat(skeleton, constants):
    if len(constants)!=skeleton.placeholder_count or any(type(c) is not int for c in constants): raise ValueError('exact parameter count and integer values required')
    return splice(skeleton.raw_wat, constants)


def _legacy_result(result):
    return ConstantSolverResult(solver_type=result.method, constants=list(result.constants), solve_duration_ms=result.elapsed_ms, is_sat=result.outcome=='verified_solution', grounded_wat=result.source if result.outcome=='verified_solution' else None, error_message=result.reason, outcome=result.outcome, evidence=result.to_dict())


def solve_linear_diophantine(skeleton, terms, runner=None):
    if not affine(skeleton.raw_wat):
        return ConstantSolverResult(solver_type='not_run', outcome='unsupported', error_message='typed affinity was not established')
    return _legacy_result(ground(skeleton.raw_wat, terms))


def solve_smt_constants(skeleton, terms, timeout_ms=250, runner=None):
    return _legacy_result(ground(skeleton.raw_wat, terms, timeout_ms=timeout_ms))


def solve_constants(skeleton, terms, runner=None, timeout_ms=240, bound=1000):
    result=ground(skeleton.raw_wat, terms, timeout_ms=timeout_ms, bound=bound)
    return GroundedCandidate(skeleton_id=source_hash(skeleton.raw_wat), constants=list(result.constants), solver_tier=result.method, is_sat=result.outcome=='verified_solution', solve_duration_ms=result.elapsed_ms, grounded_wat=result.source if result.outcome=='verified_solution' else None, outcome=result.outcome, evidence=result.to_dict())


def resolve_program_constants(wat_code, terms, timeout_ms=250, max_placeholders=4, runner=None):
    if sum(w in ('i64.const_?','i256.const_?') for w in tokenize_wat(wat_code))>max_placeholders:
        return wat_code, [], 'UNSUPPORTED', 0.0, 'parameter cap'
    result=ground(wat_code, terms, timeout_ms=timeout_ms)
    return result.source, list(result.constants), 'PASSED' if result.outcome=='verified_solution' else result.outcome.upper(), result.elapsed_ms, result.reason
